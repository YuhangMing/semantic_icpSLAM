#include "Mapping.h"
#include "Constant.h"
#include "Reduction.h"
#include "RenderScene.h"

Mapping::Mapping() :
		meshUpdated(false), hasNewKFFlag(false) {
	Create();
}

void Mapping::Create() {

	heapCounter.create(1);
	hashCounter.create(1);
	noVisibleEntries.create(1);
	heap.create(DeviceMap::NumSdfBlocks);
	sdfBlock.create(DeviceMap::NumVoxels);
	bucketMutex.create(DeviceMap::NumBuckets);
	hashEntries.create(DeviceMap::NumEntries);
	visibleEntries.create(DeviceMap::NumEntries);

	nBlocks.create(1);
	noTriangles.create(1);
	modelVertex.create(DeviceMap::MaxVertices);
	modelNormal.create(DeviceMap::MaxVertices);
	modelColor.create(DeviceMap::MaxVertices);
	blockPoses.create(DeviceMap::NumEntries);

	edgeTable.create(256);
	vertexTable.create(256);
	triangleTable.create(16, 256);
	edgeTable.upload(edgeTableHost);
	vertexTable.upload(vertexTableHost);
	triangleTable.upload(triangleTableHost);

	zRangeMin.create(80, 60);
	zRangeMax.create(80, 60);
	zRangeMinEnlarged.create(160, 120);
	zRangeMaxEnlarged.create(160, 120);
	noRenderingBlocks.create(1);
	renderingBlockList.create(DeviceMap::MaxRenderingBlocks);

	noKeys.create(1);
	mutexKeys.create(KeyMap::MaxKeys);
	mapKeys.create(KeyMap::maxEntries);
	tmpKeys.create(KeyMap::maxEntries);
	surfKeys.create(2000);
	mapKeyIndex.create(2000);

	Reset();
}

void Mapping::ForwardWarp(const Frame * last, Frame * next) {
	ForwardWarping(last->vmap[0], last->nmap[0], next->vmap[0], next->nmap[0],
			last->GpuRotation(), next->GpuInvRotation(), last->GpuTranslation(),
			next->GpuTranslation(), Frame::fx(0), Frame::fy(0), Frame::cx(0),
			Frame::cy(0));
}

void Mapping::UpdateVisibility(const Frame * f, uint & no) {

	CheckBlockVisibility(*this, noVisibleEntries, f->GpuRotation(), f->GpuInvRotation(),
			f->GpuTranslation(), Frame::cols(0), Frame::rows(0), Frame::fx(0),
			Frame::fy(0), Frame::cx(0), Frame::cy(0), DeviceMap::DepthMax,
			DeviceMap::DepthMin, &no);
}

void Mapping::UpdateVisibility(Matrix3f Rview, Matrix3f RviewInv, float3 tview,
		float depthMin, float depthMax, float fx, float fy, float cx, float cy,
		uint & no) {

	CheckBlockVisibility(*this, noVisibleEntries, Rview, RviewInv, tview, 640,
			480, fx, fy, cx, cy, depthMax, depthMin, &no);
}

void Mapping::FuseColor(const Frame * f, uint & no) {
	FuseColor(f->range, f->color, f->nmap[0], 
			  f->detected_masks, f->detected_labels,
			  f->numDetection,
			  f->GpuRotation(), f->GpuInvRotation(), f->GpuTranslation(), no);
}

void Mapping::FuseColor(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color,
		const DeviceArray2D<float4> & normal,
		const DeviceArray2D<unsigned char> & mask,
		const DeviceArray<int> & labels,
		int numDetection,
		Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, uint & no) {

	FuseMapColor(depth, color, normal, mask, labels, numDetection, 
			noVisibleEntries, Rview, RviewInv, tview, *this,
			Frame::fx(0), Frame::fy(0), Frame::cx(0), Frame::cy(0),
			DeviceMap::DepthMax, DeviceMap::DepthMin, &no);

}

void Mapping::SemanticAnalysis(const Frame *f) {
	DeviceArray2D<float> renderedDepth(f->depth[0].cols, f->depth[0].rows);
	// difference between raw and rendered depth is at 10^-3 level
	// Use rendered depth in stead of raw depth below.
	cv::Mat mVertex(f->depth[0].rows, f->depth[0].cols, CV_32FC4);
	f->vmap[0].download(mVertex.data, mVertex.step);
	cv::Mat mDepths[4];
	cv::split(mVertex, mDepths);
	renderedDepth.upload(mDepths[2].data, mDepths[2].step);
	// cv::Mat mRawDepth(f->depth[0].rows, f->depth[0].cols, CV_32FC1);
	// f->range.download(mRawDepth.data, mRawDepth.step);
	// cv::Mat mRenderDepth(f->depth[0].rows, f->depth[0].cols, CV_32FC1);
	// renderedDepth.download(mRenderDepth.data, mRenderDepth.step);
	// std::cout << "-- Compare the rendered depth with raw depth: " << std::endl;
	// for(int i=360; i<370; i++){
	// 	// std::cout << mDepths[2].at<float>(240, i) << " - "  << mRawDepth.at<float>(240,i) << std::endl;
	// 	std::cout << mRenderDepth.at<float>(240, i) << " - "  << mRawDepth.at<float>(240,i) << std::endl;
	// }
	SemanticAnalysis(renderedDepth, f->color, f->nmap[0], 
					 f->detected_masks, f->detected_labels,
					 f->numDetection,
					 f->GpuRotation(), f->GpuInvRotation(), f->GpuTranslation());
	// SemanticAnalysis(f->range, f->color, f->nmap[0], 
	// 				 f->detected_masks, f->detected_labels,
	// 				 f->numDetection,
	// 				 f->GpuRotation(), f->GpuInvRotation(), f->GpuTranslation());
}

void Mapping::SemanticAnalysis(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color,
		const DeviceArray2D<float4> & normal, 
		const DeviceArray2D<unsigned char> & mask,
		const DeviceArray<int> & labels,
		int numDetection,
		Matrix3f Rview, Matrix3f RviewInv, float3 tview) {

	AnalyzeMapSemantics(depth, color, normal, mask, labels, numDetection, 
			noVisibleEntries, Rview, RviewInv, tview, *this,
			Frame::fx(0), Frame::fy(0), Frame::cx(0), Frame::cy(0),
			DeviceMap::DepthMax, DeviceMap::DepthMin);

}

void Mapping::RayTrace(uint noVisibleBlocks, Frame * f) {
	RayTrace(noVisibleBlocks, f->GpuRotation(), f->GpuInvRotation(), f->GpuTranslation(),
			f->vmap[0], f->nmap[0], DeviceMap::DepthMin, DeviceMap::DepthMax,
			Frame::fx(0), Frame::fy(0), Frame::cx(0), Frame::cy(0));
}

void Mapping::RayTrace(uint noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, DeviceArray2D<float4> & vmap,	DeviceArray2D<float4> & nmap,
		float depthMin, float depthMax, float fx, float fy, float cx, float cy) {

	if (CreateRenderingBlocks(visibleEntries, 
			zRangeMin, zRangeMax, 
			depthMax, depthMin,
			renderingBlockList, noRenderingBlocks, 
			RviewInv, tview, noVisibleBlocks, fx, fy, cx, cy)) {

		Raycast(*this, vmap, nmap, zRangeMin, zRangeMax, Rview, RviewInv, tview,
				1.0 / fx, 1.0 / fy, cx, cy);
	}
}

std::vector<KeyFrame *> Mapping::LocalMap() const {

	std::vector<KeyFrame *> tmp;
	std::vector<const KeyFrame *>::const_iterator iter = localMap.begin();
	std::vector<const KeyFrame *>::const_iterator lend = localMap.end();
	for(; iter != lend; ++iter)
		tmp.push_back(const_cast<KeyFrame *>(*iter));
	return tmp;
}

std::vector<KeyFrame *> Mapping::GlobalMap() const {

	std::vector<KeyFrame *> tmp;
	std::set<const KeyFrame *>::const_iterator iter = keyFrames.begin();
	std::set<const KeyFrame *>::const_iterator lend = keyFrames.end();
	for(; iter != lend; ++iter)
		tmp.push_back(const_cast<KeyFrame *>(*iter));
	return tmp;
}

void Mapping::CreateModel() {

	MeshScene(nBlocks, noTriangles, *this, edgeTable, vertexTable,
			triangleTable, modelNormal, modelVertex, modelColor, blockPoses);

	noTriangles.download(&noTrianglesHost);
	if (noTrianglesHost > 0) {
		meshUpdated = true;
	}
}

void Mapping::UpdateMapKeys() {
	noKeys.clear();
	CollectKeyPoints(*this, tmpKeys, noKeys);

	noKeys.download(&noKeysHost);
	if(noKeysHost != 0) {
		hostKeys.resize(noKeysHost);
		tmpKeys.download(hostKeys.data(), noKeysHost);
	}
}

void Mapping::CreateRAM() {

	heapCounterRAM = new int[1];
	hashCounterRAM = new int[1];
	noVisibleEntriesRAM = new uint[1];
	heapRAM = new int[DeviceMap::NumSdfBlocks];
	bucketMutexRAM = new int[DeviceMap::NumBuckets];
	sdfBlockRAM = new Voxel[DeviceMap::NumVoxels];
	hashEntriesRAM = new HashEntry[DeviceMap::NumEntries];
	visibleEntriesRAM = new HashEntry[DeviceMap::NumEntries];

	mutexKeysRAM = new int[KeyMap::MaxKeys];
	mapKeysRAM = new SURF[KeyMap::maxEntries];
}

void Mapping::DownloadToRAM() {

	CreateRAM();

	heapCounter.download(heapCounterRAM);
	hashCounter.download(hashCounterRAM);
	noVisibleEntries.download(noVisibleEntriesRAM);
	heap.download(heapRAM);
	bucketMutex.download(bucketMutexRAM);
	sdfBlock.download(sdfBlockRAM);
	hashEntries.download(hashEntriesRAM);
	visibleEntries.download(visibleEntriesRAM);

	mutexKeys.download(mutexKeysRAM);
	mapKeys.download(mapKeysRAM);
}

void Mapping::UploadFromRAM() {

	heapCounter.upload(heapCounterRAM);
	hashCounter.upload(hashCounterRAM);
	noVisibleEntries.upload(noVisibleEntriesRAM);
	heap.upload(heapRAM);
	bucketMutex.upload(bucketMutexRAM);
	sdfBlock.upload(sdfBlockRAM);
	hashEntries.upload(hashEntriesRAM);
	visibleEntries.upload(visibleEntriesRAM);

	mutexKeys.upload(mutexKeysRAM);
	mapKeys.upload(mapKeysRAM);
}

void Mapping::ReleaseRAM() {

	delete [] heapCounterRAM;
	delete [] hashCounterRAM;
	delete [] noVisibleEntriesRAM;
	delete [] heapRAM;
	delete [] bucketMutexRAM;
	delete [] sdfBlockRAM;
	delete [] hashEntriesRAM;
	delete [] visibleEntriesRAM;

	delete [] mutexKeysRAM;
	delete [] mapKeysRAM;
}

bool Mapping::HasNewKF() {

	return hasNewKFFlag;
}

void Mapping::FuseKeyFrame(const KeyFrame * kf) {

	if (keyFrames.count(kf))
		return;

	keyFrames.insert(kf);
	hasNewKFFlag = true;

	std::cout << keyFrames.size() << std::endl;

	cv::Mat desc;
	std::vector<int> index;
	std::vector<int> keyIndex;
	std::vector<SURF> keyChain;
	kf->descriptors.download(desc);
	kf->outliers.resize(kf->N);
	std::fill(kf->outliers.begin(), kf->outliers.end(), true);
	int noK = std::min(kf->N, (int) surfKeys.size);

	for (int i = 0; i < noK; ++i) {

		if (kf->observations[i] > 0) {

			SURF key;
			Eigen::Vector3f pt = kf->GetWorldPoint(i);
			key.pos = { pt(0), pt(1), pt(2) };
			key.normal = kf->pointNormal[i];
			key.valid = true;

			for (int j = 0; j < 64; ++j) {
				key.descriptor[j] = desc.at<float>(i, j);
			}

			index.push_back(i);
			keyChain.push_back(key);
			keyIndex.push_back(kf->keyIndex[i]);
			kf->outliers[i] = false;
		}
	}

	surfKeys.upload(keyChain.data(), keyChain.size());
	mapKeyIndex.upload(keyIndex.data(), keyIndex.size());

	InsertKeyPoints(*this, surfKeys, mapKeyIndex, keyChain.size());

	mapKeyIndex.download(keyIndex.data(), keyIndex.size());
	surfKeys.download(keyChain.data(), keyChain.size());

	for(int i = 0; i < index.size(); ++i) {
		int idx = index[i];
		kf->keyIndex[idx] = keyIndex[i];
		float3 pos = keyChain[i].pos;
		kf->mapPoints[idx] << pos.x, pos.y, pos.z;
	}

	if(localMap.size() > 0) {
		if(localMap.size() >= 7) {
			localMap.erase(localMap.begin());
			localMap.push_back(kf);
		}
		else
			localMap.push_back(kf);
	}
	else
		localMap.push_back(kf);
}

void Mapping::FuseKeyPoints(const Frame * f) {

	std::cout << "NOT IMPLEMENTED" << std::endl;
}

void Mapping::Reset() {

	ResetMap(*this);
	ResetKeyPoints(*this);

	mapKeys.clear();
	keyFrames.clear();
}

Mapping::operator KeyMap() const {

	KeyMap map;

	map.Keys = mapKeys;
	map.Mutex = mutexKeys;

	return map;
}

Mapping::operator DeviceMap() const {

	DeviceMap map;

	map.heapMem = heap;
	map.heapCounter = heapCounter;
	map.noVisibleBlocks = noVisibleEntries;
	map.bucketMutex = bucketMutex;
	map.hashEntries = hashEntries;
	map.visibleEntries = visibleEntries;
	map.voxelBlocks = sdfBlock;
	map.entryPtr = hashCounter;

	return map;
}
