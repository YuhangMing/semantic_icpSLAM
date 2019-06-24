#include "Solver.h"
#include "Tracking.h"
#include "sophus/se3.hpp"

#include "cuda_runtime.h"
// #define CUDA_MEM

using namespace cv;

Matrix3f eigen_to_mat3f(Eigen::Matrix3d & mat) {
	Matrix3f mat3f;
	mat3f.rowx = make_float3((float) mat(0, 0), (float) mat(0, 1), (float)mat(0, 2));
	mat3f.rowy = make_float3((float) mat(1, 0), (float) mat(1, 1), (float)mat(1, 2));
	mat3f.rowz = make_float3((float) mat(2, 0), (float) mat(2, 1), (float)mat(2, 2));
	return mat3f;
}

float3 eigen_to_float3(Eigen::Vector3d & vec) {
	return make_float3((float) vec(0), (float) vec(1), (float) vec(2));
}

Tracker::Tracker(int cols_, int rows_, float fx, float fy, float cx, float cy) :
		map(NULL), viewer(NULL), noInliers(0), mappingTurnedOff(NULL),
		state(1), lastState(1), noMissedFrames(0), useGraphMatching(false),
		imageUpdated(false), mappingDisabled(false), semanticEnabled(false),
		needImages(false), ReferenceKF(NULL), LastKeyFrame(NULL) {

	renderedImage.create(cols_, rows_);
	renderedDepth.create(cols_, rows_);
	rgbaImage.create(cols_, rows_);

	edgeImage.create(cols_, rows_);
	ccImage.create(cols_, rows_);
	fuseImage.create(cols_, rows_);

	sumSE3.create(29, 96);
	outSE3.create(29);
	sumSO3.create(11, 96);
	outSO3.create(11);
	sumRes.create(2, 96);
	outRes.create(2);

	K = Intrinsics(fx, fy, cx, cy);
	matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L2);

	NextFrame = new Frame();
	LastFrame = new Frame();
	NextFrame->Create(cols_, rows_);
	LastFrame->Create(cols_, rows_);
}

void Tracker::ResetTracking() {

	state = lastState = 1;
	ReferenceKF = LastKeyFrame = NULL;
	nextPose = Eigen::Matrix4d::Identity();
	lastPose = Eigen::Matrix4d::Identity();
	NextFrame->pose = nextPose;
	LastFrame->pose = lastPose;
}

//-----------------------------------------
// Main Control Flow
//-----------------------------------------
bool Tracker::Track() {

	bool valid = false;
	std::swap(state, lastState);
	if(needImages) {
		RenderView();
	}

	switch(state) {
	case 1:
		InitTracking();
		SwapFrame();
		lastState = 0;
		return true;

	case 0:
		valid = TrackFrame();

		if(valid) {
			noMissedFrames = 0;
			if(lastState != -1) {
				lastState = 0;
				if(NeedKeyFrame())
					CreateKeyFrame();
				else
					CheckOutliers();
				SwapFrame();
				return true;
			}

			lastState = 0;
			return false;
		}

		lastState = 0;
		noMissedFrames++;
		if(noMissedFrames > 9) {
			lastState = -1;
		}

		return false;

	case -1:
		valid = Relocalise();

		if(valid) {
			lastState = 0;
			SwapFrame();
			return true;
		}

		lastState = -1;
		return false;
	}
}

void Tracker::CheckOutliers() {

	// to prevent system from crushing
	// when have loaded a pre-built map
	// TODO: get rid of this dirty hack
	if(!ReferenceKF)
		return;

	Eigen::Matrix4f deltaT = ReferenceKF->pose.inverse() * NextFrame->pose.cast<float>();
	Eigen::Matrix3f deltaR = deltaT.topLeftCorner(3, 3);
	Eigen::Vector3f deltat = deltaT.topRightCorner(3, 1);

	std::vector<cv::DMatch> matches;
	matcher->match(NextFrame->descriptors, ReferenceKF->descriptors, matches);
	for(int i = 0; i < matches.size(); ++i) {
		Eigen::Vector3f src = NextFrame->mapPoints[matches[i].queryIdx];
		Eigen::Vector3f ref = ReferenceKF->mapPoints[matches[i].trainIdx];
		double d = (src - (deltaR * ref + deltat)).norm();
		if (d <= 0.05f) {
			ReferenceKF->observations[matches[i].trainIdx]++;
		}
	}
}

void Tracker::RenderView() {
	// store map model / raw depth / input image into DeviceArray2D<uchar4> for GPU
	if (updateImageMutex.try_lock()) {
		if (state == 0 && lastState != -1)
			RenderImage(LastFrame->vmap[0], LastFrame->nmap[0], make_float3(0), renderedImage);
		DepthToImage(LastFrame->range, renderedDepth);
		RgbImageToRgba(LastFrame->color, rgbaImage);
		imageUpdated = true;
		updateImageMutex.unlock();
	}
}

bool Tracker::TrackFrame() {
	// MAIN tracking process
	bool valid = false;

	// Feature based tracking is removalbe !!!!!!!!!!!!!!!
	// valid = TrackLastFrame();

	// if(!valid) {
	// 	std::cout << "Bootstrap failed. Rolling back." << std::endl;
		NextFrame->pose = LastFrame->pose;
	// }

    //	ComputeSO3();
	valid = ComputeSE3(false, ITERATIONS_SE3, THRESH_ICP_SE3);
	// valid = ComputeSE3(true, ITERATIONS_SE3, THRESH_ICP_SE3);
	return valid;
}

//-----------------------------------------
// Track Current Frame w.r.t Last Frame
// Feature Point based tracking
//-----------------------------------------
bool Tracker::TrackLastFrame() {

	refined.clear();
	std::vector<std::vector<cv::DMatch>> rawMatches;
	matcher->knnMatch(NextFrame->descriptors, LastFrame->descriptors, rawMatches, 2);
	for (int i = 0; i < rawMatches.size(); ++i) {
		if (rawMatches[i][0].distance < 0.80 * rawMatches[i][1].distance) {
			refined.push_back(rawMatches[i][0]);
		}
	}

	noInliers = refined.size();
	if (noInliers < 3)
		return false;

	refPoints.clear();
	framePoints.clear();
	for (int i = 0; i < noInliers; ++i) {
		framePoints.push_back(NextFrame->mapPoints[refined[i].queryIdx].cast<double>());
		refPoints.push_back(LastFrame->mapPoints[refined[i].trainIdx].cast<double>());
	}

	Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
	NextFrame->outliers.resize(refined.size());
	std::fill(NextFrame->outliers.begin(), NextFrame->outliers.end(), true);

	bool result = Solver::PoseEstimate(framePoints, refPoints, NextFrame->outliers, delta, maxIter, true);
	noInliers = std::count(outliers.begin(), outliers.end(), false);

	if (result) {
		nextPose = delta.inverse() * LastFrame->pose;
		NextFrame->pose = nextPose;
	}

	return result;
}

bool Tracker::NeedKeyFrame() {

	if(mappingDisabled)
		return false;

	Eigen::Matrix4f dT = NextFrame->pose.cast<float>() * ReferenceKF->pose.inverse();
	Eigen::Matrix3f dR = dT.topLeftCorner(3, 3);
	Eigen::Vector3f dt = dT.topRightCorner(3, 1);
	Eigen::Vector3f angle = dR.eulerAngles(0, 1, 2).array().sin();
	if (angle.norm() > 0.1 || dt.norm() > 0.1)
		return true;

	return false;
}

void Tracker::CreateKeyFrame() {
	// KeyFrame here is used to store the SURF feature points in the map
	if (ReferenceKF)
		map->FuseKeyFrame(ReferenceKF);
	std::swap(ReferenceKF, LastKeyFrame);
	ReferenceKF = new KeyFrame(NextFrame);
}

void Tracker::InitTracking() {

	ResetTracking();	// set pose of current and next frame to Identity
	CreateKeyFrame();	// set NextFrame to ReferenceKF
	return;
}

bool Tracker::GrabFrame(const cv::Mat &image, const cv::Mat &depth) {

	LastFrame->ResizeImages();
	NextFrame->ClearKeyPoints();
	NextFrame->FillImages(depth, image);
	NextFrame->ExtractKeyPoints();

	// #ifdef CUDA_MEM
	// 	size_t free_t, total_t;
	// 	cudaMemGetInfo(&free_t, &total_t);
	// 	float free_m1, free_m2, total_m, used_m;
	// 	free_m1 = (uint)free_t/1048576.0 ;
	// 	total_m = (uint)total_t/1048576.0 ;
	// #endif

	// 	if (semanticEnabled && nFrames % gap == 0) {
	// 		NextFrame->ExtractObjects(detector, false, true, true);
	// 	}
	// 	// NextFrame->ExtractObjects(detector, false, true, true);
	// 	// edgeImage.upload(NextFrame->mEdge.data, NextFrame->mEdge.step);

	// #ifdef CUDA_MEM	
	// 	cudaMemGetInfo(&free_t, &total_t);
	// 	free_m2 = (uint)free_t/1048576.0 ;
	// 	total_m = (uint)total_t/1048576.0 ;
	// 	used_m = free_m1 - free_m2;
	// 	std::cout << "## Perform 1 detection uses " << used_m << " MB memory." << std::endl
	// 			  << "   with " 
	// 			  << free_m1 << " MB free mem before, "
	// 			  << free_m2 << " MB free mem after" << std::endl
	// 			  << "   out of " << total_m << " MB total memroy." << std::endl;
	// #endif

	return Track();
}

void Tracker::SemanticAnalysis(float lamb, float tao, int win_size, int thre) {
	#ifdef CUDA_MEM
		size_t free_t, total_t;
		cudaMemGetInfo(&free_t, &total_t);
		float free_m1, free_m2, total_m, used_m;
		free_m1 = (uint)free_t/1048576.0 ;
		total_m = (uint)total_t/1048576.0 ;
	#endif

	LastFrame->ExtractObjects(detector, false, true, true);
	
	// NextFrame->ExtractObjects(detector, false, true, true);
	// edgeImage.upload(NextFrame->mEdge.data, NextFrame->mEdge.step);

	#ifdef CUDA_MEM	
		cudaMemGetInfo(&free_t, &total_t);
		free_m2 = (uint)free_t/1048576.0 ;
		total_m = (uint)total_t/1048576.0 ;
		used_m = free_m1 - free_m2;
		std::cout << "## Perform 1 detection uses " << used_m << " MB memory." << std::endl
				<< "   with " 
				<< free_m1 << " MB free mem before, "
				<< free_m2 << " MB free mem after" << std::endl
				<< "   out of " << total_m << " MB total memroy." << std::endl;
	#endif
	// #### MaskFusion - GeometricBoundaryRefinement ####
	// use last frame -> vmap, nmap
	LastFrame->GeometricRefinement(lamb, tao, win_size);
	ucharImageToRgba(LastFrame->edge, edgeImage);
	ucharImageToRgba(LastFrame->cc_labeled, ccImage);
	// fuse masks
	LastFrame->FuseMasks(thre);
	if(LastFrame->numDetection > 0)
		ucharImageToRgba(LastFrame->detected_masks, fuseImage);
}

void Tracker::SwapFrame() {
	std::swap(NextFrame, LastFrame);
}

void Tracker::ComputeSO3() {

	Eigen::Matrix<double, 3, 3, Eigen::RowMajorBit> matA;
	Eigen::Vector3d vecb;
	Eigen::Vector3d result;
	Eigen::Matrix3d lastRot = LastFrame->Rotation();
	Eigen::Matrix3d nextRot = NextFrame->Rotation();
	Eigen::Matrix3d rot = nextRot;
	Eigen::Matrix3d delta = Eigen::Matrix3d::Identity();

	float so3Error = 0;
	int so3Count = 0;
	int pyrLevel = 2;
	float lastSO3Error = std::numeric_limits<float>::max();

	for(int i = 0; i < 10; ++i) {

		SO3Step(NextFrame->image[pyrLevel],
				LastFrame->image[pyrLevel],
				NextFrame->dIdx[pyrLevel],
				NextFrame->dIdy[pyrLevel],
				NextFrame->GpuInvRotation(),
				LastFrame->GpuRotation(),
				K(pyrLevel),
				sumSO3,
				outSO3,
				so3Residual,
				matA.data(),
				vecb.data());

		so3Error = sqrt(so3Residual[0]) / so3Residual[1];
		so3Count = (int) so3Residual[1];

		if(lastSO3Error < so3Error)
			return;

		result = matA.ldlt().solve(vecb);
		auto e = Sophus::SO3d::exp(result);
		auto dT = e.matrix();

		delta = dT * delta;
		nextRot = lastRot * delta.inverse();
		NextFrame->pose.topLeftCorner(3, 3) = nextRot;
		lastSO3Error = so3Error;
	}
}

bool Tracker::ComputeSE3(bool icpOnly, const int * iter, const float thresh_icp) {

	// Formulate Ax=b 6x6 linear system
	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> matA;
	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> matA_icp;
	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> matA_rgb;
	Eigen::Matrix<double, 6, 1> vecb;
	Eigen::Matrix<double, 6, 1> vecb_icp;
	Eigen::Matrix<double, 6, 1> vecb_rgb;
	Eigen::Matrix<double, 6, 1> result;
	lastPose = LastFrame->pose;
	nextPose = NextFrame->pose;
	Eigen::Matrix4d pose = NextFrame->pose;

	float rgbError = 0;
	float icpError = 0;
	int rgbCount = 0;
	int icpCount = 0;

	Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
	int breakCount = 0;
	float lastError = 100.;
	float currError = 0.;
	float w = 1e-4;
	Eigen::Matrix4d candidate;
	float candidateICP;

	// loop through the 3 layer pyramid
	for(int i = Frame::NUM_PYRS - 1; i >= 0; --i) {
		for(int j = 0; j < iter[i]; ++j) {

			if(!icpOnly) {
				// TODO: convoluted transformation
				RGBStep(NextFrame->image[i],
						LastFrame->image[i],
						NextFrame->vmap[i],
						LastFrame->vmap[i],
						NextFrame->dIdx[i],
						NextFrame->dIdy[i],
						NextFrame->GpuRotation(),
						NextFrame->GpuInvRotation(),
						LastFrame->GpuRotation(),
						LastFrame->GpuInvRotation(),
						NextFrame->GpuTranslation(),
						LastFrame->GpuTranslation(),
						K(i),
						sumSE3,
						outSE3,
						sumRes,
						outRes,
						rgbResidual,
						matA_rgb.data(),
						vecb_rgb.data());

				rgbError = sqrt(rgbResidual[0]) / rgbResidual[1];
				rgbCount = (int) rgbResidual[1];

				if (std::isnan(rgbError) || rgbCount < MIN_ICP_COUNT[i]) {
					std::cout << "track rgb failed" << std::endl;
					NextFrame->pose = lastPose;
					return false;
				}
			}

			ICPStep(NextFrame->vmap[i],
					LastFrame->vmap[i],
					NextFrame->nmap[i],
					LastFrame->nmap[i],
					NextFrame->GpuRotation(),
					NextFrame->GpuTranslation(),
					LastFrame->GpuRotation(),
					LastFrame->GpuInvRotation(),
					LastFrame->GpuTranslation(),
					K(i),
					sumSE3,
					outSE3,
					icpResidual,
					matA_icp.data(),
					vecb_icp.data());

			icpError = sqrt(icpResidual[0]) / icpResidual[1];
			icpCount = (int) icpResidual[1];

			if (std::isnan(icpError) || icpCount < MIN_ICP_COUNT[i]) {
				std::cout << "track icp failed" << std::endl;
				NextFrame->pose = lastPose;
				return false;
			}

			// Prevent solution escape from optimal
			currError = rgbError*w + icpError;
			if(lastError < currError){
				breakCount ++;
				// store last pose as candidate
				if(breakCount == 1)
					candidate = NextFrame->pose;
					candidateICP = icpError;
			} else {
				breakCount=0;
			}
			if(breakCount > 2){
				NextFrame->pose = candidate;
				// std::cout << "stops at " << i << "/" << j-breakCount 
				// 		  << " with icp error = " << candidateICP
				// 		  << " / " << icpError
				// 		  << std::endl;
				icpError = candidateICP;
				lastIcpError = icpError;
				break;
			}
			lastError = currError;

			// solve for increment
			lastIcpError = icpError;

			if (!icpOnly) {
				// float w = 1e-4;
				matA = matA_rgb * w + matA_icp;
				vecb = vecb_rgb * w + vecb_icp;
			} else {
				matA = matA_icp;
				vecb = vecb_icp;
			}

			result = matA.ldlt().solve(vecb);
			auto e = Sophus::SE3d::exp(result);
			auto dT = e.matrix();

			delta = dT * delta;
			nextPose = lastPose * delta.inverse();
			NextFrame->pose = nextPose;
		}
		if(breakCount > 2)
			break;
	}

	Eigen::Matrix4d p = pose.inverse() * NextFrame->pose;
	Eigen::Matrix3d r = p.topLeftCorner(3, 3);
	Eigen::Vector3d t = p.topRightCorner(3, 1);
	Eigen::Vector3d a = r.eulerAngles(0, 1, 2).array().sin();
	// enforce small motion and small icp error
	if ((icpError < thresh_icp) && (a.norm() <= 0.1 && t.norm() <= 0.1)) {
		return true;
	} else {
		std::cout << "bad : " << icpError << "/" << rgbError << " " << a.norm() << " " << t.norm() << std::endl;
		NextFrame->pose = lastPose;
		return false;
	}
}

bool Tracker::Relocalise() {

	if(lastState != -1) {

		map->UpdateMapKeys();

		if(map->noKeysHost == 0)
			return false;

		cv::Mat desc(map->noKeysHost, 64, CV_32FC1);
		mapKeysAll.clear();
		for(int i = 0; i < map->noKeysHost; ++i) {
			SURF & key = map->hostKeys[i];
			for(int j = 0; j < 64; ++j) {
				desc.at<float>(i, j) = key.descriptor[j];
			}
			Eigen::Vector3d pos;
			pos << key.pos.x, key.pos.y, key.pos.z;
			mapKeysAll.push_back(pos);
		}
		descriptors.upload(desc);
	}

	refined.clear();
	std::vector<std::vector<cv::DMatch>> matches;
	matcher->knnMatch(NextFrame->descriptors, descriptors, matches, 2);
	for (int i = 0; i < matches.size(); ++i) {
		if (matches[i][0].distance < 0.90 * matches[i][1].distance) {
			refined.push_back(matches[i][0]);
		} else if (useGraphMatching) {
			refined.push_back(matches[i][0]);
			refined.push_back(matches[i][1]);
		}
	}

	if (refined.size() < 3)
		return false;

	framePoints.clear();
	refPoints.clear();

	if(useGraphMatching) {
		return GenerateGraph(N_LISTS_SELECT);
	} else {
		for (int i = 0; i < refined.size(); ++i) {
			framePoints.push_back(NextFrame->mapPoints[refined[i].queryIdx].cast<double>());
			refPoints.push_back(mapKeysAll[refined[i].trainIdx]);
		}
	}

	output = refPoints;
	Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
	bool bOK = Solver::PoseEstimate(framePoints, refPoints, outliers, delta, maxIterReloc);

	if (!bOK) {
		return false;
	}

	std::cout << "Relocalisation Success." << std::endl;
	NextFrame->pose = delta.inverse();

	return true;
}

bool Tracker::GenerateGraph(int N) {

	// used for storing key points in the current frame
	frameKeys.clear();
	// used for storing key points in the map (current time slice)
	mapKeysMatched.clear();
	// store distances between all matches.
	distance.clear();
	// store graph matching result
	mapKeySelected.clear();
	frameKeySelected.clear();

	// build a list of query key points for the current frame
	for (int i = 0; i < refined.size(); ++i) {

		int trainIdx = refined[i].trainIdx;
		int queryIdx = refined[i].queryIdx;
		SURF & trainKey = map->hostKeys[trainIdx];

		if (!trainKey.valid)
			continue;

		SURF queryKey;
		Eigen::Vector3f & p = NextFrame->mapPoints[queryIdx];
		queryKey.pos = { p(0), p(1), p(2) };
		queryKey.normal = NextFrame->pointNormal[queryIdx];
		frameKeys.push_back(queryKey);
		mapKeysMatched.push_back(trainKey);
		distance.push_back(refined[i].distance);
	}

	DeviceArray<SURF> cuMapKeys(mapKeysMatched);
	DeviceArray<SURF> cuFrameKeys(frameKeys);
	DeviceArray<float> cuDistance(distance);

	// Adjacency Matrix a.k.a. Consistency Matrix
	cuda::GpuMat cuConMatrix(frameKeys.size(), frameKeys.size(), CV_32FC1);

	// build adjacency matrix from raw key point matches
	BuildAdjacencyMatrix(cuConMatrix, cuFrameKeys, cuMapKeys, cuDistance);

	// filtered out useful key points
	cv::cuda::GpuMat cuRank;
	cv::cuda::reduce(cuConMatrix, cuRank, 0, CV_REDUCE_SUM);
	cv::Mat rank, rankIndex;
	cuRank.download(rank);

	if(rank.cols == 0)
		return false;

	cv::sortIdx(rank, rankIndex, CV_SORT_DESCENDING);

	cv::Mat conMatrix(cuConMatrix);
	std::vector<cv::Mat> vmSelectedIdx;
	cv::Mat cvNoSelected;

	// Select multiple sub-graphs
	for (int i = 0; i < N_LISTS_SUB_GRAPH; ++i) {

		cv::Mat mSelectedIdx;
		int headIdx = 0;
		int nSelected = 0;

		// for every sub-graph, select as many key pairs as we can
		for (int j = i; j < rankIndex.cols; ++j) {

			int idx = rankIndex.at<int>(j);
			// always push the first pair in the sub-graph
			if (nSelected == 0) {
				mSelectedIdx.push_back(idx);
				headIdx = idx;
				nSelected++;
			} else {
				// check confidence score associated with the first pair in the sub-graph
				// this is essentially the consistency check to make sure every pair in
				// the graph is consistent with each other;
				float score = conMatrix.at<float>(headIdx, idx);
				if (score > THRESH_MIN_SCORE) {
					mSelectedIdx.push_back(idx);
					nSelected++;
				}
			}

			if (nSelected >= THRESH_N_SELECTION)
				break;
		}

		// ao* needs at least 3 points to run
		// although it should be noticed that
		// more points generally means better.
		if (nSelected >= 3) {
			cv::Mat refined;
			for (int k = 1; k < nSelected; ++k) {
				int a = mSelectedIdx.at<int>(k);
				int l = k + 1;
				for (; l < nSelected; ++l) {
					int b = mSelectedIdx.at<int>(l);
					// check if the score is close to 0
					// essentially it means multiple points has been matched to the same one
					// or vice versa
					if(conMatrix.at<float>(a, b) < 5e-3f || conMatrix.at<float>(b, a) < 5e-3f) {
						if(conMatrix.at<float>(headIdx, b) > conMatrix.at<float>(headIdx, a)) {
							break;
						}
					}
				}
				if(l >= nSelected) {
					refined.push_back(a);
				}
			}

			cvNoSelected.push_back(refined.rows);
			vmSelectedIdx.push_back(refined.t());
		}
	}

	cv::Mat tmp;
	if(cvNoSelected.rows == 0) {
		std::cout << "not enough points" << std::endl;
		return false;
	}

	const int M = (cvNoSelected.rows > N) ? N : cvNoSelected.rows;
	cv::sortIdx(cvNoSelected, tmp, CV_SORT_DESCENDING);

	// building a list of sub-graphs we are using
	for(int i = 0; i < M; ++i) {

		rankIndex = vmSelectedIdx[tmp.at<int>(i)];
		int selection = rankIndex.cols;
		if(selection <= 3)
			continue;

		std::vector<Eigen::Vector3d> FKs;
		std::vector<Eigen::Vector3d> MKs;
		for(int i = 0; i < selection; ++i) {
			int idx = rankIndex.at<int>(i);
			SURF & mapKey = mapKeysMatched[idx];
			SURF & frameKey = frameKeys[idx];
			if(mapKey.valid && frameKey.valid) {
				Eigen::Vector3d FK, MK;
				MK << mapKey.pos.x, mapKey.pos.y, mapKey.pos.z;
				FK << frameKey.pos.x, frameKey.pos.y, frameKey.pos.z;
				FKs.push_back(FK);
				MKs.push_back(MK);
			}
		}

		frameKeySelected.push_back(FKs);
		mapKeySelected.push_back(MKs);
	}

	poseRefined.clear();
	poseEstimated.clear();
	for(int i = 0; i < frameKeySelected.size(); ++i) {
		Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
		std::vector<Eigen::Vector3d> & src = frameKeySelected[i];
		std::vector<Eigen::Vector3d> & ref = mapKeySelected[i];
		bool valid = Solver::PoseEstimate(src, ref, outliers, delta, maxIterReloc);
		if(valid) {
			poseEstimated.push_back(delta.inverse().eval());
		}
	}

	return ValidatePose();
}

bool Tracker::ValidatePose() {

	cv::Mat relocIcpErrors;
	int iteration[3] = { 5, 3, 3 };
	for(int i = 0; i < poseEstimated.size(); ++i) {
		uint no = 0;
		LastFrame->pose = poseEstimated[i];
		map->UpdateVisibility(LastFrame, no);

		if(no < 512)
			continue;

		map->RayTrace(no, LastFrame);
		LastFrame->ResizeImages();
		NextFrame->pose = LastFrame->pose;
		bool valid = ComputeSE3(true, ITERATIONS_RELOC, THRESH_ICP_RELOC);

		if(valid) {
			relocIcpErrors.push_back(lastIcpError);
			poseRefined.push_back(LastFrame->pose);
		}
	}

	if(relocIcpErrors.rows == 0)
		return false;

	cv::Mat index;
	cv::sortIdx(relocIcpErrors, index, CV_SORT_DESCENDING);
	int id = index.at<int>(0);

	uint no = 0;
	LastFrame->pose = poseRefined[id];
	map->UpdateVisibility(LastFrame, no);
	map->RayTrace(no, LastFrame);
	LastFrame->ResizeImages();
	NextFrame->pose = LastFrame->pose;

	// final refinement to make sure frame is perfectly aligned
	// Note: Re-localisation Still could be failed at this stage
	return ComputeSE3(true, ITERATIONS_SE3, THRESH_ICP_SE3);
}

Eigen::Matrix4f Tracker::GetCurrentPose() const {
	return LastFrame->pose.cast<float>();
}

void Tracker::SetMap(Mapping* pMap) {
	map = pMap;
}

void Tracker::SetViewer(Viewer* pViewer) {
	viewer = pViewer;
}

void Tracker::SetDetector(MaskRCNN* pDetector) {
	detector = pDetector;
}