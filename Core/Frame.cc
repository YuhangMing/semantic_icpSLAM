#include "Frame.h"
#include "KeyFrame.h"
#include "Reduction.h"

#include <Eigen/Dense>
#include <core/cuda.hpp>
#include <unistd.h>
#include <fstream>

#include <typeinfo>

using namespace cv;
using namespace std;

// #define DETECT_INFO

Mat Frame::mK[NUM_PYRS];
bool Frame::mbFirstCall = true;
float Frame::mDepthCutoff = 3.0f;
float Frame::mDepthScale = 1000.0f;
int Frame::mCols[NUM_PYRS];
int Frame::mRows[NUM_PYRS];
unsigned long Frame::nextId = 0;
cv::cuda::SURF_CUDA Frame::surfExt;
cv::Ptr<cv::BRISK> Frame::briskExt;

Frame::Frame():frameId(0), N(0), bad(false) {}

Frame::Frame(const Frame * other):frameId(other->frameId), N(0) {

}

void Frame::Create(int cols_, int rows_) {
	row_frame = rows_;
	col_frame = cols_;

	if(mbFirstCall) {
		surfExt = cv::cuda::SURF_CUDA(20);
		briskExt = cv::BRISK::create(30, 4);
		for(int i = 0; i < NUM_PYRS; ++i) {
			mCols[i] = cols_ / (1 << i);
			mRows[i] = rows_ / (1 << i);
		}
		mbFirstCall = false;
	}

	temp.create(cols_, rows_);
	range.create(cols_, rows_);
	color.create(cols_, rows_);
	// objects related create is done in ExtractObjects()

	for(int i = 0; i < NUM_PYRS; ++i) {
		int cols = cols_ / (1 << i);
		int rows = rows_ / (1 << i);
		vmap[i].create(cols, rows);
		nmap[i].create(cols, rows);
		depth[i].create(cols, rows);
		image[i].create(cols, rows);
		dIdx[i].create(cols, rows);
		dIdy[i].create(cols, rows);
	}
}

void Frame::Clear() {

}

void Frame::FillImages(const cv::Mat & range_, const cv::Mat & color_) {

	temp.upload(range_.data, range_.step);
	color.upload(color_.data, color_.step);
	FilterDepth(temp, range, depth[0], mDepthScale, mDepthCutoff);
	ImageToIntensity(color, image[0]);
	for(int i = 1; i < NUM_PYRS; ++i) {
		PyrDownGauss(depth[i - 1], depth[i]);
		PyrDownGauss(image[i - 1], image[i]);
	}

	for(int i = 0; i < NUM_PYRS; ++i) {
		ComputeVMap(depth[i], vmap[i], fx(i), fy(i), cx(i), cy(i), mDepthCutoff);
		ComputeNMap(vmap[i], nmap[i]);
		ComputeDerivativeImage(image[i], dIdx[i], dIdy[i]);
	}

	frameId = nextId++;
	bad = false;

	detected_masks.clear();
	detected_labels.clear();
	detected_scores.clear();
	numDetection=0;
}

void Frame::ResizeImages() {
	for(int i = 1; i < NUM_PYRS; ++i) {
		ResizeMap(vmap[i - 1], nmap[i - 1], vmap[i], nmap[i]);
	}
}

void Frame::ClearKeyPoints() {
	N = 0;
	keyPoints.clear();
	mapPoints.clear();
	descriptors.release();
}

float Frame::InterpDepth(cv::Mat & map, float & x, float & y) {

	float dp = std::nanf("0x7fffffff");
	if(x <= 1 || y <= 1 || y >= map.cols - 1 || x >= map.rows - 1)
		return dp;

	float2 coeff = make_float2(x, y) - make_float2(floor(x), floor(y));

	int2 upperLeft = make_int2((int) floor(x), (int) floor(y));
	int2 lowerLeft = make_int2((int) floor(x), (int) ceil(y));
	int2 upperRight = make_int2((int) ceil(x), (int) floor(y));
	int2 lowerRight = make_int2((int) ceil(x), (int) ceil(y));

	float d00 = map.at<float>(upperLeft.y, upperLeft.x);
	if(std::isnan(d00) || d00 < 0.3 || d00 > mDepthCutoff)
		return dp;

	float d10 = map.at<float>(lowerLeft.y, lowerLeft.x);
	if(std::isnan(d10) || d10 < 0.3 || d10 > mDepthCutoff)
		return dp;

	float d01 = map.at<float>(upperRight.y, upperRight.x);
	if(std::isnan(d01) || d01 < 0.3 || d01 > mDepthCutoff)
		return dp;

	float d11 = map.at<float>(lowerRight.y, lowerRight.x);
	if(std::isnan(d11) || d11 < 0.3 || d11 > mDepthCutoff)
		return dp;

	float d0 = d01 * coeff.x + d00 * (1 - coeff.x);
	float d1 = d11 * coeff.x + d10 * (1 - coeff.x);
	float final = (1 - coeff.y) * d0 + coeff.y * d1;
	if(std::abs(final - d00) <= 0.005)
		dp = final;
	return dp;
}

float4 Frame::InterpNormal(cv::Mat & map, float & x, float & y) {

	if(x <= 1 || y <= 1 || y >= map.cols - 1 || x >= map.rows - 1)
		return make_float4(std::nanf("0x7fffffff"));

	float2 coeff = make_float2(x, y) - make_float2(floor(x), floor(y));

	int2 upperLeft = make_int2((int) floor(x), (int) floor(y));
	int2 lowerLeft = make_int2((int) floor(x), (int) ceil(y));
	int2 upperRight = make_int2((int) ceil(x), (int) floor(y));
	int2 lowerRight = make_int2((int) ceil(x), (int) ceil(y));
	cv::Vec4f n;

	n = map.at<cv::Vec4f>(upperLeft.y, upperLeft.x);
	float4 d00 = make_float4(n(0), n(1), n(2), n(3));

	n = map.at<cv::Vec4f>(lowerLeft.y, lowerLeft.x);
	float4 d10 = make_float4(n(0), n(1), n(2), n(3));

	n = map.at<cv::Vec4f>(upperRight.y, upperRight.x);
	float4 d01 = make_float4(n(0), n(1), n(2), n(3));

	n = map.at<cv::Vec4f>(lowerRight.y, lowerRight.x);
	float4 d11 = make_float4(n(0), n(1), n(2), n(3));

	float4 d0 = d01 * coeff.x + d00 * (1 - coeff.x);
	float4 d1 = d11 * coeff.x + d10 * (1 - coeff.x);
	float4 final = d0 * (1 - coeff.y) + d1 * coeff.y;

	if(norm(final - d00) <= 0.1)
		return final;
	else
		return make_float4(std::nanf("0x7fffffff"));
}

void Frame::ExtractKeyPoints() {

	cv::Mat rawDescriptors;
	cv::Mat sNormal(depth[0].rows, depth[0].cols, CV_32FC4);
	cv::Mat sDepth(depth[0].rows, depth[0].cols, CV_32FC1);
	std::vector<cv::KeyPoint> rawKeyPoints;

	depth[0].download(sDepth.data, sDepth.step);
	nmap[0].download(sNormal.data, sNormal.step);

	N = 0;
	keyPoints.clear();
	mapPoints.clear();
	descriptors.release();

	cv::cuda::GpuMat img(image[0].rows, image[0].cols, CV_8UC1, image[0].data, image[0].step);
	surfExt(img, cv::cuda::GpuMat(), rawKeyPoints, descriptors);
	descriptors.download(rawDescriptors);

	cv::Mat desc;
	N = rawKeyPoints.size();
	for(int i = 0; i < N; ++i) {
		cv::KeyPoint & kp = rawKeyPoints[i];
		float x = kp.pt.x;
		float y = kp.pt.y;
		float dp = InterpDepth(sDepth, x, y);
		if(!std::isnan(dp) && dp > 0.3 && dp < mDepthCutoff) {
			float4 n = InterpNormal(sNormal, x, y);
			if(!std::isnan(n.x)) {
				Eigen::Vector3f v;
				v(0) = dp * (x - cx(0)) / fx(0);
				v(1) = dp * (y - cy(0)) / fy(0);
				v(2) = dp;
				mapPoints.push_back(v);
				keyPoints.push_back(kp);
				pointNormal.push_back(n);
				desc.push_back(rawDescriptors.row(i));
			}
		}
	}

	N = mapPoints.size();
	if(N < MIN_KEY_POINTS)
		bad = true;

	descriptors.upload(desc);
	pose = Eigen::Matrix4d::Identity();
}

void Frame::DrawKeyPoints() {

	cv::Mat rawImage(480, 640, CV_8UC1);
	image[0].download(rawImage.data, rawImage.step);
	for (int i = 0; i < N; ++i) {
		cv::Point2f upperLeft = keyPoints[i].pt - cv::Point2f(5, 5);
		cv::Point2f lowerRight = keyPoints[i].pt + cv::Point2f(5, 5);
		cv::drawMarker(rawImage, keyPoints[i].pt, cv::Scalar(0, 125, 0), cv::MARKER_CROSS, 5);
		cv::rectangle(rawImage, upperLeft, lowerRight, cv::Scalar(0, 125, 0));
	}

	cv::imshow("img", rawImage);
	cv::waitKey(10);
}

void Frame::ExtractObjects(MaskRCNN* detector, bool bBbox, bool bContour, bool bText) {
	cv::Mat img(row_frame, col_frame, CV_8UC3);
    color.download(img.data, img.step);
	detector->performDetection(img);
	colorObject = false;

	// get detected information
    numDetection = detector->numDetection;
    pLabels = detector->pLabels;
    pScores = detector->pScores;
    pMasks = detector->pMasks;    // binary mask, 0 & 1
    pBoxes = detector->pBoxes;

	vMasks.clear();
	vLabels.clear();
	vScores.clear();
	vRemovedIdx.clear();
	for(int i=0; i<numDetection; i++) {
		// array to vector
		cv::Mat tmpMask = Array2Mat(&pMasks[i*row_frame*col_frame]);
		vMasks.push_back(tmpMask.clone());
		vLabels.push_back(int(pLabels[i]));
		vScores.push_back(pScores[i]);

		for(int j=0; j<i; j++){
			// if jth detection is already removed, move on
			if(std::find(vRemovedIdx.begin(), vRemovedIdx.end(), j) != vRemovedIdx.end()) {
				continue;
			}
			// check for intersection
			cv::Mat overlap = (vMasks[j]>0) & (tmpMask>0); // with value of 0 & 255
			// cv::imwrite("/home/lk18493/overlap.png", overlap);
			// overlap image checked out
			if (cv::sum(overlap)[0] > 0) {
			#ifdef DETECT_INFO
				std::cout << "overlap found between " 
						  << j << ": " << pLabels[j] << ", "  << pScores[j]
						  << " and "
						  << i << ": " << pLabels[i] << ", "  << pScores[i]
						  << std::endl;
			#endif
				float prob_pre = pScores[j];
				float prob_now = pScores[i];
				if (prob_pre >= prob_now) {
					vRemovedIdx.push_back(i);
					break;
				} else {
					vRemovedIdx.push_back(j);
				}
			}
		}
	}
	#ifdef DETECT_INFO
		std::cout << "££ CDEBUG: in Frame.cc " << std::endl;
		std::cout << "number of detected objects is " << numDetection << std::endl;
		for(std::vector<int>::iterator it=vRemovedIdx.begin(); it!=vRemovedIdx.end(); it++) {
			std::cout << *it << ", ";
		}
		std::cout << std::endl;
		for(int i=0; i<numDetection; i++){
			if(std::find(vRemovedIdx.begin(), vRemovedIdx.end(), i) != vRemovedIdx.end()) {
				continue;
			}
			std::cout << "Label: " << pLabels[i] << ", " << detector->CATEGORIES[pLabels[i]] << "; ";
			std::cout << "Score = " << pScores[i] << "; ";
			std::cout << std::endl;
		}
		usleep(5000000);

		// std::cout << "Labels: " << std::endl << "  ";
		// for(int i=0; i<numDetection; i++){
		//     std::cout << pLabels[i] << ": " << detector->CATEGORIES[pLabels[i]] << "; ";
		// }
		// std::cout << std::endl;
		// std::cout << "Scores: " << std::endl << "  ";
		// for(int i=0; i<numDetection; i++){
		//     std::cout << pScores[i] << ", ";
		// }
		// std::cout << std::endl;
		
		// std::cout << " Masks sample IN TXT FILE" << std::endl;
		// ofstream file2( "/home/lk18493/maskc-r.txt" );
		// for(int r=0; r<480; r++) {
		//     for (int c=0; c<640; c++) {
		// 		file2 << pMasks[c+r*640];
		// 	}
		// 	file2 << "\n";
		// }
		// file2.close();
		// std::cout << "B-Boxes: " << std::endl << "  ";
		// for(int i=0; i<numDetection*4; i++){
		//     std::cout << pBoxes[i] << ", ";
		// }
		// std::cout << std::endl;
		// std::cout << " ** ** ** ** ** \n" << std::endl;
	#endif
    // combine masks into a single mask & draw detections
	// new mask will have value as corresponding obj label
	int tmp_counter = 0;
    for(int i=0; i<numDetection; i++){
		// remove unwanted labels and scores
		if(std::find(vRemovedIdx.begin(), vRemovedIdx.end(), i) != vRemovedIdx.end()) {
			vLabels.erase(vLabels.begin()+i-tmp_counter);
			vScores.erase(vScores.begin()+i-tmp_counter);
			tmp_counter++;
			continue;
		}
		// combine masks
		if(i == 0) {
			mask = vMasks[i]*int(pLabels[i]);
		} else {
			mask += vMasks[i]*int(pLabels[i]);
		}
		// draw bboxes
    	cv::Scalar objColor = CalculateColor(pLabels[i]);
    	if (bBbox){
			cv::Point2f top_left(pBoxes[i*4], pBoxes[i*4+1]);
			cv::Point2f bottom_right(pBoxes[i*4+2], pBoxes[i*4+3]);
			cv::rectangle(img, top_left, bottom_right, objColor);
		}
		// draw mask countors
		if (bContour) {
			cv::Mat mMask = vMasks[i];
			std::vector<std::vector<cv::Point> > contours;
			std::vector<Vec4i> hierarchy;
			cv::findContours(mMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
			cv::drawContours(img, contours, -1, objColor, 4);
		}
		// display text
		if (bText){
			std::string label_text = detector->CATEGORIES[pLabels[i]];
			cv::Point2f top_left(pBoxes[i*4], pBoxes[i*4+1]);
			cv::putText(img, label_text, top_left, cv::FONT_HERSHEY_SIMPLEX, 1.0, objColor);
			// cv::Point2f bottom_right(pBoxes[i*4+2], pBoxes[i*4+3]);
			// cv::putText(img, label_text, bottom_right, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255));
		}
	}
	#ifdef DETECT_INFO
		std::cout << "Original Detected Labels are: ";
		for(int i=0; i<numDetection; i++) {
			std::cout << pLabels[i] << ", ";
		}
		std::cout << std::endl;
	#endif
	numDetection -= vRemovedIdx.size();
	#ifdef DETECT_INFO
		std::cout << "Removed labels are: ";
		for(std::vector<int>::iterator it=vRemovedIdx.begin(); it<vRemovedIdx.end(); it++){
			std::cout << pLabels[*it] << ", ";
		}
		std::cout << std::endl;
		std::cout << "Final Detection results are: \n" ;
		for(int i=0; i<numDetection; i++) {
			std::cout << vLabels[i] << " - " << vScores[i] << std::endl;
		}
		ofstream file2( "/home/lk18493/mask-all.txt" );
		file2 << mask;
		file2.close();
	#endif
	// upload detection results to device array
	if(numDetection != 0){
		if(colorObject){
			detected_labels.create(numDetection);
			detected_scores.create(numDetection);
			detected_masks.create(col_frame, row_frame);

			detected_labels.upload(vLabels);
			detected_scores.upload(vScores);
			detected_masks.upload(mask.data, mask.step);
		}
		color.upload(img.data, img.step);
	} 
	// // else {
	// //  std::cout << numDetection << std::endl;
	// // 	usleep(7000000);
	// // }
}
void Frame::GeometricRefinement(float lamb, float tao, int win_size) {
	edge.create(depth[0].cols, depth[0].rows);
	cc_labeled.create(depth[0].cols, depth[0].rows);
	int step = win_size/2; // floor
	NVmapToEdge(nmap[0], vmap[0], edge, lamb, tao, win_size, step);

	// //  CPU IMPLEMENTATION WAY TOO SLOW ########################
	// // ####### Transfer to GPU IMPLEMENTATION#### ##############
	// // get calculated vertex and normal from the rendered scene
	// cv::Mat mNormal(depth[0].rows, depth[0].cols, CV_32FC4);
	// cv::Mat mVertex(depth[0].rows, depth[0].cols, CV_32FC4);
	// nmap[0].download(mNormal.data, mNormal.step);
	// vmap[0].download(mVertex.data, mVertex.step);
	// mEdge = cv::Mat::zeros(depth[0].rows, depth[0].cols, CV_8UC1);
	
	// // calculate geometric edges/boundaries to find better fitted masks
	// for(int r=step; r<depth[0].rows-step; r++){
	// 	for(int c=step; c<depth[0].cols-step; c++){
	// 		cv::Mat sub_normal(win_size, win_size, CV_32FC4);
	// 		cv::Mat sub_vertex(win_size, win_size, CV_32FC4);
	// 		mNormal( cv::Range((r-step),(r+step+1)), cv::Range((c-step),(c+step+1)) ).copyTo(sub_normal);
	// 		mVertex( cv::Range((r-step),(r+step+1)), cv::Range((c-step),(c+step+1)) ).copyTo(sub_vertex);
	// 		// cv::Mat sub_vertex = mVertex( cv::Range((r-step),(r+step+1)),
	// 		// 							  cv::Range((c-step),(c+step+1)) );

	// 		// std::cout << "check if there is NAN in the mat" << std::endl;
	// 		if(isnan(cv::sum(sub_normal)[0]) || isnan(cv::sum(sub_vertex)[0])){
	// 			continue;
	// 		}
	// 		cv::Vec4f v = mVertex.at<cv::Vec4f>(r, c);
	// 		cv::Vec4f n = mNormal.at<cv::Vec4f>(r, c);

	// 		// depth term
	// 		cv::Mat diff = sub_vertex - v;
	// 		cv::Mat depth_term = diff.reshape(1, win_size*win_size) * cv::Mat(n);
	// 		double phi_d;
	// 		minMaxLoc(abs(depth_term), NULL, &phi_d);

	// 		// convex term
	// 		cv::Mat convex_term = 1 - sub_normal.reshape(1, win_size*win_size) * cv::Mat(n);
	// 		cv::Mat comp_term = (depth_term>=0)/255;
	// 		cv::Mat comp_term_F(convex_term.size(), convex_term.type());
	// 		comp_term.convertTo(comp_term_F, convex_term.type());
	// 		double phi_c; 
	// 		minMaxLoc(convex_term.mul(comp_term_F), NULL, &phi_c );
			
	// 		// if(r>476 && c>550){
	// 		// 	std::cout << r << ", " << c << " out of (" 
	// 		// 			  << depth[0].rows << ", " << depth[0].cols << ")" << std::endl;
	// 		// 	std::cout << sub_normal << std::endl;
	// 		// 	std::cout << sub_vertex << std::endl;
	// 		// 	std::cout << v << std::endl;
	// 		// 	std::cout << n << std::endl;
	// 		// 	std::cout << phi_d << std::endl;
	// 		// 	std::cout << phi_c << std::endl;
	// 		// 	std::cout << edge << std::endl;
	// 		// 	std::cout << typeid(phi_d).name() << std::endl;
	// 		// 	std::cout << typeid(lamb*phi_c).name() << std::endl;
	// 		// 	std::cout << typeid(phi_d+lamb*phi_c).name() << std::endl;
	// 		// 	std::cout << typeid((phi_d+lamb*phi_c)>tao).name() << std::endl;			
	// 		// }

	// 		mEdge.at<uchar>(r, c) = int((phi_d+lamb*phi_c)<tao)*255;
	// 	}
	// }
	// ##############################################################

	cv::Mat mEdge(depth[0].rows, depth[0].cols, CV_8UC1);
	edge.download(mEdge.data, mEdge.step);
	// cv::imwrite("/home/lk18493/edge_map.png", mEdge);

	nConComps = cv::connectedComponentsWithStats(mEdge, mLabeled, mStats, mCentroids, 4);
	cc_labeled.upload(mLabeled.data, mLabeled.step);
	// cv::imwrite("/home/lk18493/label_map.png", mLabeled);
	// std::ofstream outfile;
	// outfile.open("/home/lk18493/nCCs.txt", std::ios_base::app);
	// outfile << nConComps << "\n"; 
}
void Frame::FuseMasks(int thre){
	int nMasks;
	cv::Mat tmpLabeled, tmpStats, tmpCentroids;

	if(numDetection > 0){
		// TRANSFER TO GPU LATER ON
		fusedMask = cv::Mat::zeros(row_frame, col_frame, CV_8UC1);

		for(int i_comp=1; i_comp<nConComps; i_comp++)
		{
			// filter out small blobs
			int area_blob = mStats.at<int>(i_comp, CC_STAT_AREA);
			if(area_blob < thre){
				continue;
			}
			cv::Mat one_blob = (mLabeled==i_comp)/255; // binary matrix of 0/1
			
			for(int i_detect=0; i_detect<numDetection; i_detect++)
			{
				cv::Mat one_mask = (mask==vLabels[i_detect])/255; // binary
				int area_mask = cv::sum(one_mask)[0];

				cv::Mat overlap = (one_blob & one_mask);
				int area_overlap = cv::sum(overlap)[0];
				// cv::Mat uunionn = ()
				
				// accept the overlap area as an object if iou is larger than a threshold
				if(area_overlap >= area_mask*0.7 || area_overlap >= area_blob*0.7){
					// meaning blob is inside the mask, asign the blob to the mask
					fusedMask += overlap*vLabels[i_detect];
					// surjective mapping, one blob can have a only one mask map
					break;
				}
			}
		}

		// // remove partial ovserved objects
		// nMasks = cv::connectedComponentsWithStats(fusedMask, tmpLabeled, tmpStats, tmpCentroids, 4);
		// int lmost, rmost, tmost, bmost;
		// int bthre=4;
		// for(int i=1; i<nMasks; i++){
		// 	lmost = tmpStats.at<int>(i, CC_STAT_LEFT);
		// 	tmost = tmpStats.at<int>(i, CC_STAT_TOP);
		// 	rmost = lmost + tmpStats.at<int>(i, CC_STAT_WIDTH);
		// 	bmost = tmost - tmpStats.at<int>(i, CC_STAT_HEIGHT);
		// 	if(lmost < bthre || bmost < bthre || rmost > 640-bthre || tmost > 480-bthre){
		// 		fusedMask = fusedMask.mul( (tmpLabeled!=i)/255 );
		// 	}
		// }	
		// cv::imwrite("/home/lk18493/mask_map.png", fusedMask);
		// usleep(500000);
		
		// upload detection results to device array
		detected_labels.create(numDetection);
		detected_scores.create(numDetection);
		detected_masks.create(col_frame, row_frame);

		detected_labels.upload(vLabels);
		detected_scores.upload(vScores);
		detected_masks.upload(fusedMask.data, fusedMask.step);
		// color.upload(img.data, img.step);
		colorObject = true;
	} 
}
cv::Scalar Frame::CalculateColor(long int label) {
	cv::Scalar color(
			int(pallete[0]*label%255), 
			int(pallete[1]*label%255), 
			int(pallete[2]*label%255)
		);
	return color;
}
cv::Mat Frame::Array2Mat(int* aMask) {
	cv::Mat mMask(row_frame, col_frame, CV_32SC1, aMask, sizeof(int)*col_frame);
	// cv::imwrite("/home/lk18493/mask.jpg", mMask*255);
	cv::Mat mScaleMask(row_frame, col_frame, CV_8UC1);
	mMask.convertTo(mScaleMask, CV_8UC1);
	// cv::imwrite("/home/lk18493/mask_scale.jpg", mScaleMask*255);

	return mScaleMask;
}

void Frame::SetK(cv::Mat& K) {
	for(int i = 0; i < NUM_PYRS; ++i) {
		mK[i] = cv::Mat::eye(3, 3, CV_32FC1);
		mK[i].at<float>(0, 0) = K.at<float>(0, 0) / (1 << i);
		mK[i].at<float>(1, 1) = K.at<float>(1, 1) / (1 << i);
		mK[i].at<float>(0, 2) = K.at<float>(0, 2) / (1 << i);
		mK[i].at<float>(1, 2) = K.at<float>(1, 2) / (1 << i);
	}
}

float Frame::fx(int pyr) {
	assert(pyr >= 0 && pyr <= NUM_PYRS);
	return mK[pyr].at<float>(0, 0);
}

float Frame::fy(int pyr) {
	assert(pyr >= 0 && pyr <= NUM_PYRS);
	return mK[pyr].at<float>(1, 1);
}

float Frame::cx(int pyr) {
	assert(pyr >= 0 && pyr <= NUM_PYRS);
	return mK[pyr].at<float>(0, 2);
}

float Frame::cy(int pyr) {
	assert(pyr >= 0 && pyr <= NUM_PYRS);
	return mK[pyr].at<float>(1, 2);
}

int Frame::cols(int pyr) {
	assert(pyr >= 0 && pyr <= NUM_PYRS);
	return mCols[pyr];
}

int Frame::rows(int pyr) {
	assert(pyr >= 0 && pyr <= NUM_PYRS);
	return mRows[pyr];
}

Eigen::Vector3f Frame::GetWorldPoint(int i) const {
	Eigen::Matrix3f r = Rotation().cast<float>();
	Eigen::Vector3f t = Translation().cast<float>();
	return r * mapPoints[i] + t;
}

Matrix3f Frame::GpuRotation() const {
	Matrix3f Rot;
	Rot.rowx = make_float3(pose(0, 0), pose(0, 1), pose(0, 2));
	Rot.rowy = make_float3(pose(1, 0), pose(1, 1), pose(1, 2));
	Rot.rowz = make_float3(pose(2, 0), pose(2, 1), pose(2, 2));
	return Rot;
}

Matrix3f Frame::GpuInvRotation() const {
	Matrix3f Rot;
	const Eigen::Matrix3d mPoseInv = Rotation().transpose();
	Rot.rowx = make_float3(mPoseInv(0, 0), mPoseInv(0, 1), mPoseInv(0, 2));
	Rot.rowy = make_float3(mPoseInv(1, 0), mPoseInv(1, 1), mPoseInv(1, 2));
	Rot.rowz = make_float3(mPoseInv(2, 0), mPoseInv(2, 1), mPoseInv(2, 2));
	return Rot;
}

float3 Frame::GpuTranslation() const {
	return make_float3(pose(0, 3), pose(1, 3), pose(2, 3));
}

Eigen::Matrix3d Frame::Rotation() const {
	return pose.topLeftCorner(3, 3);
}

Eigen::Matrix3d Frame::RotationInv() const {
	return Rotation().transpose();
}

Eigen::Vector3d Frame::Translation() const {
	return pose.topRightCorner(3, 1);
}

Eigen::Vector3d Frame::TranslationInv() const {
	return -RotationInv() * Translation();
}
