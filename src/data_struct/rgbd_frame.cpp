#include "data_struct/rgbd_frame.h"
#include <ctime>

namespace fusion
{

RgbdFrame::RgbdFrame(const cv::Mat &depth, const cv::Mat &image, const size_t id, const double ts)
    : id(id), timeStamp(ts)
{
    this->image = image.clone();
    this->depth = depth.clone();
    row_frame = image.rows;
    col_frame = image.cols;
    numDetection = 0;
}

RgbdFrame::RgbdFrame()
{
	id = -1;
	timeStamp = -1.;
}

void RgbdFrame::copyTo(RgbdFramePtr dst){
	if(dst==NULL)
		return;

	dst->cv_key_points = cv_key_points;
	dst->key_points = key_points;
	dst->neighbours = neighbours;
	descriptors.copyTo(dst->descriptors);

	dst->id = id;
	dst->timeStamp = timeStamp;
	dst->pose = pose;

	image.copyTo(dst->image);
	depth.copyTo(dst->depth);
	vmap.copyTo(dst->vmap);
	nmap.copyTo(dst->nmap);

	dst->row_frame = row_frame;
	dst->col_frame = col_frame;
	
	dst->numDetection = numDetection;
	// dst->pMasks = pMasks;
	// dst->pLabels = pLabels;
	// dst->pScores = pScores;
	// dst->pBoxes = pBoxes;
	mask.copyTo(dst->mask);
	// fusedMask.copyTo(dst->fusedMask);

	// dst->colorObject = colorObject;
	dst->vMasks = vMasks;
	dst->vLabels = vLabels;
	dst->vScores = vScores;
	dst->vRemovedIdx = vRemovedIdx;
	dst->nConComps = nConComps;
	mLabeled.copyTo(dst->mLabeled);
	mStats.copyTo(dst->mStats);
	mCentroids.copyTo(dst->mCentroids);
}

void RgbdFrame::ExtractObjects(semantic::MaskRCNN* detector, bool bBbox, bool bContour, bool bText)
{
	// detection
	std::clock_t start = std::clock();
	detector->performDetection(image);
	std::cout << "Extract object takes "
              << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
              << " seconds" << std::endl;

    start = std::clock();
	// get detected information
	numDetection = detector->numDetection;
 //    pLabels = detector->pLabels;
 //    pScores = detector->pScores;
 //    pMasks = detector->pMasks;    // binary mask, 0 & 1
 //    pBoxes = detector->pBoxes;

    vMasks.clear();
	vLabels.clear();
	vScores.clear();
	vRemovedIdx.clear();
	// solve lable conflicts in an image
	// choose higher probability one over lower one if overlap found
	for(size_t i=0; i<numDetection; ++i)
	{
		// array to vector
		cv::Mat tmpMask = Array2Mat(&detector->pMasks[i*row_frame*col_frame]);
		vMasks.push_back(tmpMask.clone());
		vLabels.push_back(int(detector->pLabels[i]));
		vScores.push_back(detector->pScores[i]);

		for(size_t j=0; j<i; ++j)
		{
			// if jth detection is already removed, move on
			if(std::find(vRemovedIdx.begin(), vRemovedIdx.end(), j) != vRemovedIdx.end())
				continue;
			// check for intersection
			cv::Mat overlap = (vMasks[j]>0) & (tmpMask>0); // with value of 0 & 255
			// cv::imwrite("/home/lk18493/overlap.png", overlap);
			// overlap image checked out
			if (cv::sum(overlap)[0] > 0) {
				// float prob_pre = pScores[j];
				// float prob_now = pScores[i];
				float prob_pre = vScores[j];
				float prob_now = vScores[i];
				if (prob_pre >= prob_now) {
					vRemovedIdx.push_back(i);
					break;
				} else {
					vRemovedIdx.push_back(j);
				}
			} 
		} // j
	} // i

	// combine masks into a single mask & draw detections
	// new mask will have value as corresponding obj label
	int tmp_counter = 0;
    for(int i=0; i<numDetection; i++){
		// remove unwanted labels and scores
		if(std::find(vRemovedIdx.begin(), vRemovedIdx.end(), i) != vRemovedIdx.end()) {
			vLabels.erase(vLabels.begin()+i-tmp_counter);
			vScores.erase(vScores.begin()+i-tmp_counter);
			vMasks.erase(vMasks.begin()+i-tmp_counter);
			tmp_counter++;
			continue;
		}
		
		// combine masks
		// if(i == 0)
		// 	mask = vMasks[i]*int(pLabels[i]);
		// else
		// 	mask += vMasks[i]*int(pLabels[i]);
		if(i == 0)
			mask = vMasks[i-tmp_counter]*int(vLabels[i-tmp_counter]);
		else
			mask += vMasks[i-tmp_counter]*int(vLabels[i-tmp_counter]);
		
		// draw bboxes
    	cv::Scalar objColor = CalculateColor(vLabels[i-tmp_counter]);
    	if (bBbox){
			cv::Point2f top_left(detector->pBoxes[i*4], detector->pBoxes[i*4+1]);
			cv::Point2f bottom_right(detector->pBoxes[i*4+2], detector->pBoxes[i*4+3]);
			cv::rectangle(image, top_left, bottom_right, objColor);
		}
		// draw mask countors
		if (bContour) {
			cv::Mat mMask = vMasks[i-tmp_counter];
			std::vector<std::vector<cv::Point> > contours;
			std::vector<cv::Vec4i> hierarchy;
			cv::findContours(mMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
			cv::drawContours(image, contours, -1, objColor, 4);
		}
		// display text
		if (bText){
			std::string label_text = detector->CATEGORIES[vLabels[i-tmp_counter]];
			cv::Point2f top_left(detector->pBoxes[i*4], detector->pBoxes[i*4+1]);
			cv::putText(image, label_text, top_left, cv::FONT_HERSHEY_SIMPLEX, 1.0, objColor);
		}
	}

	// update numDetection
	numDetection -= vRemovedIdx.size();

	std::cout << "Post-processing takes "
              << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
              << " seconds" << std::endl;
}
// void GeometricRefinement(float lamb, float tao, int win_size);
void RgbdFrame::FuseMasks(cv::Mat edge, int thre)
{
	std::clock_t start = std::clock(); 
	// find connected components, 4-connected
	nConComps = cv::connectedComponentsWithStats(edge, mLabeled, mStats, mCentroids, 4);

	// fuse masks
	int nMasks;
	cv::Mat tmpLabeled, tmpStats, tmpCentroids;
	if(numDetection > 0){
		// TRANSFER TO GPU LATER ON
		cv::Mat fusedMask = cv::Mat::zeros(row_frame, col_frame, CV_8UC1);

		for(int i_comp=1; i_comp<nConComps; i_comp++)
		{
			// filter out small blobs
			int area_blob = mStats.at<int>(i_comp, cv::CC_STAT_AREA);
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

		mask = fusedMask;

		// debug
		// cv::imwrite("/home/lk18493/detected.png", image);
		// cv::imwrite("/home/lk18493/connected_components.png", mLabeled);
		// cv::imwrite("/home/lk18493/mask_fused.png", fusedMask*255);
		// // usleep(500000);
	} 

	std::cout << "Fuse masks takes "
              << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
              << " seconds" << std::endl;
}

cv::Scalar RgbdFrame::CalculateColor(long int label)
{
	cv::Scalar color(
			int(pallete[0]*label%255), 
			int(pallete[1]*label%255), 
			int(pallete[2]*label%255)
		);
	return color;
}

cv::Mat RgbdFrame::Array2Mat(int* aMask)
{
	cv::Mat mMask(row_frame, col_frame, CV_32SC1, aMask, sizeof(int)*col_frame);
	// cv::imwrite("/home/lk18493/mask.jpg", mMask*255);
	cv::Mat mScaleMask(row_frame, col_frame, CV_8UC1);
	mMask.convertTo(mScaleMask, CV_8UC1);
	// cv::imwrite("/home/lk18493/mask_scale.jpg", mScaleMask*255);

	return mScaleMask;
}



} // namespace fusion