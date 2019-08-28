#ifndef SUBMAP_MANAGER_H
#define SUBMAP_MANAGER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <mutex>
#include "system.h"
#include "data_struct/map_struct.h"
#include "data_struct/rgbd_frame.h"
#include "features/extractor.h"

namespace fusion
{

class DenseMapping;
class DenseOdometry;

class SubMapManager {

public:
	SubMapManager();

	void Create(const fusion::IntrinsicMatrix base, 
				int submapIdx, bool bTrack=false, bool bRender=false);
	void Create(const fusion::IntrinsicMatrix base, 
				int submapIdx, RgbdImagePtr ref_img, bool bTrack=false, bool bRender=true);

	void ResetSubmaps();

	float CheckVisPercent(int submapIdx);
	void CheckActive();

	void CheckTrackAndRender(int cur_frame_id, int max_perct_idx);

	void AddKeyFrame(RgbdFramePtr currKF);
	std::vector<Eigen::Matrix<float, 4, 4>> GetKFPoses();
	void GetPoints(float *pt3d, size_t &count, size_t max_size);

	void SetTracker(std::shared_ptr<DenseOdometry> pOdometry);
	void SetExtractor(std::shared_ptr<FeatureExtractor> pExtractor);

	std::shared_ptr<DenseOdometry> odometry;

	bool bKFCreated;
	bool bHasNewSM;
	int renderIdx;	// for active submaps, index of the submapp is constructing

	// submap storage
	// std::vector< std::shared_ptr<DenseMapping> > all_submaps;	// discard this variable, all_submaps = active + passive
	std::vector< std::shared_ptr<DenseMapping> > active_submaps;
	std::vector< std::shared_ptr<DenseMapping> > passive_submaps;
	std::vector< int > activeTOpassiveIdx;

	// std::map< int, std::vector<Eigen::Vector4f> > mPassiveMPs;
	// std::map< int, std::vector<Eigen::Vector4f> > mPassiveNs;

protected:

private:
	std::shared_ptr<FeatureExtractor> extractor;

	std::mutex mMutexDownload;
	int ref_frame_id;
};


} // namespace fusion

#endif