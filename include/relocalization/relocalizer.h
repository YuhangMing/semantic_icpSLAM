#ifndef SLAM_RELOCALIZER_H
#define SLAM_RELOCALIZER_H

#include "data_struct/rgbd_frame.h"
#include "data_struct/map_point.h"
#include "features/extractor.h"
#include "features/matcher.h"
#include "data_struct/intrinsic_matrix.h"

namespace fusion
{

class Relocalizer
{
public:
    Relocalizer(const fusion::IntrinsicMatrix K);
    void setFeatureExtractor(std::shared_ptr<FeatureExtractor>);
    void setDescriptorMatcher(std::shared_ptr<DescriptorMatcher>);
    void set_map_points(std::vector<std::shared_ptr<Point3d>> mapPoints, cv::Mat &mapDescriptors);
    void set_target_frame(std::shared_ptr<RgbdFrame> frame);
    void compute_pose_candidates(std::vector<Sophus::SE3d> &candidates);

private:
    std::shared_ptr<FeatureExtractor> extractor;
    std::shared_ptr<DescriptorMatcher> matcher;

    cv::Mat map_descriptors;
    std::shared_ptr<RgbdFrame> target_frame;
    std::vector<std::shared_ptr<Point3d>> map_points;
    fusion::IntrinsicMatrix cam_param;
};

} // namespace fusion

#endif