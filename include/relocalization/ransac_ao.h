#ifndef SLAM_POSE_ESTIMATOR_H
#define SLAM_POSE_ESTIMATOR_H

#include "data_struct/map_point.h"

namespace fusion
{

class PoseEstimator
{
public:
    //! Compute relative transformation from two sets of points
    //! outliers: indicates which points are outliers
    //! At least 3 pairs of points need to be supplied
    //! and none of them can be co-linear
    static bool absolute_orientation(
        std::vector<Eigen::Vector3f> src,
        std::vector<Eigen::Vector3f> dst,
        std::vector<bool> outliers,
        Eigen::Matrix4f &estimate);

    //! Compute relative transformation from two sets of points
    //! all points are treated like inliers
    //! At least 3 pairs of points need to be supplied
    //! and none of them can be co-linear
    static bool absolute_orientation(
        std::vector<Eigen::Vector3f> src,
        std::vector<Eigen::Vector3f> dst,
        Eigen::Matrix4f &estimate);

    static int evaluate_inliers(
        const std::vector<Eigen::Vector3f> &src,
        const std::vector<Eigen::Vector3f> &dst,
        std::vector<bool> &outliers,
        const Eigen::Matrix4f &estimate);

    static void RANSAC(
        const std::vector<Eigen::Vector3f> &src,
        const std::vector<Eigen::Vector3f> &dst,
        std::vector<bool> &outliers,
        Eigen::Matrix4f &estimate,
        float &inlier_ratio,
        float &confidence);

private:
};

} // namespace fusion

#endif