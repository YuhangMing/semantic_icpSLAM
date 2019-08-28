#include <stdlib.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "relocalization/ransac_ao.h"
#define MAX_RANSAC_ITER 200

namespace fusion
{

bool PoseEstimator::absolute_orientation(
    std::vector<Eigen::Vector3f> src,
    std::vector<Eigen::Vector3f> dst,
    std::vector<bool> outliers,
    Eigen::Matrix4f &estimate)
{
    //! Must initialize before using
    //! Eigen defaults to random numbers
    estimate = Eigen::Matrix4f::Identity();
    Eigen::Vector3f src_pts_sum = Eigen::Vector3f::Zero();
    Eigen::Vector3f dst_pts_sum = Eigen::Vector3f::Zero();
    int no_inliers = 0;

    Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
    for (int i = 0; i < src.size(); ++i)
    {
        if (!outliers[i])
        {
            no_inliers++;
            src_pts_sum += src[i];
            dst_pts_sum += dst[i];
            M += src[i] * dst[i].transpose();
        }
    }

    //! Compute centroids
    src_pts_sum /= no_inliers;
    dst_pts_sum /= no_inliers;
    M -= no_inliers * (src_pts_sum * dst_pts_sum.transpose());

    const auto svd = M.cast<double>().jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto MatU = svd.matrixU();
    const auto MatV = svd.matrixV();
    Eigen::Matrix3d R;

    //! Check if R is a valid rotation matrix
    if (MatU.determinant() * MatV.determinant() < 0)
    {
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        I(2, 2) = -1;
        R = MatV * I * MatU.transpose();
    }
    else
    {
        R = MatV * MatU.transpose();
    }

    const auto t = src_pts_sum - R.cast<float>() * dst_pts_sum;

    estimate.topLeftCorner(3, 3) = R.cast<float>();
    estimate.topRightCorner(3, 1) = t;

    return true;
}

bool PoseEstimator::absolute_orientation(
    std::vector<Eigen::Vector3f> src,
    std::vector<Eigen::Vector3f> dst,
    Eigen::Matrix4f &estimate)
{
    std::vector<bool> outliers(src.size());
    std::fill(outliers.begin(), outliers.end(), false);
    return absolute_orientation(src, dst, outliers, estimate);
}

int PoseEstimator::evaluate_inliers(
    const std::vector<Eigen::Vector3f> &src,
    const std::vector<Eigen::Vector3f> &dst,
    std::vector<bool> &outliers,
    const Eigen::Matrix4f &estimate)
{
    int no_inliers = 0;
    float dist_thresh = 0.05f;
    const auto &R = estimate.topLeftCorner(3, 3);
    const auto &t = estimate.topRightCorner(3, 1);

    std::fill(outliers.begin(), outliers.end(), true);
    for (int i = 0; i < src.size(); ++i)
    {
        float dist = (src[i] - (R * dst[i] + t)).norm();
        if (dist < dist_thresh)
        {
            no_inliers++;
            outliers[i] = false;
        }
    }

    return no_inliers;
}

static float compute_residual(
    const std::vector<Eigen::Vector3f> &src,
    const std::vector<Eigen::Vector3f> &dst,
    const Eigen::Matrix4f &pose_estimate)
{
    float residual_sum = 0;
    const auto &R = pose_estimate.topLeftCorner(3, 3);
    const auto &t = pose_estimate.topRightCorner(3, 1);

    for (int i = 0; i < src.size(); ++i)
    {
        residual_sum += (src[i] - (R * dst[i] + t)).norm();
    }
}

void PoseEstimator::RANSAC(
    const std::vector<Eigen::Vector3f> &src,
    const std::vector<Eigen::Vector3f> &dst,
    std::vector<bool> &outliers,
    Eigen::Matrix4f &estimate,
    float &inlier_ratio, float &confidence)
{
    const auto size = src.size();
    inlier_ratio = 0.f;
    int no_iter = 0;
    confidence = 0.f;
    int best_no_inlier = 0;
    float smallest_residual_sum = std::numeric_limits<float>::max();

    //! Compute pose estimate
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();

    if (outliers.size() != size)
        outliers.resize(size);

    while (no_iter < MAX_RANSAC_ITER)
    {
        no_iter++;

        //! randomly sample 3 point pairs
        std::vector<size_t> sampleIdx = {rand() % size, rand() % size, rand() % size};

        if (sampleIdx[0] == sampleIdx[1] ||
            sampleIdx[1] == sampleIdx[2] ||
            sampleIdx[2] == sampleIdx[0])
            continue;

        //! Src points corresponds to the frame
        std::vector<Eigen::Vector3f> src_pts = {src[sampleIdx[0]],
                                                src[sampleIdx[1]],
                                                src[sampleIdx[2]]};

        //! Dst points correspond to the map
        std::vector<Eigen::Vector3f> dst_pts = {dst[sampleIdx[0]],
                                                dst[sampleIdx[1]],
                                                dst[sampleIdx[2]]};

        //! Check if the 3 points are co-linear
        float src_d = (src_pts[1] - src_pts[0]).cross(src_pts[2] - src_pts[0]).norm();
        float dst_d = (dst_pts[1] - dst_pts[0]).cross(dst_pts[2] - dst_pts[0]).norm();
        if (src_d < 1e-6 || dst_d < 1e-6)
            continue;

        const auto valid = absolute_orientation(src_pts, dst_pts, pose);

        if (valid)
        {
            //! Check for outliers
            const auto no_inliers = evaluate_inliers(src, dst, outliers, pose);
            // const auto residual_sum = std::abs(compute_residual(src, dst, estimate));

            if (no_inliers > best_no_inlier)
            {
                best_no_inlier = no_inliers;
                inlier_ratio = (float)no_inliers / src.size();
                confidence = 1 - pow((1 - pow(inlier_ratio, 3)), no_iter + 1);
                estimate = pose;
            }
            // if (residual_sum < smallest_residual_sum)
            // {
            //     smallest_residual_sum = residual_sum;
            //     confidence = 1 - pow((1 - pow(inlier_ratio, 3)), no_iter + 1);
            //     estimate = pose;
            // }

            // if (inlier_ratio >= 0.8 && confidence >= 0.95f)
            //     break;
        }
    }

    const auto no_inliers = evaluate_inliers(src, dst, outliers, estimate);
    const auto valid = absolute_orientation(src, dst, outliers, estimate);
    // const auto residual_sum = std::abs(compute_residual(src, dst, pose));

    // std::cout << "Residual: " << residual_sum << std::endl;

    if (!valid)
    {
        estimate.setIdentity();
    }

    // std::cout << "NO_INLIERS: " << no_inliers << " Confidence: " << confidence << " Ration: " << inlier_ratio << std::endl;
}

}; // namespace fusion