#include "relocalization/relocalizer.h"
#include "relocalization/ransac_ao.h"
#include "tracking/cuda_imgproc.h"

namespace fusion
{

Relocalizer::Relocalizer(const fusion::IntrinsicMatrix K) : cam_param(K)
{
}

void Relocalizer::setFeatureExtractor(std::shared_ptr<FeatureExtractor> ext)
{
    extractor = ext;
}

void Relocalizer::setDescriptorMatcher(std::shared_ptr<DescriptorMatcher> mcr)
{
    matcher = mcr;
}

void Relocalizer::set_map_points(std::vector<std::shared_ptr<Point3d>> mapPoints, cv::Mat &mapDescriptors)
{
    map_points = mapPoints;
    map_descriptors = mapDescriptors;
}

void Relocalizer::set_target_frame(std::shared_ptr<RgbdFrame> frame)
{
    target_frame = frame;
}

void Relocalizer::compute_pose_candidates(std::vector<Sophus::SE3d> &candidates)
{
    target_frame->pose = Sophus::SE3d();
    std::vector<cv::KeyPoint> raw_keypoints;
    cv::Mat raw_descriptors;

    cv::cuda::GpuMat depth(target_frame->depth);
    cv::cuda::GpuMat vmap_gpu, nmap_gpu;
    backProjectDepth(depth, vmap_gpu, cam_param);
    computeNMap(vmap_gpu, nmap_gpu);

    extractor->extract_features_surf(
        target_frame->image,
        raw_keypoints,
        raw_descriptors);

    extractor->compute_3d_points(
        cv::Mat(vmap_gpu),
        cv::Mat(nmap_gpu),
        raw_keypoints,
        raw_descriptors,
        target_frame->cv_key_points,
        target_frame->descriptors,
        target_frame->key_points,
        target_frame->pose.cast<float>());

    std::vector<std::vector<cv::DMatch>> matches;
    matcher->match_hamming_knn(
        map_descriptors,
        target_frame->descriptors,
        matches, 2);

    std::vector<cv::DMatch> list;
    std::vector<std::vector<cv::DMatch>> candidate_matches;
    matcher->filter_matches_ratio_test(matches, list);
    candidate_matches.push_back(list);
    // matcher->filter_matches_pair_constraint(target_frame->key_points, map_points, matches, candidate_matches);

    for (const auto &match_list : candidate_matches)
    {
        std::vector<Eigen::Vector3f> src_pts, dst_pts;
        for (const auto &match : match_list)
        {
            src_pts.push_back(map_points[match.trainIdx]->pos);
            dst_pts.push_back(target_frame->key_points[match.queryIdx]->pos);
        }

        std::vector<bool> outliers;
        Eigen::Matrix4f estimate;
        float inlier_ratio, confidence;
        PoseEstimator::RANSAC(src_pts, dst_pts, outliers, estimate, inlier_ratio, confidence);

        const int no_inliers = std::count(outliers.begin(), outliers.end(), false);
        std::cout << estimate << std::endl
                  << no_inliers << std::endl;

        candidates.emplace_back(Sophus::SE3f(estimate).cast<double>());
    }
}

} // namespace fusion
