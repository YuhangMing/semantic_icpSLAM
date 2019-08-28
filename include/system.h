#ifndef SYSTEM_H
#define SYSTEM_H

#include <thread>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>
#include "data_struct/intrinsic_matrix.h"
#include "data_struct/rgbd_frame.h"
#include "voxel_hashing/voxel_hashing.h"
#include "tracking/rgbd_odometry.h"
#include "relocalization/relocalizer.h"
#include "keyframe_graph/keyframe_graph.h"
#include "features/extractor.h"
#include "features/matcher.h"
#include "map_manager.h"
#include "detection/MaskRCNN.h"

namespace fusion
{

class SubMapManager;
class DenseOdometry;
// class MaskRCNN;

class System
{
public:
    ~System();
    System(const fusion::IntrinsicMatrix base, const int NUM_PYR);
    void process_images(const cv::Mat depth, const cv::Mat image, const fusion::IntrinsicMatrix base);

    // get rendered ray tracing map
    cv::Mat get_detected_image();
    cv::Mat get_shaded_depth();
    cv::Mat get_rendered_scene() const;
    cv::Mat get_rendered_scene_textured() const;

    // create a mesh from the map
    // and save it to a named file
    // it only contains vertex data
    void save_mesh_to_file(const char *str);

    // create mesh and store in the address
    // users are reponsible for allocating
    // the adresses in CUDA using `cudaMalloc`
    size_t fetch_mesh_vertex_only(float *vertex);
    size_t fetch_mesh_with_normal(float *vertex, float *normal);
    size_t fetch_mesh_with_colour(float *vertex, unsigned char *colour);

    // key points
    void fetch_key_points(float *points, size_t &count, size_t max);
    void fetch_key_points_with_normal(float *points, float *normal, size_t &max_size);
    std::vector<Eigen::Matrix<float, 4, 4>> getKeyFramePoses() const;

    bool is_initialized;

    // system controls
    void change_colour_mode(int colour_mode = 0);
    void change_run_mode(int run_mode = 0);
    void restart();
    void setLost(bool lost);

    void writeMapToDisk(std::string file_name) const;
    void readMapFromDisk(std::string file_name);

    Eigen::Matrix4f get_camera_pose() const;

private:
    // no more separate keyframe structure, keyframe is frame
    RgbdFramePtr current_frame;
    RgbdFramePtr last_tracked_frame;
    RgbdFramePtr current_keyframe;

    size_t frame_id;

    // System modules
    std::shared_ptr<SubMapManager> manager;
    // std::shared_ptr<DenseMapping> mapping;
    std::shared_ptr<DenseOdometry> odometry;
    std::shared_ptr<KeyFrameGraph> graph;
    std::shared_ptr<Relocalizer> relocalizer;
    std::shared_ptr<FeatureExtractor> extractor;
    std::shared_ptr<DescriptorMatcher> matcher;
    semantic::MaskRCNN * detector;
    std::thread graphThread;

    // Return TRUE if a new key frame is desired
    // return FALSE otherwise
    // TODO: this needs to be redesigned.
    bool keyframe_needed() const;
    void create_keyframe();
    void initialization();
    bool hasNewKeyFrame;
    Sophus::SE3d initialPose;

    cv::cuda::GpuMat device_depth_float;
    cv::cuda::GpuMat device_image_uchar;
    cv::cuda::GpuMat device_vmap_cast;
    cv::cuda::GpuMat device_nmap_cast;

    int renderIdx;

    void extract_objects(float lamb, float tao, int win_size, int thre);
};

} // namespace fusion

#endif