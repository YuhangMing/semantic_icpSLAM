#ifndef DENSE_MAPPING_H
#define DENSE_MAPPING_H

#include <memory>
#include "data_struct/map_struct.h"
#include "data_struct/rgbd_frame.h"
#include "tracking/device_image.h"

namespace fusion
{

class DenseMapping
{
public:
  ~DenseMapping();
  DenseMapping(const fusion::IntrinsicMatrix &K, int idx, bool bTrack, bool bRender);

  void update(RgbdImagePtr frame);
  void update(cv::cuda::GpuMat depth, cv::cuda::GpuMat image, const Sophus::SE3d pose);
  void raycast(cv::cuda::GpuMat &vmap, cv::cuda::GpuMat &image, const Sophus::SE3d pose);

  void raycast_check_visibility(
      cv::cuda::GpuMat &vmap,
      cv::cuda::GpuMat &image,
      const Sophus::SE3d pose);

  void reset_mapping();

  size_t fetch_mesh_vertex_only(void *vertex);
  size_t fetch_mesh_with_normal(void *vertex, void *normal);
  size_t fetch_mesh_with_colour(void *vertex, void *normal);

  void writeMapToDisk(std::string file_name);
  void readMapFromDisk(std::string file_name);

  // submap data structures
  int submapIdx;
  bool bTrack, bRender;
  float visible_percent;
  Sophus::SE3d poseGlobal;
  std::vector<RgbdFramePtr> vKFs; // stored whenever a new kf is created
  float CheckVisPercent();
  void check_visibility(RgbdImagePtr frame);

  // semantic
  void color_objects(RgbdImagePtr frame);

private:
  IntrinsicMatrix cam_params;
  MapStruct<true> device_map;

  // for map udate
  cv::cuda::GpuMat flag;
  cv::cuda::GpuMat pos_array;
  uint count_visible_block;
  HashEntry *visible_blocks;

  // for raycast
  cv::cuda::GpuMat zrange_x;
  cv::cuda::GpuMat zrange_y;
  uint count_rendering_block;
  RenderingBlock *rendering_blocks;
};

} // namespace fusion

#endif