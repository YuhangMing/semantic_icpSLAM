#include <cuda_runtime_api.h>
#include "utils/safe_call.h"
#include "data_struct/map_struct.h"
#include "voxel_hashing/map_proc.h"
#include "voxel_hashing/voxel_hashing.h"

namespace fusion
{

FUSION_HOST void *deviceMalloc(size_t sizeByte)
{
  void *dev_ptr;
  safe_call(cudaMalloc((void **)&dev_ptr, sizeByte));
  return dev_ptr;
}

FUSION_HOST void deviceRelease(void **dev_ptr)
{
  if (*dev_ptr != NULL)
    safe_call(cudaFree(*dev_ptr));

  *dev_ptr = 0;
}

DenseMapping::DenseMapping(const IntrinsicMatrix &K, int idx, bool bTrack, bool bRender) 
          : cam_params(K), submapIdx(idx), bTrack(bTrack), bRender(bRender)
{
  device_map.create();
  zrange_x.create(cam_params.height / 8, cam_params.width / 8, CV_32FC1);
  zrange_y.create(cam_params.height / 8, cam_params.width / 8, CV_32FC1);

  visible_blocks = (HashEntry *)deviceMalloc(sizeof(HashEntry) * device_map.state.num_total_hash_entries_);
  rendering_blocks = (RenderingBlock *)deviceMalloc(sizeof(RenderingBlock) * 100000);

  reset_mapping();
}

DenseMapping::~DenseMapping()
{
  device_map.release();
  deviceRelease((void **)&visible_blocks);
  deviceRelease((void **)&rendering_blocks);
}

void DenseMapping::update(RgbdImagePtr frame)
{
  auto image = frame->get_image();
  auto depth = frame->get_raw_depth();
  auto normal = frame->get_nmap();
  auto pose = frame->get_reference_frame()->pose;

  count_visible_block = 0;

  cuda::update_weighted(
      device_map.map,
      device_map.state,
      depth,
      normal,
      image,
      pose,
      cam_params,
      flag,
      pos_array,
      visible_blocks,
      count_visible_block);
}

void DenseMapping::update(
    cv::cuda::GpuMat depth,
    cv::cuda::GpuMat image,
    const Sophus::SE3d pose)
{
  count_visible_block = 0;

  cuda::update(
      device_map.map,
      device_map.state,
      depth,
      image,
      pose,
      cam_params,
      flag,
      pos_array,
      visible_blocks,
      count_visible_block);
}

void DenseMapping::raycast(
    cv::cuda::GpuMat &vmap,
    cv::cuda::GpuMat &image,
    const Sophus::SE3d pose)
{
  if (count_visible_block == 0)
    return;

  cuda::create_rendering_blocks(
      count_visible_block,
      count_rendering_block,
      visible_blocks,
      zrange_x,
      zrange_y,
      rendering_blocks,
      pose,
      cam_params);

  if (count_rendering_block != 0)
  {

    cuda::raycast_with_colour(
        device_map.map,
        device_map.state,
        vmap,
        vmap,
        image,
        zrange_x,
        zrange_y,
        pose,
        cam_params);
  }
}

// --Yohann
void DenseMapping::check_visibility(RgbdImagePtr frame)
{
  auto image = frame->get_image();
  auto depth = frame->get_raw_depth();
  auto normal = frame->get_nmap();
  auto pose = frame->get_reference_frame()->pose;

  count_visible_block = 0;

  cuda::check_visibility(
      device_map.map,
      device_map.state,
      depth,
      pose,
      cam_params,
      flag,
      pos_array,
      visible_blocks,
      count_visible_block);
}

void DenseMapping::color_objects(RgbdImagePtr frame)
{
  auto image = frame->get_image();
  auto depth = frame->get_raw_depth();
  auto normal = frame->get_nmap();
  auto mask = frame->get_object_mask();
  auto pose = frame->get_reference_frame()->pose;

  count_visible_block = 0;

  cuda::color_objects(
    device_map.map,
    device_map.state,
    depth,
    image,
    mask,
    pose,
    cam_params,
    flag,
    pos_array,
    visible_blocks,
    count_visible_block);
}

void DenseMapping::raycast_check_visibility(
    cv::cuda::GpuMat &vmap,
    cv::cuda::GpuMat &image,
    const Sophus::SE3d pose)
{
  fusion::cuda::count_visible_entry(
      device_map.map,
      device_map.size,
      cam_params,
      pose.inverse(),
      visible_blocks,
      count_visible_block);

  raycast(vmap, image, pose);
}

void DenseMapping::reset_mapping()
{
  device_map.reset();
}

size_t DenseMapping::fetch_mesh_vertex_only(void *vertex)
{
  uint count_triangle = 0;

  cuda::create_mesh_vertex_only(
      device_map.map,
      device_map.state,
      count_visible_block,
      visible_blocks,
      count_triangle,
      vertex);

  return (size_t)count_triangle;
}

size_t DenseMapping::fetch_mesh_with_normal(void *vertex, void *normal)
{
  uint count_triangle = 0;

  cuda::create_mesh_with_normal(
      device_map.map,
      device_map.state,
      count_visible_block,
      visible_blocks,
      count_triangle,
      vertex,
      normal);

  return (size_t)count_triangle;
}

size_t DenseMapping::fetch_mesh_with_colour(void *vertex, void *colour)
{
  uint count_triangle = 0;

  cuda::create_mesh_with_colour(
      device_map.map,
      device_map.state,
      count_visible_block,
      visible_blocks,
      count_triangle,
      vertex,
      colour);

  return (size_t)count_triangle;
}

void DenseMapping::writeMapToDisk(std::string file_name)
{
  MapStruct<false> host_map;
  host_map.create();
  device_map.download(host_map);
  host_map.writeToDisk(file_name);
  host_map.release();
}

void DenseMapping::readMapFromDisk(std::string file_name)
{
  MapStruct<false> host_map;
  host_map.create();
  host_map.readFromDisk(file_name);
  device_map.upload(host_map);
  host_map.release();
}

float DenseMapping::CheckVisPercent() {
  // copy heap_mem_counter_ from device to host
  int *remain = new int[1];
  safe_call(cudaMemcpy(remain, device_map.map.heap_mem_counter_, sizeof(int), cudaMemcpyDeviceToHost));
  int remain_relu = remain[0] > 0 ? remain[0] : 0;

  visible_percent = float(count_visible_block) / (device_map.state.num_total_voxel_blocks_ - remain_relu);
  // std::cout << visible_percent << ": " << count_visible_block << "/" 
  //           << (device_map.state.num_total_voxel_blocks_ - remain_relu) << "/" 
  //           << remain_relu << std::endl;

  return visible_percent;
}

} // namespace fusion