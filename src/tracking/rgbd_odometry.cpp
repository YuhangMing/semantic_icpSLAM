#include "tracking/rgbd_odometry.h"
#include "tracking/icp_tracker.h"

namespace fusion
{

DenseOdometry::DenseOdometry(const fusion::IntrinsicMatrix base, int NUM_PYR)
    : tracker(new DenseTracking()),
      trackingLost(false),
      initialized(false)
{
  currDeviceMapPyramid = std::make_shared<DeviceImage>(base, NUM_PYR);
  std::shared_ptr<DeviceImage> refDeviceMapPyramid = std::make_shared<DeviceImage>(base, NUM_PYR);
  vModelDeviceMapPyramid.push_back(refDeviceMapPyramid);
  BuildIntrinsicPyramid(base, cam_params, NUM_PYR);
}

void DenseOdometry::trackFrame(std::shared_ptr<RgbdFrame> frame)
{
  // CURRENT, updated in every submap
  currDeviceMapPyramid->upload(frame);

  if (!initialized)
  {
    // std::cout << "Odometry: Initializing... " << std::endl;
    vModelFrames.push_back(frame);
    copyDeviceImage(currDeviceMapPyramid, vModelDeviceMapPyramid[submapIdx]);
    initialized = true;
    return;
  }

  // std::cout << "Odometry: Set context... " << std::endl;
  context.use_initial_guess_ = true;
  context.initial_estimate_ = Sophus::SE3d();
  context.intrinsics_pyr_ = cam_params;
  context.max_iterations_ = {10, 5, 3, 3, 3};

  // std::cout << "Odometry: Compute transform... " << std::endl;
  if(manager->active_submaps[submapIdx]->bTrack){
    result = tracker->compute_transform(vModelDeviceMapPyramid[submapIdx], currDeviceMapPyramid, context);
    result.update = vModelFrames[submapIdx]->pose * result.update;
  } else {
    Sophus::SE3d Tmf = vModelFrames[trackIdx]->pose;
    Sophus::SE3d Twm = manager->active_submaps[trackIdx]->poseGlobal;
    Sophus::SE3d Twcinv = manager->active_submaps[submapIdx]->poseGlobal.inverse();
    // pose of input frame w.r.t. current sm
    result.update = Twcinv * Twm * Tmf;
    result.sucess = true;
  }

  // std::cout << "Odometry: Update LastFrame... " << std::endl;
  if (result.sucess)
  {
    // if(manager->active_submaps[submapIdx]->bTrack)
    //   frame->pose = vModelFrames[submapIdx]->pose * result.update;
    // else
    frame->pose = result.update;
    vModelFrames[submapIdx] = frame;
    copyDeviceImage(currDeviceMapPyramid, vModelDeviceMapPyramid[submapIdx]);
    trackingLost = false;
  }
  else
  {
    trackingLost = true;
  }
}

std::shared_ptr<DeviceImage> DenseOdometry::get_current_image() const
{
  return currDeviceMapPyramid;
}

std::shared_ptr<DeviceImage> DenseOdometry::get_reference_image(int i) const
{
  return vModelDeviceMapPyramid[i];
}

// Eigen::Matrix4f DenseOdometry::get_current_pose_matrix() const
// {
//   if (currDeviceMapPyramid && currDeviceMapPyramid->get_reference_frame())
//   {
//     return currDeviceMapPyramid->get_reference_frame()->pose.matrix().cast<float>();
//   }
//   else
//     return Eigen::Matrix4f::Identity();
// }

void DenseOdometry::reset()
{
  vModelFrames.clear();
  initialized = false;
  trackingLost = false;
}

void DenseOdometry::SetManager(std::shared_ptr<SubMapManager> pManager){
  manager = pManager;
}

void DenseOdometry::setSubmapIdx(int idx){
  submapIdx = idx;
}

void DenseOdometry::setTrackIdx(int idx){
  trackIdx = idx;
}

// void DenseOdometry::SetDetector(semantic::MaskRCNN * pDetector){
//   detector = pDetector;
// }

} // namespace fusion