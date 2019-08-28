#include "tracking/build_pyramid.h"
#include "tracking/pose_estimator.h"
#include "tracking/m_estimator.h"
#include "utils/revertable.h"
#include "data_struct/rgbd_frame.h"
#include "tracking/icp_tracker.h"
#include "tracking/device_image.h"

namespace fusion
{

DenseTracking::DenseTracking()
{
  SUM_SE3.create(96, 29, CV_32FC1);
  OUT_SE3.create(1, 29, CV_32FC1);
}

DenseTracking::DenseTracking(const IntrinsicMatrix K, const int NUM_PYR) : DenseTracking()
{
  BuildIntrinsicPyramid(K, cam_params, NUM_PYR);
}

TrackingResult DenseTracking::compute_transform(const RgbdImagePtr reference, const RgbdImagePtr current, const TrackingContext &c)
{
  Revertable<Sophus::SE3d> estimate = Revertable<Sophus::SE3d>(Sophus::SE3d());

  if (c.use_initial_guess_)
    estimate = Revertable<Sophus::SE3d>(c.initial_estimate_);

  bool invalid_error = false;
  Sophus::SE3d init_estimate = estimate.get();

  for (int level = c.max_iterations_.size() - 1; level >= 0; --level)
  {
    cv::cuda::GpuMat curr_vmap = current->get_vmap(level);
    cv::cuda::GpuMat last_vmap = reference->get_vmap(level);
    cv::cuda::GpuMat curr_nmap = current->get_nmap(level);
    cv::cuda::GpuMat last_nmap = reference->get_nmap(level);
    cv::cuda::GpuMat curr_intensity = current->get_intensity(level);
    cv::cuda::GpuMat last_intensity = reference->get_intensity(level);
    cv::cuda::GpuMat intensity_dx = current->get_intensity_dx(level);
    cv::cuda::GpuMat intensity_dy = current->get_intensity_dy(level);
    IntrinsicMatrix K = c.intrinsics_pyr_[level];
    float icp_error = std::numeric_limits<float>::max();
    float rgb_error = std::numeric_limits<float>::max();
    float total_error = std::numeric_limits<float>::max();
    int icp_count = 0, rgb_count = 0;
    float stddev_estimated = 0;

    for (int iter = 0; iter < c.max_iterations_[level]; ++iter)
    {
      auto last_estimate = estimate.get();
      auto last_icp_error = icp_error;
      auto last_rgb_error = rgb_error;

      icp_reduce(
          curr_vmap,
          curr_nmap,
          last_vmap,
          last_nmap,
          SUM_SE3,
          OUT_SE3,
          last_estimate,
          K,
          icp_hessian.data(),
          icp_residual.data(),
          residual_icp_.data());

      float stdev_estimated;

      rgb_step(
          curr_intensity,
          last_intensity,
          last_vmap,
          curr_vmap,
          intensity_dx,
          intensity_dy,
          SUM_SE3,
          OUT_SE3,
          stddev_estimated,
          last_estimate,
          K,
          rgb_hessian.data(),
          rgb_residual.data(),
          residual_rgb_.data());

      stddev_estimated = sqrt(residual_rgb_[0] / (residual_rgb_[1] - 6));

      auto A = 1e6 * icp_hessian + rgb_hessian;
      auto b = 1e6 * icp_residual + rgb_residual;

      update = A.cast<double>().ldlt().solve(b.cast<double>());
      estimate = Sophus::SE3d::exp(update) * last_estimate;

      icp_error = sqrt(residual_icp_(0)) / residual_icp_(1);
      if (std::isnan(icp_error))
      {
        invalid_error = true;
        break;
      }

      if (icp_error > last_icp_error)
      {
        if (icp_count >= 2)
        {
          estimate.revert();
          break;
        }

        icp_count++;
        icp_error = last_icp_error;
      }
      else
      {
        icp_count = 0;
      }

      rgb_error = sqrt(residual_rgb_(0)) / residual_rgb_(1);
      if (std::isnan(rgb_error))
      {
        invalid_error = true;
        break;
      }

      if (rgb_error > last_rgb_error)
      {
        if (rgb_count >= 2)
        {
          estimate.revert();
          break;
        }

        rgb_count++;
        rgb_error = last_rgb_error;
      }
      else
      {
        rgb_count = 0;
      }
    }
  }

  TrackingResult result;

  if (invalid_error || (estimate.get().inverse() * init_estimate).log().norm() > 0.1)
  {
    result.sucess = false;
  }
  else
  {
    result.sucess = true;
    result.update = estimate.get().inverse();
  }

  return result;
}

void DenseTracking::swap_intensity_pyr()
{
  for (int i = 0; i < intensity_src_pyr.size(); ++i)
    intensity_src_pyr[i].swap(intensity_ref_pyr[i]);
}

void DenseTracking::set_source_vmap(cv::cuda::GpuMat vmap)
{
  fusion::build_vmap_pyr(vmap, vmap_src_pyr, cam_params.size());
  fusion::build_nmap_pyr(vmap_src_pyr, nmap_src_pyr);
}

void DenseTracking::set_source_image(cv::cuda::GpuMat image)
{
  cv::cuda::GpuMat intensity;
  cv::cuda::cvtColor(image, intensity, cv::COLOR_RGB2GRAY);
  intensity.convertTo(intensity, CV_32FC1);
  set_source_intensity(intensity);
}

void DenseTracking::set_source_depth(cv::cuda::GpuMat depth_float)
{
  fusion::build_depth_pyr(depth_float, depth_src_pyr, cam_params.size());
}

void DenseTracking::set_source_intensity(cv::cuda::GpuMat intensity)
{
  fusion::build_intensity_pyr(intensity, intensity_src_pyr, cam_params.size());
  fusion::build_intensity_dxdy_pyr(intensity_src_pyr, intensity_dx_pyr, intensity_dy_pyr);
}

void DenseTracking::set_reference_image(cv::cuda::GpuMat image)
{
  cv::cuda::GpuMat intensity;
  cv::cuda::cvtColor(image, intensity, cv::COLOR_RGB2GRAY);
  intensity.convertTo(intensity, CV_32FC1);
  set_reference_intensity(intensity);
}

void DenseTracking::set_reference_intensity(cv::cuda::GpuMat intensity)
{
  fusion::build_intensity_pyr(intensity, intensity_ref_pyr, cam_params.size());
}

void DenseTracking::set_reference_vmap(cv::cuda::GpuMat vmap)
{
  fusion::build_vmap_pyr(vmap, vmap_ref_pyr, cam_params.size());
  fusion::build_nmap_pyr(vmap_ref_pyr, nmap_ref_pyr);
}

TrackingResult DenseTracking::compute_transform(const TrackingContext &context)
{
  Revertable<Sophus::SE3d> estimate = Revertable<Sophus::SE3d>(Sophus::SE3d());

  if (context.use_initial_guess_)
    estimate = Revertable<Sophus::SE3d>(context.initial_estimate_);

  for (int level = context.max_iterations_.size() - 1; level >= 0; --level)
  {
    cv::cuda::GpuMat curr_vmap = vmap_src_pyr[level];
    cv::cuda::GpuMat last_vmap = vmap_ref_pyr[level];
    cv::cuda::GpuMat curr_nmap = nmap_src_pyr[level];
    cv::cuda::GpuMat last_nmap = nmap_ref_pyr[level];
    cv::cuda::GpuMat curr_intensity = intensity_src_pyr[level];
    cv::cuda::GpuMat last_intensity = intensity_ref_pyr[level];
    cv::cuda::GpuMat intensity_dx = intensity_dx_pyr[level];
    cv::cuda::GpuMat intensity_dy = intensity_dy_pyr[level];
    IntrinsicMatrix K = cam_params[level];
    float icp_error = std::numeric_limits<float>::max();
    float rgb_error = std::numeric_limits<float>::max();
    float total_error = std::numeric_limits<float>::max();
    int icp_count = 0, rgb_count = 0;
    float stddev_estimated = 0;

    for (int iter = 0; iter < context.max_iterations_[level]; ++iter)
    {
      auto last_estimate = estimate.get();
      last_icp_error = icp_error;
      last_rgb_error = rgb_error;

      icp_reduce(
          curr_vmap,
          curr_nmap,
          last_vmap,
          last_nmap,
          SUM_SE3,
          OUT_SE3,
          last_estimate,
          K,
          icp_hessian.data(),
          icp_residual.data(),
          residual_icp_.data());

      float stdev_estimated;

      rgb_step(
          curr_intensity,
          last_intensity,
          last_vmap,
          curr_vmap,
          intensity_dx,
          intensity_dy,
          SUM_SE3,
          OUT_SE3,
          stddev_estimated,
          last_estimate,
          K,
          rgb_hessian.data(),
          rgb_residual.data(),
          residual_rgb_.data());

      stddev_estimated = sqrt(residual_rgb_[0] / (residual_rgb_[1] - 6));

      auto A = 1e6 * icp_hessian + rgb_hessian;
      auto b = 1e6 * icp_residual + rgb_residual;
      // auto A = icp_hessian;
      // auto b = icp_residual;

      update = A.cast<double>().ldlt().solve(b.cast<double>());
      estimate = Sophus::SE3d::exp(update) * last_estimate;

      icp_error = sqrt(residual_icp_(0)) / residual_icp_(1);

      if (icp_error > last_icp_error)
      {
        if (icp_count >= 2)
        {
          estimate.revert();
          break;
        }

        icp_count++;
        icp_error = last_icp_error;
      }
      else
      {
        icp_count = 0;
      }

      rgb_error = sqrt(residual_rgb_(0)) / residual_rgb_(1);

      if (rgb_error > last_rgb_error)
      {
        if (rgb_count >= 2)
        {
          estimate.revert();
          break;
        }

        rgb_count++;
        rgb_error = last_rgb_error;
      }
      else
      {
        rgb_count = 0;
      }
    }
  }

  if (estimate.get().log().transpose().norm() > 0.1)
    std::cout << estimate.get().log().transpose().norm() << std::endl;

  TrackingResult result;
  result.sucess = true;
  result.icp_error = last_icp_error;
  result.update = estimate.get().inverse();
  return result;
}

cv::cuda::GpuMat DenseTracking::get_vmap_src(const int &level)
{
  return vmap_src_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_nmap_src(const int &level)
{
  return nmap_src_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_depth_src(const int &level)
{
  return depth_src_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_intensity_src(const int &level)
{
  return intensity_src_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_intensity_dx(const int &level)
{
  return intensity_dx_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_intensity_dy(const int &level)
{
  return intensity_dy_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_vmap_ref(const int &level)
{
  return vmap_ref_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_nmap_ref(const int &level)
{
  return nmap_ref_pyr[level];
}

cv::cuda::GpuMat DenseTracking::get_intensity_ref(const int &level)
{
  return intensity_ref_pyr[level];
}

} // namespace fusion