#ifndef FUSION_RGBD_FRAME_H
#define FUSION_RGBD_FRAME_H

#include <map>
#include <mutex>
#include <memory>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include "data_struct/map_point.h"
#include "detection/MaskRCNN.h"

namespace fusion
{

class RgbdFrame;
using RgbdFramePtr = std::shared_ptr<RgbdFrame>;

class RgbdFrame
{
public:
  RgbdFrame(const cv::Mat &depth, const cv::Mat &image, const size_t id, const double ts);
  RgbdFrame();
  void copyTo(RgbdFramePtr dst);

  std::vector<cv::KeyPoint> cv_key_points;
  std::vector<std::shared_ptr<Point3d>> key_points;
  std::map<RgbdFramePtr, Eigen::Matrix4f> neighbours;
  cv::Mat descriptors;

  std::size_t id;
  double timeStamp;
  Sophus::SE3d pose;

  cv::Mat image;
  cv::Mat depth;
  cv::Mat vmap;
  cv::Mat nmap;

  int row_frame, col_frame;

  // data structure for maskRCNN detection results
  int numDetection;
  // int *pMasks;
  // long int *pLabels;        Removing these 4 improve speed about 0.1 second
  // float *pScores, *pBoxes;

  std::vector<cv::Mat> vMasks;
  std::vector<int> vLabels;
  std::vector<float> vScores;
  std::vector<int> vRemovedIdx;

  cv::Mat mask;
  // bool colorObject;
  
  // Geometric Segmentation results
  int nConComps;
  cv::Mat mLabeled, mStats, mCentroids;
  int pallete[3] = {int(pow(2, 25)-1), int(pow(2, 15)-1), int(pow(2, 21)-1)};

  void ExtractObjects(semantic::MaskRCNN* detector, bool bBbox, bool bContour, bool bText);
  // void GeometricRefinement(float lamb, float tao, int win_size);
  void FuseMasks(cv::Mat edge, int thre);
  cv::Scalar CalculateColor(long int label);
  cv::Mat Array2Mat(int* aMask);
};

} // namespace fusion

#endif