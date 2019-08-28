#ifndef FUSION_INPUT_ONI_CAMERA_H
#define FUSION_INPUT_ONI_CAMERA_H

#include <opencv2/opencv.hpp>
#include <OpenNI2/OpenNI.h>

namespace fusion
{

class ONICamera
{
public:
    ONICamera();
    ~ONICamera();
    ONICamera(int cols, int rows, int fps);
    bool get_next_images(cv::Mat &depth, cv::Mat &image);

private:
    int width, height, frame_rate;

    openni::Device device;
    openni::VideoStream depth_stream;
    openni::VideoStream color_stream;
    openni::VideoFrameRef depth_frame;
    openni::VideoFrameRef color_frame;
};

} // namespace fusion

#endif