#ifndef CAMERA_HPP__
#define CAMERA_HPP__

#include <OpenNI.h>
#include <opencv.hpp>

using openni::Device;
using openni::VideoStream;
using openni::VideoFrameRef;

class PrimeSense {

public:

	PrimeSense();

	PrimeSense(int cols_, int rows_, int fps_);

	~PrimeSense();

	void Initialization();

	void StartStreaming();

	void StopStreaming();

	bool FetchFrame(cv::Mat & depth, cv::Mat & rgb);

	void SetAutoExposure(bool value);

	void SetAutoWhiteBalance(bool value);

protected:

	void FetchRGBFrame(cv::Mat & rgb);

	void FetchDepthFrame(cv::Mat & depth);

	openni::Device * device;

	openni::VideoStream * colorStream;

	openni::VideoStream * depthStream;

	openni::VideoFrameRef * colorFrame;

	openni::VideoFrameRef * depthFrame;

	openni::CameraSettings * setting;

	int cols;

	int rows;

	int fps;
};

#endif
