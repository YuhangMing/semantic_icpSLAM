#include "Camera.h"

PrimeSense::PrimeSense() :
		PrimeSense(640, 480, 30) {
}

PrimeSense::PrimeSense(int cols_, int rows_, int fps_)
: cols(cols_), rows(rows_), fps(fps_), device(NULL),
  colorStream(NULL), colorFrame(NULL), depthStream(NULL),
  depthFrame(NULL), setting(NULL) {

	Initialization();
	StartStreaming();
}

PrimeSense::~PrimeSense() {
	StopStreaming();
}

void PrimeSense::SetAutoExposure(bool value) {
	if(!setting)
		setting = colorStream->getCameraSettings();

	setting->setAutoExposureEnabled(value);
}

void PrimeSense::SetAutoWhiteBalance(bool value) {
	if(!setting)
		setting = colorStream->getCameraSettings();

	setting->setAutoWhiteBalanceEnabled(value);
}

void PrimeSense::Initialization() {

	if (openni::OpenNI::initialize() != openni::STATUS_OK) {
		printf("OpenNI Initialisation Failed with Error Message : %s\n", openni::OpenNI::getExtendedError());
		exit(0);
	}

	device = new openni::Device();
	if (device->open(openni::ANY_DEVICE) != openni::STATUS_OK) {
		printf("Couldn't open device\n%s\n", openni::OpenNI::getExtendedError());
		exit(0);
	}

	depthStream = new openni::VideoStream();
	colorStream = new openni::VideoStream();
	if (depthStream->create(*device, openni::SENSOR_DEPTH) != openni::STATUS_OK	||
	    colorStream->create(*device, openni::SENSOR_COLOR) != openni::STATUS_OK) {
		printf("Couldn't create streaming service\n%s\n", openni::OpenNI::getExtendedError());
		exit(0);
	}

	openni::VideoMode depth_video_mode = depthStream->getVideoMode();
	depth_video_mode.setResolution(cols, rows);
	depth_video_mode.setFps(fps);
	depth_video_mode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);

	openni::VideoMode color_video_mode = colorStream->getVideoMode();
	color_video_mode.setResolution(cols, rows);
	color_video_mode.setFps(fps);
	color_video_mode.setPixelFormat(openni::PIXEL_FORMAT_RGB888);

	// save customised mode
	depthStream->setVideoMode(depth_video_mode);
	colorStream->setVideoMode(color_video_mode);

	// Note: Doing image registration earlier than this point seems to fail
	if (device->isImageRegistrationModeSupported(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR)) {
		if (device->setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR) == openni::STATUS_OK) {
			printf("Depth To Colour Image Registration Set Success\n");
		} else {
			printf("Depth To Colour Image Registration Set FAILED\n");
		}
	} else {
		printf("Depth To Colour Image Registration is NOT Supported!!!\n");
	}

	printf("OpenNI Camera Initialisation Complete!\n");
}

void PrimeSense::StartStreaming() {

	depthStream->setMirroringEnabled(false);
	colorStream->setMirroringEnabled(false);

	if (depthStream->start() != openni::STATUS_OK) {
		printf("Couldn't start depth streaming service\n%s\n", openni::OpenNI::getExtendedError());
		exit(0);
	}

	if (colorStream->start() != openni::STATUS_OK) {
		printf("Couldn't start colour streaming service\n%s\n", openni::OpenNI::getExtendedError());
		exit(0);
	}

	depthFrame = new openni::VideoFrameRef();
	colorFrame = new openni::VideoFrameRef();

	printf("OpenNI Camera Streaming Started!\n");
}

void PrimeSense::StopStreaming() {

	depthStream->stop();
	colorStream->stop();

	depthStream->destroy();
	colorStream->destroy();

	device->close();

	openni::OpenNI::shutdown();
}

bool PrimeSense::FetchFrame(cv::Mat& depth, cv::Mat& rgb) {

	openni::VideoStream * streams[] = { depthStream, colorStream };
	int streamReady = -1;
	auto state = openni::STATUS_OK;
	while (state == openni::STATUS_OK) {

		state = openni::OpenNI::waitForAnyStream(streams, 2, &streamReady, 0);
		if (state == openni::STATUS_OK) {
			switch (streamReady) {
			case 0:
				FetchDepthFrame(depth);
				break;
			case 1:
				FetchRGBFrame(rgb);
				break;
			default:
				printf("Unexpected stream number!\n");
				return false;
			}
		}
	}

	if (!colorFrame || !depthFrame || !colorFrame->isValid() || !depthFrame->isValid())
		return false;

	return true;
}

void PrimeSense::FetchRGBFrame(cv::Mat& rgb) {
	if (colorStream->readFrame(colorFrame) != openni::STATUS_OK) {
		printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
	}
	rgb = cv::Mat(rows, cols, CV_8UC3, const_cast<void*>(colorFrame->getData()));
}

void PrimeSense::FetchDepthFrame(cv::Mat& depth) {
	if (depthStream->readFrame(depthFrame) != openni::STATUS_OK) {
		printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
	}
	depth = cv::Mat(rows, cols, CV_16UC1, const_cast<void*>(depthFrame->getData()));
}
