#include <cmath>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv.hpp>
#include <highgui.hpp>
#include <ctime>
#include <DeviceMap.h>

#include "Camera.h"
#include "Tracking.h"

// #define TIMING

int main(int argc, char** argv) {

	SysDesc desc;
	PrimeSense cam;
	cv::Mat imD, imRGB;

	desc.DepthCutoff = 3.0f;
	desc.DepthScale = 1000.0f;
	desc.cols = 640;
	desc.rows = 480;
	desc.fx = 520.149963;
	desc.fy = 516.175781;
	desc.cx = 309.993548;
	desc.cy = 227.090932;
	desc.TrackModel = true;
	desc.bUseDataset = false;

	std::cout << "Size of a voxel is " << sizeof(Voxel) << std::endl;

	System slam(&desc);
//	cam.SetAutoExposure(false);
//	cam.SetAutoWhiteBalance(false);

	while (1) {
		if (cam.FetchFrame(imD, imRGB)) {
			std::clock_t start = std::clock();

			bool valid = slam.GrabImage(imRGB, imD);
#ifdef TIMING
			std::cout << "------------ In Total Takes "
					  << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
					  << "------------" << std::endl;
#endif
			if(!valid) {
				cam.StopStreaming();
				return 0;
			}
		}
	}
}
