#include <cmath>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <unistd.h>
#include <ctime>
#include <DeviceMap.h>

#include "Tracking.h"

void load_tum_dataset(std::string & dataset_path,
		std::vector<std::string> & depth_image_list,
		std::vector<std::string> & rgb_image_list,
		std::vector<double> & time_stamp_list);

int main(int argc, char** argv) {

	if (argc != 2) {
		exit(-1);
	}

	std::string data_path = std::string(argv[1]);
	std::vector<std::string> depth_image_list;
	std::vector<std::string> rgb_image_list;
	std::vector<double> time_stamp_list;

	load_tum_dataset(data_path, depth_image_list, rgb_image_list, time_stamp_list);

	if (depth_image_list.empty()) {
		std::cout << "Error occurs while reading the dataset.\n"
				  << "Please check your input parameters." << std::endl;
		exit(-1);
	}

	SysDesc desc;

	desc.DepthCutoff = 3.0f;
	desc.DepthScale = 5000.0f;
	desc.cols = 640;
	desc.rows = 480;
	desc.fx = 517.3;
	desc.fy = 516.5;
	desc.cx = 318.6;
	desc.cy = 255.3;
	desc.TrackModel = true;
	desc.bUseDataset = false;

	std::cout << "Size of a voxel is " << sizeof(Voxel) << std::endl;

	System slam(&desc);

	int N = std::min(rgb_image_list.size(), time_stamp_list.size());
	for (int i = 0; i < N; ++i) {
		std::clock_t start = std::clock();

		cv::Mat depth = cv::imread(depth_image_list[i], cv::IMREAD_UNCHANGED);
		cv::Mat image = cv::imread(rgb_image_list[i], cv::IMREAD_UNCHANGED);
		cvtColor(image, image, cv::COLOR_BGR2RGB);
		bool nonstop = slam.GrabImage(image, depth);
		
		std::cout << "------------ In Total Takes "
				  << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
				  << "------------" << std::endl;
		
		if(!nonstop)
			return 0;

        // usleep(500000);
		// exit(-1);
	}

//
//	SysDesc desc;
//
//	desc.DepthCutoff = 3.0f;
//	desc.DepthScale = 1000.0f;
//	desc.cols = 640;
//	desc.rows = 480;
//	desc.fx = 583;
//	desc.fy = 583;
//	desc.cx = 320;
//	desc.cy = 240;
//	desc.TrackModel = true;
//	desc.bUseDataset = false;
//
//	System slam(&desc);
//
//	for (int i = 0; i < 8560; ++i) {
//		std::stringstream ss;
//		ss << std::setfill('0') << std::setw(6) << i;
//		std::string number = "";
//		std::string img_depth = "frame-";
//		std::string img_rgb = "frame-";
//		ss >> number;
//		img_depth += number;
//		img_depth += ".depth.png";
//		img_rgb += number;
//		img_rgb += ".color.jpg";
//
//		cv::Mat depth = cv::imread(std::string("/home/xyang/Downloads/apt0/") + img_depth, cv::IMREAD_UNCHANGED);
//		cv::Mat image = cv::imread(std::string("/home/xyang/Downloads/apt0/") + img_rgb, cv::IMREAD_UNCHANGED);
//		cvtColor(image, image, cv::COLOR_BGR2RGB);
//		bool nonstop = slam.GrabImage(image, depth);
//		if(!nonstop)
//			return 0;
//	}

	slam.JoinViewer();
}

void load_tum_dataset(std::string & dataset_path,
		std::vector<std::string> & depth_image_list,
		std::vector<std::string> & rgb_image_list,
		std::vector<double> & time_stamp_list) {

	std::ifstream depth_file, rgb_file;
	std::string depth_file_path, rgb_file_path;

	if (dataset_path.back() != '/')
		dataset_path += '/';

	depth_file.open(dataset_path + "depth.txt");
	rgb_file.open(dataset_path + "rgb.txt");

	std::string temp;
	for (int i = 0; i < 3; ++i) {
		getline(depth_file, temp);
		getline(rgb_file, temp);
	}

	std::string line, depth_line, rgb_line;
	while (true) {
		double t;
		depth_file >> t;
		time_stamp_list.push_back(t);
		depth_file >> depth_line;
		depth_image_list.push_back(dataset_path + depth_line);
		rgb_file >> rgb_line;
		rgb_file >> rgb_line;
		rgb_image_list.push_back(dataset_path + rgb_line);
		if (depth_file.eof() || rgb_file.eof())
			return;
	}
}
