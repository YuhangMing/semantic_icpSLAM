#ifndef FRAME_HPP__
#define FRAME_HPP__

#include "DeviceMap.h"
#include "DeviceArray.h"
#include "KeyFrame.h"
#include "MaskRCNN.h"

#include <vector>
#include <opencv.hpp>
#include <features2d.hpp>
#include <cudaarithm.hpp>
#include <Eigen/Dense>
#include <xfeatures2d/cuda.hpp>
#include <cmath>

struct ORBKey;
struct KeyFrame;

struct Frame {

	static const int NUM_PYRS = 3;
	// static const int NUM_PYRS = 5;
	static const int MIN_KEY_POINTS = 500;

	Frame();

	Frame(const Frame * other);

	void Create(int cols_, int rows_);

	void ExtractKeyPoints();

    //////////
	void ExtractObjects(MaskRCNN* detector, bool bBbox, bool bContour, bool bText);
	void GeometricRefinement(float lamb, float tao, int win_size);
	void FuseMasks(int thre);
	cv::Scalar CalculateColor(long int label);
	cv::Mat Array2Mat(int* aMask);

	void ResizeImages();

	void Clear();

	void DrawKeyPoints();

	void ClearKeyPoints();

	void FillImages(const cv::Mat & range_, const cv::Mat & color_);

	float InterpDepth(cv::Mat & map, float & x, float & y);

	float4 InterpNormal(cv::Mat & map, float & x, float & y);

	Eigen::Matrix3d Rotation() const;

	Eigen::Matrix3d RotationInv() const;

	Eigen::Vector3d Translation() const;

	Eigen::Vector3d TranslationInv() const;

	Matrix3f GpuRotation() const;

	float3 GpuTranslation() const;

	Matrix3f GpuInvRotation() const;

	Eigen::Vector3f GetWorldPoint(int i) const;

	DeviceArray2D<unsigned short> temp;
	DeviceArray2D<float> range;
	DeviceArray2D<uchar3> color;

	DeviceArray2D<float4> vmap[NUM_PYRS];
	DeviceArray2D<float4> nmap[NUM_PYRS];
	DeviceArray2D<float> depth[NUM_PYRS];
	DeviceArray2D<unsigned char> image[NUM_PYRS];
	DeviceArray2D<short> dIdx[NUM_PYRS];
	DeviceArray2D<short> dIdy[NUM_PYRS];

	unsigned long frameId;
	static unsigned long nextId;

	std::vector<bool> outliers;

	Eigen::Matrix4f deltaT;
	Eigen::Matrix4d pose;

	int N;
	bool bad;
	cv::cuda::GpuMat descriptors;
	std::vector<float4> pointNormal;
	std::vector<Eigen::Vector3f> mapPoints;
	std::vector<cv::KeyPoint> keyPoints;

	static cv::cuda::SURF_CUDA surfExt;
	static cv::Ptr<cv::BRISK> briskExt;

	static cv::Mat mK[NUM_PYRS];
	static int mCols[NUM_PYRS];
	static int mRows[NUM_PYRS];

	static float fx(int pyr);
	static float fy(int pyr);
	static float cx(int pyr);
	static float cy(int pyr);
	static int cols(int pyr);
	static int rows(int pyr);
	static void SetK(cv::Mat& K);
	static bool mbFirstCall;
	static float mDepthScale;
	static float mDepthCutoff;

	int row_frame, col_frame;
	// data structure for maskRCNN detection results
    int numDetection;
    int *pMasks;
    long int *pLabels;
    float *pScores, *pBoxes;
	cv::Mat mask;
	bool colorObject;
	// cv::Mat colorMask(480, 640, CV_8UC3);
    int pallete[3] = {int(pow(2, 25)-1), int(pow(2, 15)-1), int(pow(2, 21)-1)};
    std::vector<cv::Mat> vMasks;
	std::vector<int> vLabels;
	std::vector<float> vScores;
	std::vector<int> vRemovedIdx;

	DeviceArray2D<unsigned char> detected_masks;
	DeviceArray<int> detected_labels;
	DeviceArray<float> detected_scores;

	// Geometric Segmentation results
	int nConComps;
	cv::Mat mLabeled, mStats, mCentroids;
	// fuse masks
	cv::Mat fusedMask;

	DeviceArray2D<unsigned char> edge;
	DeviceArray2D<unsigned char> cc_labeled;
};

#endif
