#ifndef MASKRCNN_H
#define MASKRCNN_H

#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <string>

namespace fusion
{

namespace semantic
{

class MaskRCNN
{
public:
    MaskRCNN(char* module_name);
	~MaskRCNN();

	void initializeDetector(char* config_path, long val);
	void performDetection(cv::Mat image);
	void detectionWithLoad(char* img_path);

	// int getNumDetected();
	// // int* getDetectedImg();
    // // long int* getDetectedLabels();
	// // float* getDetectedScores();
	// // int* getDetectedMasks();
    // // OR
    // void getDetectedImg(int *pDetectedImg);
    // void getDetectedLabels(long int *pDetectedLabels);
	// void getDetectedScores(float *pDetectedScores);
	// void getDetectedMasks(int *pDetectedMasks);

    // c++ data structure
    int numDetection;
    int *pDetected, *pMasks;
    long int *pLabels;
    float *pScores, *pBoxes;

    std::string CATEGORIES[81] = {
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush"
    };

private:
    // PyObject
    PyObject *pPyName, *pPyModule, *pPyArgs, *pPyFunc, *pPyValue;
	PyObject *pPyImage, *pPyModel;
    // PyArrayObject
    PyArrayObject *pArrLabels, *pArrScores, *pArrMasks, *pArrBoxes;
    // PyArrayObject *pArrDetected;
};

} // namespace semantic

} // namespace semantic

#endif