#include "detection/MaskRCNN.h"
#include <typeinfo>
#include <cxxabi.h>

using namespace std;

#define VERBOSE
// #define CDEBUG
// #define IMGDEBUG

namespace fusion
{

namespace semantic
{

MaskRCNN::MaskRCNN(char* module_name)
{
#ifdef VERBOSE
    cout << "-- Constructing MaskRCNN Class...";
#endif
    // Initialise Python interpreter
    Py_Initialize();
    // Import numpy array module
    _import_array();
    if(PyArray_API == NULL)
    {
    	PyErr_Print();
    	fprintf(stderr, "Failed to import array API\n");
    }
    // else
    // {
    //     cout << "!!!!! Array API imported... " << endl;
    // }
    // Set module pathes
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.insert(0, '/home/lk18493/github/maskrcnn-benchmark/demo/')");
    // Build name object
    pPyName = PyUnicode_FromString(module_name);
    pPyModule = PyImport_Import(pPyName);
    Py_DECREF(pPyName);
    if (pPyModule == NULL){
    	PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", module_name);
    }
#ifdef VERBOSE
    cout << "    DONE." << endl;
#endif
}

MaskRCNN::~MaskRCNN()
{
#ifdef VERBOSE
    cout << "-- Destructing MaskRCNN Class...";
#endif
    Py_DECREF(pPyArgs);
    Py_DECREF(pPyImage);
    Py_DECREF(pPyValue);
    Py_DECREF(pPyModel);
    Py_XDECREF(pPyFunc);
    Py_XDECREF(pPyModule);
    Py_Finalize();
#ifdef VERBOSE
    cout << "    DONE." << endl;
#endif
}

void MaskRCNN::initializeDetector(char* config_path, long val)
{
#ifdef VERBOSE
    cout << "-- Loading Pretrained MaskRCNN Model...";
#endif
    // Retrieve the initialization function
    pPyFunc = PyObject_GetAttrString(pPyModule, "load_maskrcnn_detector");
    if (pPyFunc && PyCallable_Check(pPyFunc))
    {
        pPyArgs = PyTuple_New(2);
        pPyValue = PyUnicode_FromString(config_path);
        PyTuple_SetItem(pPyArgs, 0, pPyValue);
        pPyValue = PyBool_FromLong(val);
        PyTuple_SetItem(pPyArgs, 1, pPyValue);

        // execute the function
        pPyModel = PyObject_CallObject(pPyFunc, pPyArgs);
        if (pPyModel == NULL) {
            PyErr_Print();
            fprintf(stderr,"Failed to load the model\n");
        }
    }
    else 
    {
        if (PyErr_Occurred())
            PyErr_Print();
        fprintf(stderr, "Cannot find function \"load_maskrcnn_detector\"\n");
    }
#ifdef VERBOSE
    cout << "    DONE." << endl;
#endif
}

void MaskRCNN::performDetection(cv::Mat image)
{
#ifdef VERBOSE
    cout << "-- Performing One Detection...";
#endif
    // convert cv::Mat to numpy.ndarray
    npy_intp dimensions[3] = {image.rows, image.cols, image.channels()};
    // pPyImage = PyArray_SimpleNewFromData(3, (npy_intp*)&dimensions[0], NPY_UINT8, (void *)image.data);
    pPyImage = PyArray_SimpleNewFromData(3, dimensions, NPY_UINT8, image.data);

    // Retrieve the detection function
    pPyFunc = PyObject_GetAttrString(pPyModule, "detection_x1");
    if (pPyFunc && PyCallable_Check(pPyFunc))
    {
        // store model and image into PyTuple
        pPyArgs = PyTuple_New(2);
        PyTuple_SetItem(pPyArgs, 0, pPyModel);
        PyTuple_SetItem(pPyArgs, 1, pPyImage);
        
        // execute the function
        pPyValue = PyObject_CallObject(pPyFunc, pPyArgs);
        if (pPyValue != NULL) {
            // Get each returns from the tuple, feed into pyArrayObject
            // pArrDetected = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(pPyValue, 0));
            pArrLabels = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(pPyValue, 1));
            pArrScores = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(pPyValue, 2));
            pArrMasks = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(pPyValue, 3));
            pArrBoxes = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(pPyValue, 4));

            // Convert from PyArrayObject into C++ data structure
            numDetection = PyArray_SHAPE(pArrLabels)[0];
            // pDetected = reinterpret_cast<int*>(PyArray_DATA(pArrDetected));
            pLabels = reinterpret_cast<long int*>(PyArray_DATA(pArrLabels));
            pScores = reinterpret_cast<float*>(PyArray_DATA(pArrScores));
            pMasks = reinterpret_cast<int*>(PyArray_DATA(pArrMasks));
            pBoxes = reinterpret_cast<float*>(PyArray_DATA(pArrBoxes));
        }
        else {
            PyErr_Print();
            fprintf(stderr,"Call failed\n");
        }
    }
    else 
    {
        if (PyErr_Occurred())
            PyErr_Print();
        fprintf(stderr, "Cannot find function \"detection_x1\"\n");
    }
#ifdef VERBOSE
    cout << "    DONE." << endl;
#endif

#ifdef CDEBUG
    cout << "$$ CDEBUG: in MaskRCNN.cpp " << endl;
    // cout << " Detected image sample " << endl;
    // for (int i=70000; i<70020; i++){
    //     cout << pDetected[i] << ", ";
    // }
    // cout << endl;
    cout << " Labels: " << endl << "  ";
    for(int i=0; i<numDetection; i++){
        cout << pLabels[i] << ": " << CATEGORIES[pLabels[i]] << "; ";
    }
    cout << endl;
    cout << " Scores: " << endl << "  ";
    for(int i=0; i<numDetection; i++){
        cout << pScores[i] << ", ";
    }
    cout << endl;
    // cout << " Masks sample " << endl;
    // for (int i=70000; i<70020; i++){
    //     cout << pMasks[i] << ", ";
    // }
    // cout << endl;
    cout << " B-Boxes: " << endl << "  ";
    for(int i=0; i<numDetection*4; i++){
        cout << pBoxes[i] << ", ";
    }
    cout << endl;
    cout << " ** ** ** ** ** " << endl;
#endif
}

void MaskRCNN::detectionWithLoad(char* img_path)
{
#ifdef VERBOSE
    cout << "Performing One Detection..." << endl;
#endif
    // tests on PyArray_SimpleNewFromData ///////////////////////////////////////////////////////////////////
    // tests on the array from cv::mat 
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        cout << " Could not open or find the image." << endl;
    }
    else{
        cout << " Image successfully loaded." << endl;
    }
    npy_intp dimensions[3] = {image.rows, image.cols, image.channels()};
#ifdef IMGDEBUG
    cout << "€€ IMGDEBUG: image info" << endl;
    cout << image.dims << endl;
    cout << image.rows << ", " << image.cols << ", " << image.channels() << endl;
    cout << dimensions << endl;
    cout << dimensions[0] << ", " << dimensions[1] << ", " << dimensions[2] << endl;
    cout << abi::__cxa_demangle(typeid(dimensions).name(), 0, 0, 0) << endl;    // long [3]
    cout << abi::__cxa_demangle(typeid(image.data).name(), 0, 0, 0) << endl;    // unsigned char*
    // cout << image.data << endl;
#endif
    // pPyImage = PyArray_SimpleNewFromData(3, (npy_intp*)&dimensions[0], NPY_UINT8, (void *)image.data);
    pPyImage = PyArray_SimpleNewFromData(3, dimensions, NPY_UINT8, image.data);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Retrieve the initialization function
    pPyFunc = PyObject_GetAttrString(pPyModule, "detection_x1");
    if (pPyFunc && PyCallable_Check(pPyFunc))
    {
        // pPyArgs = PyTuple_New(2);
        // PyTuple_SetItem(pPyArgs, 0, pPyModel);
        // pPyValue = PyUnicode_FromString(img_path);
        // PyTuple_SetItem(pPyArgs, 1, pPyValue);
        pPyArgs = PyTuple_New(2);
        PyTuple_SetItem(pPyArgs, 0, pPyModel);
        PyTuple_SetItem(pPyArgs, 1, pPyImage);
        
        // execute the function
        pPyValue = PyObject_CallObject(pPyFunc, pPyArgs);
        if (pPyValue != NULL) {
            // Get each returns from the tuple, feed into pyArrayObject
            // pArrDetected = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(pPyValue, 0));
            pArrLabels = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(pPyValue, 1));
            pArrScores = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(pPyValue, 2));
            pArrMasks = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(pPyValue, 3));
            pArrBoxes = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(pPyValue, 4));

            // Convert from PyArrayObject into C++ data structure
            numDetection = PyArray_SHAPE(pArrLabels)[0];
            // pDetected = reinterpret_cast<int*>(PyArray_DATA(pArrDetected));
            pLabels = reinterpret_cast<long int*>(PyArray_DATA(pArrLabels));
            pScores = reinterpret_cast<float*>(PyArray_DATA(pArrScores));
            pMasks = reinterpret_cast<int*>(PyArray_DATA(pArrMasks));
            pBoxes = reinterpret_cast<float*>(PyArray_DATA(pArrBoxes));
        }
        else {
            PyErr_Print();
            fprintf(stderr,"Call failed\n");
        }
    }
    else 
    {
        if (PyErr_Occurred())
            PyErr_Print();
        fprintf(stderr, "Cannot find function \"detection_x1\"\n");
    }

#ifdef CDEBUG
    cout << "$$ DEBUG: results in c++ data structure " << endl;
    // cout << " Detected image sample " << endl;
    // for (int i=70000; i<70020; i++){
    //     cout << pDetected[i] << ", ";
    // }
    // cout << endl;
    cout << " Labels: " << endl;
    for(int i=0; i<numDetection; i++){
        cout << pLabels[i] << ", ";
    }
    cout << endl;
    cout << " Scores: " << endl;
    for(int i=0; i<numDetection; i++){
        cout << pScores[i] << ", ";
    }
    cout << endl;
    cout << " Masks sample " << endl;
    for (int i=70000; i<70020; i++){
        cout << pMasks[i] << ", ";
    }
    cout << endl;
    cout << " B-Boxes: " << endl;
    for(int i=0; i<numDetection*4; i++){
        cout << pBoxes[i] << ", ";
    }
    cout << endl;
    cout << " ** ** ** ** ** \n" << endl;
#endif
}

} // namespace semantic

} // namespace fusion