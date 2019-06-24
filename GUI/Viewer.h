#ifndef VIEWER_HPP__
#define VIEWER_HPP__

#include "System.h"

#include <atomic>
#include <vector>
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>

class System;
class Mapping;
class Tracker;

class Viewer {
public:

	Viewer();
	void spin();

	void setMap(Mapping * pMap);
	void setSystem(System * pSystem);
	void setTracker(Tracker * pTracker);
	void signalQuit();

private:

	void Insert(std::vector<GLfloat> & vPt, Eigen::Vector3f & pt);
	void drawCamera();
	void drawKeys();
	void followCam();
	void drawNormal();
	void drawColor();
	void drawKeyFrame();
	void drawMesh(bool bNormal);
	void showColorImage();
	void showPrediction();
	void showDepthImage();
	void topDownView();

	System * system;
	Mapping * map;
	Tracker * tracker;

	std::atomic<bool> quit;

	GLuint vao;
	GLuint vao_color;
	pangolin::OpenGlRenderState sCam;
	pangolin::GlSlProgram phongShader;
	pangolin::GlSlProgram normalShader;
	pangolin::GlSlProgram colorShader;
	pangolin::GlBufferCudaPtr vertex;
	pangolin::GlBufferCudaPtr normal;
	pangolin::GlBufferCudaPtr color;
	pangolin::CudaScopedMappedPtr * vertexMaped;
	pangolin::CudaScopedMappedPtr * normalMaped;
	pangolin::CudaScopedMappedPtr * colorMaped;

	pangolin::GlTextureCudaArray colorImage;
	pangolin::GlTextureCudaArray depthImage;
	pangolin::GlTextureCudaArray renderedImage;
	pangolin::GlTextureCudaArray topDownImage;
	pangolin::CudaScopedMappedArray * colorImageMaped;
	pangolin::CudaScopedMappedArray * depthImageMaped;
	pangolin::CudaScopedMappedArray * renderedImageMaped;
	pangolin::CudaScopedMappedArray * topDownImageMaped;

	pangolin::GlTextureCudaArray geoSegImage;
	pangolin::CudaScopedMappedArray * geoSegImageMaped;
	pangolin::GlTextureCudaArray fuseMaskImage;
	pangolin::CudaScopedMappedArray * fuseMaskImageMaped;
};

#endif
