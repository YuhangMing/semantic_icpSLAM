#ifndef GPU_REDUCTION_H__
#define GPU_REDUCTION_H__

#include "VectorMath.h"
#include "Intrinsics.h"
#include "RenderScene.h"
#include <opencv.hpp>

struct Residual {

	int diff;
	bool valid;
	int2 curr;
	int2 last;
	float3 point;
};

void FilterDepth(const DeviceArray2D<unsigned short> & depth,
		DeviceArray2D<float> & rawDepth, DeviceArray2D<float> & filteredDepth,
		float depthScale, float depthCutoff);

void ComputeVMap(const DeviceArray2D<float> & depth,
		DeviceArray2D<float4> & vmap, float fx, float fy, float cx, float cy,
		float depthCutoff);

void ComputeNMap(const DeviceArray2D<float4> & vmap,
		DeviceArray2D<float4> & nmap);

void PyrDownGauss(const DeviceArray2D<float> & src, DeviceArray2D<float> & dst);

void PyrDownGauss(const DeviceArray2D<unsigned char> & src,
		DeviceArray2D<unsigned char> & dst);

void ImageToIntensity(const DeviceArray2D<uchar3> & rgb,
		DeviceArray2D<unsigned char> & image);

void ComputeDerivativeImage(DeviceArray2D<unsigned char> & image,
		DeviceArray2D<short> & dx, DeviceArray2D<short> & dy);

void ResizeMap(const DeviceArray2D<float4> & vsrc,
		const DeviceArray2D<float4> & nsrc, DeviceArray2D<float4> & vdst,
		DeviceArray2D<float4> & ndst);

void RenderImage(const DeviceArray2D<float4>& points,
		const DeviceArray2D<float4>& normals, const float3 light_pose,
		DeviceArray2D<uchar4>& image);

void DepthToImage(const DeviceArray2D<float> & depth,
		DeviceArray2D<uchar4> & image);

void RgbImageToRgba(const DeviceArray2D<uchar3> & image,
		DeviceArray2D<uchar4> & rgba);

void ucharImageToRgba(const DeviceArray2D<unsigned char> & gray, 
		DeviceArray2D<uchar4> & rgba);

void NVmapToEdge(const DeviceArray2D<float4> & normal, const DeviceArray2D<float4> & vertex,
		DeviceArray2D<unsigned char> & edge, 
		float lamb, float tao, int win_size, int step);

void ForwardWarping(const DeviceArray2D<float4> & srcVMap,
		const DeviceArray2D<float4> & srcNMap, DeviceArray2D<float4> & dstVMap,
		DeviceArray2D<float4> & dstNMap, Matrix3f srcRot, Matrix3f dstInvRot,
		float3 srcTrans, float3 dstTrans, float fx, float fy, float cx,
		float cy);

void SO3Step(const DeviceArray2D<unsigned char> & nextImage,
		const DeviceArray2D<unsigned char> & lastImage,
		const DeviceArray2D<short> & dIdx, const DeviceArray2D<short> & dIdy,
		Matrix3f RcurrInv, Matrix3f Rlast, Intrinsics K,
		DeviceArray2D<float> & sum, DeviceArray<float> & out, float * residual,
		double * matrixA_host, double * vectorB_host);

void ICPStep(DeviceArray2D<float4> & nextVMap, DeviceArray2D<float4> & lastVMap,
		DeviceArray2D<float4> & nextNMap, DeviceArray2D<float4> & lastNMap,
		Matrix3f Rcurr, float3 tcurr, Matrix3f Rlast, Matrix3f RlastInv,
		float3 tlast, Intrinsics K, DeviceArray2D<float> & sum,
		DeviceArray<float> & out, float * residual, double * matrixA_host,
		double * vectorB_host);

void RGBStep(const DeviceArray2D<unsigned char> & nextImage,
		const DeviceArray2D<unsigned char> & lastImage,
		const DeviceArray2D<float4> & nextVMap,
		const DeviceArray2D<float4> & lastVMap,
		const DeviceArray2D<short> & dIdx, const DeviceArray2D<short> & dIdy,
		Matrix3f Rcurr, Matrix3f RcurrInv, Matrix3f Rlast, Matrix3f RlastInv,
		float3 tcurr, float3 tlast, Intrinsics K, DeviceArray2D<float> & sum,
		DeviceArray<float> & out, DeviceArray2D<int> & sumRes,
		DeviceArray<int> & outRes, float * residual, double * matrixA_host,
		double * vectorB_host);

void BuildAdjacencyMatrix(cv::cuda::GpuMat & adjecencyMatrix,
		DeviceArray<SURF> & frameKeys,
		DeviceArray<SURF> & mapKeys,
		DeviceArray<float> & dist);

void FilterKeyMatching(cv::cuda::GpuMat & adjecencyMatrix,
		DeviceArray<SURF> & trainKey,
		DeviceArray<SURF> & queryKey,
		DeviceArray<SURF> & trainKeyFiltered,
		DeviceArray<SURF> & queryKeyFiltered,
		DeviceArray<int> & QueryIdx,
		DeviceArray<int> & keyIdxFiltered);

#endif
