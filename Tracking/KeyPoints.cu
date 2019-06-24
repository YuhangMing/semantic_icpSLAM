#include "DeviceMap.h"

#include <opencv.hpp>
#include <cudaarithm.hpp>

__device__ __forceinline__ float clamp(float a) {
	a = a > -1.f ? a : -1.f;
	a = a < 1.f ? a : 1.f;
	return a;
}

__global__ void BuildAdjecencyMatrixKernel(cv::cuda::PtrStepSz<float> adjecencyMatrix,
										   PtrSz<SURF> frameKeys,
										   PtrSz<SURF> mapKeys,
										   PtrSz<float> dist) {

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if(x >= adjecencyMatrix.cols || y >= adjecencyMatrix.rows)
		return;

	float score = 0;

	if(x == y) {
		score = exp(-dist[x]);
	} else {

		SURF * mapKey00 = &mapKeys[x];
		SURF * mapKey01 = &mapKeys[y];

		SURF * frameKey00 = &frameKeys[x];
		SURF * frameKey01 = &frameKeys[y];

		float d00 = norm(frameKey00->pos - frameKey01->pos);
		float d01 = norm(mapKey00->pos - mapKey01->pos);

		float4 d10 = make_float4(normalised(frameKey00->pos - frameKey01->pos));
		float4 d11 = make_float4(normalised(mapKey00->pos - mapKey01->pos));

		if(d00 <= 1e-2 || d01 <= 1e-2) {
			score = 0;
		} else {
			float alpha00 = acos(clamp(frameKey00->normal * frameKey01->normal));
			float beta00 = acos(clamp(d10 * frameKey00->normal));
			float gamma00 = acos(clamp(d10 * frameKey01->normal));
			float alpha01 = acos(clamp(mapKey00->normal * mapKey01->normal));
			float beta01 = acos(clamp(d11 * mapKey00->normal));
			float gamma01 = acos(clamp(d11 * mapKey01->normal));
			score = exp(-(fabs(d00 - d01) + fabs(alpha00 - alpha01) + fabs(beta00 - beta01) + fabs(gamma00 - gamma01)));
		}
	}

	if(isnan(score))
		score = 0;

	adjecencyMatrix.ptr(y)[x] = score;
}

void BuildAdjacencyMatrix(cv::cuda::GpuMat & adjecencyMatrix,
						  DeviceArray<SURF> & frameKeys,
						  DeviceArray<SURF> & mapKeys,
						  DeviceArray<float> & dist) {

	int cols = adjecencyMatrix.cols;
	int rows = adjecencyMatrix.rows;

	dim3 thread(8, 8);
	dim3 block(DivUp(cols, thread.x), DivUp(rows, thread.y));

	BuildAdjecencyMatrixKernel<<<block, thread>>>(adjecencyMatrix, frameKeys, mapKeys, dist);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	cv::cuda::GpuMat result;
	cv::cuda::reduce(adjecencyMatrix, result, 0, CV_REDUCE_SUM);
}
