#include "Reduction.h"

#define WarpSize 32
#define MaxThread 1024

template<int rows, int cols> void inline CreateMatrix(float* host_data, double* host_a, double* host_b) {
	int shift = 0;
	for (int i = 0; i < rows; ++i)
		for (int j = i; j < cols; ++j) {
			double value = (double) host_data[shift++];
			if (j == rows)
				host_b[i] = value;
			else
				host_a[j * rows + i] = host_a[i * rows + j] = value;
		}
}

template<typename T, int size> __device__ inline void WarpReduce(T* val) {
	#pragma unroll
	for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
	// 16, 8, 4, 2, 1
		#pragma unroll
		for (int i = 0; i < size; ++i) {
			val[i] += __shfl_down_sync(0xffffffff, val[i], offset);
		}
	}
}

template<typename T, int size> __device__ inline void BlockReduce(T* val) {
	static __shared__ T shared[32 * size];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	WarpReduce<T, size>(val);

	if (lane == 0)
		memcpy(&shared[wid * size], val, sizeof(T) * size);

	__syncthreads();

	if (threadIdx.x < blockDim.x / warpSize)
		memcpy(val, &shared[lane * size], sizeof(T) * size);
	else
		memset(val, 0, sizeof(T) * size);

	if (wid == 0)
		WarpReduce<T, size>(val);
}

template<typename T, int size> __global__ void Reduce(PtrStep<T> in, T * out, int N) {
	T sum[size];
	memset(sum, 0, sizeof(T) * size);
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (; i < N; i += blockDim.x * gridDim.x)
	#pragma unroll
		for (int j = 0; j < size; ++j)
			sum[j] += in.ptr(i)[j];

	BlockReduce<T, size>(sum);

	if (threadIdx.x == 0)
	#pragma unroll
		for (int i = 0; i < size; ++i)
			out[i] = sum[i];
}

struct SO3Reduce {

	PtrStep<unsigned char> nextImage;
	PtrStep<unsigned char> lastImage;
	PtrStep<short> dIdx;
	PtrStep<short> dIdy;
	int cols, rows;
	int N;
	float fx, fy, cx, cy;
	Matrix3f RcurrInv;
	Matrix3f Rlast;

	mutable PtrStepSz<float> out;

	__device__ __inline__ bool findCorresp(int & x, int & y, int & u,
			int & v, float3 & vlastcurr) const {

		float3 vlast;
		vlast.x = (x - cx) / fx;
		vlast.y = (y - cy) / fy;
		vlast.z = 1.0f;
		vlastcurr = RcurrInv * (Rlast * vlast);

		u = __float2int_rn(fx * vlastcurr.x / vlastcurr.z + cx);
		v = __float2int_rn(fy * vlastcurr.y / vlastcurr.z + cy);

		if(u >= 5 && v >= 5 && u < cols - 5 && v < rows - 5 &&
		   x >= 5 && y >= 5 && x < cols - 5 && y < rows - 5) {
			return true;
		} else
			return false;
	}

	__device__ __inline__ void GetRow(int & k, float * sum) const {

		int y = k / cols;
		int x = k - y * cols;
		float row[4] = { 0, 0, 0, 0 };

		float3 point;
		int u = 0, v = 0;
		bool found = findCorresp(x, y, u, v, point);

		if(found) {
			float gx = (float)dIdx.ptr(v)[u] / 9.0f;
			float gy = (float)dIdy.ptr(v)[u] / 9.0f;
			float invz = 1.0f / point.z;

			float3 left;

			left.x = gx * fx * invz;
			left.y = gy * fy * invz;
			left.z = -(point.x * left.x + point.y * left.y) * invz;

			*(float3*) &row[0] = -cross(left, point);
			row[3] = -((float)nextImage.ptr(v)[u] - (float)lastImage.ptr(y)[x]);
		}

		int count = 0;
		#pragma unroll
		for (int i = 0; i < 4; ++i)
			#pragma unroll
			for (int j = i; j < 4; ++j)
				sum[count++] = row[i] * row[j];

		sum[count] = (float) found;
	}

	__device__ __inline__ void operator()() const {

		float sum[11] = { 0, 0, 0,
						  0, 0, 0,
						  0, 0, 0,
						  0 };
		float value[11];
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		for (; i < N; i += blockDim.x * gridDim.x) {
			GetRow(i, value);
			#pragma unroll
			for (int j = 0; j < 11; ++j)
				sum[j] += value[j];
		}

		BlockReduce<float, 11>(sum);

		if (threadIdx.x == 0)
			#pragma unroll
			for (int i = 0; i < 11; ++i)
				out.ptr(blockIdx.x)[i] = sum[i];
	}
};

__global__ void so3StepKernel(SO3Reduce so3) {
	so3();
}

void SO3Step(const DeviceArray2D<unsigned char> & nextImage,
		     const DeviceArray2D<unsigned char> & lastImage,
		     const DeviceArray2D<short> & dIdx,
		     const DeviceArray2D<short> & dIdy,
		     Matrix3f RcurrInv,
		     Matrix3f Rlast,
		     Intrinsics K,
		     DeviceArray2D<float> & sum,
		     DeviceArray<float> & out,
		     float * residual,
		     double * matrixA_host,
		     double * vectorB_host) {

	int cols = nextImage.cols;
	int rows = nextImage.rows;

	SO3Reduce so3;
	so3.nextImage = nextImage;
	so3.lastImage = lastImage;
	so3.dIdx = dIdx;
	so3.dIdy = dIdy;
	so3.RcurrInv = RcurrInv;
	so3.Rlast = Rlast;
	so3.fx = K.fx;
	so3.fy = K.fy;
	so3.cx = K.cx;
	so3.cy = K.cy;
	so3.cols = cols;
	so3.rows = rows;
	so3.N = cols * rows;
	so3.out = sum;

	so3StepKernel<<<96, 224>>>(so3);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	Reduce<float, 11> <<<1, MaxThread>>>(sum, out, 96);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	float host_data[11];
	out.download((void*) host_data);
	CreateMatrix<3, 4>(host_data, matrixA_host, vectorB_host);

	residual[0] = host_data[9];
	residual[1] = host_data[10];
}

struct ICPReduce {

	Matrix3f Rcurr;
	Matrix3f Rlast;
	Matrix3f RlastInv;
	float3 tcurr;
	float3 tlast;
	PtrStep<float4> VMapCurr, VMapLast;
	PtrStep<float4> NMapCurr, NMapLast;
	int cols, rows, N;	// N = rows*cols
	float fx, fy, cx, cy;
	float angleThresh, distThresh;

	mutable PtrStepSz<float> out;

	__device__ __inline__ bool searchPoint(int& x, int& y, float3& vcurr_g,
			float3& vlast_g, float3& nlast_g) const {

		float3 vcurr_c = make_float3(VMapCurr.ptr(y)[x]);
		if (isnan(vcurr_c.x) || vcurr_c.z < 1e-3)
			return false;

		vcurr_g = Rcurr * vcurr_c + tcurr;
		float3 vcurr_p = RlastInv * (vcurr_g - tlast);

		float invz = 1.0 / vcurr_p.z;
		int u = (int) (vcurr_p.x * invz * fx + cx + 0.5);
		int v = (int) (vcurr_p.y * invz * fy + cy + 0.5);
		if (u < 0 || v < 0 || u >= cols || v >= rows)
			return false;

		float3 vlast_c = make_float3(VMapLast.ptr(v)[u]);
		vlast_g = Rlast * vlast_c + tlast;

		float3 ncurr_c = make_float3(NMapCurr.ptr(y)[x]);
		float3 ncurr_g = Rcurr * ncurr_c;

		float3 nlast_c = make_float3(NMapLast.ptr(v)[u]);
		nlast_g = Rlast * nlast_c;

		float dist = norm(vlast_g - vcurr_g);
		float sine = norm(cross(ncurr_g, nlast_g));

		return (sine < angleThresh && dist <= distThresh && !isnan(ncurr_c.x)
				&& !isnan(nlast_c.x));
	}

	__device__ __inline__ void getRow(int & i, float * sum) const {

		int y = i / cols;
		int x = i - y * cols;

		bool found = false;
		float3 vcurr, vlast, nlast;
		// read valid vertices and normal from map and current frame
		found = searchPoint(x, y, vcurr, vlast, nlast);
		float row[7] = { 0, 0, 0, 0, 0, 0, 0 };

		// calculate Jacobian #### CORE PART ####
		if (found) {
			nlast = RlastInv * nlast;
			vcurr = RlastInv * (vcurr - tlast);
			vlast = RlastInv * (vlast - tlast);
			*(float3*) &row[0] = -nlast;
			*(float3*) &row[3] = cross(nlast, vlast);
			row[6] = -nlast * (vcurr - vlast);
		}

		// from Jacobian to values in A & b
		int count = 0;
		#pragma unroll
		for (int i = 0; i < 7; ++i)
			#pragma unroll
			for (int j = i; j < 7; ++j)
				sum[count++] = row[i] * row[j];

		sum[count] = (float) found;
	}

	__device__ __inline__ void operator()() const {

		float sum[29] = { 0, 0, 0, 0, 0,
				   	   	  0, 0, 0, 0, 0,
				   	   	  0, 0, 0, 0, 0,
				   	   	  0, 0, 0, 0, 0,
				   	   	  0, 0, 0, 0, 0,
				   	   	  0, 0, 0, 0 };
		// 0-26: upper matrix of a 6x6 symmetric matrix and a 6x1 vector
		// 27: sum of error term
		// 28: count of valid measurements

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		float val[29];
		// loop through all GRIDS to make sure all pixels are considered
		for (; i < N; i += blockDim.x * gridDim.x) {
			getRow(i, val);
			#pragma unroll
			for (int j = 0; j < 29; ++j)
				sum[j] += val[j];
		}

		// returned is "the sum over all threads inside 1 single grid"
		BlockReduce<float, 29>(sum);

		if (threadIdx.x == 0)
			#pragma unroll
			for (int i = 0; i < 29; ++i)
				out.ptr(blockIdx.x)[i] = sum[i];
	}
};

__global__ void icpStepKernel(const ICPReduce icp) {
	icp();
}

void ICPStep(DeviceArray2D<float4> & nextVMap,
			 DeviceArray2D<float4> & lastVMap,
			 DeviceArray2D<float4> & nextNMap,
			 DeviceArray2D<float4> & lastNMap,
			 Matrix3f Rcurr,
			 float3 tcurr,
			 Matrix3f Rlast,
			 Matrix3f RlastInv,
			 float3 tlast,
			 Intrinsics K,
			 DeviceArray2D<float> & sum,
			 DeviceArray<float> & out,
			 float * residual,
			 double * matrixA_host,
			 double * vectorB_host) {

	int cols = nextVMap.cols;
	int rows = nextVMap.rows;

	ICPReduce icp;
	icp.out = sum;
	icp.VMapCurr = nextVMap;
	icp.NMapCurr = nextNMap;
	icp.VMapLast = lastVMap;
	icp.NMapLast = lastNMap;
	icp.cols = cols;
	icp.rows = rows;
	icp.N = cols * rows;
	icp.Rcurr = Rcurr;
	icp.tcurr = tcurr;
	icp.Rlast = Rlast;
	icp.RlastInv = RlastInv;
	icp.tlast = tlast;
	icp.angleThresh = 0.6;
	icp.distThresh = 0.1;
	icp.fx = K.fx;
	icp.fy = K.fy;
	icp.cx = K.cx;
	icp.cy = K.cy;

	icpStepKernel<<<96, 224>>>(icp);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	// sum over all grids
	Reduce<float, 29> <<<1, MaxThread>>>(sum, out, 96);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	float host_data[29];
	out.download((void*) host_data);
	CreateMatrix<6, 7>(host_data, matrixA_host, vectorB_host);

	residual[0] = host_data[27];
	residual[1] = host_data[28];
}

struct RGBReduction {

	int N;
	int cols, rows;

	Matrix3f Rcurr;
	Matrix3f RcurrInv;
	Matrix3f Rlast;
	Matrix3f RlastInv;
	float3 tlast, tcurr;

	PtrStep<float4> nextVMap;
	PtrStep<float4> lastVMap;
	PtrStep<unsigned char> nextImage;
	PtrStep<unsigned char> lastImage;
	PtrStep<short> dIdx;
	PtrStep<short> dIdy;

	float minScale;
	float sigma;
	float sobelScale;
	float fx, fy, cx, cy;

	mutable PtrStep<int> outRes;
	mutable PtrStep<float> out;
	mutable PtrSz<Residual> RGBResidual;

	__device__ __inline__ int2 FindCorresp(int & k) const {

		int y = k / cols;
		int x = k - y * cols;

		Residual res;
		int2 value = { 0, 0 };
		res.valid = false;

		if (x >= 5 && x < cols - 5 && y >= 5 && y < rows - 5) {

			bool valid = true;
			for (int u = max(y - 2, 0); u < min(y + 2, rows); ++u) {
				for (int v = max(x - 2, 0); v < min(x + 2, cols); ++v) {
					valid = valid && (nextImage.ptr(u)[v] > 0);
				}
			}

			if (valid) {
				short gx = dIdx.ptr(y)[x];
				short gy = dIdy.ptr(y)[x];
				if (sqrtf(gx * gx + gy * gy) > 0) {
					float4 vcurr = nextVMap.ptr(y)[x];
					if (!isnan(vcurr.x)) {
						float3 vcurr_g = Rcurr * vcurr + tcurr;
						float3 vcurrlast = RlastInv * (vcurr_g - tlast);
						int u = __float2int_rd(fx * vcurrlast.x / vcurrlast.z + cx + 0.5);
						int v = __float2int_rd(fy * vcurrlast.y / vcurrlast.z + cy + 0.5);

						if (u >= 0 && v >= 0 && u < cols && v < rows) {
							float4 vlast = lastVMap.ptr(v)[u];
							if (!isnan(vlast.x) && norm(vlast - vcurr) < 0.1 && lastImage.ptr(v)[u] != 0) {
								res.last = { u, v };
								res.curr = { x, y };
								res.diff = (int)nextImage.ptr(y)[x] - (int)lastImage.ptr(v)[u];
								res.valid = true;
								res.point = make_float3(vlast);
								value = { 1, res.diff * res.diff };
							}
						}
					}
				}
			}
		}

		RGBResidual[k] = res;
		return value;
	}

	__device__ __inline__ void ComputeResidual() const {

		int sum[2] = { 0, 0 };
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		// loop through all GRIDS to make sure all pixels are considered
		for (; i < N; i += blockDim.x * gridDim.x) {
			int2 val = FindCorresp(i);
			// count of difference measured
			sum[0] += val.x;
			// squared difference
			sum[1] += val.y;
		}

		// returned is "the sum over all threads inside 1 grid"
		BlockReduce<int, 2>(sum);

		if (threadIdx.x == 0) {
			outRes.ptr(blockIdx.x)[0] = sum[0];
			outRes.ptr(blockIdx.x)[1] = sum[1];
		}
	}

	__device__ __inline__ void GetRow(int & k, float * sum) const {

		const Residual & res = RGBResidual[k];
		float row[7] = { 0, 0, 0, 0, 0, 0, 0 };

		if (res.valid) {

			float w = sigma + abs(res.diff);
			w = w < 1e-3 ? 1.0f : 1.0f / w;

			if(sigma < 1e-6)
				w = 1.0f;

			float4 vlast = lastVMap.ptr(res.last.y)[res.last.x];
			float3 point = RcurrInv * (Rlast * vlast + tlast - tcurr);
			float gx = (float)dIdx.ptr(res.curr.y)[res.curr.x] / 9.0;
			float gy = (float)dIdy.ptr(res.curr.y)[res.curr.x] / 9.0;

			float3 left;
			float invz = 1.0f / point.z;
			left.x = gx * fx * invz;
			left.y = gy * fy * invz;
			left.z = -(left.x * point.x + left.y * point.y) * invz;

			*(float3*) &row[0] = w * left;
			*(float3*) &row[3] = w * cross(point, left);
			row[6] = -w * res.diff;
		}

		int count = 0;
		#pragma unroll
		for (int i = 0; i < 7; ++i)
			#pragma unroll
			for (int j = i; j < 7; ++j)
				sum[count++] = row[i] * row[j];

		sum[count] = (float) res.valid;
	}

	__device__ __inline__ void operator()() const {

		float sum[29] = { 0, 0, 0, 0, 0,
				   	   	  0, 0, 0, 0, 0,
				   	   	  0, 0, 0, 0, 0,
				   	   	  0, 0, 0, 0, 0,
				   	   	  0, 0, 0, 0, 0,
				   	   	  0, 0, 0, 0 };

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		float value[29];

		for (; i < N; i += blockDim.x * gridDim.x) {

			GetRow(i, value);
			#pragma unroll
			for (int j = 0; j < 29; ++j)
				sum[j] += value[j];
		}

		BlockReduce<float, 29>(sum);

		if (threadIdx.x == 0)
			#pragma unroll
			for (int i = 0; i < 29; ++i)
				out.ptr(blockIdx.x)[i] = sum[i];
	}
};

__global__ void ComputeResidualKernel(RGBReduction rgb) {
	rgb.ComputeResidual();
}

__global__ void RgbStepKernel(RGBReduction rgb) {
	rgb();
}

void RGBStep(const DeviceArray2D<unsigned char> & nextImage,
			 const DeviceArray2D<unsigned char> & lastImage,
			 const DeviceArray2D<float4> & nextVMap,
			 const DeviceArray2D<float4> & lastVMap,
			 const DeviceArray2D<short> & dIdx,
			 const DeviceArray2D<short> & dIdy,
			 Matrix3f Rcurr,
			 Matrix3f RcurrInv,
			 Matrix3f Rlast,
			 Matrix3f RlastInv,
			 float3 tcurr,
			 float3 tlast,
			 Intrinsics K,
			 DeviceArray2D<float> & sum,
			 DeviceArray<float> & out,
			 DeviceArray2D<int> & sumRes,
			 DeviceArray<int> & outRes,
			 float * residual,
			 double * matrixA_host,
			 double * vectorB_host) {

	int cols = nextImage.cols;
	int rows = nextImage.rows;

	RGBReduction rgb;
	DeviceArray<Residual> rgbResidual(cols * rows);

	rgb.cols = cols;
	rgb.rows = rows;
	rgb.N = cols * rows;
	rgb.nextImage = nextImage;
	rgb.lastImage = lastImage;
	rgb.nextVMap = nextVMap;
	rgb.lastVMap = lastVMap;
	rgb.dIdx = dIdx;
	rgb.dIdy = dIdy;
	rgb.outRes = sumRes;
	rgb.Rcurr = Rcurr;
	rgb.RcurrInv = RcurrInv;
	rgb.Rlast = Rlast;
	rgb.RlastInv = RlastInv;
	rgb.tcurr = tcurr;
	rgb.tlast = tlast;
	rgb.RGBResidual = rgbResidual;
	rgb.fx = K.fx;
	rgb.fy = K.fy;
	rgb.cx = K.cx;
	rgb.cy = K.cy;

	ComputeResidualKernel<<<96, 224>>>(rgb);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	// summarize count & diff^2 in all grids and store in outRes
	Reduce<int, 2> <<<1, MaxThread>>>(sumRes, outRes, 96);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	int res_host[2];
	outRes.download(res_host);
	rgb.sigma = sqrt((float) res_host[1] / (res_host[0] == 0 ? 1 : res_host[0]));
	rgb.out = sum;

	RgbStepKernel<<<96, 224>>>(rgb);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	Reduce<float, 29> <<<1, MaxThread>>>(sum, out, 96);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	float host_data[29];
	out.download((void*) host_data);
	CreateMatrix<6, 7>(host_data, matrixA_host, vectorB_host);

	residual[0] = host_data[27];
	residual[1] = host_data[28];
}
