#ifndef MATH_LIB_H__
#define MATH_LIB_H__

#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__host__ __device__ __forceinline__ uchar3 make_uchar3(int a) {
	return make_uchar3(a, a, a);
}

__host__ __device__ __forceinline__ uchar4 make_uchar4(int a) {
	return make_uchar4(a, a, a, a);
}

__host__ __device__ __forceinline__ uchar3 make_uchar3(float3 a) {
	return make_uchar3((int) a.x, (int) a.y, (int) a.z);
}

__host__ __device__ __forceinline__ uchar4 make_uchar4(float4 a) {
	return make_uchar4((int) a.x, (int) a.y, (int) a.z, (int) a.w);
}

__host__ __device__ __forceinline__ int2 make_int2(int a) {
	return make_int2(a, a);
}

__host__ __device__ __forceinline__ int2 make_int2(float2 a) {
	return make_int2((int) a.x, (int) a.y);
}

__host__ __device__ __forceinline__ int3 make_int3(int a) {
	return make_int3(a, a, a);
}

__host__ __device__ __forceinline__ int3 make_int3(float3 a) {
	int3 b = make_int3((int)a.x, (int)a.y, (int)a.z);
	b.x = b.x > a.x ? b.x - 1 : b.x;
	b.y = b.y > a.y ? b.y - 1 : b.y;
	b.z = b.z > a.z ? b.z - 1 : b.z;
	return b;
}

__host__ __device__ __forceinline__ int4 make_int4(int3 a, int b) {
	return make_int4(a.x, a.y, a.z, b);
}


__host__ __device__ __forceinline__ uint2 make_uint2(int a) {
	return make_uint2(a, a);
}

__host__ __device__ __forceinline__ uint3 make_uint3(int a) {
	return make_uint3(a, a, a);
}

__host__ __device__ __forceinline__ float2 make_float2(float a) {
	return make_float2(a, a);
}

__host__ __device__ __forceinline__ float3 make_float3(uchar3 a) {
	return make_float3(a.x, a.y, a.z);
}

__host__ __device__ __forceinline__ float4 make_float4(uchar4 a) {
	return make_float4(a.x, a.y, a.z, a.w);
}

__host__ __device__ __forceinline__ float3 make_float3(float a) {
	return make_float3(a, a, a);
}

__host__ __device__ __forceinline__ int3 make_int3(float a) {
	return make_int3(make_float3(a));
}

__host__ __device__ __forceinline__ float3 make_float3(int3 a) {
	return make_float3(a.x, a.y, a.z);
}

__host__ __device__ __forceinline__ float3 make_float3(float4 a) {
	return make_float3(a.x, a.y, a.z);
}

__host__ __device__ __forceinline__ float4 make_float4(float a) {
	return make_float4(a, a, a, a);
}

__host__ __device__ __forceinline__ float4 make_float4(float3 a) {
	return make_float4(a.x, a.y, a.z, 1.f);
}

__host__ __device__ __forceinline__ float4 make_float4(float3 a, float b) {
	return make_float4(a.x, a.y, a.z, b);
}

__host__ __device__ __forceinline__ double4 make_double4(double a) {
	return make_double4(a, a, a, a);
}

__host__ __device__ __forceinline__ int2 operator+(int2 a, int2 b) {
	return make_int2(a.x + b.x, a.y + b.y);
}

__host__ __device__ __forceinline__ float2 operator+(float2 a, float2 b) {
	return make_float2(a.x + b.x, a.y + b.y);
}

__host__ __device__ __forceinline__ float3 operator+(int3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __forceinline__ uchar3 operator+(uchar3 a, uchar3 b) {
	return make_uchar3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __forceinline__ int3 operator+(int3 a, int3 b) {
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __forceinline__ float3 operator+(float3 a, float b) {
	return make_float3(a.x + b, a.y + b, a.z + b);
}

__host__ __device__ __forceinline__ float3 operator+(float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __forceinline__ float4 operator+(float4 a, float3 b) {
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w);
}

__host__ __device__ __forceinline__ float4 operator+(float4 a, float4 b) {
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ __forceinline__ void operator+=(float3 & a, uchar3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

__host__ __device__ __forceinline__ void operator+=(float3 & a, float3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

__host__ __device__ __forceinline__ void operator-=(float3 & a, float3 b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

__host__ __device__ __forceinline__ uchar3 operator-(uchar3 a, uchar3 b) {
	return make_uchar3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ __forceinline__ int2 operator-(int2 a, int2 b) {
	return make_int2(a.x - b.x, a.y - b.y);
}

__host__ __device__ __forceinline__ float2 operator-(float2 lhs, float2 rhs) {
	return make_float2(lhs.x - rhs.x, lhs.y - rhs.y);
}

__host__ __device__ __forceinline__ float3 operator-(float3 b) {
	return make_float3(-b.x, -b.y, -b.z);
}

__host__ __device__ __forceinline__ float4 operator-(float4 b) {
	return make_float4(-b.x, -b.y, -b.z, -b.w);
}

__host__ __device__ __forceinline__ float3 operator-(float3 a, float b) {
	return make_float3(a.x - b, a.y - b, a.z - b);
}

__host__ __device__ __forceinline__ float3 operator-(float a, float3 b) {
	return make_float3(a - b.x, a - b.y, a - b.z);
}

__host__ __device__ __forceinline__ float3 operator-(float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ __forceinline__ float4 operator-(float4 a, float3 b) {
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w);
}

__host__ __device__ __forceinline__ float4 operator-(float4 a, float4 b) {
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__host__ __device__ __forceinline__ float operator*(float3 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ __forceinline__ float operator*(float3 a, float4 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ __forceinline__ float operator*(float4 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w;
}

__host__ __device__ __forceinline__ float operator*(float4 a, float4 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__host__ __device__ __forceinline__ uchar3 operator*(uchar3 a, unsigned short b) {
	return make_uchar3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ uchar3 operator*(uchar3 a, int b) {
	return make_uchar3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ uchar3 operator*(int b, uchar3 a) {
	return make_uchar3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ int3 operator*(int3 a, unsigned int b) {
	return make_int3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ int3 operator*(int3 a, int b) {
	return make_int3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ int3 operator*(float3 a, int b) {
	return make_int3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ float3 operator*(int3 a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ float3 operator*(float3 a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ float3 operator*(float a, float3 b) {
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__ __forceinline__ float4 operator*(float4 a, float b) {
	return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

__host__ __device__ __forceinline__ int3 operator/(int3 a, int3 b) {
	return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ __forceinline__ float3 operator/(float3 a, int3 b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ __forceinline__ float3 operator/(float3 a, float3 b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ __forceinline__ float4 operator/(float4 a, float4 b) {
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}


__host__ __device__ __forceinline__ int2 operator/(int2 a, int b) {
	return make_int2(a.x / b, a.y / b);
}

__host__ __device__ __forceinline__ float2 operator/(float2 a, int b) {
	return make_float2(a.x / b, a.y / b);
}

__host__ __device__ __forceinline__ uchar3 operator/(uchar3 a, int b) {
	return make_uchar3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ __forceinline__ int3 operator/(int3 a, unsigned int b) {
	return make_int3(a.x / (int) b, a.y / (int) b, a.z / (int) b);
}

__host__ __device__ __forceinline__ int3 operator/(int3 a, int b) {
	return make_int3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ __forceinline__ float3 operator/(float3 a, int b) {
	return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ __forceinline__ float3 operator/(float3 a, float b) {
	return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ __forceinline__ float3 operator/(float a, float3 b) {
	return make_float3(a / b.x, a / b.y, a / b.z);
}

__host__ __device__ __forceinline__ float4 operator/(float4 a, float b) {
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

__host__ __device__ __forceinline__ int3 operator%(int3 a, int b) {
	return make_int3(a.x % b, a.y % b, a.z % b);
}

__host__ __device__ __forceinline__ bool operator==(int3 a, int3 b) {
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

__host__  __device__  __forceinline__ float3 cross(float3 a, float3 b) {
	return make_float3(a.y * b.z - a.z * b.y,
			           a.z * b.x - a.x * b.z,
			           a.x * b.y - a.y * b.x);
}

__host__  __device__  __forceinline__ float3 cross(float4 a, float4 b) {
	return make_float3(a.y * b.z - a.z * b.y,
			           a.z * b.x - a.x * b.z,
			           a.x * b.y - a.y * b.x);
}

__host__ __device__ __forceinline__ float norm(float3 a) {
	return sqrt(a * a);
}

__host__ __device__ __forceinline__ float norm(float4 a) {
	return sqrt(a * a);
}

__host__ __device__ __forceinline__ float inv_norm(float3 a) {
	return 1.0 / sqrt(a * a);
}

__host__ __device__ __forceinline__ float inv_norm(float4 a) {
	return 1.0 / sqrt(a * a);
}

__host__ __device__ __forceinline__ float3 normalised(float3 a) {
	return a / norm(a);
}

__host__ __device__ __forceinline__ float4 normalised(float4 a) {
	return a / norm(a);
}

__host__ __device__ __forceinline__ float3 floor(float3 a) {
	return make_float3(floor(a.x), floor(a.y), floor(a.z));
}

__host__ __device__ __forceinline__ float3 fmaxf(float3 a, float3 b) {
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__host__ __device__ __forceinline__ float3 fminf(float3 a, float3 b) {
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

struct Matrix3f {

	float3 rowx, rowy, rowz;

	__host__ __device__ __forceinline__ Matrix3f() {
		rowx = rowy = rowz = make_float3(0, 0, 0);
	}

	__host__ __device__ __forceinline__ float3 operator*(float3 a) const {
		return make_float3(rowx * a, rowy * a, rowz * a);
	}

	__host__ __device__ __forceinline__ float3 operator*(float4 a) const {
		return make_float3(rowx * a, rowy * a, rowz * a);
	}
};

#endif
