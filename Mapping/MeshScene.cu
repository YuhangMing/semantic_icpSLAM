#include "RenderScene.h"
#include "ParallelScan.h"

struct MeshEngine {

	DeviceMap map;
	mutable PtrSz<float3> vertices;
	mutable PtrSz<float3> normals;
	mutable PtrSz<uchar3> color;

	PtrStep<int> triangleTable;
	PtrSz<int> edgeTable;
	PtrSz<int> noVertexTable;

	PtrSz<int3> blockPos;

	uint* noBlocks;
	uint* noTriangles;

	__device__ inline void checkBlocks() {
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		__shared__ bool scan;
		if(x == 0)
			scan = false;
		__syncthreads();

		uint val = 0;
		if (x < DeviceMap::NumEntries && map.hashEntries[x].ptr >= 0) {
			int3 pos = map.hashEntries[x].pos * DeviceMap::BlockSize;
			scan = true;
			val = 1;
		}

		__syncthreads();
		if(scan) {
			int offset = ComputeOffset<1024>(val, noBlocks);
			if(offset != -1) {
				blockPos[offset] = map.hashEntries[x].pos;
			}
		}
	}

	__device__ inline bool readNormal(float3* n, float* sdf, int3 pos) {

		float v1, v2, v3;
		v1 = map.FindVoxel(pos + make_int3(-1, 0, 0)).sdf;
		v2 = map.FindVoxel(pos + make_int3(0, -1, 0)).sdf;
		v3 = map.FindVoxel(pos + make_int3(0, 0, -1)).sdf;
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[0] = make_float3(sdf[1] - v1, sdf[3] - v2, sdf[4] - v3);

		v1 = map.FindVoxel(pos + make_int3(2, 0, 0)).sdf;
		v2 = map.FindVoxel(pos + make_int3(1, -1, 0)).sdf;
		v3 = map.FindVoxel(pos + make_int3(1, 0, -1)).sdf;
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[1] = make_float3(v1 - sdf[0], sdf[2] - v2, sdf[5] - v3);

		v1 = map.FindVoxel(pos + make_int3(2, 1, 0)).sdf;
		v2 = map.FindVoxel(pos + make_int3(1, 2, 0)).sdf;
		v3 = map.FindVoxel(pos + make_int3(1, 1, -1)).sdf;
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[2] = make_float3(v1 - sdf[3], v2 - sdf[1], sdf[6] - v3);

		v1 = map.FindVoxel(pos + make_int3(-1, 1, 0)).sdf;
		v2 = map.FindVoxel(pos + make_int3(0, 2, 0)).sdf;
		v3 = map.FindVoxel(pos + make_int3(0, 1, -1)).sdf;
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[3] = make_float3(sdf[2] - v1, v2 - sdf[0], sdf[7] - v3);

		v1 = map.FindVoxel(pos + make_int3(-1, 0, 1)).sdf;
		v2 = map.FindVoxel(pos + make_int3(0, -1, 1)).sdf;
		v3 = map.FindVoxel(pos + make_int3(0, 0, 2)).sdf;
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[4] = make_float3(sdf[5] - v1, sdf[7] - v2, v3 - sdf[0]);

		v1 = map.FindVoxel(pos + make_int3(2, 0, 1)).sdf;
		v2 = map.FindVoxel(pos + make_int3(1, -1, 1)).sdf;
		v3 = map.FindVoxel(pos + make_int3(1, 0, 2)).sdf;
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[5] = make_float3(v1 - sdf[4], sdf[6] - v2 , v3 - sdf[1]);

		v1 = map.FindVoxel(pos + make_int3(2, 1, 1)).sdf;
		v2 = map.FindVoxel(pos + make_int3(1, 2, 1)).sdf;
		v3 = map.FindVoxel(pos + make_int3(1, 1, 2)).sdf;
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[6] = make_float3(v1 - sdf[7], v2 - sdf[5] , v3 - sdf[2]);

		v1 = map.FindVoxel(pos + make_int3(-1, 1, 1)).sdf;
		v2 = map.FindVoxel(pos + make_int3(0, 2, 1)).sdf;
		v3 = map.FindVoxel(pos + make_int3(0, 1, 2)).sdf;
		if(isnan(v1) || isnan(v2) || isnan(v3))
			return false;
		n[7] = make_float3(sdf[6] - v1, v2 - sdf[4] , v3 - sdf[3]);

		return true;
	}

	__device__ inline bool readVertexAndColor(uchar3* c, float* sdf, int3 pos) {

		map.FindVoxel(pos + make_float3(0, 0, 0)).getValue(sdf[0], c[0]);
		// if (sdf[0] > 0.7f || isnan(sdf[0]))
		if (sdf[0] == 1.0 || isnan(sdf[0]))
			return false;

		map.FindVoxel(pos + make_float3(1, 0, 0)).getValue(sdf[1], c[1]);
		if (sdf[1] == 1.0 || isnan(sdf[1]))
			return false;

		map.FindVoxel(pos + make_float3(1, 1, 0)).getValue(sdf[2], c[2]);
		if (sdf[2] == 1.0 || isnan(sdf[2]))
			return false;

		map.FindVoxel(pos + make_float3(0, 1, 0)).getValue(sdf[3], c[3]);
		if (sdf[3] == 1.0 || isnan(sdf[3]))
			return false;

		map.FindVoxel(pos + make_float3(0, 0, 1)).getValue(sdf[4], c[4]);
		if (sdf[4] == 1.0 || isnan(sdf[4]))
			return false;

		map.FindVoxel(pos + make_float3(1, 0, 1)).getValue(sdf[5], c[5]);
		if (sdf[5] == 1.0 || isnan(sdf[5]))
			return false;

		map.FindVoxel(pos + make_float3(1, 1, 1)).getValue(sdf[6], c[6]);
		if (sdf[6] == 1.0 || isnan(sdf[6]))
			return false;

		map.FindVoxel(pos + make_float3(0, 1, 1)).getValue(sdf[7], c[7]);
		if (sdf[7] == 1.0 || isnan(sdf[7]))
			return false;

		return true;
	}

	__device__ inline float interp(float & v1, float & v2) {
		if(fabs(0 - v1) < 1e-6)
			return 0;
		if(fabs(0 - v2) < 1e-6)
			return 1;
		if(fabs(v1 - v2) < 1e-6)
			return 0;
		return (0 - v1) / (v2 - v1);
	}

	__device__ inline int buildVertexList(float3* vlist, float3* nlist, uchar3* clist, const int3 & pos) {

		float3 normal[8];
		uchar3 color[8];
		float sdf[8];

		if (!readVertexAndColor(color, sdf, pos))
			return -1;

		if (!readNormal(normal, sdf, pos))
			return -1;

		int cubeIndex = 0;
		if (sdf[0] < 0)
			cubeIndex |= 1;
		if (sdf[1] < 0)
			cubeIndex |= 2;
		if (sdf[2] < 0)
			cubeIndex |= 4;
		if (sdf[3] < 0)
			cubeIndex |= 8;
		if (sdf[4] < 0)
			cubeIndex |= 16;
		if (sdf[5] < 0)
			cubeIndex |= 32;
		if (sdf[6] < 0)
			cubeIndex |= 64;
		if (sdf[7] < 0)
			cubeIndex |= 128;

		if (edgeTable[cubeIndex] == 0)
			return -1;

		if (edgeTable[cubeIndex] & 1) {
			float val = interp(sdf[0], sdf[1]);
			vlist[0] = pos + make_float3(val, 0, 0);
			nlist[0] = normal[0] + val * (normal[1] - normal[0]);
			clist[0] = color[0] + val * (color[1] - color[0]);
		}
		if (edgeTable[cubeIndex] & 2) {
			float val = interp(sdf[1], sdf[2]);
			vlist[1] = pos + make_float3(1, val, 0);
			nlist[1] = normal[1] + val * (normal[2] - normal[1]);
			clist[1] = color[1] + val * (color[2] - color[1]);
		}
		if (edgeTable[cubeIndex] & 4) {
			float val = interp(sdf[2], sdf[3]);
			vlist[2] = pos + make_float3(1 - val, 1, 0);
			nlist[2] = normal[2] + val * (normal[3] - normal[2]);
			clist[2] = color[2] + val * (color[3] - color[2]);
		}
		if (edgeTable[cubeIndex] & 8) {
			float val = interp(sdf[3], sdf[0]);
			vlist[3] = pos + make_float3(0, 1 - val, 0);
			nlist[3] = normal[3] + val * (normal[0] - normal[3]);
			clist[3] = color[3] + val * (color[0] - color[3]);
		}
		if (edgeTable[cubeIndex] & 16) {
			float val = interp(sdf[4], sdf[5]);
			vlist[4] = pos + make_float3(val, 0, 1);
			nlist[4] = normal[4] + val * (normal[5] - normal[4]);
			clist[4] = color[4] + val * (color[5] - color[4]);
		}
		if (edgeTable[cubeIndex] & 32) {
			float val = interp(sdf[5], sdf[6]);
			vlist[5] = pos + make_float3(1, val, 1);
			nlist[5] = normal[5] + val * (normal[6] - normal[5]);
			clist[5] = color[5] + val * (color[6] - color[5]);
		}
		if (edgeTable[cubeIndex] & 64) {
			float val = interp(sdf[6], sdf[7]);
			vlist[6] = pos + make_float3(1 - val, 1, 1);
			nlist[6] = normal[6] + val * (normal[7] - normal[6]);
			clist[6] = color[6] + val * (color[7] - color[6]);
		}
		if (edgeTable[cubeIndex] & 128) {
			float val = interp(sdf[7], sdf[4]);
			vlist[7] = pos + make_float3(0, 1 - val, 1);
			nlist[7] = normal[7] + val * (normal[4] - normal[7]);
			clist[7] = color[7] + val * (color[4] - color[7]);
		}
		if (edgeTable[cubeIndex] & 256) {
			float val = interp(sdf[0], sdf[4]);
			vlist[8] = pos + make_float3(0, 0, val);
			nlist[8] = normal[0] + val * (normal[4] - normal[0]);
			clist[8] = color[0] + val * (color[4] - color[0]);
		}
		if (edgeTable[cubeIndex] & 512) {
			float val = interp(sdf[1], sdf[5]);
			vlist[9] = pos + make_float3(1, 0, val);
			nlist[9] = normal[1] + val * (normal[5] - normal[1]);
			clist[9] = color[1] + val * (color[5] - color[1]);
		}
		if (edgeTable[cubeIndex] & 1024) {
			float val = interp(sdf[2], sdf[6]);
			vlist[10] = pos + make_float3(1, 1, val);
			nlist[10] = normal[2] + val * (normal[6] - normal[2]);
			clist[10] = color[2] + val * (color[6] - color[2]);
		}
		if (edgeTable[cubeIndex] & 2048) {
			float val = interp(sdf[3], sdf[7]);
			vlist[11] = pos + make_float3(0, 1, val);
			nlist[11] = normal[3] + val * (normal[7] - normal[3]);
			clist[11] = color[3] + val * (color[7] - color[3]);
		}

		return cubeIndex;
	}

	__device__ inline void MarchingCube() {
		int x = blockIdx.y * gridDim.x + blockIdx.x;
		if(*noTriangles >= DeviceMap::MaxTriangles || x >= *noBlocks)
			return;

		float3 vlist[12];
		float3 nlist[12];
		uchar3 clist[12];

		int3 pos = blockPos[x] * DeviceMap::BlockSize;
		for(int i = 0; i < DeviceMap::BlockSize; ++i) {
			int3 localPos = make_int3(threadIdx.x, threadIdx.y, i);
			int cubeIdx = buildVertexList(vlist, nlist, clist, pos + localPos);
			if(cubeIdx <= 0)
				continue;

			int noTriangleNeeded = noVertexTable[cubeIdx] / 3;
			uint offset = atomicAdd(noTriangles, noTriangleNeeded);
			for(int j = 0; j < noTriangleNeeded; ++j) {
				int tid = offset + j;
				if(tid >= DeviceMap::MaxTriangles)
					return;

				vertices[tid * 3 + 0] = vlist[triangleTable.ptr(cubeIdx)[j * 3 + 0]] * DeviceMap::VoxelSize;
				vertices[tid * 3 + 1] = vlist[triangleTable.ptr(cubeIdx)[j * 3 + 1]] * DeviceMap::VoxelSize;
				vertices[tid * 3 + 2] = vlist[triangleTable.ptr(cubeIdx)[j * 3 + 2]] * DeviceMap::VoxelSize;
				normals[tid * 3 + 0] = normalised(nlist[triangleTable.ptr(cubeIdx)[j * 3 + 0]]);
				normals[tid * 3 + 1] = normalised(nlist[triangleTable.ptr(cubeIdx)[j * 3 + 1]]);
				normals[tid * 3 + 2] = normalised(nlist[triangleTable.ptr(cubeIdx)[j * 3 + 2]]);
				color[tid * 3 + 0] = clist[triangleTable.ptr(cubeIdx)[j * 3 + 0]];
				color[tid * 3 + 1] = clist[triangleTable.ptr(cubeIdx)[j * 3 + 1]];
				color[tid * 3 + 2] = clist[triangleTable.ptr(cubeIdx)[j * 3 + 2]];
			}
		}
	}
};

__global__ void CheckBlockKernel(MeshEngine me) {
	me.checkBlocks();
}

__global__ void __launch_bounds__(64, 16) MeshSceneKernel(MeshEngine me) {
	me.MarchingCube();
}

uint MeshScene(DeviceArray<uint> & noOccupiedBlocks,
			   DeviceArray<uint> & noTotalTriangles,
			   DeviceMap map,
			   const DeviceArray<int> & edgeTable,
			   const DeviceArray<int> & vertexTable,
			   const DeviceArray2D<int> & triangleTable,
			   DeviceArray<float3> & normal,
			   DeviceArray<float3> & vertex,
			   DeviceArray<uchar3> & color,
			   DeviceArray<int3> & blockPoses) {

	noOccupiedBlocks.clear();
	noTotalTriangles.clear();

	MeshEngine engine;
	engine.map = map;
	engine.triangleTable = triangleTable;
	engine.edgeTable = edgeTable;
	engine.vertices = vertex;
	engine.noBlocks = noOccupiedBlocks;
	engine.noTriangles = noTotalTriangles;
	engine.normals = normal;
	engine.color = color;
	engine.blockPos = blockPoses;
	engine.noVertexTable = vertexTable;

	dim3 thread(1024);
	dim3 block = dim3(DivUp(DeviceMap::NumEntries, thread.x));

	CheckBlockKernel<<<block, thread>>>(engine);
	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	uint host_data;
	noOccupiedBlocks.download((void*) &host_data);
	if (host_data <= 0)
		return 0;

	thread = dim3(8, 8, 1);
	block = dim3(DivUp(host_data, 16), 16, 1);

	MeshSceneKernel<<<block, thread>>>(engine);
	SafeCall(cudaGetLastError());
	SafeCall(cudaDeviceSynchronize());

	noTotalTriangles.download((void*) &host_data);
	host_data = min(host_data, DeviceMap::MaxTriangles);

	return host_data;
}
