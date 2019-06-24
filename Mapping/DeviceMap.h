#ifndef DEVICE_STRUCT_HPP__
#define DEVICE_STRUCT_HPP__

#include "VectorMath.h"
#include "DeviceArray.h"

#define MaxThread 1024

enum ENTRYTYPE { EntryAvailable = -1, EntryOccupied = -2, EntryToDelete = -3};

struct __align__(8) RenderingBlock {
	// define the block in image coordinate system
	// in pixel
	short2 upperLeft;
	short2 lowerRight;
	// in meters
	float2 zRange;
};

struct __align__(16) HashEntry {

	__device__ __forceinline__ HashEntry() :
			pos(make_int3(0)), ptr(-1), offset(-1) {
	}

	__device__ __forceinline__ HashEntry(int3 pos_, int ptr_, int offset_) :
			pos(pos_), ptr(ptr_), offset(offset_) {
	}

	__device__ __forceinline__ HashEntry(const HashEntry & other) {
		pos = other.pos;
		ptr = other.ptr;
		offset = other.offset;
	}

	__device__ __forceinline__ void release() {
		pos = make_int3(0);
		ptr = -1;
	}

	__device__ __forceinline__ void operator=(const HashEntry & other) {
		pos = other.pos;
		ptr = other.ptr;
		offset = other.offset;
	}

	__device__ __forceinline__ bool operator==(const int3 & pos_) const {
		return pos == pos_;
	}

	__device__ __forceinline__ bool operator==(const HashEntry & other) const {
		return other.pos == pos;
	}

	int3 pos;

	int  ptr; // memory address that pointed to the block

	int  offset;
};

// struct __align__(8) Voxel {
struct __align__(16) Voxel {
// struct Voxel {

	__device__ __forceinline__ Voxel() :
			sdf(std::nanf("0x7fffffff")), weight(0), color(make_uchar3(0)), 
			label(0), count(0), label_2(0), count_2(0), color_2(make_uchar3(0)) {
	}

	__device__ __forceinline__ Voxel(float sdf_, unsigned char weight_, uchar3 color_, short label_, short count_) :
			sdf(sdf_), weight(weight_), color(color_), label(label_), count(count_),
			label_2(0), count_2(0), color_2(make_uchar3(0)) {
	}

	__device__ __forceinline__ void release() {
		sdf = std::nanf("0x7fffffff");
		weight = 0;
		color = make_uchar3(0);
		label = 0;
		count = 0;
		label_2 = 0;
		count_2 = 0;
		// label_3 = 0;
		// count_3 = 0;
		color_2 = make_uchar3(0);
	}

    // called in MeshScene.cu, add label and prob later
	__device__ __forceinline__ void getValue(float & sdf_, uchar3 & color_) const {
		sdf_ = sdf;
		color_ = color;
	}

	__device__ __forceinline__ void operator=(const Voxel & other) {
		sdf = other.sdf;
		weight = other.weight;
		color = other.color;
		label = other.label;
		count = other.count;
		label_2 = other.label_2;
		count_2 = other.count_2;
		// label_3 = other.label_3;
		// count_3 = other.count_3;
		color_2 = other.color_2;
	}

	// 4+1+ 1 +2+ 2
	float sdf;
	unsigned char weight;
	uchar3 color;

	unsigned char label, count;
	// second detection
	unsigned char label_2, count_2;
	// // third detection
	// unsigned char label_3, count_3;
	uchar3 color_2;
	// CONSIDER CHANGE backup color to Object Index. Use prefix scan to get the right idx
	unsigned char obj_idx;

	// SOMEHOW the size of the voxel is 16
};

struct KeyPoint {

};

struct SURF : public KeyPoint {

	bool valid;

	float3 pos;

	float4 normal;

	float descriptor[64];
};

struct DeviceMap {

	__device__ uint Hash(const int3 & pos);
	__device__ Voxel FindVoxel(const int3 & pos);
	__device__ Voxel FindVoxel(const float3 & pos);
	__device__ Voxel FindVoxel(const float3 & pos, HashEntry & cache, bool & valid);
	__device__ HashEntry FindEntry(const int3 & pos);
	__device__ HashEntry FindEntry(const float3 & pos);
	__device__ void CreateBlock(const int3 & blockPos);
	__device__ bool FindVoxel(const int3 & pos, Voxel & vox);
	__device__ bool FindVoxel(const float3 & pos, Voxel & vox);
	__device__ HashEntry CreateEntry(const int3 & pos, const int & offset);
	__device__ void DeleteBlock(HashEntry & entry);

	__device__ int3 worldPosToVoxelPos(float3 pos) const;
	__device__ int3 voxelPosToBlockPos(const int3 & pos) const;
	__device__ int3 blockPosToVoxelPos(const int3 & pos) const;
	__device__ int3 voxelPosToLocalPos(const int3 & pos) const;
	__device__ int3 localIdxToLocalPos(const int & idx) const;
	__device__ int3 worldPosToBlockPos(const float3 & pos) const;
	__device__ float3 worldPosToVoxelPosF(float3 pos) const;
	__device__ float3 voxelPosToWorldPos(int3 pos) const;
	__device__ float3 blockPosToWorldPos(const int3 & pos) const;
	__device__ int localPosToLocalIdx(const int3 & pos) const;
	__device__ int voxelPosToLocalIdx(const int3 & pos) const;

	static constexpr uint BlockSize = 8;
	static constexpr uint BlockSize3 = 512;
	static constexpr float DepthMin = 0.1f;			// in meters
	static constexpr float DepthMax = 3.0f;			// in meters
	static constexpr uint NumExcess = 500000;
	static constexpr uint NumBuckets = 1000000;
	// ####################################################################
	// BECAUSE the use of MaskRCNN, 
	// and the addition of object info in voxels
	// Have to reduce voxels number and map size to fit in the memory size
	// ####################################################################
	// static constexpr uint NumSdfBlocks = 700000; 
	static constexpr uint NumSdfBlocks = 500000;
	static constexpr uint NumVoxels = NumSdfBlocks * BlockSize3;
	// static constexpr uint MaxTriangles = 20000000; // roughly 700MB memory
	static constexpr uint MaxTriangles = 5000000;
	static constexpr uint MaxVertices = MaxTriangles * 3;
	static constexpr float VoxelSize = 0.004f;		// in meters
	static constexpr float TruncateDist = VoxelSize * 6;	// in meters
	static constexpr int MaxRenderingBlocks = 260000;
	static constexpr float voxelSizeInv = 1.0 / VoxelSize;
	static constexpr float blockWidth = VoxelSize * BlockSize;
	static constexpr uint NumEntries = NumBuckets + NumExcess;
	static constexpr float stepScale = 0.5 * TruncateDist * voxelSizeInv;	// # of voxels as half the truncate distance

	PtrSz<int> heapMem; // stores all predefined voxel memory
	PtrSz<int> entryPtr;
	PtrSz<int> heapCounter;
	PtrSz<int> bucketMutex;
	PtrSz<Voxel> voxelBlocks;
	PtrSz<uint> noVisibleBlocks;
	PtrSz<HashEntry> hashEntries;
	PtrSz<HashEntry> visibleEntries;
	// PtrSz<HashEntry> occupiedEntries;

	PtrSz<Voxel> objectVoxels;
};

struct KeyMap {

	static constexpr float GridSize = 0.01;
	static constexpr int MaxKeys = 100000;
	static constexpr int nBuckets = 5;
	static constexpr int maxEntries = MaxKeys * nBuckets;
	static constexpr int MaxObs = 10;
	static constexpr int MinObsThresh = -5;

	__device__ int Hash(const int3& pos);

	__device__ SURF * FindKey(const float3 & pos);

	__device__ SURF * FindKey(const float3 & pos, int & first, int & buck, int & hashIndex);

	__device__ void InsertKey(SURF* key, int & hashIndex);

	__device__ void ResetKeys(int index);

	PtrSz<SURF> Keys;

	PtrSz<int> Mutex;
};

#endif
