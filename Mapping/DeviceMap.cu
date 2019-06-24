#include "DeviceMap.h"

__device__ uint DeviceMap::Hash(const int3 & pos) {
	int res = ((pos.x * 73856093) ^ (pos.y * 19349669) ^ (pos.z * 83492791))
			% NumBuckets;

	if (res < 0)
		res += NumBuckets;
	return res;
}

__device__ HashEntry DeviceMap::CreateEntry(const int3 & pos,
		const int & offset) {
	int old = atomicSub(heapCounter, 1);
	if (old >= 0) {
		int ptr = heapMem[old];
		if (ptr != -1)
			return HashEntry(pos, ptr * BlockSize3, offset);
	}
	return HashEntry(pos, EntryAvailable, 0);
}

__device__ void DeviceMap::CreateBlock(const int3& blockPos) {
	int bucketId = Hash(blockPos);
	int* mutex = &bucketMutex[bucketId];
	HashEntry* e = &hashEntries[bucketId];
	HashEntry* eEmpty = nullptr;
	// current block entry is not available, return
	if (e->pos == blockPos && e->ptr != EntryAvailable)
		return;
	
	// if entry available, copy to eEmpty
	if (e->ptr == EntryAvailable && !eEmpty)
		eEmpty = e;
	// if entry not available, check offset/appended list
	// loop until the front-most/released entry is found
	while (e->offset > 0) {
		bucketId = NumBuckets + e->offset - 1;
		e = &hashEntries[bucketId];
		// return if current block entry is not available
		if (e->pos == blockPos && e->ptr != EntryAvailable)
			return;

		// find first available entry
		if (e->ptr == EntryAvailable && !eEmpty){
			eEmpty = e;
			// break;
		}
	}

	if (eEmpty) {
		int old = atomicExch(mutex, EntryOccupied);
		if (old == EntryAvailable) {
			*eEmpty = CreateEntry(blockPos, e->offset);
			atomicExch(mutex, EntryAvailable);
		}
	} else {
		int old = atomicExch(mutex, EntryOccupied);
		if (old == EntryAvailable) {
			int offset = atomicAdd(entryPtr, 1);
			if (offset <= NumExcess) {
				eEmpty = &hashEntries[NumBuckets + offset - 1];
				*eEmpty = CreateEntry(blockPos, 0);
				e->offset = offset;
			}
			atomicExch(mutex, EntryAvailable);
		}
	}
}

__device__ void DeviceMap::DeleteBlock(HashEntry & entry) {
	// store the assigned pointer back to the heapMem
	int re = atomicAdd(heapCounter, 1);
	// printf("re = %d \n", re);
	if(re >= NumSdfBlocks)
		return;

	heapMem[re] = entry.ptr / BlockSize3;
	// release the entry
	entry.release();
}

__device__ bool DeviceMap::FindVoxel(const float3 & pos, Voxel & vox) {
	int3 voxel_pos = worldPosToVoxelPos(pos);
	return FindVoxel(voxel_pos, vox);
}

__device__ bool DeviceMap::FindVoxel(const int3 & pos, Voxel & vox) {
	HashEntry entry = FindEntry(voxelPosToBlockPos(pos));
	if (entry.ptr == EntryAvailable)
		return false;
	int idx = voxelPosToLocalIdx(pos);
	vox = voxelBlocks[entry.ptr + idx];
	return true;
}

__device__ Voxel DeviceMap::FindVoxel(const int3 & pos) {
	HashEntry entry = FindEntry(voxelPosToBlockPos(pos));
	Voxel voxel;
	if (entry.ptr == EntryAvailable)
		return voxel;
	return voxelBlocks[entry.ptr + voxelPosToLocalIdx(pos)];
}

__device__ Voxel DeviceMap::FindVoxel(const float3 & pos) {
	int3 p = make_int3(pos);
	HashEntry entry = FindEntry(voxelPosToBlockPos(p));

	Voxel voxel;
	if (entry.ptr == EntryAvailable)
		return voxel;

	return voxelBlocks[entry.ptr + voxelPosToLocalIdx(p)];
}

__device__ Voxel DeviceMap::FindVoxel(const float3 & pos, HashEntry & cache, bool & valid) {
	int3 p = make_int3(pos);	// = "floor" here
	int3 blockPos = voxelPosToBlockPos(p);
	if(blockPos == cache.pos) {
		valid = true;
		return voxelBlocks[cache.ptr + voxelPosToLocalIdx(p)];
	}

	HashEntry entry = FindEntry(blockPos);
	if (entry.ptr == EntryAvailable) {
		valid = false;
		return Voxel();
	}

	valid = true;
	cache = entry;
	return voxelBlocks[entry.ptr + voxelPosToLocalIdx(p)];
}

__device__ HashEntry DeviceMap::FindEntry(const float3 & pos) {
	int3 blockIdx = worldPosToBlockPos(pos);

	return FindEntry(blockIdx);
}

__device__ HashEntry DeviceMap::FindEntry(const int3& blockPos) {
	uint bucketId = Hash(blockPos);
	HashEntry* e = &hashEntries[bucketId];
	if (e->ptr != EntryAvailable && e->pos == blockPos)
		return *e;

	while (e->offset > 0) {
		bucketId = NumBuckets + e->offset - 1;
		e = &hashEntries[bucketId];
		if (e->pos == blockPos && e->ptr != EntryAvailable)
			return *e;
	}
	return HashEntry(blockPos, EntryAvailable, 0);
}

__device__ int3 DeviceMap::worldPosToVoxelPos(float3 pos) const {
	float3 p = pos / VoxelSize;
	return make_int3(p);
}

__device__ float3 DeviceMap::worldPosToVoxelPosF(float3 pos) const {
	return pos / VoxelSize;
}

__device__ float3 DeviceMap::voxelPosToWorldPos(int3 pos) const {
	return pos * VoxelSize;
}

__device__ int3 DeviceMap::voxelPosToBlockPos(const int3 & pos) const {
	int3 voxel = pos;

	if (voxel.x < 0)
		voxel.x -= BlockSize - 1;
	if (voxel.y < 0)
		voxel.y -= BlockSize - 1;
	if (voxel.z < 0)
		voxel.z -= BlockSize - 1;

	return voxel / BlockSize;
}

__device__ int3 DeviceMap::blockPosToVoxelPos(const int3 & pos) const {
	return pos * BlockSize;
}

__device__ int3 DeviceMap::voxelPosToLocalPos(const int3 & pos) const {
	int3 local = pos % BlockSize;

	if (local.x < 0)
		local.x += BlockSize;
	if (local.y < 0)
		local.y += BlockSize;
	if (local.z < 0)
		local.z += BlockSize;

	return local;
}

__device__ int DeviceMap::localPosToLocalIdx(const int3 & pos) const {
	return pos.z * BlockSize * BlockSize + pos.y * BlockSize + pos.x;
}

__device__ int3 DeviceMap::localIdxToLocalPos(const int & idx) const {
	uint x = idx % BlockSize;
	uint y = idx % (BlockSize * BlockSize) / BlockSize;
	uint z = idx / (BlockSize * BlockSize);
	return make_int3(x, y, z);
}

__device__ int3 DeviceMap::worldPosToBlockPos(const float3 & pos) const {
	return voxelPosToBlockPos(worldPosToVoxelPos(pos));
}

__device__ float3 DeviceMap::blockPosToWorldPos(const int3 & pos) const {
	return voxelPosToWorldPos(blockPosToVoxelPos(pos));
}

__device__ int DeviceMap::voxelPosToLocalIdx(const int3 & pos) const {
	return localPosToLocalIdx(voxelPosToLocalPos(pos));
}

///////////////////////////////////////////////////////
// Implementation - Key Maps
///////////////////////////////////////////////////////
__device__ int KeyMap::Hash(const int3 & pos) {

	int res = ((pos.x * 73856093) ^ (pos.y * 19349669) ^ (pos.z * 83492791)) % KeyMap::MaxKeys;

	if (res < 0)
		res += KeyMap::MaxKeys;

	return res;
}

__device__ SURF * KeyMap::FindKey(const float3 & pos) {

	int3 blockPos = make_int3(pos / GridSize);
	int idx = Hash(blockPos);
	int bucketIdx = idx * nBuckets;
	for (int i = 0; i < nBuckets; ++i, ++bucketIdx) {
		SURF * key = &Keys[bucketIdx];
		if (key->valid) {
			if(make_int3(key->pos / GridSize) == blockPos)
				return key;
		}
	}
	return nullptr;
}

__device__ SURF * KeyMap::FindKey(const float3 & pos, int & first,
		int & buck, int & hashIndex) {

	first = -1;
	int3 p = make_int3(pos / GridSize);
	int idx = Hash(p);
	buck = idx;
	int bucketIdx = idx * nBuckets;
	for (int i = 0; i < nBuckets; ++i, ++bucketIdx) {
		SURF * key = &Keys[bucketIdx];
		if (!key->valid && first == -1)
			first = bucketIdx;

		if (key->valid) {
			int3 tmp = make_int3(key->pos / GridSize);
			if(tmp == p) {
				hashIndex = bucketIdx;
				return key;
			}
		}
	}

	return NULL;
}

__device__ void KeyMap::InsertKey(SURF * key, int & hashIndex) {

	int buck = 0;
	int first = -1;
	SURF * oldKey = NULL;
//	if(hashIndex >= 0 && hashIndex < Keys.size) {
//		oldKey = &Keys[hashIndex];
//		if (oldKey && oldKey->valid) {
//			key->pos = oldKey->pos;
//			return;
//		}
//	}

	oldKey = FindKey(key->pos, first, buck, hashIndex);
	if (oldKey && oldKey->valid) {
		key->pos = oldKey->pos;
		return;
	}
	else if (first != -1) {

		int lock = atomicExch(&Mutex[buck], 1);
		if (lock < 0) {
			hashIndex = first;
			SURF * oldkey = &Keys[first];
			memcpy((void*) oldkey, (void*) key, sizeof(SURF));

			atomicExch(&Mutex[buck], -1);
			return;
		}
	}
}

__device__ void KeyMap::ResetKeys(int index) {

	if (index < Mutex.size)
		Mutex[index] = -1;

	if (index < Keys.size) {
		Keys[index].valid = false;
	}
}
