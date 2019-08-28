#include "DeviceMap.h"

void ResetMap(DeviceMap map);

void ResetKeyPoints(KeyMap map);

void InsertKeyPoints(KeyMap map, DeviceArray<SURF> & keys,
		DeviceArray<int> & keyIndex, size_t size);

void CollectKeyPoints(KeyMap map, DeviceArray<SURF> & keys,
		DeviceArray<uint> & noKeys);

void Raycast(DeviceMap map, DeviceArray2D<float4> & vmap,
		DeviceArray2D<float4> & nmap,
		DeviceArray2D<float> & zRangeX,
		DeviceArray2D<float> & zRangeY,
		Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, float invfx, float invfy, float cx, float cy);

bool CreateRenderingBlocks(const DeviceArray<HashEntry> & visibleBlocks,
		DeviceArray2D<float> & zRangeX,
		DeviceArray2D<float> & zRangeY,
		const float & depthMax, const float & depthMin,
		DeviceArray<RenderingBlock> & renderingBlockList,
		DeviceArray<uint> & noRenderingBlocks,
		Matrix3f RviewInv, float3 tview,
		uint noVisibleBlocks, float fx, float fy, float cx, float cy);

uint MeshScene(DeviceArray<uint> & noOccupiedBlocks,
		DeviceArray<uint> & noTotalTriangles,
		DeviceMap map,
		const DeviceArray<int> & edgeTable,
		const DeviceArray<int> & vertexTable,
		const DeviceArray2D<int> & triangleTable,
		DeviceArray<float3> & normal,
		DeviceArray<float3> & vertex,
		DeviceArray<uchar3> & color,
		DeviceArray<int3> & blockPoses);

void CheckBlockVisibility(DeviceMap map, DeviceArray<uint> & noVisibleBlocks,
		Matrix3f Rview, Matrix3f RviewInv, float3 tview, int cols, int rows,
		float fx, float fy, float cx, float cy, float depthMax, float depthMin,
		uint * host_data);

void FuseMapColor(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color,
		const DeviceArray2D<float4> & nmap,
		const DeviceArray2D<unsigned char> & mask,
		const DeviceArray<int> & labels,
		int numDetection,
		DeviceArray<uint> & noVisibleBlocks,
		Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, DeviceMap map,
		float fx, float fy, float cx, float cy,
		float depthMax, float depthMin, uint * host_data);

void AnalyzeMapSemantics(const DeviceArray2D<float> & depth,
		const DeviceArray2D<uchar3> & color,
		const DeviceArray2D<float4> & nmap,
		const DeviceArray2D<unsigned char> & mask,
		const DeviceArray<int> & labels,
		int numDetection,
		DeviceArray<uint> & noVisibleBlocks,
		Matrix3f Rview, Matrix3f RviewInv,
		float3 tview, DeviceMap map,
		float fx, float fy, float cx, float cy,
		float depthMax, float depthMin);
