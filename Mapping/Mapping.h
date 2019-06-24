#ifndef MAPPING_HPP__
#define MAPPING_HPP__

#include "System.h"
#include "Tracking.h"
#include "KeyFrame.h"
#include "DeviceMap.h"

#include <vector>
#include <opencv.hpp>

class KeyMap;
class System;
class Tracker;

class Mapping {

public:
	
	/* 
	Following structures are not sure / should be in both
	*/

	// constructor, initialize objects of the class
	// called in System::System
	// calls Create()
	Mapping();
	// called in System::System and constructor (same)
	// initialize parameters for host and device storage
	// calls Reset()
	void Create();
	/*called in Create() & System::RebootSystem
	calls 
	ResetMap(DeviceMap *map) -> FuseMap.cu 
	  -> Release all hashEntries and visible entris, set mutex to be EntryAvailable
	  -> Store index in heapMem, release all created voxels, reset heapCounter and entryPtr.
	and 
	ResetKeyPoints(DeviceMap *map)
	clears DeviceArray <SURF> mapKeys and std::set<const KeyFrames *> keyFrames*/
	void Reset();
	// no definition. Declared but not defined
	void Release();
	// neigher called nor used
	// calls ForwardWarping -> Pyrdown.cu
	void ForwardWarp(const Frame * last, Frame * next);

	/*
	  QUESTION: where are these two function called?????
	*/
	// connect Mapping with KeyMap
	operator KeyMap() const;
	// connect Mapping with DeviceMap
	operator DeviceMap() const;

	// neither called nor used
	// convert elements in vector from "const KeyFrame *" to "KeyFrame *"
	std::vector<KeyFrame *> LocalMap() const;
	std::vector<KeyFrame *> GlobalMap() const;

	// atomic operation, bool variables
	std::atomic<bool> meshUpdated;
	std::atomic<bool> mapPointsUpdated;
	std::atomic<bool> mapUpdated;
	std::atomic<bool> hasNewKFFlag;
	bool lost;

	uint noBlocksInFrustum;

	std::vector<const KeyFrame *> localMap;
	std::set<const KeyFrame *> keyFrames;


	/* 
	Following structure should be stored in every submaps 
	*/

	/*Following 4 pairs of functions are used to perform mapping operations*/
	// called in System::GrabImage, Tracker::ValidatePose x2
	// calls CheckBlockVisibility -> FuseMap.cu -> CheckVisibleBlockKernel
	// returns uint noBlocks which will be used in RayTrace
	void UpdateVisibility(const Frame * f, uint & no);
	// called in System::RenderTopDown
	// calls CheckBlockVisibility -> FuseMap.cu -> CheckVisibleBlockKernel
	// returns uint noBlocks
	void UpdateVisibility(Matrix3f Rview, Matrix3f RviewInv, float3 tview,
			float depthMin, float depthMax, float fx, float fy, float cx,
			float cy, uint & no);
	// called in System::GrabImage x2
	// calls FuseColor
	void FuseColor(const Frame * f, uint & no);
	// called in FuseColor
	// calls FuseMapColor -> FuseMap.cu
	// create voxels and performs sdf fusion
	void FuseColor(const DeviceArray2D<float> & depth,
			const DeviceArray2D<uchar3> & color,
			const DeviceArray2D<float4> & normal, 
			const DeviceArray2D<unsigned char> & mask,
			const DeviceArray<int> & labels,
			int numDetection,
			Matrix3f Rview, Matrix3f RviewInv, float3 tview, uint & no);
	// called in System::GrabImage x3, Tracker::ValidatePose x2
	// calls RayTrace()
	void RayTrace(uint noVisibleBlocks, Frame * f);
	// called in RayTrace() and System::RenderTopDown
	// calls CreateRenderingBlocks -> RenderScene.cu
	//		   returns visibles, zRangeMin, zRangeMax, renderingBlockList, noRenderingBlocks
	//       Raycast -> RenderScene.cu
	// returns vertex map and normal map in the frame
	void RayTrace(uint noVisibleBlocks, Matrix3f Rview, Matrix3f RviewInv,
			float3 tview, DeviceArray2D<float4> & vmap,
			DeviceArray2D<float4> & nmap, float depthMin, float depthMax,
			float fx, float fy, float cx, float cy);
	// called in System::GrabImage
	// calls SemanticAnalysis()
	void SemanticAnalysis(const Frame *f);
	// called in SemanticAnalysis()
	// calls AnalyzeMapSemantics -> FuseMap.cu
	// returns modified rgb image
	void SemanticAnalysis(const DeviceArray2D<float> & depth,
			const DeviceArray2D<uchar3> & color,
			const DeviceArray2D<float4> & normal, 
			const DeviceArray2D<unsigned char> & mask,
			const DeviceArray<int> & labels,
			int numDetection,
			Matrix3f Rview, Matrix3f RviewInv, float3 tview);

	/*Following 3 functions are used to fuse key-data into the map*/
	// called in Optimizer::run, which is NOT IMPLEMENTED yet
	// returns hasNewKFFlag
	bool HasNewKF();
	// called in Tracker::CreateKeyFrame
	// fuse keyframe data into the map (complicate function)
	void FuseKeyFrame(const KeyFrame * kf);
	// neither called nor used.
	void FuseKeyPoints(const Frame * f);

	/*Following RAM related functions and variables are used to store and read map from disk*/
	// called in DownloadToRAM() & System::ReadMapFromDisk
	// initiate RAM for data storage
	void CreateRAM();
	// called in System::WriteMapToDisk
	// calls CreateRAM()
	void DownloadToRAM();
	// called in System::ReadMapFromDisk
	void UploadFromRAM();
	// called in System::ReadMapFromDisk
	void ReleaseRAM();
	// Host Memory Spaces， used to store map in disk, some should be for the whole map not the submaps
	int * heapRAM;
	int * heapCounterRAM;
	int * hashCounterRAM;
	int * bucketMutexRAM;
	Voxel * sdfBlockRAM;
	uint * noVisibleEntriesRAM;
	HashEntry * hashEntriesRAM;
	HashEntry * visibleEntriesRAM;
	int * mutexKeysRAM;
	SURF * mapKeysRAM;

	// key points in host
	uint noKeysHost;					//DeviceArray<uint> noKeys, stores the total number of keypoints
	std::vector<SURF> hostKeys;			//DeviceArray<SURF> tmpKeys, stores all the keypoints in the map


	/* 
	Following structure should be stored in the general map 
	*/

	// called in System::GrabImage/WriteMeshToDisk/ReadMapFromDisk
	// calls MeshScene -> MeshScene.cu -> MARCHING CUBES algorithm
	// returns nBlocks, noTriangles, modelNormal, modelVertex, modelColor, blockPoses
	void CreateModel();

	// called in System::GrabImage & Tracker::Relocalise
	// calls CollectKeyPoints -> FuseMap.cu 
	// collects keypoints in the map and stores noKeys & noKeysHost
	void UpdateMapKeys();

	// used for meshing
	uint noTrianglesHost;				//DeviceArray<uint> noTriangles, stores the number of triangles for meshing
	DeviceArray<float3> modelVertex;	// for MeshScene
	DeviceArray<float3> modelNormal;	// for MeshScene
	DeviceArray<uchar3> modelColor;		// for MeshScene

protected:

	/* 
	Following structure should be stored in every submaps 
	*/

	// General map structure
	// DeviceArray<int> heap;					// stores the memory necessary for all voxel blocks
	// DeviceArray<int> heapCounter;			// counts how many voxel blocks has been created
	// DeviceArray<int> hashCounter;			// counts how many hash entries are occupied
	// DeviceArray<int> bucketMutex;			// store all the mutex for single thread access
	// DeviceArray<Voxel> sdfBlock;			// stores all the voxels, index is entry's pointer + index inside the block
	DeviceArray<uint> noVisibleEntries;		// stores the number of visible hashentries for current frame
	// DeviceArray<HashEntry> hashEntries;		// stores all the hashentries
	DeviceArray<HashEntry> visibleEntries;	// stores all visible hashentries for current frame

	// Used for rendering
	DeviceArray<uint> noRenderingBlocks;			// store the number of rendering blocks
	DeviceArray<RenderingBlock> renderingBlockList;	// RenderingBlock is defined in DeviceMap.h, 
	DeviceArray2D<float> zRangeMin;					// Same for the following 4 device arrays
	DeviceArray2D<float> zRangeMax;					// Rendering depth range for every subsampled pixel
	DeviceArray2D<float> zRangeMinEnlarged;
	DeviceArray2D<float> zRangeMaxEnlarged;

	// Key Points and Re-localisation
	DeviceArray<uint> noKeys;		// noKeysHost
	DeviceArray<int> mutexKeys;		// store all the mutex for keypoints access
	DeviceArray<int> mapKeyIndex;	// store the keypoins index 
	DeviceArray<SURF> mapKeys;		// stores all the keypoints in the map
	DeviceArray<SURF> tmpKeys;		// used in UpdateMapKeys
	DeviceArray<SURF> surfKeys;		// used in FuseKeyFrame, store the keypoints to be fused into the map


	/* 
	Following structure should be stored in the general map 
	*/

	// General map structure
	DeviceArray<int> heap;					// stores the memory necessary for all voxel blocks
	DeviceArray<int> heapCounter;			// counts how many voxel blocks has been created
	DeviceArray<int> hashCounter;			// counts how many hash entries are occupied
	DeviceArray<int> bucketMutex;			// store all the mutex for single thread access
	DeviceArray<Voxel> sdfBlock;			// stores all the voxels, index is entry's pointer + index inside the block
	// DeviceArray<uint> noVisibleEntries;		// stores the number of visible hashentries for current frame
	DeviceArray<HashEntry> hashEntries;		// stores all the hashentries
	// DeviceArray<HashEntry> visibleEntries;	// stores all visible hashentries for current frame

	// Used for meshing
	DeviceArray<uint> nBlocks;			// number of occupied blocks, returned from MeshScene
	DeviceArray<int3> blockPoses;		// positions of all occupied blocks
	DeviceArray<uint> noTriangles;		// number of triangles needed
	DeviceArray<int> edgeTable;			// the following 3 table, are predefined in Constant.h for marching cubes
	DeviceArray<int> vertexTable;
	DeviceArray2D<int> triangleTable;

};

#endif
