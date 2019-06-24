#include "RenderScene.h"
#include "ParallelScan.h"

struct Fusion
{

	DeviceMap map;
	float invfx, invfy;
	float fx, fy, cx, cy;
	float minDepth, maxDepth;
	int cols, rows;
	Matrix3f Rview;
	Matrix3f RviewInv;
	float3 tview;

	uint *noVisibleBlocks;

	int pallete[3] = {int(pow(2, 25) - 1), int(pow(2, 15) - 1), int(pow(2, 21) - 1)};
	PtrSz<int> labels;
	PtrStep<unsigned char> mask;
	int numDetection;

	PtrStep<float4> nmap;
	PtrStep<float> depth;
	PtrStep<uchar3> rgb;

	__device__ inline float2 project(float3 &pt3d)
	{
		float2 pt2d;
		pt2d.x = fx * pt3d.x / pt3d.z + cx;
		pt2d.y = fy * pt3d.y / pt3d.z + cy;
		return pt2d;
	}

	__device__ inline float3 unproject(int &x, int &y, float &z)
	{
		// from pixel (in image coordinates) to meters (in camera coordinates)
		float3 pt3d;
		pt3d.z = z;
		pt3d.x = z * (x - cx) * invfx;
		pt3d.y = z * (y - cy) * invfy;
		// return result in meters (in world coordinates)
		return Rview * pt3d + tview;
	}

	__device__ inline bool CheckVertexVisibility(const float3 &pt3d)
	{
		float3 pt = RviewInv * (pt3d - tview);
		if (pt.z < 1e-3f)
			return false;
		float2 pt2d = project(pt);

		return pt2d.x >= 0 && pt2d.y >= 0 &&
			   pt2d.x < cols && pt2d.y < rows &&
			   pt.z >= minDepth && pt.z <= maxDepth;
	}

	__device__ inline bool CheckBlockVisibility(const int3 &pos)
	{

		float scale = DeviceMap::blockWidth;
		float3 corner = pos * scale;
		if (CheckVertexVisibility(corner))
			return true;
		corner.z += scale;
		if (CheckVertexVisibility(corner))
			return true;
		corner.y += scale;
		if (CheckVertexVisibility(corner))
			return true;
		corner.x += scale;
		if (CheckVertexVisibility(corner))
			return true;
		corner.z -= scale;
		if (CheckVertexVisibility(corner))
			return true;
		corner.y -= scale;
		if (CheckVertexVisibility(corner))
			return true;
		corner.x -= scale;
		corner.y += scale;
		if (CheckVertexVisibility(corner))
			return true;
		corner.x += scale;
		corner.y -= scale;
		corner.z += scale;
		if (CheckVertexVisibility(corner))
			return true;
		return false;
	}

	__device__ inline void CreateBlocks()
	{

		// get current location
		// x, y in pixels;
		// z    in meters.
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= cols || y >= rows)
			return;

		float z = depth.ptr(y)[x];
		if (isnan(z) || z < DeviceMap::DepthMin || z > DeviceMap::DepthMax)
			return;

		// choose a range of 8-voxel_size long
		float thresh = DeviceMap::TruncateDist / 2;
		float z_near = max(DeviceMap::DepthMin, z - thresh);
		float z_far = min(DeviceMap::DepthMax, z + thresh);
		if (z_near >= z_far)
			return;

		// find start and end points of this depth range
		// pt_near, pt_far all in unit of voxels (meters / voxelSize), in world coordinates
		float3 pt_near = unproject(x, y, z_near) * DeviceMap::voxelSizeInv;
		float3 pt_far = unproject(x, y, z_far) * DeviceMap::voxelSizeInv;
		float3 dir = pt_far - pt_near;

		// choose a step_size based on experience
		float length = norm(dir);
		int nSteps = (int)ceil(2.0 * length); // jing yan zhi
		dir = dir / (float)(nSteps - 1);

		// create a block around point (every step_size) in this depth range
		for (int i = 0; i < nSteps; ++i)
		{
			int3 blockPos = map.voxelPosToBlockPos(make_int3(pt_near)); // find out which block this voxel belongs to
			map.CreateBlock(blockPos);									// create block at this block position
			pt_near += dir;
		}
	}

	__device__ inline void CheckFullVisibility()
	{

		__shared__ bool bScan;
		if (threadIdx.x == 0)
			bScan = false;
		__syncthreads();
		uint val = 0;
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		if (x < map.hashEntries.size)
		{
			HashEntry &e = map.hashEntries[x];
			if (e.ptr != EntryAvailable)
			{
				if (CheckBlockVisibility(e.pos))
				{
					bScan = true;
					val = 1;	// 0 for invisible, 1 for visible
				}
			}
		}

		__syncthreads();
		if (bScan)
		{
			// label visible entries an offset for an entry
			// details refer to "prefix sum"
			int offset = ComputeOffset<1024>(val, noVisibleBlocks);
			if (offset != -1 && offset < map.visibleEntries.size && x < map.hashEntries.size)
			{
				map.visibleEntries[offset] = map.hashEntries[x];
			}
		}
		
	}

	__device__ inline void collectGarbage()
	{
		if (blockIdx.x >= map.visibleEntries.size ||
			blockIdx.x >= *noVisibleBlocks)
			return;

		HashEntry &entry = map.visibleEntries[blockIdx.x];
		if (entry.ptr == EntryAvailable)
			return;

		// int3 block_pos = map.blockPosToVoxelPos(entry.pos);

		__shared__ float minAbsTSDF;
		__shared__ unsigned char maxWeight;

		// calculate point location in camera coordinate system
		int3 localPos = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);
		int locId = map.localPosToLocalIdx(localPos);

		Voxel &prev = map.voxelBlocks[entry.ptr + locId];
		float absTSDF = prev.sdf > 0 ? prev.sdf : (-1 * prev.sdf);

		if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
			minAbsTSDF = absTSDF;
			maxWeight = prev.weight;
		}
		__syncthreads();

		minAbsTSDF = minAbsTSDF <= absTSDF ? minAbsTSDF : absTSDF;
		__syncthreads();

		maxWeight = maxWeight >= prev.weight ? maxWeight : prev.weight;
		__syncthreads();

		// printf("minTSDF = %f, maxWeight = %d", minAbsTSDF, maxWeight);
		// 0.5 could be changed later for better performance
		if(minAbsTSDF < 0.7 || maxWeight == 0){
			entry.ptr = EntryToDelete;
			// delete in second pass
			// map.DeleteBlock(entry);
		}

	}

	__device__ inline void integrateColor()
	{

		if (blockIdx.x >= map.visibleEntries.size ||
			blockIdx.x >= *noVisibleBlocks)
			return;

		HashEntry &entry = map.visibleEntries[blockIdx.x];
		if (entry.ptr == EntryAvailable)
			return;

		int3 block_pos = map.blockPosToVoxelPos(entry.pos);

		// praga unroll is used to reduce loop consumption
		#pragma unroll
		for (int i = 0; i < 8; ++i)
		{
			int3 localPos = make_int3(threadIdx.x, threadIdx.y, i);
			int locId = map.localPosToLocalIdx(localPos);
			float3 pos = map.voxelPosToWorldPos(block_pos + localPos);
			pos = RviewInv * (pos - tview);
			int2 uv = make_int2(project(pos));
			if (uv.x < 0 || uv.y < 0 || uv.x >= cols || uv.y >= rows)
				continue;

			float dp = depth.ptr(uv.y)[uv.x];
			if (isnan(dp) || dp > maxDepth || dp < minDepth)
				continue;

			float thresh = DeviceMap::TruncateDist;
			float sdf = dp - pos.z;

			if (sdf >= -thresh)
			{
				// INITIALIZE CURRENT SDF VALUES
				sdf = fmin(1.0f, sdf / thresh);
				float4 nl = nmap.ptr(uv.y)[uv.x];
				if (isnan(nl.x))
					continue;
				float w = nl * normalised(make_float4(pos));

				// INITIALIZE CURRENT OBJ VALUES
				float3 val;
				unsigned char label;
				// float prob;
				if (numDetection>0)
				{
					// if there are obj detected, use specific color to mask it
					label = mask.ptr(uv.y)[uv.x];
					if (label == 0)
					{
						val = make_float3(rgb.ptr(uv.y)[uv.x]);
					}
					else
					{
						val = make_float3(int(pallete[0] * int(label) % 255),
										  int(pallete[1] * int(label) % 255),
										  int(pallete[2] * int(label) % 255));
					}
				}
				else
				{
					// if no obj, use original color
					label = 0;
					val = make_float3(rgb.ptr(uv.y)[uv.x]);
				}
				// printf("label %d with score %f\n", int(label), prob);

				Voxel &prev = map.voxelBlocks[entry.ptr + locId];
				if (prev.weight == 0)
				{
					// create new voxel
					prev = Voxel(sdf, 1, make_uchar3(val), label, 0);
					// if(numDetection>0){
					// 	printf("Creating voxels in second pass.\n");
					// }
				}
				else
				{
					// previous voxel created
					// update color, prob & label
					if (label != 0)
					// currently object found ##########
					{
						if (label == prev.label)
						// same object found
						{
							// has same label as previous one
							// update probability, label and color stay the same
							prev.count = min(255, prev.count + 1);
							prev.color = make_uchar3(val);
						}
						else
						// different object found
						{
							// USE A NEW UPDATE APPROACH HERE ###########
							// DISCARD probability completely, use the # of detection as confidence
							if (label == prev.label_2)
							// check with back_up label
							{
								prev.count_2 = min(255, prev.count_2 + 1);
								prev.color_2 = make_uchar3(val);
								// compare count
								if (prev.count < prev.count_2)
								{
									// swap orig with backup
									short tmp = prev.label;
									prev.label = prev.label_2;
									prev.label_2 = tmp;
									tmp = prev.count;
									prev.count = prev.count_2;
									prev.count_2 = tmp;
									uchar3 tmpC = prev.color;
									prev.color = prev.color_2;
									prev.color_2 = tmpC;
								}
							}
							else
							// update back_up label if the previous one is detected few times only
							// discard new detection if previous one is already very confident
							{
								if (prev.count_2 < 7)
								{
									prev.label_2 = label;
									prev.count_2 = 1;
									prev.color_2 = make_uchar3(val);
								}
							}
						}
						// // compare count
						// if(prev.count < prev.count_2) {
						// 	// swap orig with backup
						// 	short tmp = prev.label;
						// 	prev.label = prev.label_2;
						// 	prev.label_2 = tmp;
						// 	tmp = prev.count;
						// 	prev.count = prev.count_2;
						// 	prev.count_2 = tmp;
						// 	// uchar3 tmpC = prev.color;
						// 	// prev.color = prev.color_2;
						// 	// prev.color_2 = tmpC;
						// }
					}
					else
					// currently object not found ##########
					{
						if (prev.label == 0)
						{
							// no obj found, update as icpSLAM does
							val = val / 255.f;
							float3 old = make_float3(prev.color) / 255.f;
							float3 res = (w * 0.2f * val + (1 - w * 0.2f) * old) * 255.f;
							prev.color = make_uchar3(res);
							// prev.count = min(255, prev.count + 1);
							// printf("prev.color = (%d, %d, %d) in the case of bg.",
							// 		prev.color.x, prev.color.y, prev.color.z);
						}
						else
						{
							// previous objected detected, check previous object confidence/count
							if(prev.count < 7){
								// move current obj label to back_up label and set current as bg
								prev.label_2 = prev.label;
								prev.color_2 = prev.color;
								prev.count_2 = prev.count;
								prev.label = 0;
								prev.count = 1;
								val = val / 255.f;
								float3 old = make_float3(prev.color) / 255.f;
								float3 res = (w * 0.2f * val + (1 - w * 0.2f) * old) * 255.f;
								prev.color = make_uchar3(res);
							}

							// // prev has obj detected, stay the same
							// if(prev.label_2 == 0)
							// {
							// 	val = val / 255.f;
							// 	float3 old = make_float3(prev.color) / 255.f;
							// 	float3 res = (w * 0.2f * val + (1 - w * 0.2f) * old) * 255.f;
							// 	prev.color_2 = make_uchar3(res);
							// 	prev.count_2 = min(255, prev.count + 1);
							// 	// compare count
							// 	if (prev.count < prev.count_2)
							// 	{
							// 		// swap orig with backup
							// 		short tmp = prev.label;
							// 		prev.label = prev.label_2;
							// 		prev.label_2 = tmp;
							// 		tmp = prev.count;
							// 		prev.count = prev.count_2;
							// 		prev.count_2 = tmp;
							// 		uchar3 tmpC = prev.color;
							// 		prev.color = prev.color_2;
							// 		prev.color_2 = tmpC;
							// 	}
							// }
							// else
							// {
							// 	if (prev.count_2 < 7)
							// 	{
							// 		prev.label_2 = label;
							// 		prev.count_2 = 1;
							// 		val = val / 255.f;
							// 		float3 old = make_float3(prev.color) / 255.f;
							// 		float3 res = (w * 0.2f * val + (1 - w * 0.2f) * old) * 255.f;
							// 		prev.color_2 = make_uchar3(res);
							// 	}
							// }
						}
					}
					// update sdf and voxel weights
					prev.sdf = (prev.sdf * prev.weight + w * sdf) / (prev.weight + w);
					prev.weight = min(255, prev.weight + 1);
				}
			}
		}
	}

	__device__ inline void createVoxels()
	{
		// KEEP THE SAME AS integrateColor in icpSLAM
		// do not mess with colors, only create voxels and update sdfs
		if (blockIdx.x >= map.visibleEntries.size ||
			blockIdx.x >= *noVisibleBlocks)
			return;

		HashEntry &entry = map.visibleEntries[blockIdx.x];
		if (entry.ptr == EntryToDelete){
			map.DeleteBlock(entry);
		}
		if (entry.ptr == EntryAvailable)
			return;

		int3 block_pos = map.blockPosToVoxelPos(entry.pos);

		// praga unroll is used to reduce loop consumption
		#pragma unroll
		for (int i = 0; i < 8; ++i)
		{
			// calculate point location in camera coordinate system
			int3 localPos = make_int3(threadIdx.x, threadIdx.y, i);
			int locId = map.localPosToLocalIdx(localPos);
			float3 pos = map.voxelPosToWorldPos(block_pos + localPos);
			pos = RviewInv * (pos - tview);
			// point in image coordinate
			int2 uv = make_int2(project(pos));
			if (uv.x < 0 || uv.y < 0 || uv.x >= cols || uv.y >= rows)
				continue;

			float dp = depth.ptr(uv.y)[uv.x];
			if (isnan(dp) || dp > maxDepth || dp < minDepth)
				continue;

			float thresh = DeviceMap::TruncateDist;
			float sdf = dp - pos.z;

			if (sdf >= -thresh)
			{
				// INITIALIZE CURRENT SDF VALUES
				sdf = fmin(1.0f, sdf / thresh);
				// // calculate weight for color fusion
				// float4 nl = nmap.ptr(uv.y)[uv.x];
				// if (isnan(nl.x))
				// 	continue;
				// float w = nl * normalised(make_float4(pos));

				// INITIALIZE CURRENT OBJ VALUES
				float3 val = make_float3(rgb.ptr(uv.y)[uv.x]);
				// printf("label %d with score %f\n", int(label), prob);

				Voxel &prev = map.voxelBlocks[entry.ptr + locId];
				if (prev.weight == 0)
				{
					// create new voxel, sdf's weight initialized as 1
					prev = Voxel(sdf, (unsigned char) 1, make_uchar3(val), 
								 (unsigned char) 0, 0);
				}
				else
				{
					// update color if no object detected
					if(prev.label == 0){
						// val = val / 255.f;
						// float3 old = make_float3(prev.color) / 255.f;
						// float3 res = (w * 0.2f * val + (1 - w * 0.2f) * old) * 255.f;
						float3 res = (make_float3(prev.color) * prev.weight + val) / (prev.weight + 1);
						prev.color = make_uchar3(res);
					}	
					// update sdf and voxel weights
					prev.sdf = (prev.sdf * prev.weight + sdf) / (prev.weight + 1);
					prev.weight = min(255, prev.weight + 1);
				}
			}
		}
	}

	__device__ inline void analyzeSemantics()
	{
		// printf("Start coloring map;    ");
		// semantic analysis, update object masks
		if (blockIdx.x >= map.visibleEntries.size ||
			blockIdx.x >= *noVisibleBlocks)
			return;
		// printf("Valid block index;    ");

		HashEntry &entry = map.visibleEntries[blockIdx.x];
		if (entry.ptr == EntryAvailable)
			return;
		// printf("Valid entry;    ");

		int3 block_pos = map.blockPosToVoxelPos(entry.pos);

		// praga unroll is used to reduce loop consumption
		#pragma unroll
		for (int i = 0; i < 8; ++i)
		{
			int3 localPos = make_int3(threadIdx.x, threadIdx.y, i);
			int locId = map.localPosToLocalIdx(localPos);
			float3 pos = map.voxelPosToWorldPos(block_pos + localPos);
			pos = RviewInv * (pos - tview);
			int2 uv = make_int2(project(pos));
			if (uv.x < 0 || uv.y < 0 || uv.x >= cols || uv.y >= rows)
				continue;
			// printf("Valid pixel coordinates;    ");

			float dp = depth.ptr(uv.y)[uv.x];
			// continue when it's 0 or nan, too close or too far away
			// printf("Depth value is %f;    ", dp);
			if (isnan(dp) || dp > maxDepth || dp < minDepth)
				continue;
			// printf("Pass depth threshold;    ");
			
			float4 nl = nmap.ptr(uv.y)[uv.x];
			if (isnan(nl.x))
				continue;

			float thresh = DeviceMap::TruncateDist;
			float sdf = dp - pos.z;

			// only update the first encountered voxel
			if (sdf >= -thresh)
			{
				// printf("update color;    ");
			// INITIALIZE CURRENT OBJ VALUES
			float3 val;
			unsigned char label;
			// float prob;
			if (numDetection>0)
			{
				// if there are obj detected, use specific color to mask it
				label = mask.ptr(uv.y)[uv.x];
				if (label == 0)
				{
					val = make_float3(rgb.ptr(uv.y)[uv.x]);
				}
				else
				{
					val = make_float3(int(pallete[0] * int(label) % 255),
									  int(pallete[1] * int(label) % 255),
									  int(pallete[2] * int(label) % 255));
				}
			}
			else
			{
				// if no obj, use original color
				label = 0;
				val = make_float3(rgb.ptr(uv.y)[uv.x]);
			}

			Voxel &prev = map.voxelBlocks[entry.ptr + locId];
			// UPDATE SCHEME
			if (prev.weight != 0)
			{
				// previous voxel created
				// update color, prob & label
				if (label != 0)
				// currently object found ##########
				///// !!!!!!!!!! CONSIDER store these voxels in a separate data structure
				{
					if (label == prev.label)
					// same object found
					{
						prev.count = min(255, prev.count + 1);
						prev.color = make_uchar3(val);
					}
					else
					// different object found
					{
						// USE COUNT/VOTE INSTEAD OF PROBABILITY HERE ###########
						if (label == prev.label_2)
						// check with back_up label
						{
							prev.count_2 = min(255, prev.count_2 + 1);
							prev.color_2 = make_uchar3(val);
						}
						else
						// update back_up label if the previous one is detected few times only
						// discard new detection if previous one is already very confident
						{
							if (prev.count_2 < 7)
							{
								prev.label_2 = label;
								prev.count_2 = 1;
								prev.color_2 = make_uchar3(val);
							}
						}
						// compare count
						// switch if 1) prev is bg, object detected half frequently more than the bg
						//           2) prev is ob, has more count
						if ( (prev.count < prev.count_2 && prev.label !=0) ||
						     (prev.count < prev.count_2*2 && prev.label ==0) )
						{
							// swap orig with backup
							short tmp = prev.label;
							prev.label = prev.label_2;
							prev.label_2 = tmp;
							tmp = prev.count;
							prev.count = prev.count_2;
							prev.count_2 = tmp;
							uchar3 tmpC = prev.color;
							prev.color = prev.color_2;
							prev.color_2 = tmpC;
						}
					}
				}
				else
				// currently object not found ##########
				{
					float w = nl * normalised(make_float4(pos));
					if (prev.label != 0)
					{
						// previous objected detected, check previous object confidence/count
						if(prev.count < 7){
							// move current obj label to back_up label and set current as bg
							prev.label_2 = prev.label;
							prev.color_2 = prev.color;
							prev.count_2 = prev.count;
							prev.label = 0;
							prev.count = 1;
							val = val / 255.f;
							float3 old = make_float3(prev.color) / 255.f;
							float3 res = (w * 0.2f * val + (1 - w * 0.2f) * old) * 255.f;
							prev.color = make_uchar3(res);
						}
					}
					else
					{
						// prev is bg, current is bg
						prev.count = min(255, prev.count + 1);
					}
					
				}
			}
			}
		}
	}

	__device__ inline void UnifyBlockLabel()
	{
		// 	// check if block is visible
		// 	if(blockIdx.x >= map.visibleEntries.size || blockIdx.x >= *noVisibleBlocks)
		// 		return;

		// 	// check if block is occupied
		// 	HashEntry& entry = map.visibleEntries[blockIdx.x];
		// 	if (entry.ptr == EntryAvailable)
		// 		return;

		// 	// get block position
		// 	int3 block_pos = map.blockPosToVoxelPos(entry.pos);

		// 	// loop through every voxel to find the most like label and probability
		// 	int *voxel_label = new int[4];
		// 	int *voxel_count = new int[4];
		// 	float *voxel_prob = new float[4];
		// 	uchar3 *voxel_color = new uchar3[4];
		// 	int label_count = 0;

		// 	for(int locId=0; locId<512; locId++) {
		// 		// get voxel
		// 		Voxel & cur_voxel = map.voxelBlocks[entry.ptr + locId];
		// 		int tmp_label = cur_voxel.label;
		// 		float tmp_prob = cur_voxel.prob;
		// 		uchar3 tmp_color = cur_voxel.color;

		// 		// check if this label already exist
		// 		if(label_count == 0){
		// 			voxel_label[0] = tmp_label;
		// 			voxel_prob[0] = tmp_prob;
		// 			voxel_color[0] = tmp_color;
		// 			voxel_count[0] = 1;
		// 			label_count++;
		// 		}
		// 		else
		// 		{
		// 			bool match_found = false;
		// 			for(int m=0; m<label_count; m++) {
		// 				// if match found
		// 				if(voxel_label[m] == tmp_label) {
		// 					voxel_count[m] ++;
		// 					voxel_prob[m] = ((voxel_count[m]-1)*voxel_prob[m] + tmp_prob)
		// 									/voxel_count[m];
		// 					match_found = true;
		// 					break;
		// 				}
		// 			}
		// 			// if no match found, add new label
		// 			if(!match_found) {
		// 				voxel_label[label_count] = tmp_label;
		// 				voxel_prob[label_count] = tmp_prob;
		// 				voxel_color[label_count] = tmp_color;
		// 				voxel_count[label_count] = 1;
		// 				label_count++;
		// 			}
		// 		}
		// 	}

		// 	// update voxel labels and probability for all voxels
		// 	// find max probability and associated index
		// 	int max_idx = 0;
		// 	float max_prob = voxel_prob[0];
		// 	for(int idx=1; idx<label_count; idx++) {
		// 		float next = voxel_prob[idx];
		// 		if (max_prob >= next) {
		// 			continue;
		// 		} else {
		// 			max_prob = next;
		// 			max_idx = idx;
		// 		}
		// 	}
		// 	for(int locId=0; locId<512; locId++){
		// 		Voxel & cur_voxel = map.voxelBlocks[entry.ptr + locId];
		// 		cur_voxel.prob = max_prob;
		// 		cur_voxel.label = voxel_label[max_idx];
		// 		cur_voxel.color = voxel_color[max_idx];
		// 	}

		// 	// release memory
		// 	delete[] voxel_label;
		// 	delete[] voxel_color;
		// 	delete[] voxel_prob;
		// 	delete[] voxel_count;
	}

	__device__ inline void UpdateNeighborLabel(float3 pos, int n_label[], int n_count[], int n_lc[])
	{
		/////// LOOKS LIKE sth. is WRONG here, all -1 was updated to the same neighbor label
		/////// eg. [-1, -1, -1, -1] -> [63, 63, 63, 63] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		/////// using "else if" instead of "if" magically solved the problem
		/////// the count is not correct. 
		Voxel neighbor = Voxel();
		bool neighbor_found = map.FindVoxel(pos, neighbor);
		if (neighbor_found)
		{
			// store label
			int tmp_label = neighbor.weight == 0 ? -1 : int(neighbor.label);
			// check if voxel at this location is created before
			if (tmp_label == -1)
			{
				return;
			}
			for (int l = 0; l < 4; l++)
			{
				// search for a match
				if (n_label[l] == tmp_label)
				{
					n_count[l]++;
					n_lc[l] = n_lc[l] > int(neighbor.count) ? n_lc[l] : int(neighbor.count);
					break;
				}
				// if match not found, add new if possible
				else if (n_label[l] == -1)
				{
					n_label[l] = tmp_label;
					n_count[l] = 1;
					n_lc[l] = int(neighbor.count);
					break;
				}
			}
		}
		else
		{
			// store -1
		}
	}

	__device__ inline void UnifyNeighbor()
	{
		// if (blockIdx.x >= map.hashEntries.size || blockIdx.x >= *noVisibleBlocks)
		// 	return;
		if (blockIdx.x >= map.hashEntries.size)
			return;

		// HashEntry& entry = map.visibleEntries[blockIdx.x];
		HashEntry &entry = map.hashEntries[blockIdx.x];
		if (entry.ptr == EntryAvailable)
			return;
		// found all occupied entries

		int3 block_pos = map.blockPosToVoxelPos(entry.pos);

		// praga unroll is used to reduce loop consumption
		int neighbor_label[4] = {-1, -1, -1, -1};
		int neighbor_count[4] = {0, 0, 0, 0};
		int neighbor_lc[4] = {0, 0, 0, 0};
		// Voxel & neighbor;
		// #pragma unroll
		for (int i = 0; i < 8; ++i)
		{
			int3 localPos = make_int3(threadIdx.x, threadIdx.y, i);
			int locId = map.localPosToLocalIdx(localPos);
			Voxel &cur = map.voxelBlocks[entry.ptr + locId];
			int cur_label = cur.weight == 0 ? -1 : int(cur.label);
			if (cur_label == -1)
			{
				continue;
			}

			float3 pos = map.voxelPosToWorldPos(block_pos + localPos);
			// get neighbors 26-adjacency
			float3 neighbor_pos;
			int range = 2;
			/// SOMETHINGS' wong here when trying to find the neighbor labels
			for (int r = 1; r < range; r++)
			{
				float step = map.VoxelSize * r;
				neighbor_pos = make_float3(pos.x + step, pos.y, pos.z);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x + step, pos.y + step, pos.z);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x + step, pos.y - step, pos.z);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x + step, pos.y, pos.z + step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x + step, pos.y, pos.z - step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x + step, pos.y + step, pos.z + step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x + step, pos.y + step, pos.z - step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x + step, pos.y - step, pos.z + step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x + step, pos.y - step, pos.z - step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);

				neighbor_pos = make_float3(pos.x - step, pos.y, pos.z);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x - step, pos.y + step, pos.z);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x - step, pos.y - step, pos.z);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x - step, pos.y, pos.z + step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x - step, pos.y, pos.z - step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x - step, pos.y + step, pos.z + step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x - step, pos.y + step, pos.z - step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x - step, pos.y - step, pos.z + step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x - step, pos.y - step, pos.z - step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);

				neighbor_pos = make_float3(pos.x, pos.y + step, pos.z);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x, pos.y + step, pos.z + step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x, pos.y + step, pos.z - step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x, pos.y - step, pos.z);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x, pos.y - step, pos.z + step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x, pos.y - step, pos.z - step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);

				neighbor_pos = make_float3(pos.x, pos.y, pos.z + step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
				neighbor_pos = make_float3(pos.x, pos.y, pos.z - step);
				UpdateNeighborLabel(neighbor_pos, neighbor_label, neighbor_count, neighbor_lc);
			}

			// // check neighbor count
			// printf("neighbor label-count: %d-%d, %d-%d, %d-%d, %d-%d\n",
			// 		neighbor_label[0], neighbor_count[0],
			// 		neighbor_label[1], neighbor_count[1],
			// 		neighbor_label[2], neighbor_count[2],
			// 		neighbor_label[3], neighbor_count[3]);

			int tmp_nei_label = -1;
			int tmp_nei_count = 0;
			int tmp_nei_lc = 0;
			for (int j = 0; j < 4; j++)
			{
				if (neighbor_label[j] != -1 && neighbor_lc[j] != 0)
				{
					if (neighbor_count[j] > tmp_nei_count)
					{
						tmp_nei_label = neighbor_label[j];
						tmp_nei_count = neighbor_count[j];
						// tmp_nei_lc = neighbor_lc[j] / neighbor_count[j];
						tmp_nei_lc = neighbor_lc[j];
					}
				}
			}
			// if (cur.label == 63)
			// {
			// 	printf("-- label-count: %d-%d >>>>>>>> %d-%d \n   nei_labels: %d, %d, %d, %d \n       nei_lc: %d, %d, %d, %d \n   nei_counts: %d, %d, %d, %d \n",
			// 		   cur.label, cur.count, tmp_nei_label, tmp_nei_lc,
			// 		   neighbor_label[0], neighbor_label[1], neighbor_label[2], neighbor_label[3],
			// 		   neighbor_lc[0], neighbor_lc[1], neighbor_lc[2], neighbor_lc[3],
			// 		   neighbor_count[0], neighbor_count[1], neighbor_count[2], neighbor_count[3]);
			// }
			if (cur_label != tmp_nei_label)
			{
				// called in every frames
				// printf("UPDATE voxel label from kNN\n");
				int tmp;
				if(cur_label == int(cur.label_2)){
					tmp = tmp_nei_lc > int(cur.count_2) ? tmp_nei_lc : (cur.count_2+5);
				} else {
					tmp = tmp_nei_lc;
				}
				cur.label_2 = cur.label;
				cur.count_2 = cur.count;
				cur.label = (unsigned char)tmp_nei_label;
				cur.count = (unsigned char)tmp;
			}
			else
			{
				cur.count = (unsigned char)tmp_nei_lc;
			}
			// printf("%d-%d \n", cur.label, cur.count);
		}
	}
};

__global__ void CreateBlocksKernel(Fusion fuse)
{
	fuse.CreateBlocks();
}

__global__ void CheckVisibleBlockKernel(Fusion fuse)
{
	fuse.CheckFullVisibility();
}

__global__ void FuseColorKernal(Fusion fuse)
{
	fuse.integrateColor();
}

__global__ void CreateVoxelsKernel(Fusion fuse)
{
	fuse.createVoxels();
}

__global__ void GarbageCollectKernel(Fusion fuse)
{
	fuse.collectGarbage();
}

__global__ void SemanticAnalysisKernel(Fusion fuse)
{
	fuse.analyzeSemantics();
}

__global__ void UnifyBlockLabelKernel(Fusion fuse)
{
	fuse.UnifyBlockLabel();
}

__global__ void UnifyNeighborKernel(Fusion fuse)
{
	fuse.UnifyNeighbor();
}

void CheckBlockVisibility(DeviceMap map,
						  DeviceArray<uint> &noVisibleBlocks,
						  Matrix3f Rview,
						  Matrix3f RviewInv,
						  float3 tview,
						  int cols,
						  int rows,
						  float fx,
						  float fy,
						  float cx,
						  float cy,
						  float depthMax,
						  float depthMin,
						  uint *host_data)
{

	noVisibleBlocks.clear();

	Fusion fuse;
	fuse.map = map;
	fuse.Rview = Rview;
	fuse.RviewInv = RviewInv;
	fuse.tview = tview;
	fuse.fx = fx;
	fuse.fy = fy;
	fuse.cx = cx;
	fuse.cy = cy;
	fuse.invfx = 1.0 / fx;
	fuse.invfy = 1.0 / fy;
	fuse.rows = rows;
	fuse.cols = cols;
	fuse.noVisibleBlocks = noVisibleBlocks;
	fuse.maxDepth = depthMax;
	fuse.minDepth = depthMin;

	dim3 thread = dim3(1024);
	dim3 block = dim3(DivUp((int)DeviceMap::NumEntries, thread.x));

	CheckVisibleBlockKernel<<<block, thread>>>(fuse);

	host_data[0] = 0;
	noVisibleBlocks.download((void *)host_data);
	if (host_data[0] == 0){
		printf(" ## WARNING: no visible blocks found!\n");
		return;
	}
}

void FuseMapColor(const DeviceArray2D<float> &depth,
				  const DeviceArray2D<uchar3> &color,
				  const DeviceArray2D<float4> &nmap,
				  const DeviceArray2D<unsigned char> &mask,
				  const DeviceArray<int> &labels,
				  int numDetection,
				  DeviceArray<uint> &noVisibleBlocks,
				  Matrix3f Rview,
				  Matrix3f RviewInv,
				  float3 tview,
				  DeviceMap map,
				  float fx,
				  float fy,
				  float cx,
				  float cy,
				  float depthMax,
				  float depthMin,
				  uint *host_data)
{

	int cols = depth.cols;
	int rows = depth.rows;
	noVisibleBlocks.clear();

	Fusion fuse;
	fuse.map = map;
	fuse.Rview = Rview;
	fuse.RviewInv = RviewInv;
	fuse.tview = tview;
	fuse.fx = fx;
	fuse.fy = fy;
	fuse.cx = cx;
	fuse.cy = cy;
	fuse.invfx = 1.0 / fx;
	fuse.invfy = 1.0 / fy;
	fuse.depth = depth;
	fuse.rgb = color;
	fuse.nmap = nmap;
	fuse.rows = rows;
	fuse.cols = cols;
	fuse.noVisibleBlocks = noVisibleBlocks;
	fuse.maxDepth = DeviceMap::DepthMax;
	fuse.minDepth = DeviceMap::DepthMin;
	fuse.numDetection = numDetection;
	if (numDetection>0)
	{
		fuse.mask = mask;
		fuse.labels = labels;
	}

	dim3 thread, block;

	// ####### CREATE NEW BLOCKS 
	thread = dim3(16, 8);
	block = dim3(DivUp(cols, thread.x), DivUp(rows, thread.y));
	// printf("Grid : {%d, %d} blocks. Blocks : {%d, %d} threads.\n", block.x, block.y, thread.x, thread.y);
	// 40, 60; 16, 8
	CreateBlocksKernel<<<block, thread>>>(fuse);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	// ####### FIND ALL VISIBLE BLOCKS
	thread = dim3(1024);
	block = dim3(DivUp((int)DeviceMap::NumEntries, thread.x));
	// printf("Grid : {%d} blocks. Blocks : {%d} threads.\n", block.x, thread.x);
	// 1465, 1024
	CheckVisibleBlockKernel<<<block, thread>>>(fuse);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	// old version // ####### INTEGRATE COLOR, CREATE VOXELS IN THE BLOCK
	// FuseColorKernal<<<block, thread>>>(fuse);
	// SafeCall(cudaDeviceSynchronize());
	// SafeCall(cudaGetLastError());

	// ####### CREATE VOXELS IN THE BLOCK
	host_data[0] = 0;
	noVisibleBlocks.download((void *)host_data);
	if (host_data[0] == 0){
		printf(" ## WARNING: no visible blocks found!\n");
		return;
	}
	// else{
	// 	printf(" %d visible blocks found.\n", host_data[0]);
	// }
	thread = dim3(8, 8); // a block has 8x8x8 voxels, parallely operate on 8x8 voxels and perform a 8-loop in each thread
	block = dim3(host_data[0]);
	CreateVoxelsKernel<<<block, thread>>>(fuse);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	// ####### COLLECT GARBAGE
	// assign a block for a block of voxels
	thread = dim3(8, 8, 8);
	// printf("Grid : {%d, %d} blocks. Blocks : {%d, %d, %d} threads.\n\n", block.x, block.y, thread.x, thread.y, thread.z);
	// depends(~3000), 1; 8, 8, 8
	GarbageCollectKernel<<<block, thread>>>(fuse);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	// if detection is enabled, but shouldn't for now
	if(numDetection > 0){
		// printf("Disable Mapping First!!!")

		// ####### COLOR OBJECTS IN THE MAP
		SemanticAnalysisKernel<<<block, thread>>>(fuse);
		SafeCall(cudaDeviceSynchronize());
		SafeCall(cudaGetLastError());

		// // ####### UNIFY LABELS & PROBABILITIES IN ONE BLOCK
		// // again use all visible blocks
		// thread = dim3(1);
		// UnifyBlockLabelKernel<<<block, thread>>>(fuse);
		// SafeCall(cudaDeviceSynchronize());
		// SafeCall(cudaGetLastError());

		// Utilize neighbor information after objects are colored	
		// ####### NEIGHBOR UNIFY
		// block = dim3(DeviceMap::NumEntries); // should use this one value, but super slow !!!!!!!!!!!!!!
		UnifyNeighborKernel<<<block, thread>>>(fuse);
		SafeCall(cudaDeviceSynchronize());
		SafeCall(cudaGetLastError());
	}
}

void AnalyzeMapSemantics(const DeviceArray2D<float> &depth,
						 const DeviceArray2D<uchar3> &color,
						 const DeviceArray2D<float4> &nmap,
						 const DeviceArray2D<unsigned char> &mask,
						 const DeviceArray<int> &labels,
						 int numDetection,
						 DeviceArray<uint> &noVisibleBlocks,
						 Matrix3f Rview,
						 Matrix3f RviewInv,
						 float3 tview,
						 DeviceMap map,
						 float fx,
						 float fy,
						 float cx,
						 float cy,
						 float depthMax,
						 float depthMin)
{
	int cols = depth.cols;
	int rows = depth.rows;
	noVisibleBlocks.clear();

	Fusion fuse;
	fuse.map = map;
	fuse.Rview = Rview;
	fuse.RviewInv = RviewInv;
	fuse.tview = tview;
	fuse.fx = fx;
	fuse.fy = fy;
	fuse.cx = cx;
	fuse.cy = cy;
	fuse.invfx = 1.0 / fx;
	fuse.invfy = 1.0 / fy;
	fuse.depth = depth;
	fuse.rgb = color;
	fuse.nmap = nmap;
	fuse.rows = rows;
	fuse.cols = cols;
	fuse.noVisibleBlocks = noVisibleBlocks;
	fuse.maxDepth = DeviceMap::DepthMax;
	fuse.minDepth = DeviceMap::DepthMin;
	fuse.numDetection = numDetection;
	if (numDetection>0)
	{
		fuse.mask = mask;
		fuse.labels = labels;
	}

	dim3 thread, block;
	uint host_data;

	// printf("Checking visible blocks...\n");
	// check visible blocks
	thread = dim3(1024);
	block = dim3(DivUp((int)DeviceMap::NumEntries, thread.x));
	// printf("Grid : {%d} blocks. Blocks : {%d} threads.\n", block.x, thread.x);
	// 1465, 1024
	CheckVisibleBlockKernel<<<block, thread>>>(fuse);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	// printf("Analyzing the built map...\n");
	// semantic analysis
	host_data = 0;
	noVisibleBlocks.download((void *) &host_data);
	// printf("%d visible blocks found\n", host_data);
	if (host_data == 0){
		printf(" ## WARNING: no visible blocks found!\n");
		return;
	}
	thread = dim3(8, 8); // a block has 8x8x8 voxels, parallely operate on 8x8 voxels and perform a 8-loop in each thread
	block = dim3(host_data);
	SemanticAnalysisKernel<<<block, thread>>>(fuse);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

	// ####### NEIGHBOR UNIFY
	// printf("Checking neighbors...\n");
	// block = dim3(DeviceMap::NumEntries);  // should use this one value, but super slow !!!!!!!!!!!!!!
	UnifyNeighborKernel<<<block, thread>>>(fuse);
	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());

}



__global__ void ResetHashKernel(DeviceMap map)
{
	// printf("HashKernel block %d, thread %d\n", blockIdx.x, threadIdx.x);

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < map.hashEntries.size)
	{
		map.hashEntries[x].release();
		map.visibleEntries[x].release();
	}

	if (x < DeviceMap::NumBuckets)
	{
		map.bucketMutex[x] = EntryAvailable;
	}
}

__global__ void ResetSdfBlockKernel(DeviceMap map)
{
	// printf("SdfBlockKernel block %d, thread %d\n", blockIdx.x, threadIdx.x);

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < DeviceMap::NumSdfBlocks)
	{
		map.heapMem[x] = DeviceMap::NumSdfBlocks - x - 1;
	}

	int idx = x * DeviceMap::BlockSize3;
	for (int i = 0; i < DeviceMap::BlockSize3 && idx < map.voxelBlocks.size; ++i, ++idx)
	{
		map.voxelBlocks[idx].release();
	}

	if (x == 0)
	{
		map.heapCounter[0] = DeviceMap::NumSdfBlocks - 1;
		map.entryPtr[0] = 1;
	}
}

void ResetMap(DeviceMap map)
{

	dim3 thread(1024);
	dim3 block(DivUp((int)DeviceMap::NumEntries, thread.x));

	ResetHashKernel<<<block, thread>>>(map);

	block = dim3(DivUp((int)DeviceMap::NumSdfBlocks, thread.x));
	ResetSdfBlockKernel<<<block, thread>>>(map);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

__global__ void ResetKeyPointsKernel(KeyMap map)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	map.ResetKeys(x);
}

void ResetKeyPoints(KeyMap map)
{

	dim3 thread(1024);
	dim3 block(DivUp((int)KeyMap::maxEntries, thread.x));

	ResetKeyPointsKernel<<<block, thread>>>(map);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

struct KeyFusion
{

	__device__ __forceinline__ void CollectKeys()
	{

		__shared__ bool scan;
		if (threadIdx.x == 0)
			scan = false;
		__syncthreads();

		uint val = 0;
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		if (x < map.Keys.size)
		{
			SURF *key = &map.Keys[x];
			if (key->valid)
			{
				scan = true;
				val = 1;
			}
		}
		__syncthreads();

		if (scan)
		{
			int offset = ComputeOffset<1024>(val, nokeys);
			if (offset > 0 && x < map.Keys.size)
			{
				memcpy(&keys[offset], &map.Keys[x], sizeof(SURF));
			}
		}
	}

	__device__ __forceinline__ void InsertKeys()
	{

		int x = blockDim.x * blockIdx.x + threadIdx.x;
		if (x < size)
			map.InsertKey(&keys[x], index[x]);
	}

	KeyMap map;

	uint *nokeys;

	PtrSz<SURF> keys;

	size_t size;

	PtrSz<int> index;
};

__global__ void CollectKeyPointsKernel(KeyFusion fuse)
{
	fuse.CollectKeys();
}

__global__ void InsertKeyPointsKernel(KeyFusion fuse)
{
	fuse.InsertKeys();
}

void CollectKeyPoints(KeyMap map, DeviceArray<SURF> &keys, DeviceArray<uint> &noKeys)
{

	KeyFusion fuse;
	fuse.map = map;
	fuse.keys = keys;
	fuse.nokeys = noKeys;

	dim3 thread(1024);
	dim3 block(DivUp(map.Keys.size, thread.x));

	CollectKeyPointsKernel<<<block, thread>>>(fuse);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}

void InsertKeyPoints(KeyMap map, DeviceArray<SURF> &keys,
					 DeviceArray<int> &keyIndex, size_t size)
{

	if (size == 0)
		return;

	KeyFusion fuse;

	fuse.map = map;
	fuse.keys = keys;
	fuse.size = size;
	fuse.index = keyIndex;

	dim3 thread(1024);
	dim3 block(DivUp(size, thread.x));

	InsertKeyPointsKernel<<<block, thread>>>(fuse);

	SafeCall(cudaDeviceSynchronize());
	SafeCall(cudaGetLastError());
}
