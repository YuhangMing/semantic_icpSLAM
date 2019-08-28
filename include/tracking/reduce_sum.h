#ifndef FUSION_ICP_REDUCE_SUM_H
#define FUSION_ICP_REDUCE_SUM_H

#include "macros.h"

#define MAX_WARP_SIZE 32

namespace fusion
{

template <typename T, int size>
FUSION_DEVICE inline void WarpReduce(T *val)
{
#pragma unroll
    for (int offset = MAX_WARP_SIZE / 2; offset > 0; offset /= 2)
    {
#pragma unroll
        for (int i = 0; i < size; ++i)
        {
            val[i] += __shfl_down_sync(0xffffffff, val[i], offset);
        }
    }
}

template <typename T, int size>
FUSION_DEVICE inline void BlockReduce(T *val)
{
    static __shared__ T shared[32 * size];
    int lane = threadIdx.x % MAX_WARP_SIZE;
    int wid = threadIdx.x / MAX_WARP_SIZE;

    WarpReduce<T, size>(val);

    if (lane == 0)
        memcpy(&shared[wid * size], val, sizeof(T) * size);

    __syncthreads();

    if (threadIdx.x < blockDim.x / MAX_WARP_SIZE)
        memcpy(val, &shared[lane * size], sizeof(T) * size);
    else
        memset(val, 0, sizeof(T) * size);

    if (wid == 0)
        WarpReduce<T, size>(val);
}

template <int rows, int cols>
void inline create_jtjjtr(cv::Mat &host_data, float *host_a, float *host_b)
{
    int shift = 0;
    for (int i = 0; i < rows; ++i)
        for (int j = i; j < cols; ++j)
        {
            float value = host_data.ptr<float>(0)[shift++];
            if (j == rows)
                host_b[i] = value;
            else
                host_a[j * rows + i] = host_a[i * rows + j] = value;
        }
}

} // namespace fusion

#endif