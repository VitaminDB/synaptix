#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>

// Numerically stable softmax по последнему dim.
// 2-pass (max-subtract + exp+sum + normalize). Один block обрабатывает одну
// row из `hidden` элементов; threads cooperatively reduce'ят через warp shuffle.
//
// Grid: (batch, 1, 1); block: (BLOCK, 1, 1) — BLOCK достаточно велик чтобы
// покрыть row через strided loop.

__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ float ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void softmax_impl(
    const T* __restrict__ x,
    T*       __restrict__ y,
    int hidden,
    int batch
) {
    int row = blockIdx.x;
    if (row >= batch) return;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const T* x_row = x + (long long)row * hidden;
    T*       y_row = y + (long long)row * hidden;

    __shared__ float warp_red[32];
    __shared__ float s_max;
    __shared__ float s_sum;

    // ---- Pass 1: max ----
    float local_max = (-CUDART_INF_F);
    for (int i = tid; i < hidden; i += block_size) {
        float v = ld(x_row + i);
        if (v > local_max) local_max = v;
    }
    unsigned int mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(mask, local_max, off, 32);
        if (other > local_max) local_max = other;
    }
    int warp = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) warp_red[warp] = local_max;
    __syncthreads();
    int num_warps = (block_size + 31) >> 5;
    if (warp == 0) {
        float v = (lane < num_warps) ? warp_red[lane] : (-CUDART_INF_F);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_down_sync(mask, v, off, 32);
            if (other > v) v = other;
        }
        if (lane == 0) s_max = v;
    }
    __syncthreads();
    float mx = s_max;

    // ---- Pass 2: sum(exp(x - max)) ----
    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += block_size) {
        local_sum += expf(ld(x_row + i) - mx);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, off, 32);
    }
    if (lane == 0) warp_red[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < num_warps) ? warp_red[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(mask, v, off, 32);
        }
        if (lane == 0) s_sum = v;
    }
    __syncthreads();
    float inv_sum = 1.0f / s_sum;

    // ---- Pass 3: normalize ----
    for (int i = tid; i < hidden; i += block_size) {
        float v = expf(ld(x_row + i) - mx) * inv_sum;
        st(y_row + i, v);
    }
}

extern "C" __global__ void softmax_f32(
    const float* x, float* y, int hidden, int batch
) { softmax_impl<float>(x, y, hidden, batch); }

extern "C" __global__ void softmax_f16(
    const __half* x, __half* y, int hidden, int batch
) { softmax_impl<__half>(x, y, hidden, batch); }

extern "C" __global__ void softmax_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* y, int hidden, int batch
) { softmax_impl<__nv_bfloat16>(x, y, hidden, batch); }
