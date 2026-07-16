#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Fast 4D permute: NCHW [B,C,H,W] ↔ NHWC [B,H,W,C] через shmem-tile (32×32).
// Заменяет generic permute()+contiguous() (медленный strided-copy) для входов
// implicit-GEMM. +1 padding в shmem против bank conflicts. Coalesced reads
// в NCHW (по HW), coalesced writes в NHWC (по C — для прямого пути), и vice
// versa для обратного.

#define TILE 32

template <typename T>
__device__ __forceinline__ void nchw_to_nhwc_impl(
    const T* __restrict__ src, T* __restrict__ dst,
    int C, int H, int W
) {
    __shared__ T tile[TILE][TILE + 1];
    long long HW = (long long)H * W;
    int c_base = blockIdx.y * TILE;
    int hw_base = blockIdx.x * TILE;
    int b = blockIdx.z;

    int c = c_base + threadIdx.y;
    int hw = hw_base + threadIdx.x;
    if (c < C && hw < HW) {
        tile[threadIdx.y][threadIdx.x] = src[((long long)b * C + c) * HW + hw];
    }
    __syncthreads();

    // Транспонированная запись: thread (tx, ty) пишет dst[b, hw_base+ty, c_base+tx].
    int hw_out = hw_base + threadIdx.y;
    int c_out = c_base + threadIdx.x;
    if (c_out < C && hw_out < HW) {
        dst[((long long)b * HW + hw_out) * C + c_out] = tile[threadIdx.x][threadIdx.y];
    }
}

template <typename T>
__device__ __forceinline__ void nhwc_to_nchw_impl(
    const T* __restrict__ src, T* __restrict__ dst,
    int C, int H, int W
) {
    __shared__ T tile[TILE][TILE + 1];
    long long HW = (long long)H * W;
    int c_base = blockIdx.y * TILE;
    int hw_base = blockIdx.x * TILE;
    int b = blockIdx.z;

    // Read NHWC: dst[b, hw, c]. Thread (tx=c, ty=hw) reads coalesced по c.
    int hw_in = hw_base + threadIdx.y;
    int c_in = c_base + threadIdx.x;
    if (c_in < C && hw_in < HW) {
        tile[threadIdx.y][threadIdx.x] = src[((long long)b * HW + hw_in) * C + c_in];
    }
    __syncthreads();

    // Write NCHW: dst[b, c, hw]. Thread (tx=hw, ty=c) writes coalesced по hw.
    int c_out = c_base + threadIdx.y;
    int hw_out = hw_base + threadIdx.x;
    if (c_out < C && hw_out < HW) {
        dst[((long long)b * C + c_out) * HW + hw_out] = tile[threadIdx.x][threadIdx.y];
    }
}

extern "C" __global__ void nchw_to_nhwc_f32(const float* s, float* d, int C, int H, int W)
{ nchw_to_nhwc_impl<float>(s, d, C, H, W); }
extern "C" __global__ void nchw_to_nhwc_f16(const __half* s, __half* d, int C, int H, int W)
{ nchw_to_nhwc_impl<__half>(s, d, C, H, W); }
extern "C" __global__ void nchw_to_nhwc_bf16(const __nv_bfloat16* s, __nv_bfloat16* d, int C, int H, int W)
{ nchw_to_nhwc_impl<__nv_bfloat16>(s, d, C, H, W); }

extern "C" __global__ void nhwc_to_nchw_f32(const float* s, float* d, int C, int H, int W)
{ nhwc_to_nchw_impl<float>(s, d, C, H, W); }
extern "C" __global__ void nhwc_to_nchw_f16(const __half* s, __half* d, int C, int H, int W)
{ nhwc_to_nchw_impl<__half>(s, d, C, H, W); }
extern "C" __global__ void nhwc_to_nchw_bf16(const __nv_bfloat16* s, __nv_bfloat16* d, int C, int H, int W)
{ nhwc_to_nchw_impl<__nv_bfloat16>(s, d, C, H, W); }
