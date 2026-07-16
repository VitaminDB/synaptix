#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Fused: residual = x + residual; y = LayerNorm(residual) * gamma + beta.
//
// Заменяет 2 launch (residual add + layernorm) одним — экономит memory pass по
// hidden buffer (аналог rmsnorm_residual.cu, но полный LayerNorm: mean + var).
//
// Layout: x, residual, y — (batch, hidden) row-major; gamma/beta — (hidden,).
// beta опционален: has_beta=0 ⟹ beta-указатель не читается (можно placeholder).
// Grid: (batch, 1, 1), block: (BLOCK, 1, 1). mean и var собираются в одном
// pass (sum + sumsq) в F32; var = max(sumsq/N - mean*mean, 0).

__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ float ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void layernorm_residual_impl(
    const T* __restrict__ x,
    T*       __restrict__ residual,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    int has_beta,
    T*       __restrict__ y,
    int batch,
    int hidden,
    float eps
) {
    int row = blockIdx.x;
    if (row >= batch) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    const T* x_row = x + (long long)row * hidden;
    T*       r_row = residual + (long long)row * hidden;
    T*       y_row = y + (long long)row * hidden;

    __shared__ float warp_sum[32];
    __shared__ float warp_sumsq[32];
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    // ---- Pass 1: residual += x; собираем sum и sumsq в F32. ----
    float lsum = 0.0f;
    float lsq = 0.0f;
    for (int t = tid; t < hidden; t += bs) {
        float v = ld(x_row + t) + ld(r_row + t);
        st(r_row + t, v);
        lsum += v;
        lsq += v * v;
    }
    unsigned int mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        lsum += __shfl_down_sync(mask, lsum, off, 32);
        lsq += __shfl_down_sync(mask, lsq, off, 32);
    }
    int warp = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) {
        warp_sum[warp] = lsum;
        warp_sumsq[warp] = lsq;
    }
    __syncthreads();
    int num_warps = (bs + 31) >> 5;
    if (warp == 0) {
        float s = (lane < num_warps) ? warp_sum[lane] : 0.0f;
        float sq = (lane < num_warps) ? warp_sumsq[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s += __shfl_down_sync(mask, s, off, 32);
            sq += __shfl_down_sync(mask, sq, off, 32);
        }
        if (lane == 0) {
            float mean = s / (float)hidden;
            float var = sq / (float)hidden - mean * mean;
            if (var < 0.0f) var = 0.0f;
            s_mean = mean;
            s_inv_std = rsqrtf(var + eps);
        }
    }
    __syncthreads();
    float mean = s_mean;
    float inv = s_inv_std;

    // ---- Pass 2: y = ((residual - mean) * inv_std) * gamma + beta ----
    for (int t = tid; t < hidden; t += bs) {
        float rv = ld(r_row + t);
        float g = ld(gamma + t);
        float b = has_beta ? ld(beta + t) : 0.0f;
        st(y_row + t, ((rv - mean) * inv) * g + b);
    }
}

extern "C" __global__ void layernorm_residual_f32(
    const float* x, float* residual, const float* gamma, const float* beta,
    int has_beta, float* y, int batch, int hidden, float eps
) { layernorm_residual_impl<float>(x, residual, gamma, beta, has_beta, y, batch, hidden, eps); }

extern "C" __global__ void layernorm_residual_f16(
    const __half* x, __half* residual, const __half* gamma, const __half* beta,
    int has_beta, __half* y, int batch, int hidden, float eps
) { layernorm_residual_impl<__half>(x, residual, gamma, beta, has_beta, y, batch, hidden, eps); }

extern "C" __global__ void layernorm_residual_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* residual, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, int has_beta, __nv_bfloat16* y,
    int batch, int hidden, float eps
) { layernorm_residual_impl<__nv_bfloat16>(x, residual, gamma, beta, has_beta, y, batch, hidden, eps); }
