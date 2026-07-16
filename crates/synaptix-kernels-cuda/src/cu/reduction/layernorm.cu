#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Полный LayerNorm: y = ((x - mean) / sqrt(var + eps)) * gamma + beta.
// One block per row, BLOCK=256 threads. Сумма и сумма квадратов считаются
// в одном pass; var = max(sumsq/N - mean*mean, 0). Для типичных DiT/encoder
// inputs (zero-centered weights) этого достаточно. Если caller хочет
// LayerNorm без bias — передаёт has_beta=0 и beta_base может быть nullptr.

struct LayerNormParams {
    int       batch;
    int       hidden;
    float     eps;
    int       has_beta;
    long long x_offset;
    long long w_offset;
    long long b_offset;
    long long y_offset;
    long long x_row_stride;
    long long y_row_stride;
};

__device__ __forceinline__ float load_f32(const float* p) { return *p; }
__device__ __forceinline__ float load_f32(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_f32(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void store_t(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_t(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_t(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void layernorm_impl(
    const T* __restrict__ x_base,
    const T* __restrict__ w_base,
    const T* __restrict__ b_base,
    T*       __restrict__ y_base,
    LayerNormParams p)
{
    int row = blockIdx.x;
    if (row >= p.batch) return;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const T* x_row = x_base + p.x_offset + (long long)row * p.x_row_stride;
    T*       y_row = y_base + p.y_offset + (long long)row * p.y_row_stride;
    const T* w_vec = w_base + p.w_offset;
    const T* b_vec = (p.has_beta && b_base != nullptr) ? b_base + p.b_offset : nullptr;

    float local_sum = 0.f;
    float local_sumsq = 0.f;
    for (int t = tid; t < p.hidden; t += block_size) {
        float v = load_f32(x_row + t);
        local_sum   += v;
        local_sumsq += v * v;
    }

    unsigned int mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sum   += __shfl_down_sync(mask, local_sum,   off, 32);
        local_sumsq += __shfl_down_sync(mask, local_sumsq, off, 32);
    }

    __shared__ float warp_sum[32];
    __shared__ float warp_sumsq[32];
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) {
        warp_sum[warp_id] = local_sum;
        warp_sumsq[warp_id] = local_sumsq;
    }
    __syncthreads();

    __shared__ float s_mean;
    __shared__ float s_inv_std;
    if (warp_id == 0) {
        int num_warps = block_size >> 5;
        float vs = (lane < num_warps) ? warp_sum[lane] : 0.f;
        float vq = (lane < num_warps) ? warp_sumsq[lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            vs += __shfl_down_sync(mask, vs, off, 32);
            vq += __shfl_down_sync(mask, vq, off, 32);
        }
        if (lane == 0) {
            float n = (float)p.hidden;
            float mean = vs / n;
            float var = vq / n - mean * mean;
            if (var < 0.f) var = 0.f;
            s_mean = mean;
            s_inv_std = rsqrtf(var + p.eps);
        }
    }
    __syncthreads();
    float mean = s_mean;
    float inv_std = s_inv_std;

    for (int t = tid; t < p.hidden; t += block_size) {
        float xv = load_f32(x_row + t);
        float wv = load_f32(w_vec + t);
        float norm = (xv - mean) * inv_std * wv;
        if (b_vec != nullptr) {
            norm += load_f32(b_vec + t);
        }
        store_t(y_row + t, norm);
    }
}

extern "C" __global__ void layernorm_f32(
    const float* x, const float* w, const float* b, float* y, LayerNormParams p
) { layernorm_impl<float>(x, w, b, y, p); }

extern "C" __global__ void layernorm_f16(
    const __half* x, const __half* w, const __half* b, __half* y, LayerNormParams p
) { layernorm_impl<__half>(x, w, b, y, p); }

extern "C" __global__ void layernorm_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* b, __nv_bfloat16* y,
    LayerNormParams p
) { layernorm_impl<__nv_bfloat16>(x, w, b, y, p); }
