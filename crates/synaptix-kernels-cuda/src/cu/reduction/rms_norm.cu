#include <cuda_fp16.h>
#include <cuda_bf16.h>

struct RmsNormParams {
    int       batch;
    int       hidden;
    float     eps;
    int       variant;
    long long x_offset;
    long long w_offset;
    long long g_offset;
    long long y_offset;
    long long x_row_stride;
    long long g_row_stride;
    long long y_row_stride;
};

__device__ __forceinline__ float load_f32(const float* p) { return *p; }
__device__ __forceinline__ float load_f32(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_f32(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void store_t(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_t(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_t(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ float row_sumsq(
    const T* __restrict__ x_row,
    int hidden,
    int tid,
    int block_size,
    float* warp_sums
) {
    float local = 0.0f;
    for (int t = tid; t < hidden; t += block_size) {
        float v = load_f32(x_row + t);
        local += v * v;
    }
    unsigned int mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local += __shfl_down_sync(mask, local, off, 32);
    }
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) {
        warp_sums[warp_id] = local;
    }
    __syncthreads();
    int num_warps = block_size >> 5;
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(mask, v, off, 32);
        }
        if (lane == 0) {
            warp_sums[0] = v;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

template <typename T>
__device__ __forceinline__ void rms_norm_impl(
    const T* __restrict__ x_base,
    const T* __restrict__ w_base,
    T*       __restrict__ y_base,
    RmsNormParams p
) {
    int row = blockIdx.x;
    if (row >= p.batch) return;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    __shared__ float warp_sums[32];

    const T* x_row = x_base + p.x_offset + (long long)row * p.x_row_stride;
    T*       y_row = y_base + p.y_offset + (long long)row * p.y_row_stride;
    const T* w_vec = w_base + p.w_offset;

    float sumsq = row_sumsq(x_row, p.hidden, tid, block_size, warp_sums);
    __shared__ float s_rms;
    if (tid == 0) {
        float mean = sumsq / (float)p.hidden;
        s_rms = rsqrtf(mean + p.eps);
    }
    __syncthreads();
    float rms = s_rms;

    bool qwen = p.variant == 1;
    for (int t = tid; t < p.hidden; t += block_size) {
        float xv = load_f32(x_row + t);
        float wv = load_f32(w_vec + t);
        float scale = qwen ? (1.0f + wv) : wv;
        store_t(y_row + t, scale * xv * rms);
    }
}

template <typename T>
__device__ __forceinline__ void rms_norm_gated_impl(
    const T* __restrict__ x_base,
    const T* __restrict__ g_base,
    const T* __restrict__ w_base,
    T*       __restrict__ y_base,
    RmsNormParams p
) {
    int row = blockIdx.x;
    if (row >= p.batch) return;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    __shared__ float warp_sums[32];

    const T* x_row = x_base + p.x_offset + (long long)row * p.x_row_stride;
    const T* g_row = g_base + p.g_offset + (long long)row * p.g_row_stride;
    T*       y_row = y_base + p.y_offset + (long long)row * p.y_row_stride;
    const T* w_vec = w_base + p.w_offset;

    float sumsq = row_sumsq(x_row, p.hidden, tid, block_size, warp_sums);
    __shared__ float s_rms;
    if (tid == 0) {
        float mean = sumsq / (float)p.hidden;
        s_rms = rsqrtf(mean + p.eps);
    }
    __syncthreads();
    float rms = s_rms;

    for (int t = tid; t < p.hidden; t += block_size) {
        float xv = load_f32(x_row + t);
        float gv = load_f32(g_row + t);
        float wv = load_f32(w_vec + t);
        float sig = 1.0f / (1.0f + expf(-gv));
        float silu_g = gv * sig;
        store_t(y_row + t, wv * silu_g * xv * rms);
    }
}

extern "C" __global__ void rms_norm_f32(
    const float* x, const float* w, float* y, RmsNormParams p
) { rms_norm_impl<float>(x, w, y, p); }

extern "C" __global__ void rms_norm_f16(
    const __half* x, const __half* w, __half* y, RmsNormParams p
) { rms_norm_impl<__half>(x, w, y, p); }

extern "C" __global__ void rms_norm_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* y, RmsNormParams p
) { rms_norm_impl<__nv_bfloat16>(x, w, y, p); }

extern "C" __global__ void rms_norm_gated_f32(
    const float* x, const float* gate, const float* w, float* y, RmsNormParams p
) { rms_norm_gated_impl<float>(x, gate, w, y, p); }

extern "C" __global__ void rms_norm_gated_f16(
    const __half* x, const __half* gate, const __half* w, __half* y, RmsNormParams p
) { rms_norm_gated_impl<__half>(x, gate, w, y, p); }

extern "C" __global__ void rms_norm_gated_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gate, const __nv_bfloat16* w, __nv_bfloat16* y,
    RmsNormParams p
) { rms_norm_gated_impl<__nv_bfloat16>(x, gate, w, y, p); }
