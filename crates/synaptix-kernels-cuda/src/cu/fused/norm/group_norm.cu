#include <cuda_fp16.h>
#include <cuda_bf16.h>

struct GroupNormParams {
    int       C;
    int       HW;
    int       G;
    float     eps;
    int       has_affine;
    int       apply_silu;
    long long x_offset;
    long long w_offset;
    long long b_offset;
    long long y_offset;
};

__device__ __forceinline__ float load_f32(const float* p) { return *p; }
__device__ __forceinline__ float load_f32(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_f32(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void store_t(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_t(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_t(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T> struct VecCfg;
template <> struct VecCfg<float> { static constexpr int N = 4; };
template <> struct VecCfg<__half> { static constexpr int N = 8; };
template <> struct VecCfg<__nv_bfloat16> { static constexpr int N = 8; };

__device__ __forceinline__ void load_vec(const float* p, float* o) {
    float4 r = *reinterpret_cast<const float4*>(p);
    o[0] = r.x; o[1] = r.y; o[2] = r.z; o[3] = r.w;
}
__device__ __forceinline__ void store_vec(float* p, const float* o) {
    float4 r; r.x = o[0]; r.y = o[1]; r.z = o[2]; r.w = o[3];
    *reinterpret_cast<float4*>(p) = r;
}
__device__ __forceinline__ void load_vec(const __nv_bfloat16* p, float* o) {
    uint4 r = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 f = __bfloat1622float2(h[j]);
        o[2 * j] = f.x;
        o[2 * j + 1] = f.y;
    }
}
__device__ __forceinline__ void store_vec(__nv_bfloat16* p, const float* o) {
    __nv_bfloat162 h[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) h[j] = __floats2bfloat162_rn(o[2 * j], o[2 * j + 1]);
    *reinterpret_cast<uint4*>(p) = *reinterpret_cast<const uint4*>(h);
}
__device__ __forceinline__ void load_vec(const __half* p, float* o) {
    uint4 r = *reinterpret_cast<const uint4*>(p);
    const __half2* h = reinterpret_cast<const __half2*>(&r);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 f = __half22float2(h[j]);
        o[2 * j] = f.x;
        o[2 * j + 1] = f.y;
    }
}
__device__ __forceinline__ void store_vec(__half* p, const float* o) {
    __half2 h[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) h[j] = __floats2half2_rn(o[2 * j], o[2 * j + 1]);
    *reinterpret_cast<uint4*>(p) = *reinterpret_cast<const uint4*>(h);
}

template <typename T>
__device__ __forceinline__ void groupnorm_impl(
    const T* __restrict__ x_base,
    const T* __restrict__ w_base,
    const T* __restrict__ b_base,
    T*       __restrict__ y_base,
    GroupNormParams p)
{
    int bg = blockIdx.x;
    int g = bg % p.G;
    int per_group = p.C / p.G;
    long long n = (long long)per_group * p.HW;
    long long base = (long long)bg * n;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    constexpr int VEC = VecCfg<T>::N;
    bool vectorize = (p.HW % VEC == 0) && (p.x_offset % VEC == 0) && (p.y_offset % VEC == 0);

    const T* x = x_base + p.x_offset + base;
    T*       y = y_base + p.y_offset + base;

    float local_sum = 0.f;
    float local_sumsq = 0.f;
    if (vectorize) {
        long long n_vec = n / VEC;
        for (long long vi = tid; vi < n_vec; vi += block_size) {
            float vals[VEC];
            load_vec(x + vi * VEC, vals);
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                local_sum += vals[j];
                local_sumsq += vals[j] * vals[j];
            }
        }
    } else {
        for (long long i = tid; i < n; i += block_size) {
            float v = load_f32(x + i);
            local_sum += v;
            local_sumsq += v * v;
        }
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
            float nn = (float)n;
            float mean = vs / nn;
            float var = vq / nn - mean * mean;
            if (var < 0.f) var = 0.f;
            s_mean = mean;
            s_inv_std = rsqrtf(var + p.eps);
        }
    }
    __syncthreads();
    float mean = s_mean;
    float inv_std = s_inv_std;

    const T* w_vec = (p.has_affine) ? w_base + p.w_offset : nullptr;
    const T* b_vec = (p.has_affine) ? b_base + p.b_offset : nullptr;
    int c_base = g * per_group;

    if (vectorize) {
        long long n_vec = n / VEC;
        for (long long vi = tid; vi < n_vec; vi += block_size) {
            long long i0 = vi * VEC;
            float vals[VEC];
            load_vec(x + i0, vals);
            float gw = 1.f, gb = 0.f;
            if (w_vec != nullptr) {
                int c = c_base + (int)(i0 / p.HW);
                gw = load_f32(w_vec + c);
                gb = load_f32(b_vec + c);
            }
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                float norm = (vals[j] - mean) * inv_std;
                if (w_vec != nullptr) norm = norm * gw + gb;
                if (p.apply_silu) {
                    float s = 1.f / (1.f + __expf(-norm));
                    norm = norm * s;
                }
                vals[j] = norm;
            }
            store_vec(y + i0, vals);
        }
    } else {
        for (long long i = tid; i < n; i += block_size) {
            float xv = load_f32(x + i);
            float norm = (xv - mean) * inv_std;
            if (w_vec != nullptr) {
                int c = c_base + (int)(i / p.HW);
                norm = norm * load_f32(w_vec + c) + load_f32(b_vec + c);
            }
            if (p.apply_silu) {
                float s = 1.f / (1.f + __expf(-norm));
                norm = norm * s;
            }
            store_t(y + i, norm);
        }
    }
}

extern "C" __global__ void group_norm_f32(
    const float* x, const float* w, const float* b, float* y, GroupNormParams p
) { groupnorm_impl<float>(x, w, b, y, p); }

extern "C" __global__ void group_norm_f16(
    const __half* x, const __half* w, const __half* b, __half* y, GroupNormParams p
) { groupnorm_impl<__half>(x, w, b, y, p); }

extern "C" __global__ void group_norm_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* b, __nv_bfloat16* y,
    GroupNormParams p
) { groupnorm_impl<__nv_bfloat16>(x, w, b, y, p); }

// NHWC GroupNorm: x/y в layout [B,H,W,C]. Блок = (b,g). Группа g = каналы
// [g*pg, (g*pg+pg)); per spatial s они контигуальны (pg подряд), spatial шаг C.
// Редукция по pg*HW элементам, normalize + опц affine(gamma/beta по каналу) + silu.
template <typename T>
__device__ __forceinline__ void groupnorm_nhwc_impl(
    const T* __restrict__ x_base, const T* __restrict__ w_base,
    const T* __restrict__ b_base, T* __restrict__ y_base, GroupNormParams p)
{
    int bg = blockIdx.x;
    int b = bg / p.G;
    int g = bg % p.G;
    int per_group = p.C / p.G;
    long long n = (long long)per_group * p.HW;
    long long base_b = (long long)b * p.HW * p.C;
    int c0 = g * per_group;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const T* x = x_base + p.x_offset;
    T* y = y_base + p.y_offset;

    float local_sum = 0.f, local_sumsq = 0.f;
    for (long long e = tid; e < n; e += block_size) {
        int s = (int)(e / per_group);
        int cc = (int)(e % per_group);
        float v = load_f32(x + base_b + (long long)s * p.C + c0 + cc);
        local_sum += v;
        local_sumsq += v * v;
    }
    unsigned int mask = 0xFFFFFFFFu;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, off, 32);
        local_sumsq += __shfl_down_sync(mask, local_sumsq, off, 32);
    }
    __shared__ float warp_sum[32];
    __shared__ float warp_sumsq[32];
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) { warp_sum[warp_id] = local_sum; warp_sumsq[warp_id] = local_sumsq; }
    __syncthreads();
    __shared__ float s_mean, s_inv_std;
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
            float nn = (float)n;
            float mean = vs / nn;
            float var = vq / nn - mean * mean;
            if (var < 0.f) var = 0.f;
            s_mean = mean;
            s_inv_std = rsqrtf(var + p.eps);
        }
    }
    __syncthreads();
    float mean = s_mean, inv_std = s_inv_std;
    const T* w_vec = (p.has_affine) ? w_base + p.w_offset : nullptr;
    const T* b_vec = (p.has_affine) ? b_base + p.b_offset : nullptr;
    for (long long e = tid; e < n; e += block_size) {
        int s = (int)(e / per_group);
        int cc = (int)(e % per_group);
        long long gmem = base_b + (long long)s * p.C + c0 + cc;
        float norm = (load_f32(x + gmem) - mean) * inv_std;
        if (w_vec != nullptr) norm = norm * load_f32(w_vec + c0 + cc) + load_f32(b_vec + c0 + cc);
        if (p.apply_silu) { float sg = 1.f / (1.f + __expf(-norm)); norm = norm * sg; }
        store_t(y + gmem, norm);
    }
}

extern "C" __global__ void group_norm_nhwc_f32(
    const float* x, const float* w, const float* b, float* y, GroupNormParams p
) { groupnorm_nhwc_impl<float>(x, w, b, y, p); }
extern "C" __global__ void group_norm_nhwc_f16(
    const __half* x, const __half* w, const __half* b, __half* y, GroupNormParams p
) { groupnorm_nhwc_impl<__half>(x, w, b, y, p); }
extern "C" __global__ void group_norm_nhwc_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* b, __nv_bfloat16* y,
    GroupNormParams p
) { groupnorm_nhwc_impl<__nv_bfloat16>(x, w, b, y, p); }
