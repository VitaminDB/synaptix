#include <cuda_fp16.h>
#include <cuda_bf16.h>

struct PixelNormParams {
    int       C;
    long long S;
    float     eps;
    int       apply_silu;
};

__device__ __forceinline__ void pn_load_vec(const float* p, float* o) {
    float4 r = *reinterpret_cast<const float4*>(p);
    o[0] = r.x; o[1] = r.y; o[2] = r.z; o[3] = r.w;
}
__device__ __forceinline__ void pn_store_vec(float* p, const float* o) {
    float4 r; r.x = o[0]; r.y = o[1]; r.z = o[2]; r.w = o[3];
    *reinterpret_cast<float4*>(p) = r;
}
__device__ __forceinline__ void pn_load_vec(const __nv_bfloat16* p, float* o) {
    uint4 r = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 f = __bfloat1622float2(h[j]);
        o[2 * j] = f.x;
        o[2 * j + 1] = f.y;
    }
}
__device__ __forceinline__ void pn_store_vec(__nv_bfloat16* p, const float* o) {
    __nv_bfloat162 h[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) h[j] = __floats2bfloat162_rn(o[2 * j], o[2 * j + 1]);
    *reinterpret_cast<uint4*>(p) = *reinterpret_cast<const uint4*>(h);
}
__device__ __forceinline__ void pn_load_vec(const __half* p, float* o) {
    uint4 r = *reinterpret_cast<const uint4*>(p);
    const __half2* h = reinterpret_cast<const __half2*>(&r);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 f = __half22float2(h[j]);
        o[2 * j] = f.x;
        o[2 * j + 1] = f.y;
    }
}
__device__ __forceinline__ void pn_store_vec(__half* p, const float* o) {
    __half2 h[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) h[j] = __floats2half2_rn(o[2 * j], o[2 * j + 1]);
    *reinterpret_cast<uint4*>(p) = *reinterpret_cast<const uint4*>(h);
}

template <typename T> struct PnVec;
template <> struct PnVec<float> { static constexpr int N = 4; };
template <> struct PnVec<__half> { static constexpr int N = 8; };
template <> struct PnVec<__nv_bfloat16> { static constexpr int N = 8; };

// y[c,s] = silu_opt(x[c,s] / sqrt(mean_c(x^2) + eps)), на каждую локацию s
// независимо. NCHW: канал strided по S, локации контигуозны -> warp читает
// соседние s (коалесцентно), цикл по C. blockIdx.y = batch. VEC локаций на
// thread (uint4); хвост S%VEC -- скалярная ветка.
template <typename T, int VEC>
__device__ __forceinline__ void pixel_norm_impl(
    const T* __restrict__ x_base,
    T*       __restrict__ y_base,
    PixelNormParams p)
{
    long long s0 = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (s0 >= p.S) return;
    long long base = (long long)blockIdx.y * p.C * p.S;
    const T* x = x_base + base;
    T*       y = y_base + base;
    float inv_c = 1.0f / (float)p.C;
    // uint4-вектор только при S%VEC==0: адрес строки канала = c*S + s0 —
    // при некратном S нечётные c дают невыровненный 16Б-доступ (ловилось на
    // VAE-тайлах вида [14,17,22], S=5236 -> CUDA_ERROR_MISALIGNED_ADDRESS).
    bool vec_ok = (p.S % VEC) == 0;
    if (vec_ok && s0 + VEC <= p.S) {
        float ss[VEC];
#pragma unroll
        for (int j = 0; j < VEC; ++j) ss[j] = 0.0f;
        for (int c = 0; c < p.C; ++c) {
            float v[VEC];
            pn_load_vec(x + (long long)c * p.S + s0, v);
#pragma unroll
            for (int j = 0; j < VEC; ++j) ss[j] += v[j] * v[j];
        }
        float den[VEC];
#pragma unroll
        for (int j = 0; j < VEC; ++j) den[j] = sqrtf(ss[j] * inv_c + p.eps);
        for (int c = 0; c < p.C; ++c) {
            float v[VEC];
            pn_load_vec(x + (long long)c * p.S + s0, v);
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                float o = v[j] / den[j];
                if (p.apply_silu) o = o / (1.0f + __expf(-o));
                v[j] = o;
            }
            pn_store_vec(y + (long long)c * p.S + s0, v);
        }
    } else {
        long long s_end = s0 + VEC < p.S ? s0 + VEC : p.S;
        for (long long s = s0; s < s_end; ++s) {
            float ss = 0.0f;
            for (int c = 0; c < p.C; ++c) {
                float v = (float)x[(long long)c * p.S + s];
                ss += v * v;
            }
            float den = sqrtf(ss * inv_c + p.eps);
            for (int c = 0; c < p.C; ++c) {
                float o = (float)x[(long long)c * p.S + s] / den;
                if (p.apply_silu) o = o / (1.0f + __expf(-o));
                y[(long long)c * p.S + s] = (T)o;
            }
        }
    }
}

extern "C" __global__ void pixel_norm_f32(const float* x, float* y, PixelNormParams p)
{ pixel_norm_impl<float, PnVec<float>::N>(x, y, p); }
extern "C" __global__ void pixel_norm_f16(const __half* x, __half* y, PixelNormParams p)
{ pixel_norm_impl<__half, PnVec<__half>::N>(x, y, p); }
extern "C" __global__ void pixel_norm_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, PixelNormParams p)
{ pixel_norm_impl<__nv_bfloat16, PnVec<__nv_bfloat16>::N>(x, y, p); }
