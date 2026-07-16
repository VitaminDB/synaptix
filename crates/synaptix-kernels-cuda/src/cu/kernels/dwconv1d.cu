// Depthwise conv1d / conv_transpose1d (groups == C): один поток = один
// выходной элемент, f32-аккумулятор. Раньше depthwise шёл Rust-циклом по
// каналам (а convT — ещё и по тапам): C×K микро-launch'ей — вокодер LTX
// (BigVGAN Act1d, C до 1536, K=12) жёг 14s на 5с аудио.
//
// conv:  out[b,c,t] = bias[c] + Σ_k x[b,c,t·s+k−pad]·w[c,k] (вне диапазона = 0)
// convT: out[b,c,t] = bias[c] + Σ_k [(t−k)%s==0] x[b,c,(t−k)/s]·w[c,k],
//        длина out = (L−1)·s + K (кроп по padding делает вызывающий).
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float dw_to_f32(float x) { return x; }
__device__ __forceinline__ float dw_to_f32(__half x) { return __half2float(x); }
__device__ __forceinline__ float dw_to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ void dw_from_f32(float v, float* o) { *o = v; }
__device__ __forceinline__ void dw_from_f32(float v, __half* o) { *o = __float2half(v); }
__device__ __forceinline__ void dw_from_f32(float v, __nv_bfloat16* o) { *o = __float2bfloat16(v); }

#define DWCONV1D_KERNEL(name, T) \
extern "C" __global__ void name( \
    const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ bias, \
    T* __restrict__ out, \
    int c, int l, int k, int lo, int stride, int pad, long long total \
) { \
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (gid >= total) return; \
    int t = (int)(gid % lo); \
    long long bc = gid / lo; \
    int ch = (int)(bc % c); \
    float acc = bias ? dw_to_f32(bias[ch]) : 0.0f; \
    const T* xr = x + bc * (long long)l; \
    const T* wr = w + (long long)ch * k; \
    int base = t * stride - pad; \
    _Pragma("unroll 4") \
    for (int kk = 0; kk < k; ++kk) { \
        int xi = base + kk; \
        if (xi >= 0 && xi < l) acc += dw_to_f32(xr[xi]) * dw_to_f32(wr[kk]); \
    } \
    dw_from_f32(acc, out + gid); \
}

#define DWCONVT1D_KERNEL(name, T) \
extern "C" __global__ void name( \
    const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ bias, \
    T* __restrict__ out, \
    int c, int l, int k, int lo, int stride, long long total \
) { \
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (gid >= total) return; \
    int t = (int)(gid % lo); \
    long long bc = gid / lo; \
    int ch = (int)(bc % c); \
    float acc = bias ? dw_to_f32(bias[ch]) : 0.0f; \
    const T* xr = x + bc * (long long)l; \
    const T* wr = w + (long long)ch * k; \
    for (int kk = 0; kk < k; ++kk) { \
        int tt = t - kk; \
        if (tt < 0) break; \
        if (tt % stride == 0) { \
            int xi = tt / stride; \
            if (xi < l) acc += dw_to_f32(xr[xi]) * dw_to_f32(wr[kk]); \
        } \
    } \
    dw_from_f32(acc, out + gid); \
}

DWCONV1D_KERNEL(dwconv1d_f32,  float)
DWCONV1D_KERNEL(dwconv1d_f16,  __half)
DWCONV1D_KERNEL(dwconv1d_bf16, __nv_bfloat16)
DWCONVT1D_KERNEL(dwconvt1d_f32,  float)
DWCONVT1D_KERNEL(dwconvt1d_f16,  __half)
DWCONVT1D_KERNEL(dwconvt1d_bf16, __nv_bfloat16)
