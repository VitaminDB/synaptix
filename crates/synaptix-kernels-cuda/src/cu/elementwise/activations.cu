#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Standalone 1D contiguous activations с fp32 accumulator. Дополняют
// strided UnaryOp opcodes в elementwise.cu более удобными inline-функциями
// + bias-add варианты для частого FC-bias + activation паттерна.

constexpr int BLOCK = 256;

__device__ __forceinline__ float load_f32(const float* p) { return *p; }
__device__ __forceinline__ float load_f32(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_f32(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void store_t(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_t(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_t(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

__device__ __forceinline__ float silu_f(float v) { return v / (1.f + __expf(-v)); }
__device__ __forceinline__ float gelu_exact_f(float v) {
    return 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
}
__device__ __forceinline__ float gelu_tanh_f(float v) {
    float v3 = v * v * v;
    return 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v3)));
}
__device__ __forceinline__ float quick_gelu_f(float v) {
    return v / (1.f + __expf(-1.702f * v));
}
__device__ __forceinline__ float swish_beta_f(float v, float beta) {
    return v / (1.f + __expf(-beta * v));
}
__device__ __forceinline__ float softplus_f(float v) {
    return v > 20.f ? v : log1pf(__expf(v));
}
__device__ __forceinline__ float mish_f(float v) {
    return v * tanhf(softplus_f(v));
}
__device__ __forceinline__ float softsign_f(float v) {
    return v / (1.f + fabsf(v));
}
__device__ __forceinline__ float relu_f(float v) { return v > 0.f ? v : 0.f; }

template <typename T, typename F>
__device__ __forceinline__ void apply_1d(
    const T* __restrict__ x, T* __restrict__ y, int n, F f)
{
    int tid = blockIdx.x * BLOCK + threadIdx.x;
    int stride = gridDim.x * BLOCK;
    for (int i = tid; i < n; i += stride) {
        store_t(y + i, f(load_f32(x + i)));
    }
}

#define DEF_ACT_1D(NAME, FN)                                                                       \
extern "C" __global__ void NAME##_f32(const float* x, float* y, int n)                             \
{ apply_1d(x, y, n, [](float v){ return FN(v); }); }                                               \
extern "C" __global__ void NAME##_f16(const __half* x, __half* y, int n)                           \
{ apply_1d(x, y, n, [](float v){ return FN(v); }); }                                               \
extern "C" __global__ void NAME##_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, int n)            \
{ apply_1d(x, y, n, [](float v){ return FN(v); }); }

DEF_ACT_1D(silu_act,       silu_f)
DEF_ACT_1D(gelu_exact_act, gelu_exact_f)
DEF_ACT_1D(gelu_tanh_act,  gelu_tanh_f)
DEF_ACT_1D(quick_gelu_act, quick_gelu_f)
DEF_ACT_1D(softplus_act,   softplus_f)
DEF_ACT_1D(mish_act,       mish_f)
DEF_ACT_1D(softsign_act,   softsign_f)

extern "C" __global__ void swish_beta_act_f32(const float* x, float* y, int n, float beta)
{ apply_1d(x, y, n, [=](float v){ return swish_beta_f(v, beta); }); }
extern "C" __global__ void swish_beta_act_f16(const __half* x, __half* y, int n, float beta)
{ apply_1d(x, y, n, [=](float v){ return swish_beta_f(v, beta); }); }
extern "C" __global__ void swish_beta_act_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, int n, float beta)
{ apply_1d(x, y, n, [=](float v){ return swish_beta_f(v, beta); }); }

// Bias-add + activation: y[r, c] = act(x[r, c] + bias[c]). rows × cols flat.

template <typename T, typename F>
__device__ __forceinline__ void apply_bias_act_2d(
    const T* __restrict__ x, const T* __restrict__ bias, T* __restrict__ y,
    int rows, int cols, F f)
{
    int total = rows * cols;
    int tid = blockIdx.x * BLOCK + threadIdx.x;
    int stride = gridDim.x * BLOCK;
    for (int i = tid; i < total; i += stride) {
        int col = i % cols;
        float v = load_f32(x + i) + load_f32(bias + col);
        store_t(y + i, f(v));
    }
}

#define DEF_BIAS_ACT(NAME, FN)                                                                     \
extern "C" __global__ void NAME##_f32(const float* x, const float* bias, float* y, int R, int C)  \
{ apply_bias_act_2d(x, bias, y, R, C, [](float v){ return FN(v); }); }                             \
extern "C" __global__ void NAME##_f16(const __half* x, const __half* bias, __half* y, int R, int C) \
{ apply_bias_act_2d(x, bias, y, R, C, [](float v){ return FN(v); }); }                             \
extern "C" __global__ void NAME##_bf16(const __nv_bfloat16* x, const __nv_bfloat16* bias, __nv_bfloat16* y, int R, int C) \
{ apply_bias_act_2d(x, bias, y, R, C, [](float v){ return FN(v); }); }

DEF_BIAS_ACT(bias_silu,      silu_f)
DEF_BIAS_ACT(bias_gelu_tanh, gelu_tanh_f)
DEF_BIAS_ACT(bias_relu,      relu_f)

// Snake activation (Oobleck VAE): y = x + sin(a[c]*x)^2 * binv[c], где
// a = exp(alpha), binv = 1/(exp(beta)+eps) — per-channel [C], предвычислены
// caller'ом. x/y — [.., C, Tlen] contiguous (channel = (i / Tlen) % C).
// Заменяет decomposed exp/mul/sin/sqr/recip/add (≈5 крупных проходов → 1).

template <typename T>
__device__ __forceinline__ void snake_1d(
    const T* __restrict__ x, const float* __restrict__ a,
    const float* __restrict__ binv, T* __restrict__ y,
    int n, int C, int Tlen)
{
    int tid = blockIdx.x * BLOCK + threadIdx.x;
    int stride = gridDim.x * BLOCK;
    for (int i = tid; i < n; i += stride) {
        int c = (i / Tlen) % C;
        float v = load_f32(x + i);
        float s = sinf(a[c] * v);
        store_t(y + i, v + s * s * binv[c]);
    }
}

extern "C" __global__ void snake_act_f32(const float* x, const float* a, const float* binv, float* y, int n, int C, int Tlen)
{ snake_1d(x, a, binv, y, n, C, Tlen); }
extern "C" __global__ void snake_act_f16(const __half* x, const float* a, const float* binv, __half* y, int n, int C, int Tlen)
{ snake_1d(x, a, binv, y, n, C, Tlen); }
extern "C" __global__ void snake_act_bf16(const __nv_bfloat16* x, const float* a, const float* binv, __nv_bfloat16* y, int n, int C, int Tlen)
{ snake_1d(x, a, binv, y, n, C, Tlen); }
