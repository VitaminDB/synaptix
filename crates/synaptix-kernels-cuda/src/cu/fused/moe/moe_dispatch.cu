#include <cuda_fp16.h>
#include <cuda_bf16.h>

// MoE token routing (row-permutation copy), целочисленные индексы (u32):
//   scatter: out[i, :]      = x[idx[i], :]   — раскладка токенов под экспертов
//   gather:  out[idx[i], :] = x[i, :]        — возврат выходов на исходные позиции
// x [N, D], idx [N] (u32), out [N, D]. Один thread = один элемент.
// OOB idx: scatter → строка нулей; gather → запись пропускается (out нужно
// предварительно занулить, если indices не полная перестановка).
// Копирование через f32 round-trip (для f16/bf16 lossless).

__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ float ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void moe_scatter_impl(
    const T* __restrict__ x, const unsigned int* __restrict__ idx,
    T* __restrict__ out, int n, int d) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)n * d;
  if (i >= total) return;
  int row = (int)(i / d);
  int col = (int)(i % d);
  unsigned int src = idx[row];
  if (src >= (unsigned int)n) {
    st(out + i, 0.0f);
    return;
  }
  st(out + i, ld(x + (long)src * d + col));
}

template <typename T>
__device__ __forceinline__ void moe_gather_impl(
    const T* __restrict__ x, const unsigned int* __restrict__ idx,
    T* __restrict__ out, int n, int d) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)n * d;
  if (i >= total) return;
  int row = (int)(i / d);
  int col = (int)(i % d);
  unsigned int dst = idx[row];
  if (dst >= (unsigned int)n) return;
  st(out + (long)dst * d + col, ld(x + i));
}

extern "C" __global__ void moe_scatter_f32(const float* x, const unsigned int* idx, float* out, int n, int d) {
  moe_scatter_impl<float>(x, idx, out, n, d);
}
extern "C" __global__ void moe_scatter_f16(const __half* x, const unsigned int* idx, __half* out, int n, int d) {
  moe_scatter_impl<__half>(x, idx, out, n, d);
}
extern "C" __global__ void moe_scatter_bf16(const __nv_bfloat16* x, const unsigned int* idx, __nv_bfloat16* out, int n, int d) {
  moe_scatter_impl<__nv_bfloat16>(x, idx, out, n, d);
}
extern "C" __global__ void moe_gather_f32(const float* x, const unsigned int* idx, float* out, int n, int d) {
  moe_gather_impl<float>(x, idx, out, n, d);
}
extern "C" __global__ void moe_gather_f16(const __half* x, const unsigned int* idx, __half* out, int n, int d) {
  moe_gather_impl<__half>(x, idx, out, n, d);
}
extern "C" __global__ void moe_gather_bf16(const __nv_bfloat16* x, const unsigned int* idx, __nv_bfloat16* out, int n, int d) {
  moe_gather_impl<__nv_bfloat16>(x, idx, out, n, d);
}

// ─── combine: out[t, :] = Σ_s w[t·k+s] · y[inv[t·k+s], :] ───
// Сборка выхода MoE одним ядром вместо «взвесить строки → собрать по
// обратной перестановке → просуммировать по k». Блок — токен, нить — 8
// соседних элементов (16 байт); d кратно 8.
template <typename T2, typename T>
__device__ __forceinline__ void moe_combine_impl(
    const T* __restrict__ y, const unsigned int* __restrict__ inv, const float* __restrict__ w,
    T* __restrict__ out, int t, int k, int d);

__device__ __forceinline__ float2 mc_to_f2(__nv_bfloat162 v) { return __bfloat1622float2(v); }
__device__ __forceinline__ float2 mc_to_f2(__half2 v) { return __half22float2(v); }
__device__ __forceinline__ __nv_bfloat162 mc_from_f2(float2 v, __nv_bfloat162) { return __float22bfloat162_rn(v); }
__device__ __forceinline__ __half2 mc_from_f2(float2 v, __half2) { return __float22half2_rn(v); }

template <typename T2, typename T>
__device__ __forceinline__ void moe_combine_impl(
    const T* __restrict__ y, const unsigned int* __restrict__ inv, const float* __restrict__ w,
    T* __restrict__ out, int t, int k, int d) {
  int row = blockIdx.x;
  if (row >= t) return;
  int d8 = d >> 3;
  for (int c = threadIdx.x; c < d8; c += blockDim.x) {
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    for (int s = 0; s < k; ++s) {
      unsigned int src = inv[row * k + s];
      float ws = w[row * k + s];
      uint4 v = *reinterpret_cast<const uint4*>(y + (size_t)src * d + (size_t)c * 8);
      const T2* p = reinterpret_cast<const T2*>(&v);
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 f = mc_to_f2(p[i]);
        acc[2 * i] += ws * f.x;
        acc[2 * i + 1] += ws * f.y;
      }
    }
    uint4 o;
    T2* q = reinterpret_cast<T2*>(&o);
    #pragma unroll
    for (int i = 0; i < 4; ++i) q[i] = mc_from_f2(make_float2(acc[2 * i], acc[2 * i + 1]), T2{});
    *reinterpret_cast<uint4*>(out + (size_t)row * d + (size_t)c * 8) = o;
  }
}

extern "C" __global__ void moe_combine_bf16(
    const __nv_bfloat16* y, const unsigned int* inv, const float* w, __nv_bfloat16* out, int t, int k, int d) {
  moe_combine_impl<__nv_bfloat162, __nv_bfloat16>(y, inv, w, out, t, k, d);
}
extern "C" __global__ void moe_combine_f16(
    const __half* y, const unsigned int* inv, const float* w, __half* out, int t, int k, int d) {
  moe_combine_impl<__half2, __half>(y, inv, w, out, t, k, d);
}
