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
