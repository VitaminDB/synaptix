#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Token-embedding gather: out[t, :] = table[ids[t], :].
//   table [V, D] row-major, ids [N] (u32), out [N, D] row-major.
// Один thread = один выходной элемент (t, d). OOB id (>= V) → строка нулей.
// Чистое копирование через f32-round-trip (для f16/bf16 lossless), как в остальных ядрах.

__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ float ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void embed_gather_impl(
    const T* __restrict__ table, const unsigned int* __restrict__ ids,
    T* __restrict__ out, int n_ids, int dim, int vocab) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)n_ids * dim;
  if (idx >= total) return;
  int t = (int)(idx / dim);
  int d = (int)(idx % dim);
  unsigned int row = ids[t];
  if (row >= (unsigned int)vocab) {
    st(out + idx, 0.0f);
    return;
  }
  st(out + idx, ld(table + (long)row * dim + d));
}

extern "C" __global__ void embed_gather_f32(
    const float* table, const unsigned int* ids, float* out, int n_ids, int dim, int vocab) {
  embed_gather_impl<float>(table, ids, out, n_ids, dim, vocab);
}
extern "C" __global__ void embed_gather_f16(
    const __half* table, const unsigned int* ids, __half* out, int n_ids, int dim, int vocab) {
  embed_gather_impl<__half>(table, ids, out, n_ids, dim, vocab);
}
extern "C" __global__ void embed_gather_bf16(
    const __nv_bfloat16* table, const unsigned int* ids, __nv_bfloat16* out, int n_ids, int dim, int vocab) {
  embed_gather_impl<__nv_bfloat16>(table, ids, out, n_ids, dim, vocab);
}

extern "C" __global__ void embed_gather_mxfp8_f16(
    const __nv_fp8_e4m3* __restrict__ table, const unsigned char* __restrict__ scales,
    const unsigned int* __restrict__ ids, __half* __restrict__ out,
    int n_ids, int dim, int vocab) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)n_ids * dim;
  if (idx >= total) return;
  int t = (int)(idx / dim);
  int d = (int)(idx % dim);
  unsigned int row = ids[t];
  if (row >= (unsigned int)vocab) {
    out[idx] = __float2half(0.0f);
    return;
  }
  float sv = __uint_as_float(((unsigned)scales[(long)row * (dim / 32) + (d / 32)]) << 23);
  out[idx] = __float2half(float(table[(long)row * dim + d]) * sv);
}
