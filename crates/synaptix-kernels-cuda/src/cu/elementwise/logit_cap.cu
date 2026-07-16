#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Logit / attention soft-cap (Gemma2/Gemma3): out = cap * tanh(x / cap).
// Один thread = один элемент, f32-аккумулятор. Поддерживает in-place (x == out):
// каждый thread читает свой элемент до записи.

__device__ __forceinline__ float load_f(const float* p) { return *p; }
__device__ __forceinline__ float load_f(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_f(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void store_f(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_f(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_f(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void logit_cap_impl(
    const T* __restrict__ x, T* __restrict__ out, float cap, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = load_f(x + i);
  store_f(out + i, cap * tanhf(v / cap));
}

extern "C" __global__ void logit_cap_f32(const float* x, float* out, float cap, int n) {
  logit_cap_impl<float>(x, out, cap, n);
}
extern "C" __global__ void logit_cap_f16(const __half* x, __half* out, float cap, int n) {
  logit_cap_impl<__half>(x, out, cap, n);
}
extern "C" __global__ void logit_cap_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, float cap, int n) {
  logit_cap_impl<__nv_bfloat16>(x, out, cap, n);
}
