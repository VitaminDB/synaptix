#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Depthwise conv1d (groups == C, симметричный padding):
//   x [B, C, L], weight [C, 1, K] (= [C, K] row-major), bias [C] (optional),
//   out [B, C, L_out], L_out = (L + 2*padding - K) / stride + 1.
// Каждый канал свёртывается независимо:
//   out[b,c,i] = sum_ki w[c,ki] * x[b,c, i*stride - padding + ki]  (OOB → 0)  + bias[c].
// f32-аккумулятор. Один thread = один output element (i).
// Grid: (B*C, ceil(L_out/BLOCK), 1); block: (BLOCK, 1, 1).

__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ float ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void depthwise_conv1d_impl(
    const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ bias,
    int has_bias, T* __restrict__ out,
    int B, int C, int L, int K, int stride, int padding, int L_out) {
  int bc = blockIdx.x;
  int b = bc / C;
  int c = bc % C;
  int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= L_out || b >= B) return;

  const T* x_row = x + ((long)b * C + c) * L;
  const T* w_row = w + (long)c * K;
  float acc = 0.0f;
  int base = i * stride - padding;
  for (int ki = 0; ki < K; ki++) {
    int l_in = base + ki;
    if (l_in < 0 || l_in >= L) continue;
    acc += ld(x_row + l_in) * ld(w_row + ki);
  }
  if (has_bias) acc += ld(bias + c);
  st(out + ((long)b * C + c) * L_out + i, acc);
}

extern "C" __global__ void depthwise_conv1d_f32(
    const float* x, const float* w, const float* bias, int has_bias, float* out,
    int B, int C, int L, int K, int stride, int padding, int L_out) {
  depthwise_conv1d_impl<float>(x, w, bias, has_bias, out, B, C, L, K, stride, padding, L_out);
}
extern "C" __global__ void depthwise_conv1d_f16(
    const __half* x, const __half* w, const __half* bias, int has_bias, __half* out,
    int B, int C, int L, int K, int stride, int padding, int L_out) {
  depthwise_conv1d_impl<__half>(x, w, bias, has_bias, out, B, C, L, K, stride, padding, L_out);
}
extern "C" __global__ void depthwise_conv1d_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* bias, int has_bias,
    __nv_bfloat16* out, int B, int C, int L, int K, int stride, int padding, int L_out) {
  depthwise_conv1d_impl<__nv_bfloat16>(x, w, bias, has_bias, out, B, C, L, K, stride, padding, L_out);
}
