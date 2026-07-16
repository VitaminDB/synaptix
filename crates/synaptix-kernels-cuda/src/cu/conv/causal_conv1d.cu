#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Causal depthwise conv1d (Mamba-style):
//   x [B, C, L], weight [C, 1, K] (= [C, K] row-major), bias [C] (optional),
//   out [B, C, out_len], out_len = ceil(L / stride).
// Каждый канал свёртывается независимо (depthwise). Causal: левый pad K-1, т.е.
//   out[b,c,i] = sum_ki w[c,ki] * x[b,c, i*stride - (K-1) + ki]  (OOB → 0)  + bias[c].
// f32-аккумулятор. Один thread = один output element (i).
// Grid: (B*C, ceil(out_len/BLOCK), 1); block: (BLOCK, 1, 1).

__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ float ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void causal_conv1d_impl(
    const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ bias,
    int has_bias, T* __restrict__ out,
    int B, int C, int L, int K, int stride, int out_len) {
  int bc = blockIdx.x;
  int b = bc / C;
  int c = bc % C;
  int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= out_len || b >= B) return;

  const T* x_row = x + ((long)b * C + c) * L;
  const T* w_row = w + (long)c * K;
  float acc = 0.0f;
  int base = i * stride - (K - 1);
  for (int ki = 0; ki < K; ki++) {
    int l_in = base + ki;
    if (l_in < 0 || l_in >= L) continue;
    acc += ld(x_row + l_in) * ld(w_row + ki);
  }
  if (has_bias) acc += ld(bias + c);
  st(out + ((long)b * C + c) * out_len + i, acc);
}

extern "C" __global__ void causal_conv1d_f32(
    const float* x, const float* w, const float* bias, int has_bias, float* out,
    int B, int C, int L, int K, int stride, int out_len) {
  causal_conv1d_impl<float>(x, w, bias, has_bias, out, B, C, L, K, stride, out_len);
}
extern "C" __global__ void causal_conv1d_f16(
    const __half* x, const __half* w, const __half* bias, int has_bias, __half* out,
    int B, int C, int L, int K, int stride, int out_len) {
  causal_conv1d_impl<__half>(x, w, bias, has_bias, out, B, C, L, K, stride, out_len);
}
extern "C" __global__ void causal_conv1d_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* bias, int has_bias,
    __nv_bfloat16* out, int B, int C, int L, int K, int stride, int out_len) {
  causal_conv1d_impl<__nv_bfloat16>(x, w, bias, has_bias, out, B, C, L, K, stride, out_len);
}

// Stateful single-step update (decode T=1) для GatedDeltaNet conv1d:
//   x [conv_dim] — новый сэмпл; state [K-1, conv_dim] (row-major, FIFO oldest-first)
//   обновляется in-place; w [conv_dim, K] (= [C, K] row-major); out [conv_dim].
//   Окно канала c: [state[0,c], .., state[K-2,c], x[c]] (последние K сэмплов);
//   out[c] = act(sum_j w[c,j]*window[j]); затем сдвиг: state[j,c]=state[j+1,c]
//   (j<K-2), state[K-2,c]=x[c]. apply_silu → out *= sigmoid. f32-аккумулятор.
//   Один thread = один канал c. Семантика == s=1 случай causal_conv1d_stateful.
template <typename T>
__device__ __forceinline__ void causal_conv1d_update_impl(
    const T* __restrict__ x, T* __restrict__ state, const T* __restrict__ w,
    T* __restrict__ out, int conv_dim, int K, int apply_silu) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= conv_dim) return;
  int km1 = K - 1;
  const T* wc = w + (long)c * K;
  float xc = ld(x + c);
  float acc = 0.0f;
  for (int j = 0; j < km1; j++) {
    acc += ld(state + (long)j * conv_dim + c) * ld(wc + j);
  }
  acc += xc * ld(wc + km1);
  if (apply_silu) acc = acc / (1.0f + expf(-acc));
  st(out + c, acc);
  for (int j = 0; j + 1 < km1; j++) {
    st(state + (long)j * conv_dim + c, ld(state + (long)(j + 1) * conv_dim + c));
  }
  if (km1 > 0) st(state + (long)(km1 - 1) * conv_dim + c, xc);
}

extern "C" __global__ void causal_conv1d_update_f32(
    const float* x, float* state, const float* w, float* out,
    int conv_dim, int K, int apply_silu) {
  causal_conv1d_update_impl<float>(x, state, w, out, conv_dim, K, apply_silu);
}
extern "C" __global__ void causal_conv1d_update_f16(
    const __half* x, __half* state, const __half* w, __half* out,
    int conv_dim, int K, int apply_silu) {
  causal_conv1d_update_impl<__half>(x, state, w, out, conv_dim, K, apply_silu);
}
