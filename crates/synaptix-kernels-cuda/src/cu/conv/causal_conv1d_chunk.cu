#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Stateful chunked causal conv1d (prefill, T>=1) для GatedDeltaNet/Mamba.
// Layout идентичен host-эталону synaptix_ops::conv::causal_conv1d_stateful:
//   x      [T, C] row-major (time-major: x[t*C + c])
//   state  [(K-1), C] row-major (FIFO oldest-first: state[j*C + c])
//   w      [C, K] row-major (w[c*K + j])
//   out    [T, C] row-major
//
// Семантика:
//   ext = concat(state, x) ∈ R^{(K-1+T), C}
//   out[t,c] = sum_{j=0..K} w[c,j] * ext[t+j, c]   (causal: ext[t+j] видим в момент t)
//   apply_silu → out[t,c] *= sigmoid(out[t,c])
//   state ← ext[T..T+K-1, :]  (т.е. последние K-1 строк ext)
//
// Разделено на 2 launch'а, чтобы избежать global-sync для записи state:
//   1) compute: grid (ceil(C/BLOCK), T, 1), один thread = (c, t)
//   2) state_update: grid (ceil(C/BLOCK), 1, 1), один thread = c
//
// f32-аккумулятор. K шаблонно не фиксирован (хост передаёт K runtime,
// типично K=3 или 4). Полный compatibility с causal_conv1d_update_impl (T=1).

__device__ __forceinline__ float cf_ld(const float* p) { return *p; }
__device__ __forceinline__ float cf_ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float cf_ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void cf_st(float* p, float v) { *p = v; }
__device__ __forceinline__ void cf_st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void cf_st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void chunk_compute_impl(
    const T* __restrict__ x,        // [T_in, C]
    const T* __restrict__ state,    // [K-1, C]
    const T* __restrict__ w,        // [C, K]
    T* __restrict__ out,            // [T_in, C]
    int T_in, int C, int K, int apply_silu) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int t = blockIdx.y;
  if (c >= C || t >= T_in) return;
  int km1 = K - 1;
  const T* wc = w + (long)c * K;
  float acc = 0.0f;
  // ext[t+j, c]: t+j ∈ [0, K-1+T_in). t+j < K-1 → state[t+j, c]; иначе x[(t+j)-(K-1), c].
  for (int j = 0; j < K; ++j) {
    int e = t + j;
    float val;
    if (e < km1) {
      val = cf_ld(state + (long)e * C + c);
    } else {
      val = cf_ld(x + (long)(e - km1) * C + c);
    }
    acc += cf_ld(wc + j) * val;
  }
  if (apply_silu) acc = acc / (1.0f + expf(-acc));
  cf_st(out + (long)t * C + c, acc);
}

template <typename T>
__device__ __forceinline__ void chunk_state_update_impl(
    const T* __restrict__ x,        // [T_in, C]
    T* __restrict__ state,          // [K-1, C] — in-place, читается затем пишется
    int T_in, int C, int K) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  int km1 = K - 1;
  // ext[T_in + j, c] для j ∈ [0, km1): T_in+j < km1 → старое state[T_in+j, c];
  // T_in+j ≥ km1 → x[(T_in+j) - km1, c]. Каждый thread = свой c (нет race между
  // threads); read-then-write через reg-buf избегает self-race по каналу.
  float buf[8];  // K ≤ 8 в обозримых архитектурах
  for (int j = 0; j < km1; ++j) {
    int e = T_in + j;
    if (e < km1) {
      buf[j] = cf_ld(state + (long)e * C + c);
    } else {
      buf[j] = cf_ld(x + (long)(e - km1) * C + c);
    }
  }
  for (int j = 0; j < km1; ++j) {
    cf_st(state + (long)j * C + c, buf[j]);
  }
}

extern "C" __global__ void causal_conv1d_chunk_compute_f32(
    const float* x, const float* state, const float* w, float* out,
    int T_in, int C, int K, int apply_silu) {
  chunk_compute_impl<float>(x, state, w, out, T_in, C, K, apply_silu);
}
extern "C" __global__ void causal_conv1d_chunk_compute_f16(
    const __half* x, const __half* state, const __half* w, __half* out,
    int T_in, int C, int K, int apply_silu) {
  chunk_compute_impl<__half>(x, state, w, out, T_in, C, K, apply_silu);
}
extern "C" __global__ void causal_conv1d_chunk_compute_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* state, const __nv_bfloat16* w,
    __nv_bfloat16* out, int T_in, int C, int K, int apply_silu) {
  chunk_compute_impl<__nv_bfloat16>(x, state, w, out, T_in, C, K, apply_silu);
}

extern "C" __global__ void causal_conv1d_chunk_update_state_f32(
    const float* x, float* state, int T_in, int C, int K) {
  chunk_state_update_impl<float>(x, state, T_in, C, K);
}
extern "C" __global__ void causal_conv1d_chunk_update_state_f16(
    const __half* x, __half* state, int T_in, int C, int K) {
  chunk_state_update_impl<__half>(x, state, T_in, C, K);
}
extern "C" __global__ void causal_conv1d_chunk_update_state_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* state, int T_in, int C, int K) {
  chunk_state_update_impl<__nv_bfloat16>(x, state, T_in, C, K);
}
