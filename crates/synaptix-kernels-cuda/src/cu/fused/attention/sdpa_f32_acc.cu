#include <cuda_fp16.h>
#include <cuda_bf16.h>

// NVRTC компилирует без стандартных хедеров — math-built-ins (fmaxf/__expf/
// rsqrtf/__shfl_*) доступны и так, а INFINITY/isfinite (макросы math.h) — нет.
// Используем конечный sentinel для маски: exp(-1e30 - m) = 0.
#define SDPA_NEG_INF (-1e30f)

// Наивный scaled-dot-product attention с полной F32-аккумуляцией (точный
// baseline/reference, НЕ flash). Один block = одна q-позиция (bi, h, ti);
// scores[Tkv] материализуются в dynamic shared memory. GQA + causal.
//   q (B, NH, Tq, D), k/v (B, NKV, Tkv, D), out (B, NH, Tq, D) — row-major.
//   out = softmax(scale * Q·Kᵀ + causal_mask) · V.
// Лимит: Tkv * 4 байт должно влезать в shared memory (≈ ≤ 8192 для 32KB).

__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ float ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

__device__ __forceinline__ float block_reduce_max(float val, float* warp_red, int tid, int bs) {
  unsigned int mask = 0xFFFFFFFFu;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) val = fmaxf(val, __shfl_down_sync(mask, val, off, 32));
  int warp = tid >> 5, lane = tid & 31;
  if (lane == 0) warp_red[warp] = val;
  __syncthreads();
  int num_warps = (bs + 31) >> 5;
  float r = SDPA_NEG_INF;
  if (warp == 0) {
    r = (lane < num_warps) ? warp_red[lane] : SDPA_NEG_INF;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) r = fmaxf(r, __shfl_down_sync(mask, r, off, 32));
  }
  return r;  // валиден только в lane 0 warp 0
}

__device__ __forceinline__ float block_reduce_sum(float val, float* warp_red, int tid, int bs) {
  unsigned int mask = 0xFFFFFFFFu;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(mask, val, off, 32);
  int warp = tid >> 5, lane = tid & 31;
  if (lane == 0) warp_red[warp] = val;
  __syncthreads();
  int num_warps = (bs + 31) >> 5;
  float r = 0.0f;
  if (warp == 0) {
    r = (lane < num_warps) ? warp_red[lane] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) r += __shfl_down_sync(mask, r, off, 32);
  }
  return r;
}

template <typename T>
__device__ __forceinline__ void sdpa_f32_acc_impl(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ out, int B, int NH, int NKV, int Tq, int Tkv, int D,
    float scale, int causal) {
  long row = blockIdx.x;            // 0 .. B*NH*Tq
  int ti = (int)(row % Tq);
  int h = (int)((row / Tq) % NH);
  int bi = (int)(row / ((long)Tq * NH));
  if (bi >= B) return;
  int tid = threadIdx.x, bs = blockDim.x;
  int n_rep = NH / NKV;
  int h_kv = h / n_rep;

  extern __shared__ float scores[];   // [Tkv]
  __shared__ float warp_red[32];
  __shared__ float s_max, s_sum;

  const T* q_row = q + (((long)(bi * NH + h) * Tq + ti) * D);
  int q_pos = (Tkv >= Tq) ? (Tkv - Tq + ti) : ti;

  // Pass 1: scores[j] = scale * dot(q, k_j), causal-mask → -inf.
  for (int j = tid; j < Tkv; j += bs) {
    if (causal && j > q_pos) { scores[j] = SDPA_NEG_INF; continue; }
    const T* k_row = k + (((long)(bi * NKV + h_kv) * Tkv + j) * D);
    float s = 0.0f;
    for (int kk = 0; kk < D; kk++) s += ld(q_row + kk) * ld(k_row + kk);
    scores[j] = s * scale;
  }
  __syncthreads();

  // Block max.
  float lmax = SDPA_NEG_INF;
  for (int j = tid; j < Tkv; j += bs) lmax = fmaxf(lmax, scores[j]);
  float bmax = block_reduce_max(lmax, warp_red, tid, bs);
  if (tid == 0) s_max = bmax;
  __syncthreads();
  float m = s_max;

  // exp(scores - max) обратно в smem + block sum.
  float lsum = 0.0f;
  for (int j = tid; j < Tkv; j += bs) {
    float e = __expf(scores[j] - m);  // маска: exp(-1e30 - m) = 0
    scores[j] = e;
    lsum += e;
  }
  __syncthreads();
  float bsum = block_reduce_sum(lsum, warp_red, tid, bs);
  if (tid == 0) s_sum = bsum;
  __syncthreads();
  float denom = s_sum;

  // Pass 2: out[dd] = (sum_j exp_scores[j] * v_j[dd]) / sum.
  T* out_row = out + (((long)(bi * NH + h) * Tq + ti) * D);
  for (int dd = tid; dd < D; dd += bs) {
    float acc = 0.0f;
    for (int j = 0; j < Tkv; j++) {
      float e = scores[j];
      if (e != 0.0f) {
        const T* v_row = v + (((long)(bi * NKV + h_kv) * Tkv + j) * D);
        acc += e * ld(v_row + dd);
      }
    }
    st(out_row + dd, denom > 0.0f ? acc / denom : 0.0f);
  }
}

extern "C" __global__ void sdpa_f32_acc_f32(
    const float* q, const float* k, const float* v, float* out,
    int B, int NH, int NKV, int Tq, int Tkv, int D, float scale, int causal) {
  sdpa_f32_acc_impl<float>(q, k, v, out, B, NH, NKV, Tq, Tkv, D, scale, causal);
}
extern "C" __global__ void sdpa_f32_acc_f16(
    const __half* q, const __half* k, const __half* v, __half* out,
    int B, int NH, int NKV, int Tq, int Tkv, int D, float scale, int causal) {
  sdpa_f32_acc_impl<__half>(q, k, v, out, B, NH, NKV, Tq, Tkv, D, scale, causal);
}
extern "C" __global__ void sdpa_f32_acc_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* out,
    int B, int NH, int NKV, int Tq, int Tkv, int D, float scale, int causal) {
  sdpa_f32_acc_impl<__nv_bfloat16>(q, k, v, out, B, NH, NKV, Tq, Tkv, D, scale, causal);
}
