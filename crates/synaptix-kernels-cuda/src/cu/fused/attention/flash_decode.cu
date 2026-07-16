#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Flash-decoding split-K (decode path, T_q обычно = 1). KV-измерение разбивается
// на split_k сегментов по grid.y; каждый блок считает online-softmax по своему
// сегменту и пишет НЕнормализованный partial (m, l, acc[D]) в F32. Merge-ядро
// объединяет partial'ы через online-softmax-merge и нормализует выход.
//
//   q   (B, NH,  Tq,  D)  row-major
//   k/v (B, NKV, Tkv, D)  row-major   (GQA: h_kv = h / (NH/NKV))
//   out (B, NH,  Tq,  D)  row-major
//   out = softmax(scale * Q·Kᵀ + causal_mask) · V
//   causal q_pos = (Tkv >= Tq) ? Tkv - Tq + ti : ti
//
// NVRTC компилирует без <math.h>: реальный -inf берём через __int_as_float
// (intrinsic, без хедеров), что позволяет fa_is_finite корректно отбраковывать
// полностью замаскированные тайлы (exp(-inf - (-inf)) был бы NaN).

#define FD_NEG_INF (__int_as_float(0xFF800000))
#define FD_BLOCK   128   // = TILE_KV: один thread считает один score за тайл
#define TILE_KV    128
#define SPLIT_K_MAX 32

__device__ __forceinline__ bool fd_is_finite(float x) {
  return (__float_as_int(x) & 0x7F800000) != 0x7F800000;
}

__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ float ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// FP8 E4M3 dequant (KV-кеш). Header-free: 2^exp_raw собираем битами экспоненты
// (exp_raw∈[-6,8] → биас 127 в рамках normal range), вместо exp2f/<math.h>.
// Совпадает с fp8_quant.cu::fp8_decode_e4m3 (тот компилируется тем же NVRTC).
__device__ __forceinline__ float fp8_dec_e4m3(unsigned char byte) {
  bool sign = (byte & 0x80) != 0;
  int e = (byte >> 3) & 0x0F;
  int m = byte & 0x07;
  if (e == 15 && m == 7) return __int_as_float(0x7FC00000);  // NaN
  float val;
  if (e == 0) {
    val = (float)m * 0.001953125f;  // subnormal: m·2^-9
  } else {
    int exp_raw = e - 7;
    float frac = 1.0f + (float)m * 0.125f;
    val = frac * __int_as_float((exp_raw + 127) << 23);  // frac·2^exp_raw
  }
  return sign ? -val : val;
}

__device__ __forceinline__ float fd_block_reduce_max(float val, float* warp_red, int tid, int bs) {
  unsigned int mask = 0xFFFFFFFFu;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) val = fmaxf(val, __shfl_down_sync(mask, val, off, 32));
  int warp = tid >> 5, lane = tid & 31;
  if (lane == 0) warp_red[warp] = val;
  __syncthreads();
  int num_warps = (bs + 31) >> 5;
  float r = FD_NEG_INF;
  if (warp == 0) {
    r = (lane < num_warps) ? warp_red[lane] : FD_NEG_INF;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) r = fmaxf(r, __shfl_down_sync(mask, r, off, 32));
  }
  return r;  // валиден только в lane 0 warp 0
}

__device__ __forceinline__ float fd_block_reduce_sum(float val, float* warp_red, int tid, int bs) {
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
  return r;  // валиден только в lane 0 warp 0
}

// ─────────────────────────────────────────────────────────────────────────────
// Split-ядро: один блок = (bi, h, ti, split_id). Online-softmax по KV-сегменту
// [kv_start, kv_end), результат — НЕнормализованный partial.
template <typename T>
__device__ __forceinline__ void flash_decode_split_impl(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    float* __restrict__ partial_acc,  // (B*NH*Tq*split_k, D)
    float* __restrict__ partial_m,    // (B*NH*Tq*split_k,)
    float* __restrict__ partial_l,    // (B*NH*Tq*split_k,)
    int B, int NH, int NKV, int Tq, int Tkv, int D,
    float scale, int causal, int split_k, int t_stride) {
  long row = blockIdx.x;            // 0 .. B*NH*Tq
  int split_id = (int)blockIdx.y;
  int ti = (int)(row % Tq);
  int h = (int)((row / Tq) % NH);
  int bi = (int)(row / ((long)Tq * NH));
  if (bi >= B || split_id >= split_k) return;

  int tid = threadIdx.x, bs = blockDim.x;
  int n_rep = NH / NKV;
  int h_kv = h / n_rep;
  int q_pos = (Tkv >= Tq) ? (Tkv - Tq + ti) : ti;
  // physical row-stride dim-T в K/V (preallocated fixed buffer). 0 → contiguous = Tkv.
  long kv_stride = (t_stride > 0) ? (long)t_stride : (long)Tkv;

  // Равномерное разбиение Tkv на split_k сегментов.
  int seg = (Tkv + split_k - 1) / split_k;
  int kv_start = split_id * seg;
  int kv_end_unb = kv_start + seg;
  int kv_end = (kv_end_unb < Tkv) ? kv_end_unb : Tkv;

  long partial_idx = (((long)bi * NH + h) * Tq + ti) * split_k + split_id;

  extern __shared__ float smem[];
  float* q_sh = smem;            // [D]
  float* acc_sh = q_sh + D;      // [D]
  float* s_sh = acc_sh + D;      // [TILE_KV]
  __shared__ float warp_red[32];
  __shared__ float meta[2];      // [0]=m_new, [1]=alpha
  __shared__ float s_run_m, s_run_l;

  // Пустой сегмент → partial = (-inf, 0, 0).
  if (kv_start >= kv_end) {
    if (tid == 0) { partial_m[partial_idx] = FD_NEG_INF; partial_l[partial_idx] = 0.0f; }
    for (int d = tid; d < D; d += bs) partial_acc[partial_idx * D + d] = 0.0f;
    return;
  }

  const T* q_row = q + (((long)(bi * NH + h) * Tq + ti) * D);
  for (int d = tid; d < D; d += bs) { q_sh[d] = ld(q_row + d); acc_sh[d] = 0.0f; }
  if (tid == 0) { s_run_m = FD_NEG_INF; s_run_l = 0.0f; }
  __syncthreads();

  for (int tile_base = kv_start; tile_base < kv_end; tile_base += TILE_KV) {
    int tile_count = kv_end - tile_base;
    if (tile_count > TILE_KV) tile_count = TILE_KV;

    // Stage 1: thread tid считает score для ключа kv_t = tile_base + tid.
    float s = FD_NEG_INF;
    if (tid < tile_count) {
      int kv_t = tile_base + tid;
      if (causal && kv_t > q_pos) {
        s = FD_NEG_INF;
      } else {
        const T* k_row = k + (((long)(bi * NKV + h_kv) * kv_stride + kv_t) * D);
        float dot = 0.0f;
        for (int d = 0; d < D; d++) dot += q_sh[d] * ld(k_row + d);
        s = dot * scale;
      }
    }
    s_sh[tid] = s;
    __syncthreads();

    // Stage 2: block-max → online m_new + alpha (в tid 0).
    float m_tile = fd_block_reduce_max(s_sh[tid], warp_red, tid, bs);
    if (tid == 0) {
      float m_curr = s_run_m;
      float m_new = fmaxf(m_curr, m_tile);
      float alpha;
      if (!fd_is_finite(m_curr)) alpha = 0.0f;
      else if (!fd_is_finite(m_new)) alpha = 1.0f;
      else alpha = __expf(m_curr - m_new);
      meta[0] = m_new; meta[1] = alpha;
    }
    __syncthreads();
    float m_new = meta[0], alpha = meta[1];

    // Stage 3: p[j] = exp(s[j] - m_new) обратно в s_sh + block-sum.
    float sj = s_sh[tid];
    float p = (!fd_is_finite(m_new) || sj == FD_NEG_INF) ? 0.0f : __expf(sj - m_new);
    s_sh[tid] = p;
    float p_sum = fd_block_reduce_sum(p, warp_red, tid, bs);
    if (tid == 0) {
      s_run_l = s_run_l * alpha + p_sum;
      s_run_m = m_new;
    }
    __syncthreads();

    // Stage 4: acc[d] = acc[d]*alpha + sum_j p[j] * V[tile_base+j, d].
    for (int d = tid; d < D; d += bs) {
      float a = acc_sh[d] * alpha;
      for (int j = 0; j < tile_count; j++) {
        float pj = s_sh[j];
        if (pj != 0.0f) {
          const T* v_row = v + (((long)(bi * NKV + h_kv) * kv_stride + tile_base + j) * D);
          a += pj * ld(v_row + d);
        }
      }
      acc_sh[d] = a;
    }
    __syncthreads();
  }

  // Запись НЕнормализованного partial.
  if (tid == 0) { partial_m[partial_idx] = s_run_m; partial_l[partial_idx] = s_run_l; }
  for (int d = tid; d < D; d += bs) partial_acc[partial_idx * D + d] = acc_sh[d];
}

// E8M0 decode: байт b → 2^(b-127), пол 1e-12 (совпадает с CPU e8m0_decode и
// append-квантом, где sv=max(...,1e-12)).
__device__ __forceinline__ float fd_dec_e8m0(unsigned char b) {
  return fmaxf(__uint_as_float(((unsigned)b) << 23), 1e-12f);
}

// MXFP8-KV flash-decode: как fp8-impl, но K/V scale — per-32-block E8M0 (U8),
// scale зависит от d/32 → K-dot группируется по 32-блокам (НЕ выносится из dot),
// V-scale читается per-block из DRAM (L2-resident, без vscale_sh).
template <typename T>
__device__ __forceinline__ void flash_decode_split_mxfp8_impl(
    const T* __restrict__ q,
    const unsigned char* __restrict__ k, const unsigned char* __restrict__ v,
    const unsigned char* __restrict__ k_scale, const unsigned char* __restrict__ v_scale,
    float* __restrict__ partial_acc, float* __restrict__ partial_m, float* __restrict__ partial_l,
    int B, int NH, int NKV, int Tq, int Tkv, int D,
    float scale, int causal, int split_k, int t_stride) {
  long row = blockIdx.x;
  int split_id = (int)blockIdx.y;
  int ti = (int)(row % Tq);
  int h = (int)((row / Tq) % NH);
  int bi = (int)(row / ((long)Tq * NH));
  if (bi >= B || split_id >= split_k) return;

  int tid = threadIdx.x, bs = blockDim.x;
  int n_rep = NH / NKV;
  int h_kv = h / n_rep;
  int q_pos = (Tkv >= Tq) ? (Tkv - Tq + ti) : ti;
  long kv_stride = (t_stride > 0) ? (long)t_stride : (long)Tkv;
  long kv_head_base = (long)(bi * NKV + h_kv) * kv_stride;  // в "T-row" единицах
  int nb = D / 32;

  int seg = (Tkv + split_k - 1) / split_k;
  int kv_start = split_id * seg;
  int kv_end_unb = kv_start + seg;
  int kv_end = (kv_end_unb < Tkv) ? kv_end_unb : Tkv;

  long partial_idx = (((long)bi * NH + h) * Tq + ti) * split_k + split_id;

  extern __shared__ float smem[];
  float* q_sh = smem;                  // [D]
  float* acc_sh = q_sh + D;            // [D]
  float* s_sh = acc_sh + D;            // [TILE_KV]
  __shared__ float warp_red[32];
  __shared__ float meta[2];
  __shared__ float s_run_m, s_run_l;

  if (kv_start >= kv_end) {
    if (tid == 0) { partial_m[partial_idx] = FD_NEG_INF; partial_l[partial_idx] = 0.0f; }
    for (int d = tid; d < D; d += bs) partial_acc[partial_idx * D + d] = 0.0f;
    return;
  }

  const T* q_row = q + (((long)(bi * NH + h) * Tq + ti) * D);
  for (int d = tid; d < D; d += bs) { q_sh[d] = ld(q_row + d); acc_sh[d] = 0.0f; }
  if (tid == 0) { s_run_m = FD_NEG_INF; s_run_l = 0.0f; }
  __syncthreads();

  for (int tile_base = kv_start; tile_base < kv_end; tile_base += TILE_KV) {
    int tile_count = kv_end - tile_base;
    if (tile_count > TILE_KV) tile_count = TILE_KV;

    float s = FD_NEG_INF;
    if (tid < tile_count) {
      int kv_t = tile_base + tid;
      if (causal && kv_t > q_pos) {
        s = FD_NEG_INF;
      } else {
        const unsigned char* k_row = k + (kv_head_base + kv_t) * D;
        const unsigned char* ksc_row = k_scale + (kv_head_base + kv_t) * nb;
        float dot = 0.0f;
        for (int blk = 0; blk < nb; blk++) {
          float sv = fd_dec_e8m0(ksc_row[blk]);
          float bsum = 0.0f;
          for (int i = 0; i < 32; i++) {
            int d = blk * 32 + i;
            bsum += q_sh[d] * fp8_dec_e4m3(k_row[d]);
          }
          dot += bsum * sv;
        }
        s = dot * scale;  // attn_scale; k_scale УЖЕ внутри dot
      }
    }
    s_sh[tid] = s;
    __syncthreads();

    float m_tile = fd_block_reduce_max(s_sh[tid], warp_red, tid, bs);
    if (tid == 0) {
      float m_curr = s_run_m;
      float m_new = fmaxf(m_curr, m_tile);
      float alpha;
      if (!fd_is_finite(m_curr)) alpha = 0.0f;
      else if (!fd_is_finite(m_new)) alpha = 1.0f;
      else alpha = __expf(m_curr - m_new);
      meta[0] = m_new; meta[1] = alpha;
    }
    __syncthreads();
    float m_new = meta[0], alpha = meta[1];

    float sj = s_sh[tid];
    float p = (!fd_is_finite(m_new) || sj == FD_NEG_INF) ? 0.0f : __expf(sj - m_new);
    s_sh[tid] = p;
    float p_sum = fd_block_reduce_sum(p, warp_red, tid, bs);
    if (tid == 0) {
      s_run_l = s_run_l * alpha + p_sum;
      s_run_m = m_new;
    }
    __syncthreads();

    // acc[d] = acc[d]·alpha + Σ_j p[j]·dec(V[tile_base+j,d])·dec_e8m0(vsc[...,d/32])
    for (int d = tid; d < D; d += bs) {
      int blk = d / 32;
      float a = acc_sh[d] * alpha;
      for (int j = 0; j < tile_count; j++) {
        float pj = s_sh[j];
        if (pj != 0.0f) {
          int kv_t = tile_base + j;
          const unsigned char* v_row = v + (kv_head_base + kv_t) * D;
          float vsv = fd_dec_e8m0(v_scale[(kv_head_base + kv_t) * nb + blk]);
          a += pj * fp8_dec_e4m3(v_row[d]) * vsv;
        }
      }
      acc_sh[d] = a;
    }
    __syncthreads();
  }

  if (tid == 0) { partial_m[partial_idx] = s_run_m; partial_l[partial_idx] = s_run_l; }
  for (int d = tid; d < D; d += bs) partial_acc[partial_idx * D + d] = acc_sh[d];
}

// Merge-ядро: один блок = (bi, h, ti). Объединяет split_k partial'ов через
// online-softmax-merge и нормализует.
template <typename T>
__device__ __forceinline__ void flash_decode_merge_impl(
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    T* __restrict__ out,
    int B, int NH, int Tq, int D, int split_k) {
  long row = blockIdx.x;            // 0 .. B*NH*Tq
  int ti = (int)(row % Tq);
  int h = (int)((row / Tq) % NH);
  int bi = (int)(row / ((long)Tq * NH));
  if (bi >= B) return;

  int tid = threadIdx.x, bs = blockDim.x;
  long base_idx = (((long)bi * NH + h) * Tq + ti) * split_k;

  __shared__ float m_max_sh, l_global_sh, corr_sh[SPLIT_K_MAX];

  if (tid == 0) {
    float m_max = FD_NEG_INF;
    for (int i = 0; i < split_k; i++) {
      float mi = partial_m[base_idx + i];
      if (mi > m_max) m_max = mi;
    }
    m_max_sh = m_max;
    float l_sum = 0.0f;
    bool m_finite = fd_is_finite(m_max);
    for (int i = 0; i < split_k; i++) {
      float mi = partial_m[base_idx + i];
      float li = partial_l[base_idx + i];
      float c = (!m_finite || !fd_is_finite(mi)) ? 0.0f : __expf(mi - m_max);
      corr_sh[i] = c;
      l_sum += li * c;
    }
    l_global_sh = l_sum;
  }
  __syncthreads();

  float inv = (l_global_sh > 0.0f) ? (1.0f / l_global_sh) : 0.0f;
  T* out_row = out + (((long)(bi * NH + h) * Tq + ti) * D);
  for (int d = tid; d < D; d += bs) {
    float acc = 0.0f;
    for (int i = 0; i < split_k; i++) {
      acc += partial_acc[(base_idx + i) * D + d] * corr_sh[i];
    }
    st(out_row + d, acc * inv);
  }
}

extern "C" {

__global__ void flash_decode_split_f32(
    const float* q, const float* k, const float* v,
    float* partial_acc, float* partial_m, float* partial_l,
    int B, int NH, int NKV, int Tq, int Tkv, int D, float scale, int causal, int split_k,
    int t_stride) {
  flash_decode_split_impl<float>(q, k, v, partial_acc, partial_m, partial_l,
      B, NH, NKV, Tq, Tkv, D, scale, causal, split_k, t_stride);
}
__global__ void flash_decode_split_f16(
    const __half* q, const __half* k, const __half* v,
    float* partial_acc, float* partial_m, float* partial_l,
    int B, int NH, int NKV, int Tq, int Tkv, int D, float scale, int causal, int split_k,
    int t_stride) {
  flash_decode_split_impl<__half>(q, k, v, partial_acc, partial_m, partial_l,
      B, NH, NKV, Tq, Tkv, D, scale, causal, split_k, t_stride);
}
__global__ void flash_decode_split_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    float* partial_acc, float* partial_m, float* partial_l,
    int B, int NH, int NKV, int Tq, int Tkv, int D, float scale, int causal, int split_k,
    int t_stride) {
  flash_decode_split_impl<__nv_bfloat16>(q, k, v, partial_acc, partial_m, partial_l,
      B, NH, NKV, Tq, Tkv, D, scale, causal, split_k, t_stride);
}

// MXFP8-KV split-ядра: k_scale/v_scale — E8M0 (U8), per-32-block.
__global__ void flash_decode_split_mxfp8_f32(
    const float* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale,
    float* partial_acc, float* partial_m, float* partial_l,
    int B, int NH, int NKV, int Tq, int Tkv, int D, float scale, int causal, int split_k,
    int t_stride) {
  flash_decode_split_mxfp8_impl<float>(q, k, v, k_scale, v_scale, partial_acc, partial_m, partial_l,
      B, NH, NKV, Tq, Tkv, D, scale, causal, split_k, t_stride);
}
__global__ void flash_decode_split_mxfp8_f16(
    const __half* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale,
    float* partial_acc, float* partial_m, float* partial_l,
    int B, int NH, int NKV, int Tq, int Tkv, int D, float scale, int causal, int split_k,
    int t_stride) {
  flash_decode_split_mxfp8_impl<__half>(q, k, v, k_scale, v_scale, partial_acc, partial_m, partial_l,
      B, NH, NKV, Tq, Tkv, D, scale, causal, split_k, t_stride);
}
__global__ void flash_decode_split_mxfp8_bf16(
    const __nv_bfloat16* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale,
    float* partial_acc, float* partial_m, float* partial_l,
    int B, int NH, int NKV, int Tq, int Tkv, int D, float scale, int causal, int split_k,
    int t_stride) {
  flash_decode_split_mxfp8_impl<__nv_bfloat16>(q, k, v, k_scale, v_scale, partial_acc, partial_m, partial_l,
      B, NH, NKV, Tq, Tkv, D, scale, causal, split_k, t_stride);
}

// Device-resident-length split-ядра: Tkv читается из device-памяти (*Tkv_ptr)
// вместо immediate. Launch config (grid/smem) НЕ зависит от значения Tkv →
// один CUDA-graph валиден для всех decode-позиций (значение обновляется
// memcpy_htod в Tkv_ptr перед каждым replay'ем). Merge-ядро Tkv не использует.
__global__ void flash_decode_split_f32_dev(
    const float* q, const float* k, const float* v,
    float* partial_acc, float* partial_m, float* partial_l,
    int B, int NH, int NKV, int Tq, const int* Tkv_ptr, int D, float scale, int causal, int split_k,
    int t_stride) {
  // Per-row KV length: Tkv_ptr is [B] (batch-1 passes [1] → bi=0, unchanged).
  int bi = (int)(blockIdx.x / ((long)Tq * NH));
  int Tkv = Tkv_ptr[bi < B ? bi : 0];
  flash_decode_split_impl<float>(q, k, v, partial_acc, partial_m, partial_l,
      B, NH, NKV, Tq, Tkv, D, scale, causal, split_k, t_stride);
}
__global__ void flash_decode_split_f16_dev(
    const __half* q, const __half* k, const __half* v,
    float* partial_acc, float* partial_m, float* partial_l,
    int B, int NH, int NKV, int Tq, const int* Tkv_ptr, int D, float scale, int causal, int split_k,
    int t_stride) {
  int bi = (int)(blockIdx.x / ((long)Tq * NH));
  int Tkv = Tkv_ptr[bi < B ? bi : 0];
  flash_decode_split_impl<__half>(q, k, v, partial_acc, partial_m, partial_l,
      B, NH, NKV, Tq, Tkv, D, scale, causal, split_k, t_stride);
}
__global__ void flash_decode_split_bf16_dev(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    float* partial_acc, float* partial_m, float* partial_l,
    int B, int NH, int NKV, int Tq, const int* Tkv_ptr, int D, float scale, int causal, int split_k,
    int t_stride) {
  int bi = (int)(blockIdx.x / ((long)Tq * NH));
  int Tkv = Tkv_ptr[bi < B ? bi : 0];
  flash_decode_split_impl<__nv_bfloat16>(q, k, v, partial_acc, partial_m, partial_l,
      B, NH, NKV, Tq, Tkv, D, scale, causal, split_k, t_stride);
}

// MXFP8-KV device-Tkv split-ядра (для CUDA-graph decode): Tkv из *Tkv_ptr.
__global__ void flash_decode_split_mxfp8_f32_dev(
    const float* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale,
    float* partial_acc, float* partial_m, float* partial_l,
    int B, int NH, int NKV, int Tq, const int* Tkv_ptr, int D, float scale, int causal, int split_k,
    int t_stride) {
  flash_decode_split_mxfp8_impl<float>(q, k, v, k_scale, v_scale, partial_acc, partial_m, partial_l,
      B, NH, NKV, Tq, *Tkv_ptr, D, scale, causal, split_k, t_stride);
}
__global__ void flash_decode_split_mxfp8_f16_dev(
    const __half* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale,
    float* partial_acc, float* partial_m, float* partial_l,
    int B, int NH, int NKV, int Tq, const int* Tkv_ptr, int D, float scale, int causal, int split_k,
    int t_stride) {
  flash_decode_split_mxfp8_impl<__half>(q, k, v, k_scale, v_scale, partial_acc, partial_m, partial_l,
      B, NH, NKV, Tq, *Tkv_ptr, D, scale, causal, split_k, t_stride);
}
__global__ void flash_decode_split_mxfp8_bf16_dev(
    const __nv_bfloat16* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale,
    float* partial_acc, float* partial_m, float* partial_l,
    int B, int NH, int NKV, int Tq, const int* Tkv_ptr, int D, float scale, int causal, int split_k,
    int t_stride) {
  flash_decode_split_mxfp8_impl<__nv_bfloat16>(q, k, v, k_scale, v_scale, partial_acc, partial_m, partial_l,
      B, NH, NKV, Tq, *Tkv_ptr, D, scale, causal, split_k, t_stride);
}

__global__ void flash_decode_merge_f32(
    const float* partial_acc, const float* partial_m, const float* partial_l,
    float* out, int B, int NH, int Tq, int D, int split_k) {
  flash_decode_merge_impl<float>(partial_acc, partial_m, partial_l, out, B, NH, Tq, D, split_k);
}
__global__ void flash_decode_merge_f16(
    const float* partial_acc, const float* partial_m, const float* partial_l,
    __half* out, int B, int NH, int Tq, int D, int split_k) {
  flash_decode_merge_impl<__half>(partial_acc, partial_m, partial_l, out, B, NH, Tq, D, split_k);
}
__global__ void flash_decode_merge_bf16(
    const float* partial_acc, const float* partial_m, const float* partial_l,
    __nv_bfloat16* out, int B, int NH, int Tq, int D, int split_k) {
  flash_decode_merge_impl<__nv_bfloat16>(partial_acc, partial_m, partial_l, out, B, NH, Tq, D, split_k);
}

}  // extern "C"
