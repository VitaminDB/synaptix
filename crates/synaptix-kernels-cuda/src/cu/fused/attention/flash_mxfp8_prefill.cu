#include <cuda_fp16.h>
#include <cuda_bf16.h>

// MXFP8-KV prefill attention: K/V — E4M3 байты + per-32-block E8M0 scale
// (U8, [B,NKV,Tkv,D/32]), Q/out — F16/BF16. Block-деквант при загрузке в
// smem, далее tensor-core MMA m16n8k16, online softmax, F32-аккумулятор.
// Унаследован от удалённого flash_v4 (BM=16, softmax серийно на warp 0) —
// СТРУКТУРНО МЕДЛЕННЫЙ на больших S. TODO: портировать на схему
// flash_splitq (split-Q, softmax в регистрах) как dense-путь.
//
//   q   (B, NH,  Tq,  HD)  row-major
//   k/v (B, NKV, Tkv, HD)  row-major, E4M3  (GQA: kv_h = h / (NH/NKV))
//   out (B, NH,  Tq,  HD)  row-major
//   causal q_pos = (Tkv >= Tq) ? Tkv - Tq + ti : ti
//
// NVRTC: реальный -inf через __int_as_float (без <math.h>).

#define FV4_NEG_INF (__int_as_float(0xFF800000))

#define BM 16
#define BN 32
#define N_WARPS 4
#define WMMA_BLOCK_D 128

__device__ __forceinline__ bool fv4_is_finite(float x) {
  return (__float_as_int(x) & 0x7F800000) != 0x7F800000;
}

// FP8 E4M3 dequant (header-free, как flash_decode.cu::fp8_dec_e4m3): 2^exp через
// биты экспоненты. Для FP8-KV tensor-core пути (Phase 3+4).
__device__ __forceinline__ float fv4_fp8_dec_e4m3(unsigned char byte) {
  bool sign = (byte & 0x80) != 0;
  int e = (byte >> 3) & 0x0F;
  int m = byte & 0x07;
  if (e == 15 && m == 7) return __int_as_float(0x7FC00000);  // NaN
  float val;
  if (e == 0) {
    val = (float)m * 0.001953125f;
  } else {
    int exp_raw = e - 7;
    float frac = 1.0f + (float)m * 0.125f;
    val = frac * __int_as_float((exp_raw + 127) << 23);
  }
  return sign ? -val : val;
}

// E8M0 decode (MXFP8 block-scale): байт b → 2^(b-127), пол 1e-12 (совпадает с
// CPU e8m0_decode и mxfp8_kv.cu append-квантом).
__device__ __forceinline__ float fv4_dec_e8m0(unsigned char b) {
  return fmaxf(__int_as_float((int)((unsigned)b << 23)), 1e-12f);
}

// ─── Тип-зависимые хелперы (overload на скалярном типе) ───
__device__ __forceinline__ float to_f(__half h) { return __half2float(h); }
__device__ __forceinline__ float to_f(__nv_bfloat16 h) { return __bfloat162float(h); }

__device__ __forceinline__ void store_f(__half* p, float f) { *p = __float2half(f); }
__device__ __forceinline__ void store_f(__nv_bfloat16* p, float f) { *p = __float2bfloat16(f); }

__device__ __forceinline__ __half from_f(float f, __half) { return __float2half(f); }
__device__ __forceinline__ __nv_bfloat16 from_f(float f, __nv_bfloat16) { return __float2bfloat16(f); }

// Упаковать 2 скаляра в 32-битный регистр (half2 / bf162 bit-layout для mma).
__device__ __forceinline__ unsigned int pack2(__half a, __half b) {
  union { __half2 h; unsigned int u; } x;
  x.h = __halves2half2(a, b);
  return x.u;
}
__device__ __forceinline__ unsigned int pack2(__nv_bfloat16 a, __nv_bfloat16 b) {
  union { __nv_bfloat162 h; unsigned int u; } x;
  x.h = __halves2bfloat162(a, b);
  return x.u;
}

// Загрузить 2 contiguous T из shared как packed unsigned int.
__device__ __forceinline__ unsigned int load2_smem(const __half* p) {
  union { __half2 h; unsigned int u; } x;
  x.h = *reinterpret_cast<const __half2*>(p);
  return x.u;
}
__device__ __forceinline__ unsigned int load2_smem(const __nv_bfloat16* p) {
  union { __nv_bfloat162 h; unsigned int u; } x;
  x.h = *reinterpret_cast<const __nv_bfloat162*>(p);
  return x.u;
}

// mma.sync.m16n8k16 — специализация на тип операндов A/B.
template <typename T>
__device__ __forceinline__ void mma16x8x16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3);

template <>
__device__ __forceinline__ void mma16x8x16<__half>(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

template <>
__device__ __forceinline__ void mma16x8x16<__nv_bfloat16>(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// ─── cp.async helpers ───
__device__ __forceinline__ unsigned int fv4_smem_ptr(const void* ptr) {
  unsigned int smem_ptr;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1;"
      " cvt.u32.u64 %0, smem_ptr; }"
      : "=r"(smem_ptr) : "l"(ptr));
  return smem_ptr;
}
__device__ __forceinline__ void fv4_cp_async_16(unsigned int smem_dst, const void* gmem_src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_dst), "l"(gmem_src));
}
__device__ __forceinline__ void fv4_cp_async_16_zero(unsigned int smem_dst) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;\n" :: "r"(smem_dst), "l"((const void*)nullptr));
}
__device__ __forceinline__ void fv4_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
#define FV4_CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group " #N ";\n")


// MXFP8-KV prefill: но K/V scale — per-32-block E8M0 (U8).
// Меняется ТОЛЬКО fill_kv (block-scale lookup); MMA/softmax/P@V идентичны (K/V
// деквантятся в smem до MMA). Scale-индекс: (kv_scale_base+kv_t)*NB + d/32.
template <typename T, int HD>
__device__ __forceinline__ void flash_mxfp8_impl(
    const T* __restrict__ q,
    const unsigned char* __restrict__ k, const unsigned char* __restrict__ v,
    const unsigned char* __restrict__ k_scale, const unsigned char* __restrict__ v_scale,
    T* __restrict__ out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  constexpr int K_STEPS    = HD / 16;
  constexpr int N_TILES    = HD / 32;
  constexpr int DWARP      = HD / N_WARPS;
  constexpr int Q_PASSES   = (BM * HD / 2) / WMMA_BLOCK_D;
  constexpr int FILL_PASSES = (BN * HD) / WMMA_BLOCK_D;
  constexpr int NB         = HD / 32;  // 32-блоков на строку head_dim
  unsigned int bh     = blockIdx.x;
  unsigned int b      = bh / NH;
  unsigned int h      = bh % NH;
  unsigned int q_tile = blockIdx.y;
  unsigned int tid    = threadIdx.x;
  int warp_id = (int)(tid >> 5);
  int lane    = (int)(tid & 31);

  if ((int)b >= B) return;
  unsigned int q_base = q_tile * BM;
  if ((int)q_base >= Tq) return;
  int q_count = (int)((Tq - (int)q_base) < BM ? (Tq - (int)q_base) : BM);

  int n_rep = NH / NKV;
  unsigned int kv_h = h / n_rep;
  int q_pos_base = (Tkv >= Tq) ? (Tkv - Tq) : 0;

  long kv_stride = (t_stride > 0) ? (long)t_stride : (long)Tkv;
  size_t kv_scale_base = ((size_t)b * NKV + kv_h) * (size_t)kv_stride;  // per-token base (T-row)
  size_t kv_base_offset = kv_scale_base * HD;                          // e4m3 byte base

  size_t q_row_stride = HD;
  size_t q_base_offset  = ((size_t)b * NH + h) * (size_t)Tq * HD;

  extern __shared__ unsigned char smem[];
  T* q_sm = (T*)smem;
  T* k_sm = q_sm + BM * HD;
  T* v_sm = k_sm + BN * HD;
  float* s_f32 = (float*)(v_sm + BN * HD);   // [BM][BN]
  T* p_sm  = (T*)(s_f32 + BM * BN);          // [BM][BN]
  float* m_sm = (float*)(p_sm + BM * BN);    // [BM]
  float* l_sm = m_sm + BM;                   // [BM]
  float* alpha_sm = l_sm + BM;               // [BM]

  // ─── Stage 0a: cooperative load Q → q_sm ───
  for (int pass = 0; pass < Q_PASSES; ++pass) {
    int linear = pass * WMMA_BLOCK_D + (int)tid;
    int r = linear / (HD / 2);
    int d_h2 = linear % (HD / 2);
    int d = d_h2 * 2;
    T v0, v1;
    if (r < q_count) {
      size_t off = q_base_offset + (size_t)(q_base + r) * HD + d;
      v0 = q[off]; v1 = q[off + 1];
    } else {
      v0 = from_f(0.0f, T{}); v1 = from_f(0.0f, T{});
    }
    q_sm[r * HD + d]     = v0;
    q_sm[r * HD + d + 1] = v1;
  }
  if (tid < BM) { m_sm[tid] = FV4_NEG_INF; l_sm[tid] = 0.0f; }
  __syncthreads();

  // ─── Pre-load Q fragments ───
  unsigned int q_frag[K_STEPS][4];
  {
    int row_lo = lane / 4;
    int row_hi = row_lo + 8;
    int col_lo = (lane % 4) * 2;
    int col_hi = col_lo + 8;
    #pragma unroll
    for (int k_step = 0; k_step < K_STEPS; ++k_step) {
      int base_k = k_step * 16;
      const T* p_row_lo = q_sm + row_lo * HD + base_k;
      const T* p_row_hi = q_sm + row_hi * HD + base_k;
      q_frag[k_step][0] = load2_smem(p_row_lo + col_lo);
      q_frag[k_step][1] = load2_smem(p_row_hi + col_lo);
      q_frag[k_step][2] = load2_smem(p_row_lo + col_hi);
      q_frag[k_step][3] = load2_smem(p_row_hi + col_hi);
    }
  }

  float acc[N_TILES][4];
  #pragma unroll
  for (int n = 0; n < N_TILES; ++n) {
    #pragma unroll
    for (int r = 0; r < 4; ++r) acc[n][r] = 0.0f;
  }

  // dequant-fill K/V тайла: MXFP8 E4M3 → T · 2^(E8M0[блок]-127), per-32-block scale.
  auto fill_kv = [&](int kv_block_idx) {
    int kv_base_local = kv_block_idx * BN;
    for (int pass = 0; pass < FILL_PASSES; ++pass) {
      int linear = pass * WMMA_BLOCK_D + (int)tid;
      int kv_t_local = linear / HD;
      int d = linear % HD;
      int kv_t = kv_base_local + kv_t_local;
      if (kv_t < Tkv) {
        size_t sc_idx = (kv_scale_base + kv_t) * NB + (d / 32);
        float ksc = fv4_dec_e8m0(k_scale[sc_idx]);
        float vsc = fv4_dec_e8m0(v_scale[sc_idx]);
        size_t idx = kv_base_offset + (size_t)kv_t * HD + d;
        k_sm[kv_t_local * HD + d] = from_f(fv4_fp8_dec_e4m3(k[idx]) * ksc, T{});
        v_sm[kv_t_local * HD + d] = from_f(fv4_fp8_dec_e4m3(v[idx]) * vsc, T{});
      } else {
        k_sm[kv_t_local * HD + d] = from_f(0.0f, T{});
        v_sm[kv_t_local * HD + d] = from_f(0.0f, T{});
      }
    }
  };

  int n_kv_blocks = (Tkv + BN - 1) / BN;
  for (int kv_block = 0; kv_block < n_kv_blocks; ++kv_block) {
    fill_kv(kv_block);
    __syncthreads();

    T* k_tile = k_sm;
    T* v_tile = v_sm;
    int kv_base_local = kv_block * BN;
    int rem = Tkv - kv_base_local;
    int kv_count_local = rem < BN ? rem : BN;

    // ─── Stage 1: S = scale·Q@Kᵀ ───
    float s_frag[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    int col_n_idx = lane / 4;
    int row_k_lo  = (lane % 4) * 2;
    int kv_col    = warp_id * 8 + col_n_idx;
    #pragma unroll
    for (int k_step = 0; k_step < K_STEPS; ++k_step) {
      int base_k = k_step * 16;
      const T* k_row = k_tile + kv_col * HD + base_k;
      unsigned int b0 = load2_smem(k_row + row_k_lo);
      unsigned int b1 = load2_smem(k_row + row_k_lo + 8);
      float d0, d1, d2, d3;
      mma16x8x16<T>(d0, d1, d2, d3,
          q_frag[k_step][0], q_frag[k_step][1], q_frag[k_step][2], q_frag[k_step][3],
          b0, b1, s_frag[0], s_frag[1], s_frag[2], s_frag[3]);
      s_frag[0] = d0; s_frag[1] = d1; s_frag[2] = d2; s_frag[3] = d3;
    }
    {
      int row_lo = lane / 4;
      int row_hi = row_lo + 8;
      int col_a  = warp_id * 8 + (lane % 4) * 2;
      int col_b  = col_a + 1;
      int q_pos_lo = q_pos_base + (int)q_base + row_lo;
      int q_pos_hi = q_pos_base + (int)q_base + row_hi;
      auto apply_mask = [&](float s, int q_row_idx, int q_pos, int kv_c) {
        if (q_row_idx >= q_count) return FV4_NEG_INF;
        if (kv_c >= kv_count_local) return FV4_NEG_INF;
        if (causal && kv_c + kv_base_local > q_pos) return FV4_NEG_INF;
        return s * scale;
      };
      s_frag[0] = apply_mask(s_frag[0], row_lo, q_pos_lo, col_a);
      s_frag[1] = apply_mask(s_frag[1], row_lo, q_pos_lo, col_b);
      s_frag[2] = apply_mask(s_frag[2], row_hi, q_pos_hi, col_a);
      s_frag[3] = apply_mask(s_frag[3], row_hi, q_pos_hi, col_b);
      s_f32[row_lo * BN + col_a] = s_frag[0];
      s_f32[row_lo * BN + col_b] = s_frag[1];
      s_f32[row_hi * BN + col_a] = s_frag[2];
      s_f32[row_hi * BN + col_b] = s_frag[3];
    }
    __syncthreads();

    // ─── Stage 2: warp 0 — online softmax + cast to T ───
    if (warp_id == 0 && lane < BM) {
      int r = lane;
      float row[BN];
      #pragma unroll
      for (int j = 0; j < BN; ++j) row[j] = s_f32[r * BN + j];
      float m_block = FV4_NEG_INF;
      #pragma unroll
      for (int j = 0; j < BN; ++j) if (row[j] > m_block) m_block = row[j];
      float m_curr = m_sm[r];
      float m_new = (m_block > m_curr) ? m_block : m_curr;
      float alpha;
      if (!fv4_is_finite(m_curr)) alpha = 0.0f;
      else if (!fv4_is_finite(m_new)) alpha = 1.0f;
      else alpha = expf(m_curr - m_new);
      float row_sum = 0.0f;
      #pragma unroll
      for (int j = 0; j < BN; ++j) {
        float p = (!fv4_is_finite(m_new) || row[j] == FV4_NEG_INF) ? 0.0f : expf(row[j] - m_new);
        row[j] = p;
        row_sum += p;
      }
      float l_curr = l_sm[r];
      m_sm[r] = m_new;
      l_sm[r] = l_curr * alpha + row_sum;
      alpha_sm[r] = alpha;
      #pragma unroll
      for (int j = 0; j < BN; ++j) p_sm[r * BN + j] = from_f(row[j], T{});
    }
    __syncthreads();

    // ─── Stage 3a: acc *= alpha ───
    {
      int row_lo = lane / 4;
      int row_hi = row_lo + 8;
      float alpha_lo = alpha_sm[row_lo];
      float alpha_hi = alpha_sm[row_hi];
      #pragma unroll
      for (int n = 0; n < N_TILES; ++n) {
        acc[n][0] *= alpha_lo; acc[n][1] *= alpha_lo;
        acc[n][2] *= alpha_hi; acc[n][3] *= alpha_hi;
      }
    }
    // ─── Stage 3b: acc += P@V ───
    {
      int row_lo = lane / 4;
      int row_hi = row_lo + 8;
      int col_lo = (lane % 4) * 2;
      int col_hi = col_lo + 8;
      int v_col_n = lane / 4;
      int v_row_k_lo = (lane % 4) * 2;
      #pragma unroll
      for (int k_step = 0; k_step < 2; ++k_step) {
        int base_k = k_step * 16;
        unsigned int a0 = load2_smem(p_sm + row_lo * BN + base_k + col_lo);
        unsigned int a1 = load2_smem(p_sm + row_hi * BN + base_k + col_lo);
        unsigned int a2 = load2_smem(p_sm + row_lo * BN + base_k + col_hi);
        unsigned int a3 = load2_smem(p_sm + row_hi * BN + base_k + col_hi);
        #pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
          int n_col = warp_id * DWARP + n * 8 + v_col_n;
          int k0 = base_k + v_row_k_lo;
          int k1 = k0 + 1;
          int k8 = k0 + 8;
          int k9 = k1 + 8;
          T v0 = v_tile[k0 * HD + n_col];
          T v1 = v_tile[k1 * HD + n_col];
          T v8 = v_tile[k8 * HD + n_col];
          T v9 = v_tile[k9 * HD + n_col];
          unsigned int b0 = pack2(v0, v1);
          unsigned int b1 = pack2(v8, v9);
          float d0, d1, d2, d3;
          mma16x8x16<T>(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1,
              acc[n][0], acc[n][1], acc[n][2], acc[n][3]);
          acc[n][0] = d0; acc[n][1] = d1; acc[n][2] = d2; acc[n][3] = d3;
        }
      }
    }
    __syncthreads();
  }

  // ─── Epilogue ───
  {
    int row_lo = lane / 4;
    int row_hi = row_lo + 8;
    int col_lo = (lane % 4) * 2;
    int col_hi = col_lo + 1;
    float l_lo = l_sm[row_lo];
    float l_hi = l_sm[row_hi];
    float inv_lo = (l_lo > 0.0f) ? 1.0f / l_lo : 0.0f;
    float inv_hi = (l_hi > 0.0f) ? 1.0f / l_hi : 0.0f;
    bool row_lo_valid = row_lo < q_count;
    bool row_hi_valid = row_hi < q_count;
    #pragma unroll
    for (int n = 0; n < N_TILES; ++n) {
      int d_lo = warp_id * DWARP + n * 8 + col_lo;
      int d_hi = warp_id * DWARP + n * 8 + col_hi;
      if (row_lo_valid) {
        store_f(&out[q_base_offset + (size_t)(q_base + row_lo) * q_row_stride + d_lo], acc[n][0] * inv_lo);
        store_f(&out[q_base_offset + (size_t)(q_base + row_lo) * q_row_stride + d_hi], acc[n][1] * inv_lo);
      }
      if (row_hi_valid) {
        store_f(&out[q_base_offset + (size_t)(q_base + row_hi) * q_row_stride + d_lo], acc[n][2] * inv_hi);
        store_f(&out[q_base_offset + (size_t)(q_base + row_hi) * q_row_stride + d_hi], acc[n][3] * inv_hi);
      }
    }
  }
}

extern "C" {

__global__ void flash_mxfp8_f16_hd256(
    const __half* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_mxfp8_impl<__half, 256>(q, k, v, k_scale, v_scale, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void flash_mxfp8_bf16_hd256(
    const __nv_bfloat16* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale, __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_mxfp8_impl<__nv_bfloat16, 256>(q, k, v, k_scale, v_scale, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void flash_mxfp8_f16_hd128(
    const __half* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_mxfp8_impl<__half, 128>(q, k, v, k_scale, v_scale, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void flash_mxfp8_bf16_hd128(
    const __nv_bfloat16* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale, __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_mxfp8_impl<__nv_bfloat16, 128>(q, k, v, k_scale, v_scale, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride);
}

}  // extern "C"
