#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// MXFP8-KV prefill (sm_80+; cvt E4M3 аппаратный с sm_89): FA-2 split-Q
// поверх mma.sync.m16n8k16, online softmax в регистрах, exp2-домен,
// causal-скип блоков. BM=64 строк Q на блок (4 warp'а), BN=32 ключа на тайл.
//
// Конвейер тайла KV:
//   * сырые байты E4M3 + E8M0 следующего тайла едут cp.async в smem, пока
//     считается текущий (задержка global скрыта);
//   * деквант smem → smem: cvt e4m3x2 → f16x2 и hmul2 на степень двойки
//     блока — точно в f16 (у E4M3 три бита мантиссы), две инструкции на пару;
//   * f16-тайлы K/V — с XOR-перестановкой 16-байтных кусков по строке
//     (ldmatrix без конфликтов банков без паддинга: 49.7 КБ smem на блок,
//     два блока на SM при 255 регистрах).
// Внутри — всегда f16 (Q из bf16 переводится без потерь в диапазоне f16, у
// f16 мантисса длиннее bf16 → P и K/V точнее, чем в bf16-варианте).
// P·V тайла — MMA с f16-накоплением (32 слагаемых, ~5e-4 отн. — ниже
// округления выхода), тайлы складываются в f32.
// Замер RTX 5090 Laptop, Qwen3.8-27B (24Q/4KV×256), Tq=2048, Tkv=128К:
// 131 → 75 мс (50 → 87 TFLOP/s), ошибка к эталону f64 1.9e-3 → 1.7e-3.
//
//   q/out (B, NH,  Tq,  HD)  row-major, T = f16/bf16
//   k/v   (B, NKV, Tkv, HD)  row-major, E4M3-байты, физический T-шаг t_stride
//   k/v_scale (B, NKV, T, HD/32) — E8M0 (U8), тот же t_stride; адреса — %4
//   causal q_pos = (Tkv >= Tq) ? Tkv - Tq + ti : ti
//
// NVRTC: без <math.h>; -inf через __int_as_float.

#define FM3_NEG_INF (__int_as_float(0xFF800000))
#define FM3_LOG2E 1.4426950408889634f

#define FM3_BM 64
#define FM3_BN 32
#define FM3_WARPS 4

__device__ __forceinline__ bool fm3_is_finite(float x) {
  return (__float_as_int(x) & 0x7F800000) != 0x7F800000;
}

__device__ __forceinline__ void fm3_store_f(__half* p, float f) { *p = __float2half(f); }
__device__ __forceinline__ void fm3_store_f(__nv_bfloat16* p, float f) { *p = __float2bfloat16(f); }

// Упаковка пары f32 → b16x2 одной инструкцией (как fsq_pack2).
__device__ __forceinline__ unsigned int fm3_pack2(float a, float b, __half) {
  unsigned int u;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(u) : "f"(b), "f"(a));
  return u;
}
__device__ __forceinline__ unsigned int fm3_pack2(float a, float b, __nv_bfloat16) {
  unsigned int u;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(u) : "f"(b), "f"(a));
  return u;
}
__device__ __forceinline__ float fm3_exp2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// 2×E4M3 → half2 одной cvt-инструкцией.
__device__ __forceinline__ __half2 fm3_cvt2(unsigned short two) {
  return __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)two, __NV_E4M3));
}
// E8M0: байт → 2^(b-127), пол 1e-12 (совпадает с append-квантом и CPU).
__device__ __forceinline__ float fm3_e8m0(unsigned char b) {
  return fmaxf(__uint_as_float(((unsigned)b) << 23), 1e-12f);
}

template <typename T>
__device__ __forceinline__ void fm3_mma16x8x16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3);

template <>
__device__ __forceinline__ void fm3_mma16x8x16<__half>(
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
__device__ __forceinline__ void fm3_mma16x8x16<__nv_bfloat16>(
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

__device__ __forceinline__ unsigned int fm3_smem_ptr(const void* ptr) {
  unsigned int smem_ptr;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1;"
      " cvt.u32.u64 %0, smem_ptr; }"
      : "=r"(smem_ptr) : "l"(ptr));
  return smem_ptr;
}
__device__ __forceinline__ void fm3_ldmatrix_x4(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    unsigned int addr) {
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void fm3_ldmatrix_x4_trans(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    unsigned int addr) {
  asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}


// cp.async: 16 и 4 байта, недостающее — нулями (src_bytes < size).
__device__ __forceinline__ void fm3_cp16(void* dst, const void* src, int src_bytes) {
  unsigned int d = fm3_smem_ptr(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void fm3_cp4(void* dst, const void* src, int src_bytes) {
  unsigned int d = fm3_smem_ptr(dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(d), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void fm3_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void fm3_cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }

// E8M0 → half2 множитель: 2^(b-127), вне диапазона f16 — к 0 / 2^15.
__device__ __forceinline__ __half2 fm3_scale_h2(unsigned char b) {
  float f = fminf(__uint_as_float(((unsigned)b) << 23), 32768.0f);
  return __float2half2_rn(f);
}

// Q (T) → пара f16 одним словом.
__device__ __forceinline__ unsigned int fm3_q2h(unsigned int raw, __half) { return raw; }
__device__ __forceinline__ unsigned int fm3_q2h(unsigned int raw, __nv_bfloat16) {
  __nv_bfloat162 b = *reinterpret_cast<__nv_bfloat162*>(&raw);
  float2 f = __bfloat1622float2(b);
  __half2 h = __floats2half2_rn(f.x, f.y);
  return *reinterpret_cast<unsigned int*>(&h);
}

// Смещение 16-байтного куска `c` строки `r` f16-тайла (строка HD*2 байт,
// XOR-перестановка кусков по r&7 — ldmatrix без конфликтов банков).
template <int HD>
__device__ __forceinline__ int fm3_sw(int r, int c) {
  return r * (HD * 2) + ((c ^ (r & 7)) << 4);
}

template <typename T, int HD>
__device__ __forceinline__ void flash_mxfp8_splitq_impl(
    const T* __restrict__ q,
    const unsigned char* __restrict__ k, const unsigned char* __restrict__ v,
    const unsigned char* __restrict__ k_scale, const unsigned char* __restrict__ v_scale,
    T* __restrict__ out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  constexpr int BM = FM3_BM, BN = FM3_BN, WARPS = FM3_WARPS;
  constexpr int THREADS   = WARPS * 32;
  constexpr int K_STEPS   = HD / 16;
  constexpr int SN_TILES  = BN / 8;
  constexpr int ON_TILES  = HD / 8;
  constexpr int PV_KSTEPS = BN / 16;
  constexpr int NB        = HD / 32;
  constexpr int ROW_CH    = HD / 16;            // 16-элементных кусков в строке
  constexpr int FILL_PASSES = BN * ROW_CH / THREADS;
  constexpr int RAW_CH    = BN * HD / 16;       // 16-байтных кусков сырого тайла
  constexpr int RAW_PASSES = RAW_CH / THREADS;
  constexpr int SC_CH     = BN * NB / 4;        // 4-байтных кусков скейлов тайла

  unsigned int q_tile = blockIdx.x;
  unsigned int bh     = blockIdx.y;
  unsigned int b      = bh / NH;
  unsigned int h      = bh % NH;
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
  size_t kv_scale_base  = ((size_t)b * NKV + kv_h) * (size_t)kv_stride;
  size_t kv_base_offset = kv_scale_base * HD;
  size_t q_base_offset  = ((size_t)b * NH + h) * (size_t)Tq * HD;

  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* kf  = smem;                       // f16 K [BN][HD], перестановка
  unsigned char* vf  = kf + BN * HD * 2;           // f16 V
  unsigned char* k8  = vf + BN * HD * 2;           // сырой E4M3 K [BN][HD]
  unsigned char* v8  = k8 + BN * HD;
  unsigned char* ks8 = v8 + BN * HD;               // E8M0 [BN][NB]
  unsigned char* vs8 = ks8 + BN * NB;

  // ─── Q-фрагменты (f16) в регистры ───
  unsigned int q_frag[K_STEPS][4];
  {
    int row_lo = warp_id * 16 + lane / 4;
    int row_hi = row_lo + 8;
    int col_lo = (lane % 4) * 2;
    int col_hi = col_lo + 8;
    size_t off_lo = q_base_offset + (size_t)(q_base + row_lo) * HD;
    size_t off_hi = q_base_offset + (size_t)(q_base + row_hi) * HD;
    bool lo_ok = row_lo < q_count;
    bool hi_ok = row_hi < q_count;
    #pragma unroll
    for (int ks = 0; ks < K_STEPS; ++ks) {
      int base_k = ks * 16;
      q_frag[ks][0] = lo_ok ? fm3_q2h(*(const unsigned int*)(q + off_lo + base_k + col_lo), T{}) : 0u;
      q_frag[ks][1] = hi_ok ? fm3_q2h(*(const unsigned int*)(q + off_hi + base_k + col_lo), T{}) : 0u;
      q_frag[ks][2] = lo_ok ? fm3_q2h(*(const unsigned int*)(q + off_lo + base_k + col_hi), T{}) : 0u;
      q_frag[ks][3] = hi_ok ? fm3_q2h(*(const unsigned int*)(q + off_hi + base_k + col_hi), T{}) : 0u;
    }
  }

  float o_acc[ON_TILES][4];
  #pragma unroll
  for (int n = 0; n < ON_TILES; ++n) {
    o_acc[n][0] = 0.0f; o_acc[n][1] = 0.0f; o_acc[n][2] = 0.0f; o_acc[n][3] = 0.0f;
  }
  float m_lo = FM3_NEG_INF, m_hi = FM3_NEG_INF;
  float l_lo = 0.0f, l_hi = 0.0f;

  int wrow_lo = warp_id * 16 + lane / 4;
  int wrow_hi = wrow_lo + 8;
  int q_pos_lo = q_pos_base + (int)q_base + wrow_lo;
  int q_pos_hi = q_pos_base + (int)q_base + wrow_hi;

  const float sc2 = scale * FM3_LOG2E;

  // ─── сырой тайл KV (E4M3 + E8M0) → smem асинхронно ───
  auto issue_raw = [&](int kv_block_idx) {
    int t0 = kv_block_idx * BN;
    #pragma unroll
    for (int pass = 0; pass < RAW_PASSES; ++pass) {
      int ch = pass * THREADS + (int)tid;
      int r = ch / ROW_CH;
      int c = (ch % ROW_CH) * 16;
      int t = t0 + r;
      bool ok = t < Tkv;
      size_t go = kv_base_offset + (size_t)(ok ? t : 0) * HD + c;
      fm3_cp16(k8 + r * HD + c, k + go, ok ? 16 : 0);
      fm3_cp16(v8 + r * HD + c, v + go, ok ? 16 : 0);
    }
    // скейлы: 4-байтные куски, по строке (NB=8 — два куска, NB=4 — один)
    for (int ch = (int)tid; ch < 2 * SC_CH; ch += THREADS) {
      bool is_v = ch >= SC_CH;
      int cc = is_v ? ch - SC_CH : ch;
      int byte = cc * 4;
      int r = byte / NB;
      int t = t0 + r;
      bool ok = t < Tkv;
      size_t go = (kv_scale_base + (size_t)(ok ? t : 0)) * NB + (byte % NB);
      if (is_v) fm3_cp4(vs8 + byte, v_scale + go, ok ? 4 : 0);
      else      fm3_cp4(ks8 + byte, k_scale + go, ok ? 4 : 0);
    }
    fm3_cp_commit();
  };

  // ─── деквант сырого тайла smem → f16 smem ───
  auto dequant = [&]() {
    #pragma unroll
    for (int pass = 0; pass < FILL_PASSES; ++pass) {
      int ch = pass * THREADS + (int)tid;
      int r = ch / ROW_CH;
      int cidx = ch % ROW_CH;           // 16 элементов = два 16-байтных f16-куска
      int d = cidx * 16;
      #pragma unroll
      for (int kvsel = 0; kvsel < 2; ++kvsel) {
        const unsigned char* src = kvsel ? v8 : k8;
        const unsigned char* ssc = kvsel ? vs8 : ks8;
        unsigned char* dst = kvsel ? vf : kf;
        uint4 w = *reinterpret_cast<const uint4*>(src + r * HD + d);
        __half2 s2 = fm3_scale_h2(ssc[r * NB + (d >> 5)]);
        unsigned int u[4] = {w.x, w.y, w.z, w.w};
        unsigned int pk[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          __half2 lo = __hmul2(fm3_cvt2((unsigned short)(u[i] & 0xFFFFu)), s2);
          __half2 hi = __hmul2(fm3_cvt2((unsigned short)(u[i] >> 16)), s2);
          pk[i * 2]     = *reinterpret_cast<unsigned int*>(&lo);
          pk[i * 2 + 1] = *reinterpret_cast<unsigned int*>(&hi);
        }
        *reinterpret_cast<uint4*>(dst + fm3_sw<HD>(r, cidx * 2))     = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        *reinterpret_cast<uint4*>(dst + fm3_sw<HD>(r, cidx * 2 + 1)) = make_uint4(pk[4], pk[5], pk[6], pk[7]);
      }
    }
  };

  int n_kv_blocks = (Tkv + BN - 1) / BN;
  if (causal) {
    int kv_hi = q_pos_base + (int)q_base + BM;
    int blocks_needed = (kv_hi + BN - 1) / BN;
    if (blocks_needed < n_kv_blocks) n_kv_blocks = blocks_needed;
  }

  issue_raw(0);
  fm3_cp_wait_all();
  __syncthreads();
  dequant();
  __syncthreads();
  if (n_kv_blocks > 1) issue_raw(1);

  const unsigned int kf_s = fm3_smem_ptr(kf);
  const unsigned int vf_s = fm3_smem_ptr(vf);

  for (int kv_block = 0; kv_block < n_kv_blocks; ++kv_block) {
    int kv_base_local = kv_block * BN;
    int rem = Tkv - kv_base_local;
    int kv_count_local = rem < BN ? rem : BN;

    // ─── S = Q·Kᵀ ───
    float s_frag[SN_TILES][4];
    #pragma unroll
    for (int n = 0; n < SN_TILES; ++n) {
      s_frag[n][0] = 0.0f; s_frag[n][1] = 0.0f; s_frag[n][2] = 0.0f; s_frag[n][3] = 0.0f;
    }
    {
      constexpr int QK_ITERS = SN_TILES * (K_STEPS / 2);
      auto kb_addr = [&](int i) {
        int n = i / (K_STEPS / 2);
        int kp = i % (K_STEPS / 2);
        int r = n * 8 + (lane & 7);
        int c = kp * 4 + ((lane >> 3) & 3);
        return kf_s + (unsigned)fm3_sw<HD>(r, c);
      };
      unsigned int kb[2][4];
      fm3_ldmatrix_x4(kb[0][0], kb[0][1], kb[0][2], kb[0][3], kb_addr(0));
      #pragma unroll
      for (int i = 0; i < QK_ITERS; ++i) {
        int cur = i & 1;
        if (i + 1 < QK_ITERS) {
          fm3_ldmatrix_x4(kb[cur ^ 1][0], kb[cur ^ 1][1], kb[cur ^ 1][2], kb[cur ^ 1][3], kb_addr(i + 1));
        }
        int n = i / (K_STEPS / 2);
        int kp = i % (K_STEPS / 2);
        fm3_mma16x8x16<__half>(s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3],
            q_frag[2 * kp][0], q_frag[2 * kp][1], q_frag[2 * kp][2], q_frag[2 * kp][3],
            kb[cur][0], kb[cur][1], s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3]);
        fm3_mma16x8x16<__half>(s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3],
            q_frag[2 * kp + 1][0], q_frag[2 * kp + 1][1], q_frag[2 * kp + 1][2], q_frag[2 * kp + 1][3],
            kb[cur][2], kb[cur][3], s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3]);
      }
    }

    bool full_tile = (kv_count_local == BN) && (q_count == BM)
        && (!causal || kv_base_local + BN - 1 <= q_pos_base + (int)q_base + warp_id * 16);
    float exp_mul = full_tile ? sc2 : 1.0f;
    if (!full_tile) {
      int col_base = (lane % 4) * 2;
      #pragma unroll
      for (int n = 0; n < SN_TILES; ++n) {
        int c0 = n * 8 + col_base;
        int c1 = c0 + 1;
        auto msk = [&](float s, int q_row_idx, int q_pos, int kv_c) {
          if (q_row_idx >= q_count) return FM3_NEG_INF;
          if (kv_c >= kv_count_local) return FM3_NEG_INF;
          if (causal && kv_c + kv_base_local > q_pos) return FM3_NEG_INF;
          return s * sc2;
        };
        s_frag[n][0] = msk(s_frag[n][0], wrow_lo, q_pos_lo, c0);
        s_frag[n][1] = msk(s_frag[n][1], wrow_lo, q_pos_lo, c1);
        s_frag[n][2] = msk(s_frag[n][2], wrow_hi, q_pos_hi, c0);
        s_frag[n][3] = msk(s_frag[n][3], wrow_hi, q_pos_hi, c1);
      }
    }

    float bm_lo = FM3_NEG_INF, bm_hi = FM3_NEG_INF;
    #pragma unroll
    for (int n = 0; n < SN_TILES; ++n) {
      bm_lo = fmaxf(bm_lo, fmaxf(s_frag[n][0], s_frag[n][1]));
      bm_hi = fmaxf(bm_hi, fmaxf(s_frag[n][2], s_frag[n][3]));
    }
    bm_lo = fmaxf(bm_lo, __shfl_xor_sync(0xffffffffu, bm_lo, 1));
    bm_lo = fmaxf(bm_lo, __shfl_xor_sync(0xffffffffu, bm_lo, 2));
    bm_hi = fmaxf(bm_hi, __shfl_xor_sync(0xffffffffu, bm_hi, 1));
    bm_hi = fmaxf(bm_hi, __shfl_xor_sync(0xffffffffu, bm_hi, 2));
    bm_lo *= exp_mul;
    bm_hi *= exp_mul;

    float mn_lo = fmaxf(m_lo, bm_lo);
    float mn_hi = fmaxf(m_hi, bm_hi);
    float alpha_lo, alpha_hi;
    if (!fm3_is_finite(m_lo)) alpha_lo = 0.0f;
    else if (!fm3_is_finite(mn_lo)) alpha_lo = 1.0f;
    else alpha_lo = fm3_exp2(m_lo - mn_lo);
    if (!fm3_is_finite(m_hi)) alpha_hi = 0.0f;
    else if (!fm3_is_finite(mn_hi)) alpha_hi = 1.0f;
    else alpha_hi = fm3_exp2(m_hi - mn_hi);

    float rs_lo = 0.0f, rs_hi = 0.0f;
    #pragma unroll
    for (int n = 0; n < SN_TILES; ++n) {
      float p0 = (!fm3_is_finite(mn_lo) || s_frag[n][0] == FM3_NEG_INF) ? 0.0f : fm3_exp2(fmaf(s_frag[n][0], exp_mul, -mn_lo));
      float p1 = (!fm3_is_finite(mn_lo) || s_frag[n][1] == FM3_NEG_INF) ? 0.0f : fm3_exp2(fmaf(s_frag[n][1], exp_mul, -mn_lo));
      float p2 = (!fm3_is_finite(mn_hi) || s_frag[n][2] == FM3_NEG_INF) ? 0.0f : fm3_exp2(fmaf(s_frag[n][2], exp_mul, -mn_hi));
      float p3 = (!fm3_is_finite(mn_hi) || s_frag[n][3] == FM3_NEG_INF) ? 0.0f : fm3_exp2(fmaf(s_frag[n][3], exp_mul, -mn_hi));
      s_frag[n][0] = p0; s_frag[n][1] = p1; s_frag[n][2] = p2; s_frag[n][3] = p3;
      rs_lo += p0 + p1;
      rs_hi += p2 + p3;
    }
    rs_lo += __shfl_xor_sync(0xffffffffu, rs_lo, 1);
    rs_lo += __shfl_xor_sync(0xffffffffu, rs_lo, 2);
    rs_hi += __shfl_xor_sync(0xffffffffu, rs_hi, 1);
    rs_hi += __shfl_xor_sync(0xffffffffu, rs_hi, 2);

    m_lo = mn_lo; m_hi = mn_hi;
    l_lo = l_lo * alpha_lo + rs_lo;
    l_hi = l_hi * alpha_hi + rs_hi;

    if (!__all_sync(0xffffffffu, alpha_lo == 1.0f && alpha_hi == 1.0f)) {
      #pragma unroll
      for (int n = 0; n < ON_TILES; ++n) {
        o_acc[n][0] *= alpha_lo; o_acc[n][1] *= alpha_lo;
        o_acc[n][2] *= alpha_hi; o_acc[n][3] *= alpha_hi;
      }
    }

    // ─── O += P·V: частичная сумма тайла (BN ключей) — MMA с f16-накоплением
    // (вдвое быстрее f32 на этой карте), тайлы складываются в f32. Ошибка
    // частичной суммы ~5e-4 отн. — в разы ниже округления выхода до bf16. ───
    {
      constexpr int PV_NP = ON_TILES / 2;
      unsigned int pa[PV_KSTEPS][4];
      #pragma unroll
      for (int kk = 0; kk < PV_KSTEPS; ++kk) {
        pa[kk][0] = fm3_pack2(s_frag[2 * kk][0],     s_frag[2 * kk][1],     __half{});
        pa[kk][1] = fm3_pack2(s_frag[2 * kk][2],     s_frag[2 * kk][3],     __half{});
        pa[kk][2] = fm3_pack2(s_frag[2 * kk + 1][0], s_frag[2 * kk + 1][1], __half{});
        pa[kk][3] = fm3_pack2(s_frag[2 * kk + 1][2], s_frag[2 * kk + 1][3], __half{});
      }
      auto vb_addr = [&](int i) {
        int np = i / PV_KSTEPS;
        int kk = i % PV_KSTEPS;
        int r = kk * 16 + (lane & 15);
        int c = np * 2 + ((lane >> 4) & 1);
        return vf_s + (unsigned)fm3_sw<HD>(r, c);
      };
      unsigned int vb[2][4];
      fm3_ldmatrix_x4_trans(vb[0][0], vb[0][1], vb[0][2], vb[0][3], vb_addr(0));
      unsigned int h0[2] = {0u, 0u}, h1[2] = {0u, 0u};
      #pragma unroll
      for (int i = 0; i < PV_KSTEPS * PV_NP; ++i) {
        int cur = i & 1;
        if (i + 1 < PV_KSTEPS * PV_NP) {
          fm3_ldmatrix_x4_trans(vb[cur ^ 1][0], vb[cur ^ 1][1], vb[cur ^ 1][2], vb[cur ^ 1][3], vb_addr(i + 1));
        }
        int np = i / PV_KSTEPS;
        int kk = i % PV_KSTEPS;
        if (kk == 0) { h0[0] = h0[1] = h1[0] = h1[1] = 0u; }
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
            : "+r"(h0[0]), "+r"(h0[1])
            : "r"(pa[kk][0]), "r"(pa[kk][1]), "r"(pa[kk][2]), "r"(pa[kk][3]), "r"(vb[cur][0]), "r"(vb[cur][1]));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
            : "+r"(h1[0]), "+r"(h1[1])
            : "r"(pa[kk][0]), "r"(pa[kk][1]), "r"(pa[kk][2]), "r"(pa[kk][3]), "r"(vb[cur][2]), "r"(vb[cur][3]));
        if (kk == PV_KSTEPS - 1) {
          float2 f;
          f = __half22float2(*reinterpret_cast<__half2*>(&h0[0])); o_acc[2 * np][0] += f.x; o_acc[2 * np][1] += f.y;
          f = __half22float2(*reinterpret_cast<__half2*>(&h0[1])); o_acc[2 * np][2] += f.x; o_acc[2 * np][3] += f.y;
          f = __half22float2(*reinterpret_cast<__half2*>(&h1[0])); o_acc[2 * np + 1][0] += f.x; o_acc[2 * np + 1][1] += f.y;
          f = __half22float2(*reinterpret_cast<__half2*>(&h1[1])); o_acc[2 * np + 1][2] += f.x; o_acc[2 * np + 1][3] += f.y;
        }
      }
    }

    if (kv_block + 1 < n_kv_blocks) {
      fm3_cp_wait_all();
      __syncthreads();   // сырой тайл пришёл у всех; f16-тайлы дочитаны
      dequant();
      __syncthreads();
      if (kv_block + 2 < n_kv_blocks) issue_raw(kv_block + 2);
    }
  }

  // ─── Epilogue: нормировка и запись парами ───
  {
    int col_lo = (lane % 4) * 2;
    float inv_lo = (l_lo > 0.0f) ? 1.0f / l_lo : 0.0f;
    float inv_hi = (l_hi > 0.0f) ? 1.0f / l_hi : 0.0f;
    bool lo_valid = wrow_lo < q_count;
    bool hi_valid = wrow_hi < q_count;
    #pragma unroll
    for (int n = 0; n < ON_TILES; ++n) {
      int d_lo = n * 8 + col_lo;
      if (lo_valid) {
        size_t off = q_base_offset + (size_t)(q_base + wrow_lo) * HD + d_lo;
        *reinterpret_cast<unsigned int*>(&out[off]) = fm3_pack2(o_acc[n][0] * inv_lo, o_acc[n][1] * inv_lo, T{});
      }
      if (hi_valid) {
        size_t off = q_base_offset + (size_t)(q_base + wrow_hi) * HD + d_lo;
        *reinterpret_cast<unsigned int*>(&out[off]) = fm3_pack2(o_acc[n][2] * inv_hi, o_acc[n][3] * inv_hi, T{});
      }
    }
  }
}

extern "C" {
__global__ void __launch_bounds__(128, 2) flash_mxfp8_splitq_f16_hd128(
    const __half* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_mxfp8_splitq_impl<__half, 128>(q, k, v, k_scale, v_scale, out, scale,
      B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void __launch_bounds__(128, 2) flash_mxfp8_splitq_f16_hd256(
    const __half* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_mxfp8_splitq_impl<__half, 256>(q, k, v, k_scale, v_scale, out, scale,
      B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void __launch_bounds__(128, 2) flash_mxfp8_splitq_bf16_hd128(
    const __nv_bfloat16* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale, __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_mxfp8_splitq_impl<__nv_bfloat16, 128>(q, k, v, k_scale, v_scale, out, scale,
      B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void __launch_bounds__(128, 2) flash_mxfp8_splitq_bf16_hd256(
    const __nv_bfloat16* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale, __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_mxfp8_splitq_impl<__nv_bfloat16, 256>(q, k, v, k_scale, v_scale, out, scale,
      B, NH, NKV, Tq, Tkv, causal, t_stride);
}
}  // extern "C"
