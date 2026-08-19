#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// MXFP8-KV prefill v2 (sm_120a): схема flash_splitq (FA-2 split-Q поверх
// mma.sync.m16n8k16, online softmax в регистрах, exp2-домен, causal-скип
// блоков) + fill-стадия, деквантующая K/V из E4M3+E8M0 прямо в smem:
// uint4-загрузка (16 элементов), аппаратный cvt.rn.f16x2.e4m3x2, умножение
// на float-скейл блока, упаковка cvt.rn.{f16x2,bf16x2}.f32 → два uint4-стора.
// Дальше MMA/softmax/PV байт в байт как в flash_splitq_impl (v1: BM=64,
// BN=32, 4 warp'а, single-buffer).
//
// Заменяет структурно медленный flash_mxfp8_prefill.cu (BM=16, softmax
// серийно на warp 0, скалярный ветвистый деквант): его сетка B·NH блоков
// на длинном контексте давала 0,7-1,4 ГБ/с эффективного чтения KV.
//
//   q/out (B, NH,  Tq,  HD)  row-major, T = f16/bf16
//   k/v   (B, NKV, Tkv, HD)  row-major, E4M3-байты, физический T-шаг t_stride
//   k/v_scale (B, NKV, T, HD/32) — E8M0 (U8), тот же t_stride
//   causal q_pos = (Tkv >= Tq) ? Tkv - Tq + ti : ti
//
// NVRTC: без <math.h>; -inf через __int_as_float.

#define FMS_NEG_INF (__int_as_float(0xFF800000))
#define FMS_LOG2E 1.4426950408889634f

#define FMS_BM 64
#define FMS_BN 32
#define FMS_WARPS 4

__device__ __forceinline__ bool fms_is_finite(float x) {
  return (__float_as_int(x) & 0x7F800000) != 0x7F800000;
}

__device__ __forceinline__ void fms_store_f(__half* p, float f) { *p = __float2half(f); }
__device__ __forceinline__ void fms_store_f(__nv_bfloat16* p, float f) { *p = __float2bfloat16(f); }

// Упаковка пары f32 → b16x2 одной инструкцией (как fsq_pack2).
__device__ __forceinline__ unsigned int fms_pack2(float a, float b, __half) {
  unsigned int u;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(u) : "f"(b), "f"(a));
  return u;
}
__device__ __forceinline__ unsigned int fms_pack2(float a, float b, __nv_bfloat16) {
  unsigned int u;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(u) : "f"(b), "f"(a));
  return u;
}
__device__ __forceinline__ float fms_exp2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// 2×E4M3 → half2 одной cvt-инструкцией.
__device__ __forceinline__ __half2 fms_cvt2(unsigned short two) {
  return __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)two, __NV_E4M3));
}
// E8M0: байт → 2^(b-127), пол 1e-12 (совпадает с append-квантом и CPU).
__device__ __forceinline__ float fms_e8m0(unsigned char b) {
  return fmaxf(__uint_as_float(((unsigned)b) << 23), 1e-12f);
}

template <typename T>
__device__ __forceinline__ void fms_mma16x8x16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3);

template <>
__device__ __forceinline__ void fms_mma16x8x16<__half>(
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
__device__ __forceinline__ void fms_mma16x8x16<__nv_bfloat16>(
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

__device__ __forceinline__ unsigned int fms_smem_ptr(const void* ptr) {
  unsigned int smem_ptr;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1;"
      " cvt.u32.u64 %0, smem_ptr; }"
      : "=r"(smem_ptr) : "l"(ptr));
  return smem_ptr;
}
__device__ __forceinline__ void fms_ldmatrix_x4(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    unsigned int addr) {
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void fms_ldmatrix_x4_trans(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    unsigned int addr) {
  asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

template <typename T, int HD>
__device__ __forceinline__ void flash_mxfp8_splitq_impl(
    const T* __restrict__ q,
    const unsigned char* __restrict__ k, const unsigned char* __restrict__ v,
    const unsigned char* __restrict__ k_scale, const unsigned char* __restrict__ v_scale,
    T* __restrict__ out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  constexpr int BM = FMS_BM, BN = FMS_BN, WARPS = FMS_WARPS;
  constexpr int THREADS   = WARPS * 32;
  constexpr int K_STEPS   = HD / 16;
  constexpr int SN_TILES  = BN / 8;
  constexpr int ON_TILES  = HD / 8;
  constexpr int PV_KSTEPS = BN / 16;
  constexpr int NB        = HD / 32;
  // fill: 16-элементных чанков на K- или V-тайл; HD∈{128,256} → кратно THREADS.
  constexpr int FILL_CHUNKS = BN * HD / 16;
  constexpr int FILL_PASSES = FILL_CHUNKS / THREADS;
  // KV-строки в smem с паддингом +8 эл. (16B), как в flash_splitq — иначе
  // row-stride кратен 128B и все строки ldmatrix бьют в банк 0.
  constexpr int KV_LD = HD + 8;

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
  size_t kv_scale_base  = ((size_t)b * NKV + kv_h) * (size_t)kv_stride;  // T-строки
  size_t kv_base_offset = kv_scale_base * HD;                            // E4M3-байты
  size_t q_base_offset  = ((size_t)b * NH + h) * (size_t)Tq * HD;

  extern __shared__ unsigned char smem[];
  T* k_sm = (T*)smem;
  T* v_sm = k_sm + BN * KV_LD;

  // ─── Q-фрагменты warp'а в регистры прямо из global (как в flash_splitq) ───
  unsigned int q_frag[K_STEPS][4];
  {
    int row_lo = warp_id * 16 + lane / 4;
    int row_hi = row_lo + 8;
    int col_lo = (lane % 4) * 2;
    int col_hi = col_lo + 8;
    const unsigned int zero2 = fms_pack2(0.0f, 0.0f, T{});
    size_t off_lo = q_base_offset + (size_t)(q_base + row_lo) * HD;
    size_t off_hi = q_base_offset + (size_t)(q_base + row_hi) * HD;
    bool lo_ok = row_lo < q_count;
    bool hi_ok = row_hi < q_count;
    #pragma unroll
    for (int ks = 0; ks < K_STEPS; ++ks) {
      int base_k = ks * 16;
      q_frag[ks][0] = lo_ok ? *(const unsigned int*)(q + off_lo + base_k + col_lo) : zero2;
      q_frag[ks][1] = hi_ok ? *(const unsigned int*)(q + off_hi + base_k + col_lo) : zero2;
      q_frag[ks][2] = lo_ok ? *(const unsigned int*)(q + off_lo + base_k + col_hi) : zero2;
      q_frag[ks][3] = hi_ok ? *(const unsigned int*)(q + off_hi + base_k + col_hi) : zero2;
    }
  }

  float o_acc[ON_TILES][4];
  #pragma unroll
  for (int n = 0; n < ON_TILES; ++n) {
    o_acc[n][0] = 0.0f; o_acc[n][1] = 0.0f; o_acc[n][2] = 0.0f; o_acc[n][3] = 0.0f;
  }
  float m_lo = FMS_NEG_INF, m_hi = FMS_NEG_INF;
  float l_lo = 0.0f, l_hi = 0.0f;

  int wrow_lo = warp_id * 16 + lane / 4;
  int wrow_hi = wrow_lo + 8;
  int q_pos_lo = q_pos_base + (int)q_base + wrow_lo;
  int q_pos_hi = q_pos_base + (int)q_base + wrow_hi;

  const float sc2 = scale * FMS_LOG2E;  // exp2-домен

  // ─── fill: деквант E4M3+E8M0 → T в smem, 16 элементов на проход ───
  auto fill_one = [&](const unsigned char* __restrict__ src,
                      const unsigned char* __restrict__ sscale,
                      T* dst, int kv_block_idx) {
    int kv_base_local = kv_block_idx * BN;
    #pragma unroll
    for (int pass = 0; pass < FILL_PASSES; ++pass) {
      int chunk = pass * THREADS + (int)tid;
      int kv_t_local = chunk / (HD / 16);
      int d = (chunk % (HD / 16)) * 16;
      int kv_t = kv_base_local + kv_t_local;
      unsigned int* drow = reinterpret_cast<unsigned int*>(dst + kv_t_local * KV_LD + d);
      if (kv_t < Tkv) {
        uint4 w = *reinterpret_cast<const uint4*>(
            src + kv_base_offset + (size_t)kv_t * HD + d);
        float sv = fms_e8m0(sscale[(kv_scale_base + kv_t) * NB + (d >> 5)]);
        unsigned int u[4] = {w.x, w.y, w.z, w.w};
        unsigned int packed[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          float2 flo = __half22float2(fms_cvt2((unsigned short)(u[i] & 0xFFFFu)));
          float2 fhi = __half22float2(fms_cvt2((unsigned short)(u[i] >> 16)));
          packed[i * 2]     = fms_pack2(flo.x * sv, flo.y * sv, T{});
          packed[i * 2 + 1] = fms_pack2(fhi.x * sv, fhi.y * sv, T{});
        }
        *reinterpret_cast<uint4*>(drow)     = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        *reinterpret_cast<uint4*>(drow + 4) = make_uint4(packed[4], packed[5], packed[6], packed[7]);
      } else {
        *reinterpret_cast<uint4*>(drow)     = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(drow + 4) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };

  int n_kv_blocks = (Tkv + BN - 1) / BN;
  if (causal) {
    int kv_hi = q_pos_base + (int)q_base + BM;  // exclusive
    int blocks_needed = (kv_hi + BN - 1) / BN;
    if (blocks_needed < n_kv_blocks) n_kv_blocks = blocks_needed;
  }

  for (int kv_block = 0; kv_block < n_kv_blocks; ++kv_block) {
    fill_one(k, k_scale, k_sm, kv_block);
    fill_one(v, v_scale, v_sm, kv_block);
    __syncthreads();

    int kv_base_local = kv_block * BN;
    int rem = Tkv - kv_base_local;
    int kv_count_local = rem < BN ? rem : BN;

    // ─── S = Q·Kᵀ (ldmatrix.x4 c ping-pong префетчем, как flash_splitq) ───
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
        return fms_smem_ptr(k_sm + (n * 8 + (lane & 7)) * KV_LD + kp * 32 + ((lane >> 3) & 3) * 8);
      };
      unsigned int kb[2][4];
      fms_ldmatrix_x4(kb[0][0], kb[0][1], kb[0][2], kb[0][3], kb_addr(0));
      #pragma unroll
      for (int i = 0; i < QK_ITERS; ++i) {
        int cur = i & 1;
        if (i + 1 < QK_ITERS) {
          fms_ldmatrix_x4(kb[cur ^ 1][0], kb[cur ^ 1][1], kb[cur ^ 1][2], kb[cur ^ 1][3], kb_addr(i + 1));
        }
        int n = i / (K_STEPS / 2);
        int kp = i % (K_STEPS / 2);
        fms_mma16x8x16<T>(s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3],
            q_frag[2 * kp][0], q_frag[2 * kp][1], q_frag[2 * kp][2], q_frag[2 * kp][3],
            kb[cur][0], kb[cur][1], s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3]);
        fms_mma16x8x16<T>(s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3],
            q_frag[2 * kp + 1][0], q_frag[2 * kp + 1][1], q_frag[2 * kp + 1][2], q_frag[2 * kp + 1][3],
            kb[cur][2], kb[cur][3], s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3]);
      }
    }

    // ─── маски/масштаб: fast-path полного тайла (масштаб уезжает в exp2-FMA) ───
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
          if (q_row_idx >= q_count) return FMS_NEG_INF;
          if (kv_c >= kv_count_local) return FMS_NEG_INF;
          if (causal && kv_c + kv_base_local > q_pos) return FMS_NEG_INF;
          return s * sc2;
        };
        s_frag[n][0] = msk(s_frag[n][0], wrow_lo, q_pos_lo, c0);
        s_frag[n][1] = msk(s_frag[n][1], wrow_lo, q_pos_lo, c1);
        s_frag[n][2] = msk(s_frag[n][2], wrow_hi, q_pos_hi, c0);
        s_frag[n][3] = msk(s_frag[n][3], wrow_hi, q_pos_hi, c1);
      }
    }

    // ─── online softmax в регистрах (строка = 4 лейна → shfl_xor 1,2) ───
    float bm_lo = FMS_NEG_INF, bm_hi = FMS_NEG_INF;
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
    if (!fms_is_finite(m_lo)) alpha_lo = 0.0f;
    else if (!fms_is_finite(mn_lo)) alpha_lo = 1.0f;
    else alpha_lo = fms_exp2(m_lo - mn_lo);
    if (!fms_is_finite(m_hi)) alpha_hi = 0.0f;
    else if (!fms_is_finite(mn_hi)) alpha_hi = 1.0f;
    else alpha_hi = fms_exp2(m_hi - mn_hi);

    float rs_lo = 0.0f, rs_hi = 0.0f;
    #pragma unroll
    for (int n = 0; n < SN_TILES; ++n) {
      float p0 = (!fms_is_finite(mn_lo) || s_frag[n][0] == FMS_NEG_INF) ? 0.0f : fms_exp2(fmaf(s_frag[n][0], exp_mul, -mn_lo));
      float p1 = (!fms_is_finite(mn_lo) || s_frag[n][1] == FMS_NEG_INF) ? 0.0f : fms_exp2(fmaf(s_frag[n][1], exp_mul, -mn_lo));
      float p2 = (!fms_is_finite(mn_hi) || s_frag[n][2] == FMS_NEG_INF) ? 0.0f : fms_exp2(fmaf(s_frag[n][2], exp_mul, -mn_hi));
      float p3 = (!fms_is_finite(mn_hi) || s_frag[n][3] == FMS_NEG_INF) ? 0.0f : fms_exp2(fmaf(s_frag[n][3], exp_mul, -mn_hi));
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

    // ─── O *= alpha (пропуск, если max не вырос ни у одной строки warp'а) ───
    if (!__all_sync(0xffffffffu, alpha_lo == 1.0f && alpha_hi == 1.0f)) {
      #pragma unroll
      for (int n = 0; n < ON_TILES; ++n) {
        o_acc[n][0] *= alpha_lo; o_acc[n][1] *= alpha_lo;
        o_acc[n][2] *= alpha_hi; o_acc[n][3] *= alpha_hi;
      }
    }

    // ─── O += P·V (P из s_frag, V через ldmatrix.x4.trans c префетчем) ───
    {
      constexpr int PV_NP = ON_TILES / 2;
      auto vb_addr = [&](int i) {
        int kk = i / PV_NP;
        int np = i % PV_NP;
        return fms_smem_ptr(v_sm + (kk * 16 + (lane & 15)) * KV_LD + np * 16 + ((lane >> 4) & 1) * 8);
      };
      unsigned int vb[2][4];
      fms_ldmatrix_x4_trans(vb[0][0], vb[0][1], vb[0][2], vb[0][3], vb_addr(0));
      unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      #pragma unroll
      for (int i = 0; i < PV_KSTEPS * PV_NP; ++i) {
        int cur = i & 1;
        if (i + 1 < PV_KSTEPS * PV_NP) {
          fms_ldmatrix_x4_trans(vb[cur ^ 1][0], vb[cur ^ 1][1], vb[cur ^ 1][2], vb[cur ^ 1][3], vb_addr(i + 1));
        }
        int kk = i / PV_NP;
        int np = i % PV_NP;
        if (np == 0) {
          a0 = fms_pack2(s_frag[2 * kk][0],     s_frag[2 * kk][1],     T{});
          a1 = fms_pack2(s_frag[2 * kk][2],     s_frag[2 * kk][3],     T{});
          a2 = fms_pack2(s_frag[2 * kk + 1][0], s_frag[2 * kk + 1][1], T{});
          a3 = fms_pack2(s_frag[2 * kk + 1][2], s_frag[2 * kk + 1][3], T{});
        }
        fms_mma16x8x16<T>(o_acc[2 * np][0], o_acc[2 * np][1], o_acc[2 * np][2], o_acc[2 * np][3],
            a0, a1, a2, a3, vb[cur][0], vb[cur][1],
            o_acc[2 * np][0], o_acc[2 * np][1], o_acc[2 * np][2], o_acc[2 * np][3]);
        fms_mma16x8x16<T>(o_acc[2 * np + 1][0], o_acc[2 * np + 1][1], o_acc[2 * np + 1][2], o_acc[2 * np + 1][3],
            a0, a1, a2, a3, vb[cur][2], vb[cur][3],
            o_acc[2 * np + 1][0], o_acc[2 * np + 1][1], o_acc[2 * np + 1][2], o_acc[2 * np + 1][3]);
      }
    }
    __syncthreads();  // MMA дочитал k_sm/v_sm → можно перезаписывать
  }

  // ─── Epilogue: normalize + store ───
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
        fms_store_f(&out[off], o_acc[n][0] * inv_lo);
        fms_store_f(&out[off + 1], o_acc[n][1] * inv_lo);
      }
      if (hi_valid) {
        size_t off = q_base_offset + (size_t)(q_base + wrow_hi) * HD + d_lo;
        fms_store_f(&out[off], o_acc[n][2] * inv_hi);
        fms_store_f(&out[off + 1], o_acc[n][3] * inv_hi);
      }
    }
  }
}

extern "C" {

__global__ void flash_mxfp8_splitq_f16_hd128(
    const __half* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_mxfp8_splitq_impl<__half, 128>(q, k, v, k_scale, v_scale, out, scale,
      B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void flash_mxfp8_splitq_bf16_hd128(
    const __nv_bfloat16* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale, __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_mxfp8_splitq_impl<__nv_bfloat16, 128>(q, k, v, k_scale, v_scale, out, scale,
      B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void flash_mxfp8_splitq_f16_hd256(
    const __half* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_mxfp8_splitq_impl<__half, 256>(q, k, v, k_scale, v_scale, out, scale,
      B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void flash_mxfp8_splitq_bf16_hd256(
    const __nv_bfloat16* q, const unsigned char* k, const unsigned char* v,
    const unsigned char* k_scale, const unsigned char* v_scale, __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_mxfp8_splitq_impl<__nv_bfloat16, 256>(q, k, v, k_scale, v_scale, out, scale,
      B, NH, NKV, Tq, Tkv, causal, t_stride);
}

}  // extern "C"
