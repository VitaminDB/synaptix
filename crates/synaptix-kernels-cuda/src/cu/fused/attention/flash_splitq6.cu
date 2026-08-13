#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define FSQ6_NEG_INF (__int_as_float(0xFF800000))
#define FSQ6_LOG2E 1.4426950408889634f

__device__ __forceinline__ bool fsq6_is_finite(float x) {
  return (__float_as_int(x) & 0x7F800000) != 0x7F800000;
}

__device__ __forceinline__ void fsq6_store_f(__half* p, float f) { *p = __float2half(f); }
__device__ __forceinline__ void fsq6_store_f(__nv_bfloat16* p, float f) { *p = __float2bfloat16(f); }

__device__ __forceinline__ unsigned int fsq6_pack2(float a, float b, __half) {
  unsigned int u;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(u) : "f"(b), "f"(a));
  return u;
}
__device__ __forceinline__ unsigned int fsq6_pack2(float a, float b, __nv_bfloat16) {
  unsigned int u;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(u) : "f"(b), "f"(a));
  return u;
}

__device__ __forceinline__ float fsq6_exp2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

template <typename T>
__device__ __forceinline__ void fsq6_mma16x8x16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3);

template <>
__device__ __forceinline__ void fsq6_mma16x8x16<__half>(
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
__device__ __forceinline__ void fsq6_mma16x8x16<__nv_bfloat16>(
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

__device__ __forceinline__ unsigned int fsq6_smem_ptr(const void* ptr) {
  unsigned int smem_ptr;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1;"
      " cvt.u32.u64 %0, smem_ptr; }"
      : "=r"(smem_ptr) : "l"(ptr));
  return smem_ptr;
}
__device__ __forceinline__ void fsq6_cp_async_16(unsigned int smem_dst, const void* gmem_src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_dst), "l"(gmem_src));
}
__device__ __forceinline__ void fsq6_cp_async_16_zero(unsigned int smem_dst) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;\n" :: "r"(smem_dst), "l"((const void*)nullptr));
}

__device__ __forceinline__ void fsq6_ldmatrix_x4(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    unsigned int addr) {
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void fsq6_ldmatrix_x4_trans(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    unsigned int addr) {
  asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void fsq6_mbar_init(unsigned long long* bar, unsigned int count) {
  unsigned int addr = fsq6_smem_ptr(bar);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(addr), "r"(count));
}
__device__ __forceinline__ void fsq6_mbar_arrive(unsigned long long* bar) {
  unsigned int addr = fsq6_smem_ptr(bar);
  asm volatile("{.reg .b64 t; mbarrier.arrive.shared.b64 t, [%0];}" :: "r"(addr));
}
__device__ __forceinline__ void fsq6_mbar_cp_arrive(unsigned long long* bar) {
  unsigned int addr = fsq6_smem_ptr(bar);
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(addr));
}
__device__ __forceinline__ void fsq6_mbar_wait(unsigned long long* bar, unsigned int parity) {
  unsigned int addr = fsq6_smem_ptr(bar);
  unsigned int ok = 0;
  do {
    asm volatile(
        "{.reg .pred P; mbarrier.try_wait.parity.shared.b64 P, [%1], %2; selp.b32 %0, 1, 0, P;}"
        : "=r"(ok) : "r"(addr), "r"(parity));
    if (!ok) __nanosleep(32);
  } while (!ok);
}

template <typename T, int HD, int BM = 64, int BN = 64, int CWARPS = 4, int STAGES = 2>
__device__ __forceinline__ void flash_splitq6_impl(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride, int bshd = 0) {
  constexpr int K_STEPS   = HD / 16;
  constexpr int SN_TILES  = BN / 8;
  constexpr int ON_TILES  = HD / 8;
  constexpr int PV_KSTEPS = BN / 16;
  constexpr int KV_CHUNKS = BN * HD / 8;
  constexpr int KV_PASSES = KV_CHUNKS / 32;
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
  size_t q_row_stride  = bshd ? (size_t)NH * HD : (size_t)HD;
  size_t kv_row_stride = bshd ? (size_t)NKV * HD : (size_t)HD;
  size_t kv_base_offset = bshd
      ? ((size_t)b * Tkv * NKV + kv_h) * HD
      : ((size_t)b * NKV + kv_h) * (size_t)kv_stride * HD;
  size_t q_base_offset = bshd
      ? ((size_t)b * Tq * NH + h) * HD
      : ((size_t)b * NH + h) * (size_t)Tq * HD;

  extern __shared__ unsigned char smem[];
  T* k_sm_s[STAGES];
  T* v_sm_s[STAGES];
  #pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    k_sm_s[s] = (T*)smem + (size_t)s * 2 * BN * KV_LD;
    v_sm_s[s] = k_sm_s[s] + BN * KV_LD;
  }
  __shared__ unsigned long long full_k_bar[STAGES];
  __shared__ unsigned long long full_v_bar[STAGES];
  __shared__ unsigned long long empty_bar[STAGES];
  if (tid == 0) {
    #pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      fsq6_mbar_init(&full_k_bar[s], 32);
      fsq6_mbar_init(&full_v_bar[s], 32);
      fsq6_mbar_init(&empty_bar[s], CWARPS * 32);
    }
  }
  __syncthreads();

  int n_kv_blocks = (Tkv + BN - 1) / BN;
  if (causal) {
    int kv_hi = q_pos_base + (int)q_base + BM;
    int blocks_needed = (kv_hi + BN - 1) / BN;
    if (blocks_needed < n_kv_blocks) n_kv_blocks = blocks_needed;
  }

  if (warp_id == CWARPS) {
    unsigned int phase_empty[STAGES];
    #pragma unroll
    for (int s = 0; s < STAGES; ++s) phase_empty[s] = 0;
    for (int i = 0; i < n_kv_blocks; ++i) {
      int s = i % STAGES;
      if (i >= STAGES) {
        fsq6_mbar_wait(&empty_bar[s], phase_empty[s]);
        phase_empty[s] ^= 1u;
      }
      int kv_base_local = i * BN;
      const T* srcs[2] = {k, v};
      T* dsts[2] = {k_sm_s[s], v_sm_s[s]};
      unsigned long long* bars[2] = {&full_k_bar[s], &full_v_bar[s]};
      #pragma unroll
      for (int kv = 0; kv < 2; ++kv) {
        for (int pass = 0; pass < KV_PASSES; ++pass) {
          int chunk = pass * 32 + lane;
          int kv_t_local = chunk / (HD / 8);
          int d = (chunk % (HD / 8)) * 8;
          unsigned int smem_dst = fsq6_smem_ptr(dsts[kv] + kv_t_local * KV_LD + d);
          int kv_t = kv_base_local + kv_t_local;
          if (kv_t < Tkv) {
            fsq6_cp_async_16(smem_dst, &srcs[kv][kv_base_offset + (size_t)kv_t * kv_row_stride + d]);
          } else {
            fsq6_cp_async_16_zero(smem_dst);
          }
        }
        fsq6_mbar_cp_arrive(bars[kv]);
      }
    }
    return;
  }

  unsigned int q_frag[K_STEPS][4];
  {
    int row_lo = warp_id * 16 + lane / 4;
    int row_hi = row_lo + 8;
    int col_lo = (lane % 4) * 2;
    int col_hi = col_lo + 8;
    const unsigned int zero2 = fsq6_pack2(0.0f, 0.0f, T{});
    size_t off_lo = q_base_offset + (size_t)(q_base + row_lo) * q_row_stride;
    size_t off_hi = q_base_offset + (size_t)(q_base + row_hi) * q_row_stride;
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
  float m_lo = FSQ6_NEG_INF, m_hi = FSQ6_NEG_INF;
  float l_lo = 0.0f, l_hi = 0.0f;

  int wrow_lo = warp_id * 16 + lane / 4;
  int wrow_hi = wrow_lo + 8;
  int q_pos_lo = q_pos_base + (int)q_base + wrow_lo;
  int q_pos_hi = q_pos_base + (int)q_base + wrow_hi;

  const float sc2 = scale * FSQ6_LOG2E;

  unsigned int phase_full[STAGES];
  #pragma unroll
  for (int s = 0; s < STAGES; ++s) phase_full[s] = 0;

  for (int kv_block = 0; kv_block < n_kv_blocks; ++kv_block) {
    int stg = kv_block % STAGES;
    fsq6_mbar_wait(&full_k_bar[stg], phase_full[stg]);
    fsq6_mbar_wait(&full_v_bar[stg], phase_full[stg]);
    phase_full[stg] ^= 1u;
    T* k_sm = k_sm_s[stg];
    T* v_sm = v_sm_s[stg];

    int kv_base_local = kv_block * BN;
    int rem = Tkv - kv_base_local;
    int kv_count_local = rem < BN ? rem : BN;

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
        return fsq6_smem_ptr(k_sm + (n * 8 + (lane & 7)) * KV_LD + kp * 32 + ((lane >> 3) & 3) * 8);
      };
      unsigned int kb[2][4];
      fsq6_ldmatrix_x4(kb[0][0], kb[0][1], kb[0][2], kb[0][3], kb_addr(0));
      #pragma unroll
      for (int i = 0; i < QK_ITERS; ++i) {
        int cur = i & 1;
        if (i + 1 < QK_ITERS) {
          fsq6_ldmatrix_x4(kb[cur ^ 1][0], kb[cur ^ 1][1], kb[cur ^ 1][2], kb[cur ^ 1][3], kb_addr(i + 1));
        }
        int n = i / (K_STEPS / 2);
        int kp = i % (K_STEPS / 2);
        fsq6_mma16x8x16<T>(s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3],
            q_frag[2 * kp][0], q_frag[2 * kp][1], q_frag[2 * kp][2], q_frag[2 * kp][3],
            kb[cur][0], kb[cur][1], s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3]);
        fsq6_mma16x8x16<T>(s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3],
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
        auto msk = [&](float sv, int q_row_idx, int q_pos, int kv_c) {
          if (q_row_idx >= q_count) return FSQ6_NEG_INF;
          if (kv_c >= kv_count_local) return FSQ6_NEG_INF;
          if (causal && kv_c + kv_base_local > q_pos) return FSQ6_NEG_INF;
          return sv * sc2;
        };
        s_frag[n][0] = msk(s_frag[n][0], wrow_lo, q_pos_lo, c0);
        s_frag[n][1] = msk(s_frag[n][1], wrow_lo, q_pos_lo, c1);
        s_frag[n][2] = msk(s_frag[n][2], wrow_hi, q_pos_hi, c0);
        s_frag[n][3] = msk(s_frag[n][3], wrow_hi, q_pos_hi, c1);
      }
    }

    float bm_lo = FSQ6_NEG_INF, bm_hi = FSQ6_NEG_INF;
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
    if (!fsq6_is_finite(m_lo)) alpha_lo = 0.0f;
    else if (!fsq6_is_finite(mn_lo)) alpha_lo = 1.0f;
    else alpha_lo = fsq6_exp2(m_lo - mn_lo);
    if (!fsq6_is_finite(m_hi)) alpha_hi = 0.0f;
    else if (!fsq6_is_finite(mn_hi)) alpha_hi = 1.0f;
    else alpha_hi = fsq6_exp2(m_hi - mn_hi);

    float rs_lo = 0.0f, rs_hi = 0.0f;
    #pragma unroll
    for (int n = 0; n < SN_TILES; ++n) {
      float p0 = (!fsq6_is_finite(mn_lo) || s_frag[n][0] == FSQ6_NEG_INF) ? 0.0f : fsq6_exp2(fmaf(s_frag[n][0], exp_mul, -mn_lo));
      float p1 = (!fsq6_is_finite(mn_lo) || s_frag[n][1] == FSQ6_NEG_INF) ? 0.0f : fsq6_exp2(fmaf(s_frag[n][1], exp_mul, -mn_lo));
      float p2 = (!fsq6_is_finite(mn_hi) || s_frag[n][2] == FSQ6_NEG_INF) ? 0.0f : fsq6_exp2(fmaf(s_frag[n][2], exp_mul, -mn_hi));
      float p3 = (!fsq6_is_finite(mn_hi) || s_frag[n][3] == FSQ6_NEG_INF) ? 0.0f : fsq6_exp2(fmaf(s_frag[n][3], exp_mul, -mn_hi));
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

    {
      constexpr int PV_NP = ON_TILES / 2;
      auto vb_addr = [&](int i) {
        int kk = i / PV_NP;
        int np = i % PV_NP;
        return fsq6_smem_ptr(v_sm + (kk * 16 + (lane & 15)) * KV_LD + np * 16 + ((lane >> 4) & 1) * 8);
      };
      unsigned int vb[2][4];
      fsq6_ldmatrix_x4_trans(vb[0][0], vb[0][1], vb[0][2], vb[0][3], vb_addr(0));
      unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      #pragma unroll
      for (int i = 0; i < PV_KSTEPS * PV_NP; ++i) {
        int cur = i & 1;
        if (i + 1 < PV_KSTEPS * PV_NP) {
          fsq6_ldmatrix_x4_trans(vb[cur ^ 1][0], vb[cur ^ 1][1], vb[cur ^ 1][2], vb[cur ^ 1][3], vb_addr(i + 1));
        }
        int kk = i / PV_NP;
        int np = i % PV_NP;
        if (np == 0) {
          a0 = fsq6_pack2(s_frag[2 * kk][0],     s_frag[2 * kk][1],     T{});
          a1 = fsq6_pack2(s_frag[2 * kk][2],     s_frag[2 * kk][3],     T{});
          a2 = fsq6_pack2(s_frag[2 * kk + 1][0], s_frag[2 * kk + 1][1], T{});
          a3 = fsq6_pack2(s_frag[2 * kk + 1][2], s_frag[2 * kk + 1][3], T{});
        }
        fsq6_mma16x8x16<T>(o_acc[2 * np][0], o_acc[2 * np][1], o_acc[2 * np][2], o_acc[2 * np][3],
            a0, a1, a2, a3, vb[cur][0], vb[cur][1],
            o_acc[2 * np][0], o_acc[2 * np][1], o_acc[2 * np][2], o_acc[2 * np][3]);
        fsq6_mma16x8x16<T>(o_acc[2 * np + 1][0], o_acc[2 * np + 1][1], o_acc[2 * np + 1][2], o_acc[2 * np + 1][3],
            a0, a1, a2, a3, vb[cur][2], vb[cur][3],
            o_acc[2 * np + 1][0], o_acc[2 * np + 1][1], o_acc[2 * np + 1][2], o_acc[2 * np + 1][3]);
      }
    }
    fsq6_mbar_arrive(&empty_bar[stg]);
  }

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
        size_t off = q_base_offset + (size_t)(q_base + wrow_lo) * q_row_stride + d_lo;
        fsq6_store_f(&out[off], o_acc[n][0] * inv_lo);
        fsq6_store_f(&out[off + 1], o_acc[n][1] * inv_lo);
      }
      if (hi_valid) {
        size_t off = q_base_offset + (size_t)(q_base + wrow_hi) * q_row_stride + d_lo;
        fsq6_store_f(&out[off], o_acc[n][2] * inv_hi);
        fsq6_store_f(&out[off + 1], o_acc[n][3] * inv_hi);
      }
    }
  }
}

extern "C" {

__global__ void flash_splitq6_f16_hd128(
    const __half* q, const __half* k, const __half* v, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq6_impl<__half, 128, 64, 32>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void flash_splitq6_bf16_hd128(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq6_impl<__nv_bfloat16, 128, 64, 32>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void flash_splitq6_f16_hd128_bshd(
    const __half* q, const __half* k, const __half* v, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq6_impl<__half, 128, 64, 32>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride, 1);
}
__global__ void flash_splitq6_bf16_hd128_bshd(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq6_impl<__nv_bfloat16, 128, 64, 32>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride, 1);
}

}
