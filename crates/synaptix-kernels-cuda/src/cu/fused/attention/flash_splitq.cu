#include <cuda_fp16.h>
#include <cuda_bf16.h>

// flash_splitq (sm_120): split-Q FA-2-схема поверх mma.sync.m16n8k16.
// Замена flash_v4 для prefill больших Tq: v4 (BM=16, softmax серийно на warp0,
// P через smem) упирается в ~27 TFLOPS; FA-2/SDPA на той же форме ~93 TF.
//
// Ключевые отличия от v4:
//   • BM=64: 4 warp'а × 16 q-строк (split-Q) — каждый warp владеет своим
//     S-тайлом 16×BN и softmax'ом, 128/128 потоков активны (v4: 16/128).
//   • Online softmax в РЕГИСТРАХ: строка S размазана по 4 лейнам (acc-layout
//     m16n8) → row-max/row-sum через __shfl_xor_sync(1|2), без smem-roundtrip.
//   • P (=exp(S−m)) переиспользуется из S-аккумуляторов как A-фрагменты PV-mma
//     (m16n8-acc и m16k16-A раскладки совпадают поэлементно) — P не пишется.
//   • exp2f вместо expf (1 SFU-инструкция): S заранее в exp2-домене
//     (scale·log2e), softmax инвариантен к замене основания.
//   • causal: KV-блоки целиком позади диагонали пропускаются (v4 их считал).
//
//   q   (B, NH,  Tq,  HD)  row-major          (bshd=1: [B,S,H,D])
//   k/v (B, NKV, Tkv, HD)  row-major          (GQA: kv_h = h / (NH/NKV))
//   out = softmax(scale·Q·Kᵀ + causal_mask)·V
//
// NVRTC: без <math.h>; -inf через __int_as_float.
//
// TODO(flash-потенциал, 2026-06-06): v5 = 91.2 TF/s на 14080²×32×128 bf16 —
// 92% cuDNN-якоря (98.8) при ISA-потолке mma.sync bf16→f32 117.8 (bench_mma_peak;
// на рабочем клоке ~2.4GHz ≈105-110). НЕРЕАЛИЗОВАННЫЙ потенциал до cuDNN (+8%):
//   1. Ядро cuDNN-класса: глубокий multistage (3-4 стадии K/V) на 100KB smem с
//      ОДНИМ блоком/SM и 8 варпами, где конвейер компенсирует потерю 2-блочной
//      структуры. Наши попытки (v2 74.7, mbarrier v11 82.4) теряли на барьерах
//      8 варпов / спинах — нужен warp-specialized producer (выделенный варп DMA
//      + mbarrier), как CUTLASS sm80 pipeline, а не симметричный producer-consumer.
//   2. Persistent-CTA + работа очередью тайлов: срезает хвост волны и launch
//      (мелочь ~2%, но в сумме с (1) может дотянуть к ~98-100).
//   3. ldmatrix-давление: K-фрагменты перечитываются каждым варпом (split-Q
//      дублирует чтение smem ×WARPS) — schema с register-passing между
//      варпами недоступна на mma.sync; вариант — BN=64 c warp-tile 32×64
//      (2 варпа на 16 строк, split по N) сократит дубли чтения K вдвое, но
//      удвоит PV-дубли — нужен счёт на бумаге.
//   4. Бейзлайн-сдвиги: exp2-плотность (одна MUFU на 2 элемента через
//      ex2.approx.f16x2 — потеря точности softmax, мерить), и причёсанный
//      эпилог (STG.128 вместо STG.32×2 — там ~0.5%).
//   5. Удвоение пика — только f16-acc (232 TF, точность) или FP8/NVFP4
//      (отдельная квант-сессия): mma.sync bf16 быстрее не выдаёт.

#define FSQ_NEG_INF (__int_as_float(0xFF800000))
#define FSQ_LOG2E 1.4426950408889634f

#define FSQ_BM 64
#define FSQ_BN 32
#define FSQ_WARPS 4
#define FSQ_THREADS 128

__device__ __forceinline__ bool fsq_is_finite(float x) {
  return (__float_as_int(x) & 0x7F800000) != 0x7F800000;
}

__device__ __forceinline__ void fsq_store_f(__half* p, float f) { *p = __float2half(f); }
__device__ __forceinline__ void fsq_store_f(__nv_bfloat16* p, float f) { *p = __float2bfloat16(f); }

__device__ __forceinline__ __half fsq_from_f(float f, __half) { return __float2half(f); }
__device__ __forceinline__ __nv_bfloat16 fsq_from_f(float f, __nv_bfloat16) { return __float2bfloat16(f); }

// Упаковка пары f32 → b16x2 ОДНОЙ инструкцией (cvt.rn.{f16x2,bf16x2}.f32) —
// вместо 2×cvt + prmt (ALU-диета PV-фазы).
__device__ __forceinline__ unsigned int fsq_pack2(float a, float b, __half) {
  unsigned int u;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(u) : "f"(b), "f"(a));
  return u;
}
__device__ __forceinline__ unsigned int fsq_pack2(float a, float b, __nv_bfloat16) {
  unsigned int u;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(u) : "f"(b), "f"(a));
  return u;
}
// Голый аппаратный EX2 (MUFU.EX2) без libm-обвязки exp2f.
__device__ __forceinline__ float fsq_exp2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ unsigned int fsq_pack2t(__half a, __half b) {
  union { __half2 h; unsigned int u; } x;
  x.h = __halves2half2(a, b);
  return x.u;
}
__device__ __forceinline__ unsigned int fsq_pack2t(__nv_bfloat16 a, __nv_bfloat16 b) {
  union { __nv_bfloat162 h; unsigned int u; } x;
  x.h = __halves2bfloat162(a, b);
  return x.u;
}

__device__ __forceinline__ unsigned int fsq_load2_smem(const __half* p) {
  union { __half2 h; unsigned int u; } x;
  x.h = *reinterpret_cast<const __half2*>(p);
  return x.u;
}
__device__ __forceinline__ unsigned int fsq_load2_smem(const __nv_bfloat16* p) {
  union { __nv_bfloat162 h; unsigned int u; } x;
  x.h = *reinterpret_cast<const __nv_bfloat162*>(p);
  return x.u;
}

template <typename T>
__device__ __forceinline__ void fsq_mma16x8x16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3);

template <>
__device__ __forceinline__ void fsq_mma16x8x16<__half>(
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
__device__ __forceinline__ void fsq_mma16x8x16_f16acc(
    unsigned int& d0, unsigned int& d1,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    unsigned int c0, unsigned int c1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(d0), "=r"(d1)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "r"(c0), "r"(c1));
}

template <>
__device__ __forceinline__ void fsq_mma16x8x16<__nv_bfloat16>(
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

__device__ __forceinline__ unsigned int fsq_smem_ptr(const void* ptr) {
  unsigned int smem_ptr;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1;"
      " cvt.u32.u64 %0, smem_ptr; }"
      : "=r"(smem_ptr) : "l"(ptr));
  return smem_ptr;
}
__device__ __forceinline__ void fsq_cp_async_16(unsigned int smem_dst, const void* gmem_src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_dst), "l"(gmem_src));
}
__device__ __forceinline__ void fsq_cp_async_16_zero(unsigned int smem_dst) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;\n" :: "r"(smem_dst), "l"((const void*)nullptr));
}
__device__ __forceinline__ void fsq_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
#define FSQ_CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group " #N ";\n")

// ldmatrix: 4×(8×8 b16) тайла за инструкцию (адреса от лейнов 0..31).
__device__ __forceinline__ void fsq_ldmatrix_x4(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    unsigned int addr) {
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void fsq_ldmatrix_x4_trans(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    unsigned int addr) {
  asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// ─────────────────────────────────────────────────────────────────────────────
// BM/BN/WARPS/STAGES параметризованы: v1 = 64/32/4/1 (single-buffer, 128
// потоков), v2 = 128/64/8/2 (double-buffer конвейер cp.async, 256 потоков —
// load(blk+1) перекрывается с compute(blk), wait_group(1)).
template <typename T, int HD, int BM = FSQ_BM, int BN = FSQ_BN, int WARPS = FSQ_WARPS, int STAGES = 1, int PV16 = 0>
__device__ __forceinline__ void flash_splitq_impl(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride, int bshd = 0, int window = 0) {
  constexpr int THREADS   = WARPS * 32;
  constexpr int K_STEPS   = HD / 16;          // QKᵀ reduction steps (= Q-фрагментов)
  constexpr int SN_TILES  = BN / 8;           // S n-tiles на warp (вся ширина BN)
  constexpr int ON_TILES  = HD / 8;           // O n-tiles на warp (вся ширина HD)
  constexpr int PV_KSTEPS = BN / 16;          // PV reduction steps
  constexpr int KV_CHUNKS = BN * HD / 8;
  constexpr int KV_PASSES = (KV_CHUNKS + THREADS - 1) / THREADS;
  // KV-строки в smem с паддингом +8 эл. (16B): row-stride 2·HD байт кратен 128B
  // → все строки в банке 0 → 4-8-way конфликты (ncu: mem 96%, DRAM 0.3%).
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
  // STAGES=1: single-buffer (v1 BM=64/BN=32 — double-buffer там терял occupancy:
  // 78.1 vs 81.5 TF). STAGES=2 (v2): double-buffer, issue(blk+1) до compute(blk).
  T* k_sm_s[STAGES];
  T* v_sm_s[STAGES];
  #pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    k_sm_s[s] = (T*)smem + (size_t)s * 2 * BN * KV_LD;
    v_sm_s[s] = k_sm_s[s] + BN * KV_LD;
  }

  // ─── Q-фрагменты warp'а (его 16 строк) в регистры прямо из global (Q читается
  // один раз — staging в smem не окупается), m16k16 A-layout ───
  unsigned int q_frag[K_STEPS][4];
  {
    int row_lo = warp_id * 16 + lane / 4;
    int row_hi = row_lo + 8;
    int col_lo = (lane % 4) * 2;
    int col_hi = col_lo + 8;
    const unsigned int zero2 = fsq_pack2(0.0f, 0.0f, T{});
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

  // O-аккумулятор warp'а: 16 строк × HD, ON_TILES n-tiles × 4 f32.
  float o_acc[PV16 ? 1 : ON_TILES][4];
  unsigned int o_acc16[PV16 ? ON_TILES : 1][2];
  if constexpr (PV16) {
    #pragma unroll
    for (int n = 0; n < ON_TILES; ++n) { o_acc16[n][0] = 0u; o_acc16[n][1] = 0u; }
  } else {
    #pragma unroll
    for (int n = 0; n < ON_TILES; ++n) {
      o_acc[n][0] = 0.0f; o_acc[n][1] = 0.0f; o_acc[n][2] = 0.0f; o_acc[n][3] = 0.0f;
    }
  }
  // online-softmax состояние (exp2-домен): на поток — 2 строки (row_lo, row_hi).
  float m_lo = FSQ_NEG_INF, m_hi = FSQ_NEG_INF;
  float l_lo = 0.0f, l_hi = 0.0f;

  // строки warp'а в координатах BM-тайла / глобальных q-позициях
  int wrow_lo = warp_id * 16 + lane / 4;
  int wrow_hi = wrow_lo + 8;
  int q_pos_lo = q_pos_base + (int)q_base + wrow_lo;
  int q_pos_hi = q_pos_base + (int)q_base + wrow_hi;

  const float sc2 = scale * FSQ_LOG2E;  // exp2-домен

  // K и V — РАЗДЕЛЬНЫЕ commit-группы: перед QK достаточно K (wait_group(1),
  // V ещё летит и прячется за QK-фазой), перед PV — wait_group(0).
  auto issue_one_load = [&](const T* __restrict__ src, T* dst, int kv_block_idx) {
    int kv_base_local = kv_block_idx * BN;
    for (int pass = 0; pass < KV_PASSES; ++pass) {
      int chunk = pass * THREADS + (int)tid;
      if (KV_CHUNKS % THREADS != 0 && chunk >= KV_CHUNKS) break;
      int kv_t_local = chunk / (HD / 8);
      int d = (chunk % (HD / 8)) * 8;
      unsigned int smem_dst = fsq_smem_ptr(dst + kv_t_local * KV_LD + d);
      int kv_t = kv_base_local + kv_t_local;
      if (kv_t < Tkv) {
        fsq_cp_async_16(smem_dst, &src[kv_base_offset + (size_t)kv_t * kv_row_stride + d]);
      } else {
        fsq_cp_async_16_zero(smem_dst);
      }
    }
    fsq_cp_async_commit();
  };
  auto issue_kv_load = [&](int kv_block_idx, int stage) {
    issue_one_load(k, k_sm_s[stage], kv_block_idx);
    issue_one_load(v, v_sm_s[stage], kv_block_idx);
  };

  int n_kv_blocks = (Tkv + BN - 1) / BN;
  if (causal) {
    // последняя q-позиция блока видит kv ≤ q_pos_base+q_base+BM-1 → блоки дальше пусты
    int kv_hi = q_pos_base + (int)q_base + BM;  // exclusive
    int blocks_needed = (kv_hi + BN - 1) / BN;
    if (blocks_needed < n_kv_blocks) n_kv_blocks = blocks_needed;
  }
  int kv_block_start = 0;
  int kv_block_end = n_kv_blocks;
  if (window > 0) {
    int qlo = q_pos_base + (int)q_base;
    int qhi = qlo + q_count - 1;
    int lo = qlo - window;
    if (lo < 0) lo = 0;
    kv_block_start = lo / BN;
    int be = (qhi + window) / BN + 1;
    if (be < kv_block_end) kv_block_end = be;
  }
  if (STAGES == 2) issue_kv_load(kv_block_start, 0);

  for (int kv_block = kv_block_start; kv_block < kv_block_end; ++kv_block) {
    T* k_sm;
    T* v_sm;
    if (STAGES == 2) {
      // конвейер: выдать загрузку следующего блока ДО ожидания текущего —
      // DMA(blk+1) перекрывается с compute(blk). K и V — отдельные группы (по 2).
      int cur = kv_block & 1;
      if (kv_block + 1 < kv_block_end) {
        issue_kv_load(kv_block + 1, cur ^ 1);
        FSQ_CP_ASYNC_WAIT_GROUP(2);
      } else {
        FSQ_CP_ASYNC_WAIT_GROUP(0);
      }
      k_sm = k_sm_s[cur];
      v_sm = v_sm_s[cur];
    } else {
      // (split-wait K/V с третьим барьером проверен: 87.7→84.2 — барьер дороже.)
      issue_kv_load(kv_block, 0);
      FSQ_CP_ASYNC_WAIT_GROUP(0);
      k_sm = k_sm_s[0];
      v_sm = v_sm_s[0];
    }
    __syncthreads();

    int kv_base_local = kv_block * BN;
    int rem = Tkv - kv_base_local;
    int kv_count_local = rem < BN ? rem : BN;

    // ─── S = Q·Kᵀ: warp считает все SN_TILES столбцовых тайлов своих 16 строк ───
    float s_frag[SN_TILES][4];
    #pragma unroll
    for (int n = 0; n < SN_TILES; ++n) {
      s_frag[n][0] = 0.0f; s_frag[n][1] = 0.0f; s_frag[n][2] = 0.0f; s_frag[n][3] = 0.0f;
    }
    {
      // ldmatrix.x4: B-фрагменты K сразу для пары k-step'ов (16 строк-«адресов»
      // на 4 тайла 8×8: лейны 0..7 → k=base, 8..15 → base+8, 16..23 → base+16,
      // 24..31 → base+24; строка n*8 + lane%8).
      // (свап петель kp↔n проверен: 87.7→84.7 — хуже; компилятор при n-снаружи
      // сам интерливит unroll-нутые итерации, а kp-снаружи раздул живые kb-рег.)
      // SW-префетч ldmatrix на 1 шаг (ping-pong kb): ld(i+1) выдан ДО mma(i) —
      // short-scoreboard ldmatrix→mma прячется за парой mma.
      constexpr int QK_ITERS = SN_TILES * (K_STEPS / 2);
      auto kb_addr = [&](int i) {
        int n = i / (K_STEPS / 2);
        int kp = i % (K_STEPS / 2);
        return fsq_smem_ptr(k_sm + (n * 8 + (lane & 7)) * KV_LD + kp * 32 + ((lane >> 3) & 3) * 8);
      };
      unsigned int kb[2][4];
      fsq_ldmatrix_x4(kb[0][0], kb[0][1], kb[0][2], kb[0][3], kb_addr(0));
      #pragma unroll
      for (int i = 0; i < QK_ITERS; ++i) {
        int cur = i & 1;
        if (i + 1 < QK_ITERS) {
          fsq_ldmatrix_x4(kb[cur ^ 1][0], kb[cur ^ 1][1], kb[cur ^ 1][2], kb[cur ^ 1][3], kb_addr(i + 1));
        }
        int n = i / (K_STEPS / 2);
        int kp = i % (K_STEPS / 2);
        fsq_mma16x8x16<T>(s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3],
            q_frag[2 * kp][0], q_frag[2 * kp][1], q_frag[2 * kp][2], q_frag[2 * kp][3],
            kb[cur][0], kb[cur][1], s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3]);
        fsq_mma16x8x16<T>(s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3],
            q_frag[2 * kp + 1][0], q_frag[2 * kp + 1][1], q_frag[2 * kp + 1][2], q_frag[2 * kp + 1][3],
            kb[cur][2], kb[cur][3], s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3]);
      }
    }

    // ─── scale (exp2-домен) + маски: OOB q-строк, OOB kv, causal.
    // Fast-path: полный тайл без масок (на больших S — почти все тайлы). ───
    // warp-инвариантное условие (causal — по минимальной q-строке warp'а), без divergence
    bool full_tile = (kv_count_local == BN) && (q_count == BM) && (window == 0)
        && (!causal || kv_base_local + BN - 1 <= q_pos_base + (int)q_base + warp_id * 16);
    // full-тайл: S остаётся RAW — масштаб уезжает в exp2-FMA (p = exp2(s·sc2 −
    // mn)) и в bm·sc2 (1 fmul вместо 16; max монотонен при sc2>0). masked-путь
    // масштабирует в msk() как раньше → exp_mul=1.
    float exp_mul = full_tile ? sc2 : 1.0f;
    if (!full_tile) {
      int col_base = (lane % 4) * 2;
      #pragma unroll
      for (int n = 0; n < SN_TILES; ++n) {
        int c0 = n * 8 + col_base;
        int c1 = c0 + 1;
        auto msk = [&](float s, int q_row_idx, int q_pos, int kv_c) {
          if (q_row_idx >= q_count) return FSQ_NEG_INF;
          if (kv_c >= kv_count_local) return FSQ_NEG_INF;
          if (causal && kv_c + kv_base_local > q_pos) return FSQ_NEG_INF;
          if (window > 0) {
            int d = q_pos - (kv_c + kv_base_local);
            if (d > window || d < -window) return FSQ_NEG_INF;
          }
          return s * sc2;
        };
        s_frag[n][0] = msk(s_frag[n][0], wrow_lo, q_pos_lo, c0);
        s_frag[n][1] = msk(s_frag[n][1], wrow_lo, q_pos_lo, c1);
        s_frag[n][2] = msk(s_frag[n][2], wrow_hi, q_pos_hi, c0);
        s_frag[n][3] = msk(s_frag[n][3], wrow_hi, q_pos_hi, c1);
      }
    }

    // ─── online softmax в регистрах (строка = 4 лейна → shfl_xor 1,2) ───
    float bm_lo = FSQ_NEG_INF, bm_hi = FSQ_NEG_INF;
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
    // (безусловный exp2 без is_finite/select проверен: 89.4→87.1 — селекты
    // дешевле; SASS-скедулинг с ветками складывается удачнее.)
    float alpha_lo, alpha_hi;
    if (!fsq_is_finite(m_lo)) alpha_lo = 0.0f;
    else if (!fsq_is_finite(mn_lo)) alpha_lo = 1.0f;
    else alpha_lo = fsq_exp2(m_lo - mn_lo);
    if (!fsq_is_finite(m_hi)) alpha_hi = 0.0f;
    else if (!fsq_is_finite(mn_hi)) alpha_hi = 1.0f;
    else alpha_hi = fsq_exp2(m_hi - mn_hi);

    float rs_lo = 0.0f, rs_hi = 0.0f;
    #pragma unroll
    for (int n = 0; n < SN_TILES; ++n) {
      float p0 = (!fsq_is_finite(mn_lo) || s_frag[n][0] == FSQ_NEG_INF) ? 0.0f : fsq_exp2(fmaf(s_frag[n][0], exp_mul, -mn_lo));
      float p1 = (!fsq_is_finite(mn_lo) || s_frag[n][1] == FSQ_NEG_INF) ? 0.0f : fsq_exp2(fmaf(s_frag[n][1], exp_mul, -mn_lo));
      float p2 = (!fsq_is_finite(mn_hi) || s_frag[n][2] == FSQ_NEG_INF) ? 0.0f : fsq_exp2(fmaf(s_frag[n][2], exp_mul, -mn_hi));
      float p3 = (!fsq_is_finite(mn_hi) || s_frag[n][3] == FSQ_NEG_INF) ? 0.0f : fsq_exp2(fmaf(s_frag[n][3], exp_mul, -mn_hi));
      s_frag[n][0] = p0; s_frag[n][1] = p1; s_frag[n][2] = p2; s_frag[n][3] = p3;
      rs_lo += p0 + p1;
      rs_hi += p2 + p3;
    }
    rs_lo += __shfl_xor_sync(0xffffffffu, rs_lo, 1);
    rs_lo += __shfl_xor_sync(0xffffffffu, rs_lo, 2);
    rs_hi += __shfl_xor_sync(0xffffffffu, rs_hi, 1);
    rs_hi += __shfl_xor_sync(0xffffffffu, rs_hi, 2);

    float l_old_lo = l_lo, l_old_hi = l_hi;
    m_lo = mn_lo; m_hi = mn_hi;
    l_lo = l_lo * alpha_lo + rs_lo;
    l_hi = l_hi * alpha_hi + rs_hi;
    float pnorm_lo = 1.0f, pnorm_hi = 1.0f;

    // ─── O *= alpha (пропуск, если max не вырос ни у одной строки warp'а —
    // на поздних тайлах частый случай: 2·ON_TILES FMUL/поток экономии) ───
    if constexpr (PV16) {
      float f_lo = (l_lo > 0.0f) ? (l_old_lo * alpha_lo / l_lo) : 0.0f;
      float f_hi = (l_hi > 0.0f) ? (l_old_hi * alpha_hi / l_hi) : 0.0f;
      pnorm_lo = (l_lo > 0.0f) ? (1.0f / l_lo) : 0.0f;
      pnorm_hi = (l_hi > 0.0f) ? (1.0f / l_hi) : 0.0f;
      if (!__all_sync(0xffffffffu, f_lo == 1.0f && f_hi == 1.0f)) {
        unsigned int fl2 = fsq_pack2(f_lo, f_lo, __half{});
        unsigned int fh2 = fsq_pack2(f_hi, f_hi, __half{});
        #pragma unroll
        for (int n = 0; n < ON_TILES; ++n) {
          union { __half2 h; unsigned int u; } a, b;
          a.u = o_acc16[n][0];
          a.h = __hmul2(a.h, *reinterpret_cast<__half2*>(&fl2));
          o_acc16[n][0] = a.u;
          b.u = o_acc16[n][1];
          b.h = __hmul2(b.h, *reinterpret_cast<__half2*>(&fh2));
          o_acc16[n][1] = b.u;
        }
      }
    } else if (!__all_sync(0xffffffffu, alpha_lo == 1.0f && alpha_hi == 1.0f)) {
      {
        #pragma unroll
        for (int n = 0; n < ON_TILES; ++n) {
          o_acc[n][0] *= alpha_lo; o_acc[n][1] *= alpha_lo;
          o_acc[n][2] *= alpha_hi; o_acc[n][3] *= alpha_hi;
        }
      }
    }

    // ─── O += P·V: P-фрагменты прямо из s_frag (m16n8-acc ≡ m16k16-A поэлементно);
    // V через ldmatrix.x4.trans — пара выходных n-tile за инструкцию ───
    {
      // SW-префетч ldmatrix.trans на 1 шаг (ping-pong vb), как в QK-фазе.
      constexpr int PV_NP = ON_TILES / 2;
      auto vb_addr = [&](int i) {
        int kk = i / PV_NP;
        int np = i % PV_NP;
        return fsq_smem_ptr(v_sm + (kk * 16 + (lane & 15)) * KV_LD + np * 16 + ((lane >> 4) & 1) * 8);
      };
      unsigned int vb[2][4];
      fsq_ldmatrix_x4_trans(vb[0][0], vb[0][1], vb[0][2], vb[0][3], vb_addr(0));
      unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      #pragma unroll
      for (int i = 0; i < PV_KSTEPS * PV_NP; ++i) {
        int cur = i & 1;
        if (i + 1 < PV_KSTEPS * PV_NP) {
          fsq_ldmatrix_x4_trans(vb[cur ^ 1][0], vb[cur ^ 1][1], vb[cur ^ 1][2], vb[cur ^ 1][3], vb_addr(i + 1));
        }
        int kk = i / PV_NP;
        int np = i % PV_NP;
        if (np == 0) {
          if constexpr (PV16) {
            a0 = fsq_pack2(s_frag[2 * kk][0] * pnorm_lo,     s_frag[2 * kk][1] * pnorm_lo,     T{});
            a1 = fsq_pack2(s_frag[2 * kk][2] * pnorm_hi,     s_frag[2 * kk][3] * pnorm_hi,     T{});
            a2 = fsq_pack2(s_frag[2 * kk + 1][0] * pnorm_lo, s_frag[2 * kk + 1][1] * pnorm_lo, T{});
            a3 = fsq_pack2(s_frag[2 * kk + 1][2] * pnorm_hi, s_frag[2 * kk + 1][3] * pnorm_hi, T{});
          } else {
            a0 = fsq_pack2(s_frag[2 * kk][0],     s_frag[2 * kk][1],     T{});
            a1 = fsq_pack2(s_frag[2 * kk][2],     s_frag[2 * kk][3],     T{});
            a2 = fsq_pack2(s_frag[2 * kk + 1][0], s_frag[2 * kk + 1][1], T{});
            a3 = fsq_pack2(s_frag[2 * kk + 1][2], s_frag[2 * kk + 1][3], T{});
          }
        }
        if constexpr (PV16) {
          fsq_mma16x8x16_f16acc(o_acc16[2 * np][0], o_acc16[2 * np][1],
              a0, a1, a2, a3, vb[cur][0], vb[cur][1],
              o_acc16[2 * np][0], o_acc16[2 * np][1]);
          fsq_mma16x8x16_f16acc(o_acc16[2 * np + 1][0], o_acc16[2 * np + 1][1],
              a0, a1, a2, a3, vb[cur][2], vb[cur][3],
              o_acc16[2 * np + 1][0], o_acc16[2 * np + 1][1]);
        } else {
          fsq_mma16x8x16<T>(o_acc[2 * np][0], o_acc[2 * np][1], o_acc[2 * np][2], o_acc[2 * np][3],
              a0, a1, a2, a3, vb[cur][0], vb[cur][1],
              o_acc[2 * np][0], o_acc[2 * np][1], o_acc[2 * np][2], o_acc[2 * np][3]);
          fsq_mma16x8x16<T>(o_acc[2 * np + 1][0], o_acc[2 * np + 1][1], o_acc[2 * np + 1][2], o_acc[2 * np + 1][3],
              a0, a1, a2, a3, vb[cur][2], vb[cur][3],
              o_acc[2 * np + 1][0], o_acc[2 * np + 1][1], o_acc[2 * np + 1][2], o_acc[2 * np + 1][3]);
        }
      }
    }
    __syncthreads();  // MMA дочитал k_sm/v_sm → можно перезаписывать
  }

  // ─── Epilogue: normalize + store ───
  {
    int col_lo = (lane % 4) * 2;
    float inv_lo = PV16 ? 1.0f : ((l_lo > 0.0f) ? 1.0f / l_lo : 0.0f);
    float inv_hi = PV16 ? 1.0f : ((l_hi > 0.0f) ? 1.0f / l_hi : 0.0f);
    bool lo_valid = wrow_lo < q_count;
    bool hi_valid = wrow_hi < q_count;
    #pragma unroll
    for (int n = 0; n < ON_TILES; ++n) {
      int d_lo = n * 8 + col_lo;
      float e0, e1, e2, e3;
      if constexpr (PV16) {
        union { __half2 h; unsigned int u; } a, b;
        a.u = o_acc16[n][0];
        b.u = o_acc16[n][1];
        float2 fa = __half22float2(a.h);
        float2 fb = __half22float2(b.h);
        e0 = fa.x; e1 = fa.y; e2 = fb.x; e3 = fb.y;
      } else {
        e0 = o_acc[n][0]; e1 = o_acc[n][1]; e2 = o_acc[n][2]; e3 = o_acc[n][3];
      }
      if (lo_valid) {
        size_t off = q_base_offset + (size_t)(q_base + wrow_lo) * q_row_stride + d_lo;
        fsq_store_f(&out[off], e0 * inv_lo);
        fsq_store_f(&out[off + 1], e1 * inv_lo);
      }
      if (hi_valid) {
        size_t off = q_base_offset + (size_t)(q_base + wrow_hi) * q_row_stride + d_lo;
        fsq_store_f(&out[off], e2 * inv_hi);
        fsq_store_f(&out[off + 1], e3 * inv_hi);
      }
    }
  }
}

// (128,3) ≈ 170 рег: после ldmatrix компилятор раздул времянки до >170 →
// 2 блока/SM (occupancy 16.6%). Жёсткое (128,4)=128 рег спиллит (71.4→68 TF).
#define FSQ_BOUNDS

// Device-резидентная длина KV (CUDA-graph prefill): Tkv из device-буфера —
// грид статичен по Tq, маска/границы считаются от *Tkv_ptr (рецепт v4_dev).
template <typename T, int HD>
__device__ __forceinline__ void flash_splitq_dev_impl(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ out, float scale,
    int B, int NH, int NKV, int Tq, const int* __restrict__ Tkv_ptr, int causal, int t_stride) {
  __shared__ int Tkv_sh;
  if (threadIdx.x == 0) Tkv_sh = *Tkv_ptr;
  __syncthreads();
  flash_splitq_impl<T, HD>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv_sh, causal, t_stride);
}

extern "C" {

__global__ void FSQ_BOUNDS flash_splitq_f16_hd64_dev(
    const __half* q, const __half* k, const __half* v, __half* out, float scale,
    int B, int NH, int NKV, int Tq, const int* Tkv_ptr, int causal, int t_stride) {
  flash_splitq_dev_impl<__half, 64>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv_ptr, causal, t_stride);
}
__global__ void FSQ_BOUNDS flash_splitq_bf16_hd64_dev(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, const int* Tkv_ptr, int causal, int t_stride) {
  flash_splitq_dev_impl<__nv_bfloat16, 64>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv_ptr, causal, t_stride);
}
__global__ void FSQ_BOUNDS flash_splitq_f16_hd128_dev(
    const __half* q, const __half* k, const __half* v, __half* out, float scale,
    int B, int NH, int NKV, int Tq, const int* Tkv_ptr, int causal, int t_stride) {
  flash_splitq_dev_impl<__half, 128>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv_ptr, causal, t_stride);
}
__global__ void FSQ_BOUNDS flash_splitq_bf16_hd128_dev(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, const int* Tkv_ptr, int causal, int t_stride) {
  flash_splitq_dev_impl<__nv_bfloat16, 128>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv_ptr, causal, t_stride);
}
__global__ void FSQ_BOUNDS flash_splitq_f16_hd256_dev(
    const __half* q, const __half* k, const __half* v, __half* out, float scale,
    int B, int NH, int NKV, int Tq, const int* Tkv_ptr, int causal, int t_stride) {
  flash_splitq_dev_impl<__half, 256>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv_ptr, causal, t_stride);
}
__global__ void FSQ_BOUNDS flash_splitq_bf16_hd256_dev(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, const int* Tkv_ptr, int causal, int t_stride) {
  flash_splitq_dev_impl<__nv_bfloat16, 256>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv_ptr, causal, t_stride);
}

__global__ void FSQ_BOUNDS flash_splitq_f16_hd64(
    const __half* q, const __half* k, const __half* v, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq_impl<__half, 64>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void FSQ_BOUNDS flash_splitq_bf16_hd64(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq_impl<__nv_bfloat16, 64>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void FSQ_BOUNDS flash_splitq_f16_hd128(
    const __half* q, const __half* k, const __half* v, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq_impl<__half, 128>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void FSQ_BOUNDS flash_splitq_bf16_hd128(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq_impl<__nv_bfloat16, 128>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void FSQ_BOUNDS flash_splitq_f16_hd256(
    const __half* q, const __half* k, const __half* v, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq_impl<__half, 256>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void FSQ_BOUNDS flash_splitq_bf16_hd256(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq_impl<__nv_bfloat16, 256>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride);
}
// ── v5: BM=64 / BN=64 / 4 warps single-buffer — вдвое меньше барьер-фаз, чем
// BN=32; 2 блока/SM сохраняются (smem 34.8KB, ~170 рег). Карта тупиков свипа
// (изолированный бенч 14080²×32×128 bf16): BN=32 база 84.7; BM128/8w/2st 74.7;
// +конвейер к базе 81.8; BN64+конвейер 57.3 (1 блок/SM); 3 блока рег-диетой
// 83.5 (спилл); BM96/6w 67.0; BN=80 82.9; v5 = 91.2 (split K/V commit-групп,
// fma-слияние масштаба в exp2, cvt.bf16x2/ex2.approx asm). mbarrier-конвейер
// (producer/consumer, варпы расползаются на тайл; cp.async.mbarrier.arrive.noinc
// + try_wait.parity, sm_90-компиляция) проверен ТРЕМЯ точками: BN32/2st 82.8,
// BN64/2st/4w 59.6, BM128/BN64/8w 82.4 — ВСЕ хуже v5: спин-накладные на тайл
// дороже syncthreads, а глубина расползания (1 тайл) мала; больше стадий не
// лезет в smem без потери 2 блоков/SM. cuDNN 98.8 на этой форме — иной класс
// ядра (глубокий multistage); потолок mma.sync bf16-f32 = 117.8 (bench_mma_peak).
__global__ void FSQ_BOUNDS flash_splitq5_f16_hd128(
    const __half* q, const __half* k, const __half* v, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq_impl<__half, 128, 64, 64, 4, 1>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride);
}
__global__ void FSQ_BOUNDS flash_splitq5_bf16_hd128(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq_impl<__nv_bfloat16, 128, 64, 64, 4, 1>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride);
}

// BSHD-layout ([B,S,H,D]) — image/video attn без permute+contiguous.
__global__ void FSQ_BOUNDS flash_splitq_f16_hd64_bshd(
    const __half* q, const __half* k, const __half* v, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq_impl<__half, 64>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride, 1);
}
__global__ void FSQ_BOUNDS flash_splitq_bf16_hd64_bshd(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq_impl<__nv_bfloat16, 64>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride, 1);
}
__global__ void FSQ_BOUNDS flash_splitq_f16_hd128_bshd(
    const __half* q, const __half* k, const __half* v, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq_impl<__half, 128>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride, 1);
}
__global__ void FSQ_BOUNDS flash_splitq_bf16_hd128_bshd(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq_impl<__nv_bfloat16, 128>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride, 1);
}

__global__ void FSQ_BOUNDS flash_splitq_f16_hd128_bshd_facc(
    const __half* q, const __half* k, const __half* v, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq_impl<__half, 128, FSQ_BM, FSQ_BN, FSQ_WARPS, 1, 1>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride, 1);
}
__global__ void FSQ_BOUNDS flash_splitq5_f16_hd128_facc(
    const __half* q, const __half* k, const __half* v, __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride) {
  flash_splitq_impl<__half, 128, 64, 64, 4, 1, 1>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride);
}

__global__ void FSQ_BOUNDS flash_splitq_bf16_hd128_win(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride, int window) {
  flash_splitq_impl<__nv_bfloat16, 128>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride, 0, window);
}
__global__ void FSQ_BOUNDS flash_splitq5_bf16_hd128_win(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride, int window) {
  flash_splitq_impl<__nv_bfloat16, 128, 64, 64, 4, 1>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride, 0, window);
}

__global__ void FSQ_BOUNDS flash_splitq_f16_hd128_win(
    const __half* q, const __half* k, const __half* v,
    __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride, int window) {
  flash_splitq_impl<__half, 128>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride, 0, window);
}
__global__ void FSQ_BOUNDS flash_splitq5_f16_hd128_win(
    const __half* q, const __half* k, const __half* v,
    __half* out, float scale,
    int B, int NH, int NKV, int Tq, int Tkv, int causal, int t_stride, int window) {
  flash_splitq_impl<__half, 128, 64, 64, 4, 1>(q, k, v, out, scale, B, NH, NKV, Tq, Tkv, causal, t_stride, 0, window);
}

}  // extern "C"
