#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// Тип входа/выхода: bf16 (по умолчанию) или f16 (-DFA8_F16, FLUX.1 и др.).
#ifdef FA8_F16
typedef __half fa8_t;
typedef __half2 fa8_t2;
#define FA8_MMA_T "f16"
#define FA8_CVT2 "cvt.rn.f16x2.f32"
__device__ __forceinline__ float fa8_tof(fa8_t x) { return __half2float(x); }
#else
typedef __nv_bfloat16 fa8_t;
typedef __nv_bfloat162 fa8_t2;
#define FA8_MMA_T "bf16"
#define FA8_CVT2 "cvt.rn.bf16x2.f32"
__device__ __forceinline__ float fa8_tof(fa8_t x) { return __bfloat162float(x); }
#endif

// FP8-внимание (sm_89+/sm_120): softmax(scale·Q·Kᵀ)·V, раскладка BSHD
// [B,S,H,128], без маски (DiT: видео/картинки). Схема — FA-2 split-Q
// (как flash_splitq): блок = 64 строки Q (4 варпа × 16), ключи блоками по 64,
// online softmax в регистрах (exp2-домен).
//
// • QKᵀ — mma.sync.m16n8k32 e4m3×e4m3→f32. Q и K квантуются по токену
//   (шкала = amax/448); у K вычитается среднее по токенам (softmax инвариантен
//   к сдвигу ключей на вектор: Q·(K−k̄) = Q·K − Q·k̄, второе — константа строки),
//   что убирает общий канальный выброс K и бережёт мантиссу e4m3.
//   Шкала Q уже содержит scale·log2e.
// • P·V — PV8=0: bf16 mma m16n8k16 (P из аккумуляторов S, V bf16 как есть);
//   PV8=1: e4m3 mma m16n8k32: P·448 → e4m3, V — по каналу (amax по токенам),
//   транспонирована [B,H,128,Spad] с перестановкой ключей в группах по 16 так,
//   что C-фрагменты S ложатся A-фрагментами без обменов (порядок суммы по
//   ключам не важен). Вклад блока копится в отдельном f32-аккумуляторе.
//
// Вспомогательные ядра: среднее/amax по токенам на канал, квант строк Q/K,
// транспонирующий квант V.

#define FA8_BM 64
#define FA8_BN 64
#define FA8_D 128
#define FA8_NEG_INF (__int_as_float(0xff800000))
#define FA8_LOG2E 1.4426950408889634f

__device__ __forceinline__ unsigned fa8_smem(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void fa8_cp16(unsigned dst, const void* src, bool pred) {
  int n = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void fa8_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void fa8_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }
__device__ __forceinline__ void fa8_ldsm4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void fa8_ldsm4t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void fa8_mma_e4m3(float* d, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// e4m3 с f16-накоплением: на sm_120 вдвое быстрее f32 (473 против 238 TFLOP/s).
// d[0] — строка g (столбцы 2t, 2t+1), d[1] — строка g+8.
__device__ __forceinline__ void fa8_mma_e4m3_h(unsigned* d, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float2 fa8_h2f(unsigned u) {
  return __half22float2(*reinterpret_cast<__half2*>(&u));
}
__device__ __forceinline__ void fa8_mma_bf16(float* d, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32." FA8_MMA_T "." FA8_MMA_T ".f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float fa8_exp2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// Два f32 → e4m3x2 (lo — младший байт).
__device__ __forceinline__ unsigned short fa8_e4m3x2(float lo, float hi) {
  unsigned short r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ unsigned fa8_pack4(float x0, float x1, float x2, float x3) {
  return (unsigned)fa8_e4m3x2(x0, x1) | ((unsigned)fa8_e4m3x2(x2, x3) << 16);
}
__device__ __forceinline__ unsigned fa8_pack2(float lo, float hi) {
  unsigned u;
  asm(FA8_CVT2 " %0, %1, %2;" : "=r"(u) : "f"(hi), "f"(lo));
  return u;
}

// ── Вспомогательные ядра ──────────────────────────────────────────────────

// По каналу (b,h,d): MODE 0 — среднее по токенам, 1 — amax. Блок = 128 нитей
// (по d), грид (B·H, SPLIT): частичные результаты атомиками в out (обнулён /
// для amax — 0). Среднее делится на S в главном квант-ядре.
template <int MODE>
__device__ __forceinline__ void fa8_col_reduce_impl(const fa8_t* x, float* out, int S, int H) {
  int bh = blockIdx.x, d = threadIdx.x;
  int b = bh / H, h = bh % H;
  int split = gridDim.y, per = (S + split - 1) / split;
  int s0 = blockIdx.y * per, s1 = min(S, s0 + per);
  float acc = 0.f;
  const fa8_t* p = x + ((size_t)b * S * H + h) * FA8_D + d;
  for (int s = s0; s < s1; s++) {
    float v = fa8_tof(p[(size_t)s * H * FA8_D]);
    if (MODE == 0) acc += v; else acc = fmaxf(acc, fabsf(v));
  }
  // Сумма — частичная на сплит (детерминизм: складываются в кванте по порядку);
  // amax — atomicMax (max порядконезависим).
  if (MODE == 0) out[((size_t)blockIdx.y * gridDim.x + bh) * FA8_D + d] = acc;
  else atomicMax(reinterpret_cast<int*>(out + (size_t)bh * FA8_D + d), __float_as_int(acc));
}
extern "C" __global__ void fa8_col_sum(const fa8_t* x, float* out, int S, int H) { fa8_col_reduce_impl<0>(x, out, S, H); }
extern "C" __global__ void fa8_col_amax(const fa8_t* x, float* out, int S, int H) { fa8_col_reduce_impl<1>(x, out, S, H); }

// Частичные суммы сплитов → сумма канала [B·H,128], сложенная по порядку
// сплитов — как раньше в каждой нити кванта, но один раз на канал, а не на
// каждую строку (результат бит в бит).
extern "C" __global__ void fa8_col_total(const float* sum, float* total, int BH, int nsplit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= BH * FA8_D) return;
  float t = 0.f;
  for (int k = 0; k < nsplit; k++) t += sum[(size_t)k * BH * FA8_D + i];
  total[i] = t;
}

// Строки [B,S,H,128] → e4m3 той же раскладки + шкала [B,H,S] (·mul).
// total != null — вычесть среднее канала (сумма [B·H,128] / S). Варп = строка.
extern "C" __global__ void fa8_quant_rows(const fa8_t* x, const float* total, unsigned char* out,
                                          float* sc, int B, int S, int H, float mul, float qmax, int sstride,
                                          signed char* ke) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  long long r = (long long)blockIdx.x * 8 + warp;
  if (r >= (long long)B * S * H) return;
  int h = (int)(r % H);
  int s = (int)((r / H) % S);
  int b = (int)(r / ((long long)S * H));
  const fa8_t* p = x + r * FA8_D + lane * 4;
  float v[4];
  uint2 raw = *reinterpret_cast<const uint2*>(p);
  fa8_t2 lo = *reinterpret_cast<fa8_t2*>(&raw.x), hi = *reinterpret_cast<fa8_t2*>(&raw.y);
  v[0] = __low2float(lo); v[1] = __high2float(lo); v[2] = __low2float(hi); v[3] = __high2float(hi);
  if (total) {
    float inv_s = 1.0f / (float)S;
    float4 t = *reinterpret_cast<const float4*>(total + ((size_t)b * H + h) * FA8_D + lane * 4);
    v[0] -= t.x * inv_s; v[1] -= t.y * inv_s; v[2] -= t.z * inv_s; v[3] -= t.w * inv_s;
  }
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  float scale = fmaxf(amax / qmax, 1e-12f), inv = 1.0f / scale;
  int e2 = 0;
  if (ke) {
    // Шкала — степень двойки (байт экспоненты): |x| ≤ qmax сохраняется,
    // умножение на шкалу в ядре точное, а буфер шкал в smem вчетверо меньше.
    e2 = max(-126, min(126, (int)ceilf(log2f(scale))));
    scale = __int_as_float((e2 + 127) << 23);
    inv = __int_as_float((127 - e2) << 23);
  }
  *reinterpret_cast<unsigned*>(out + r * FA8_D + lane * 4) = fa8_pack4(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv);
  if (lane == 0) {
    sc[((size_t)b * H + h) * sstride + s] = scale * mul;
    if (ke) ke[((size_t)b * H + h) * sstride + s] = (signed char)e2;
  }
}

// Ключ на позиции p (0..63) блока для PV8: в группе 16 позиция q = t*4+i ↔
// ключ (i<2 ? t*2+i : 8+t*2+i-2) — так C-фрагменты двух n8-тайлов S дают
// A-фрагмент m16n8k32 (4 соседних k на нить) без обменов.
__device__ __forceinline__ int fa8_perm(int p) {
  int g = p >> 4, q = p & 15, t = q >> 2, i = q & 3;
  return g * 16 + (i < 2 ? t * 2 + i : 8 + t * 2 + (i - 2));
}

// V [B,S,H,128] bf16 → V̂ᵀ [B,H,128,Spad] e4m3 (ключи переставлены fa8_perm),
// шкала канала = amax/448; vsc[b,h,d] = шкала/448 (масштаб P·448 учтён).
// Грид (Spad/64, B·H), блок 128 (по d).
extern "C" __global__ void fa8_quant_vt(const fa8_t* v, const float* amax, unsigned char* vt,
                                        float* vsc, int S, int H, int Spad, float vq, float ps) {
  __shared__ fa8_t tile[FA8_BN][FA8_D + 8];
  int bh = blockIdx.y, b = bh / H, h = bh % H, d = threadIdx.x;
  int s0 = blockIdx.x * FA8_BN;
  for (int c = threadIdx.x; c < FA8_BN * 16; c += 128) {
    int r = c >> 4, ch = c & 15, s = s0 + r;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (s < S) val = *reinterpret_cast<const uint4*>(v + (((size_t)b * S + s) * H + h) * FA8_D + ch * 8);
    *reinterpret_cast<uint4*>(&tile[r][ch * 8]) = val;
  }
  __syncthreads();
  float a = amax[(size_t)bh * FA8_D + d];
  float scale = fmaxf(a / vq, 1e-12f), inv = 1.0f / scale;
  if (blockIdx.x == 0) vsc[(size_t)bh * FA8_D + d] = scale / ps;
  unsigned w[16];
  for (int k = 0; k < 16; k++) {
    float x[4];
    for (int i = 0; i < 4; i++) x[i] = fa8_tof(tile[fa8_perm(k * 4 + i)][d]) * inv;
    w[k] = fa8_pack4(x[0], x[1], x[2], x[3]);
  }
  uint4* dst = reinterpret_cast<uint4*>(vt + ((size_t)bh * FA8_D + d) * Spad + s0);
  for (int k = 0; k < 4; k++) dst[k] = make_uint4(w[4 * k], w[4 * k + 1], w[4 * k + 2], w[4 * k + 3]);
}

// ── Блочный роутинг для разреженного внимания (Sol-стиль, блоки 64×64) ───

// Средний вектор 64-токенного блока: [B, S, H, 128] → [B·H, n_blk, 128] f32.
// Грид (n_blk, B·H), блок 128 (по d).
extern "C" __global__ void fa8_pool_blocks(const fa8_t* x, float* pooled, int S, int H, int n_blk) {
  int blk = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
  int b = bh / H, h = bh % H;
  int s0 = blk * FA8_BN;
  int s1 = min(S, s0 + FA8_BN);
  int cnt = s1 - s0;
  float acc = 0.f;
  const fa8_t* p = x + (((size_t)b * S + s0) * H + h) * FA8_D + d;
  for (int i = 0; i < cnt; i++) {
    acc += fa8_tof(p[(size_t)i * H * FA8_D]);
  }
  pooled[((size_t)bh * n_blk + blk) * FA8_D + d] = cnt > 0 ? acc / (float)cnt : 0.f;
}

// Выбор KV-блоков для каждого Q-блока и головы:
//  - Q в [0..prot_prefix) → плотно (все KV-блоки);
//  - KV в [0..prot_prefix) (Text/Cond/Refs/Audio) → всегда;
//  - локальное окно |k_center - q_center| ≤ local_win → всегда;
//  - остальные дальние видео-блоки → top-k по dot(q_pool, k_pool) с долей keep_ratio.
// Грид (n_qblk, B·H), блок 256, дин. smem = n_kblk*4 + n_kblk + 2048 Б.
extern "C" __global__ void fa8_sparse_route(
    const float* __restrict__ q_pool, const float* __restrict__ k_pool,
    unsigned short* __restrict__ blk_idx, int* __restrict__ blk_cnt,
    int n_qblk, int n_kblk, int q_offset, int prot_prefix, int local_win, float keep_ratio) {
  extern __shared__ unsigned char rsmem[];
  float* s_q = reinterpret_cast<float*>(rsmem);                    // 128 floats = 512 B
  float* s_red = reinterpret_cast<float*>(rsmem + 512);            // 16 floats (8 min + 8 max) = 64 B
  int* s_ired = reinterpret_cast<int*>(rsmem + 576);               // 8 ints = 32 B
  float* scores = reinterpret_cast<float*>(rsmem + 640);           // n_kblk floats
  unsigned char* flags = rsmem + 640 + n_kblk * 4;                 // n_kblk bytes

  int qb = blockIdx.x, bh = blockIdx.y, tid = threadIdx.x;
  int warp = tid >> 5, lane = tid & 31;
  unsigned short* out_idx = blk_idx + ((size_t)bh * n_qblk + qb) * n_kblk;
  int q_start = q_offset + qb * FA8_BM;
  int q_center = q_start + FA8_BM / 2;

  // Защищённый префикс (Text/Cond/Refs/Audio): запрос смотрит на все KV-блоки.
  if (q_start < prot_prefix || keep_ratio >= 0.999f) {
    for (int kb = tid; kb < n_kblk; kb += 256) out_idx[kb] = (unsigned short)kb;
    if (tid == 0) blk_cnt[(size_t)bh * n_qblk + qb] = n_kblk;
    return;
  }

  if (tid < FA8_D) s_q[tid] = q_pool[((size_t)bh * n_qblk + qb) * FA8_D + tid];
  __syncthreads();

  int prot_kblks = (prot_prefix + FA8_BN - 1) / FA8_BN;
  const float* kp_base = k_pool + (size_t)bh * n_kblk * FA8_D;

  // Коалесцированный расчёт dot(q_pool, k_pool[kb]): один варп (32 нити × float4 = 128)
  // обрабатывает один kb за один 128-байтный проход по глобальной памяти.
  const float4 q4 = *reinterpret_cast<const float4*>(s_q + lane * 4);
  float lmin = 1e30f, lmax = -1e30f;
  int lrem = 0;
  for (int kb = warp; kb < n_kblk; kb += 8) {
    int k_center = kb * FA8_BN + FA8_BN / 2;
    bool mand = (kb < prot_kblks) || (abs(k_center - q_center) <= local_win);
    if (!mand) {
      float4 k4 = *reinterpret_cast<const float4*>(kp_base + (size_t)kb * FA8_D + lane * 4);
      float dot = q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
      for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
      if (lane == 0) {
        flags[kb] = 0;
        scores[kb] = dot;
        lmin = fminf(lmin, dot);
        lmax = fmaxf(lmax, dot);
        lrem++;
      }
    } else if (lane == 0) {
      flags[kb] = 1;
      scores[kb] = -1e30f;
    }
  }
  // Редукция min, max, count по 8 варпам (значения у lane 0 каждого варпа).
  if (lane == 0) {
    s_red[warp] = lmin;
    s_red[8 + warp] = lmax;
    s_ired[warp] = lrem;
  }
  __syncthreads();
  float gmin = s_red[0], gmax = s_red[8];
  int total_rem = 0;
  if (warp == 0) {
    float vmin = (lane < 8) ? s_red[lane] : 1e30f;
    float vmax = (lane < 8) ? s_red[8 + lane] : -1e30f;
    int vrem = (lane < 8) ? s_ired[lane] : 0;
    for (int o = 4; o > 0; o >>= 1) {
      vmin = fminf(vmin, __shfl_xor_sync(0xffffffffu, vmin, o));
      vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
      vrem += __shfl_xor_sync(0xffffffffu, vrem, o);
    }
    if (lane == 0) {
      s_red[0] = vmin;
      s_red[1] = vmax;
      s_ired[0] = vrem;
    }
  }
  __syncthreads();
  gmin = s_red[0];
  gmax = s_red[1];
  total_rem = s_ired[0];

  int target_rem = max(1, (int)ceilf((float)total_rem * keep_ratio));
  if (total_rem == 0 || target_rem >= total_rem) {
    for (int kb = tid; kb < n_kblk; kb += 256) out_idx[kb] = (unsigned short)kb;
    if (tid == 0) blk_cnt[(size_t)bh * n_qblk + qb] = n_kblk;
    return;
  }

  // 10 шагов бинарного поиска порога tau по [gmin, gmax].
  float lo = gmin, hi = gmax;
  for (int it = 0; it < 10; it++) {
    float mid = 0.5f * (lo + hi);
    int c = 0;
    for (int kb = tid; kb < n_kblk; kb += 256) {
      if (flags[kb] == 0 && scores[kb] >= mid) c++;
    }
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
    if (lane == 0) s_ired[warp] = c;
    __syncthreads();
    if (warp == 0) {
      int vc = (lane < 8) ? s_ired[lane] : 0;
      for (int o = 4; o > 0; o >>= 1) vc += __shfl_xor_sync(0xffffffffu, vc, o);
      if (lane == 0) s_ired[0] = vc;
    }
    __syncthreads();
    int ge = s_ired[0];
    if (ge > target_rem) lo = mid; else hi = mid;
  }
  float tau = 0.5f * (lo + hi);
  // Разметка: 1 — обязательный, 2 — строго выше порога, 3 — на границе.
  int strict_cnt = 0;
  for (int kb = tid; kb < n_kblk; kb += 256) {
    if (flags[kb] == 0) {
      if (scores[kb] > tau) {
        flags[kb] = 2;
        strict_cnt++;
      } else if (scores[kb] >= lo) {
        flags[kb] = 3;
      }
    }
  }
  for (int o = 16; o > 0; o >>= 1) strict_cnt += __shfl_xor_sync(0xffffffffu, strict_cnt, o);
  if (lane == 0) s_ired[warp] = strict_cnt;
  __syncthreads();
  if (warp == 0) {
    int vc = (lane < 8) ? s_ired[lane] : 0;
    for (int o = 4; o > 0; o >>= 1) vc += __shfl_xor_sync(0xffffffffu, vc, o);
    if (lane == 0) s_ired[0] = vc;
  }
  __syncthreads();
  // Детерминированный проход в порядке возрастания kb (n_kblk ≤ 4096 → ~1.5 мкс).
  if (tid == 0) {
    int tie_budget = max(0, target_rem - s_ired[0]);
    int out_n = 0;
    for (int kb = 0; kb < n_kblk; kb++) {
      unsigned char f = flags[kb];
      if (f == 1 || f == 2) {
        out_idx[out_n++] = (unsigned short)kb;
      } else if (f == 3 && tie_budget > 0) {
        out_idx[out_n++] = (unsigned short)kb;
        tie_budget--;
      }
    }
    blk_cnt[(size_t)bh * n_qblk + qb] = out_n;
  }
}

// ── Главное ядро ──────────────────────────────────────────────────────────
// ACC16: обе свёртки — e4m3 с f16-накоплением. Чтобы f16 (≤ 65504) не
// переполнился: Q и K квантуются в |x| ≤ 16 (128 произведений ≤ 32768), P —
// ·256, V — в |x| ≤ 3.9 (64 ключа плитки ≤ 63898); вклад плитки P·V копится в
// f32, как и прежде. SPARSE=1: цикл идёт только по списку индексов blk_idx.
template <int PV8, int ACC16, int SPARSE>
__device__ __forceinline__ void fa8_main_impl(
    const unsigned char* __restrict__ qq, const float* __restrict__ qs,
    const unsigned char* __restrict__ kq, const float* __restrict__ ks,
    const void* __restrict__ vptr, const float* __restrict__ vsc,
    fa8_t* __restrict__ out, int B, int H, int Sq, int Skv, int Spad,
    const unsigned short* __restrict__ blk_idx, const int* __restrict__ blk_cnt, int n_kblk_stride) {
  extern __shared__ __align__(128) unsigned char smem[];
  // PV8: K 2×8К | V 2×8К | шкалы K 2×256 Б; Q (8К) живёт во втором буфере V
  // до первой подкачки — 33 КБ на блок, три блока на SM (было 41 КБ и два).
  // Шкалы ключей приезжают с плиткой K (раньше — ld.global в горячем цикле).
  unsigned char* sK = PV8 ? smem : smem + 8192;
  unsigned char* sV = PV8 ? smem + 16384 : smem + 8192 + 16384;
  unsigned char* sQ = PV8 ? sV + 8192 : smem;
  float* sKs = reinterpret_cast<float*>(smem + 32768);
  // ACC16: экспоненты шкал K (байт на ключ), 2×64 Б — блок 32.9 КБ, три на SM.
  const signed char* sKe = reinterpret_cast<const signed char*>(smem + 32768);
  const int VBUF = PV8 ? 8192 : 16384;

  int bh = blockIdx.y, b = bh / H, h = bh % H;
  int q0 = blockIdx.x * FA8_BM;
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  size_t rs = (size_t)H * FA8_D;
  const unsigned char* qbase = qq + ((size_t)b * Sq * H + h) * FA8_D;
  const unsigned char* kbase = kq + ((size_t)b * Skv * H + h) * FA8_D;

  int n_qblk = (Sq + FA8_BM - 1) / FA8_BM;
  const unsigned short* idx_row = SPARSE ? (blk_idx + ((size_t)bh * n_qblk + blockIdx.x) * n_kblk_stride) : nullptr;
  int nblk = SPARSE ? blk_cnt[(size_t)bh * n_qblk + blockIdx.x] : ((Skv + FA8_BN - 1) / FA8_BN);
  if (nblk <= 0) return;

  for (int c = tid; c < 512; c += 128) {
    int r = c >> 3, ch = c & 7, s = q0 + r;
    fa8_cp16(fa8_smem(sQ + r * 128 + ((ch ^ (r & 7)) << 4)), qbase + (size_t)min(s, Sq - 1) * rs + ch * 16, s < Sq);
  }
  auto load_kv = [&](int kblk_id, int buf) {
    int k0 = kblk_id * FA8_BN;
    unsigned char* dk = sK + buf * 8192;
    for (int c = tid; c < 512; c += 128) {
      int r = c >> 3, ch = c & 7, s = k0 + r;
      fa8_cp16(fa8_smem(dk + r * 128 + ((ch ^ (r & 7)) << 4)), kbase + (size_t)min(s, Skv - 1) * rs + ch * 16, s < Skv);
    }
    unsigned char* dv = sV + buf * VBUF;
    if (ACC16) {
      if (tid < 4)
        fa8_cp16(fa8_smem(smem + 32768 + buf * 64 + tid * 16),
                 reinterpret_cast<const unsigned char*>(ks) + (size_t)bh * Spad + k0 + tid * 16, true);
    } else if (PV8 && tid < 16) {
      // шкалы K: строка bh шагом Spad (кратно 64) — 16 копий по 16 Б
      fa8_cp16(fa8_smem(sKs + buf * 64 + tid * 4), ks + (size_t)bh * Spad + k0 + tid * 4, true);
    }
    if (PV8) {
      const unsigned char* vt = (const unsigned char*)vptr + (size_t)bh * FA8_D * Spad + k0;
      for (int c = tid; c < 512; c += 128) {
        int d = c >> 2, ch = c & 3;
        fa8_cp16(fa8_smem(dv + d * 64 + ((ch ^ ((d >> 1) & 3)) << 4)), vt + (size_t)d * Spad + ch * 16, true);
      }
    } else {
      const fa8_t* vb = (const fa8_t*)vptr + ((size_t)b * Skv * H + h) * FA8_D;
      for (int c = tid; c < 1024; c += 128) {
        int r = c >> 4, ch = c & 15, s = k0 + r;
        fa8_cp16(fa8_smem(dv + r * 256 + ((ch ^ (r & 7)) << 4)), vb + (size_t)min(s, Skv - 1) * rs + ch * 8, s < Skv);
      }
    }
  };
  int cur_kblk = SPARSE ? (int)idx_row[0] : 0;
  load_kv(cur_kblk, 0);
  fa8_commit();

  int row0 = q0 + warp * 16 + g, row1 = row0 + 8;
  float qsc0 = row0 < Sq ? qs[(size_t)bh * Sq + row0] : 0.f;
  float qsc1 = row1 < Sq ? qs[(size_t)bh * Sq + row1] : 0.f;

  unsigned qf[4][4];
  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = FA8_NEG_INF, m1 = FA8_NEG_INF, l0 = 0.f, l1 = 0.f;

  for (int j = 0; j < nblk; j++) {
    int kblk_id = cur_kblk;
    fa8_wait0();
    __syncthreads();
    if (j == 0) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        int r = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        int ch = k * 2 + (lane >> 4);
        fa8_ldsm4(qf[k][0], qf[k][1], qf[k][2], qf[k][3], fa8_smem(sQ + r * 128 + ((ch ^ (r & 7)) << 4)));
      }
      // Q лежит в буфере V №1, куда сейчас поедет следующая плитка.
      if (PV8) __syncthreads();
    }
    if (j + 1 < nblk) {
      cur_kblk = SPARSE ? (int)idx_row[j + 1] : (j + 1);
      load_kv(cur_kblk, (j + 1) & 1);
      fa8_commit();
    }
    int buf = j & 1;
    const unsigned char* cK = sK + buf * 8192;
    const unsigned char* cV = sV + buf * VBUF;

    float s[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
      unsigned sh[2] = {0u, 0u};
#pragma unroll
      for (int kp = 0; kp < 2; kp++) {
        int r = nt * 8 + (lane & 7);
        int ch = kp * 4 + (lane >> 3);
        unsigned b0, b1, b2, b3;
        fa8_ldsm4(b0, b1, b2, b3, fa8_smem(cK + r * 128 + ((ch ^ (r & 7)) << 4)));
        if (ACC16) {
          fa8_mma_e4m3_h(sh, qf[kp * 2], b0, b1);
          fa8_mma_e4m3_h(sh, qf[kp * 2 + 1], b2, b3);
        } else {
          fa8_mma_e4m3(s[nt], qf[kp * 2], b0, b1);
          fa8_mma_e4m3(s[nt], qf[kp * 2 + 1], b2, b3);
        }
      }
      if (ACC16) {
        float2 a = fa8_h2f(sh[0]), c = fa8_h2f(sh[1]);
        s[nt][0] = a.x; s[nt][1] = a.y; s[nt][2] = c.x; s[nt][3] = c.y;
      }
    }
    // Шкалы ключей и маска хвоста.
    float mx0 = FA8_NEG_INF, mx1 = FA8_NEG_INF;
    int kb = kblk_id * FA8_BN;
    if (ACC16) {
      // s = acc·k̂ (шкала Q > 0 выносится: max(s·q̂) = q̂·max(s)); q̂ входит в
      // FFMA показателя, множитель P (·256) — в его сдвиг (+8 в log2).
#pragma unroll
      for (int nt = 0; nt < 8; nt++) {
#pragma unroll
        for (int e = 0; e < 2; e++) {
          int col = kb + nt * 8 + t * 2 + e;
          if (col < Skv) {
            float kv = __int_as_float(((int)sKe[buf * 64 + nt * 8 + t * 2 + e] + 127) << 23);
            s[nt][e] *= kv;
            s[nt][2 + e] *= kv;
          } else {
            s[nt][e] = FA8_NEG_INF;
            s[nt][2 + e] = FA8_NEG_INF;
          }
          mx0 = fmaxf(mx0, s[nt][e]);
          mx1 = fmaxf(mx1, s[nt][2 + e]);
        }
      }
      mx0 *= qsc0;
      mx1 *= qsc1;
    } else {
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
#pragma unroll
      for (int e = 0; e < 2; e++) {
        int col = kb + nt * 8 + t * 2 + e;
        if (col < Skv) {
          float kv = PV8 ? sKs[buf * 64 + nt * 8 + t * 2 + e] : ks[(size_t)bh * Skv + col];
          s[nt][e] *= qsc0 * kv;
          s[nt][2 + e] *= qsc1 * kv;
        } else {
          s[nt][e] = FA8_NEG_INF;
          s[nt][2 + e] = FA8_NEG_INF;
        }
        mx0 = fmaxf(mx0, s[nt][e]);
        mx1 = fmaxf(mx1, s[nt][2 + e]);
      }
    }
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    float sh0 = (mn0 == FA8_NEG_INF) ? 0.f : mn0, sh1 = (mn1 == FA8_NEG_INF) ? 0.f : mn1;
    float al0 = fa8_exp2(m0 - sh0), al1 = fa8_exp2(m1 - sh1);
    m0 = mn0;
    m1 = mn1;
    l0 *= al0;
    l1 *= al1;
    if (ACC16) {
      // p·256 = 2^(s·q̂ − m + 8); l копится в тех же единицах.
      float o0 = 8.f - sh0, o1 = 8.f - sh1;
#pragma unroll
      for (int nt = 0; nt < 8; nt++) {
        s[nt][0] = fa8_exp2(fmaf(s[nt][0], qsc0, o0));
        s[nt][1] = fa8_exp2(fmaf(s[nt][1], qsc0, o0));
        s[nt][2] = fa8_exp2(fmaf(s[nt][2], qsc1, o1));
        s[nt][3] = fa8_exp2(fmaf(s[nt][3], qsc1, o1));
        l0 += s[nt][0] + s[nt][1];
        l1 += s[nt][2] + s[nt][3];
      }
    } else {
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      s[nt][0] = fa8_exp2(s[nt][0] - sh0);
      s[nt][1] = fa8_exp2(s[nt][1] - sh0);
      s[nt][2] = fa8_exp2(s[nt][2] - sh1);
      s[nt][3] = fa8_exp2(s[nt][3] - sh1);
      l0 += s[nt][0] + s[nt][1];
      l1 += s[nt][2] + s[nt][3];
    }
    }
    // Максимум строки обычно уже устоялся — умножение на 1 пропускается
    // (побитово то же).
    if (__any_sync(0xffffffffu, al0 != 1.f || al1 != 1.f)) {
#pragma unroll
    for (int dt = 0; dt < 16; dt++) {
      o[dt][0] *= al0;
      o[dt][1] *= al0;
      o[dt][2] *= al1;
      o[dt][3] *= al1;
    }
    }

    if (PV8) {
      unsigned pa[2][4];
      const float PS = ACC16 ? 1.f : 448.f;  // ACC16: ·256 уже в экспоненте
#pragma unroll
      for (int c = 0; c < 2; c++) {
        const float* t0 = s[c * 4], *t1 = s[c * 4 + 1], *t2 = s[c * 4 + 2], *t3 = s[c * 4 + 3];
        pa[c][0] = fa8_pack4(t0[0] * PS, t0[1] * PS, t1[0] * PS, t1[1] * PS);
        pa[c][1] = fa8_pack4(t0[2] * PS, t0[3] * PS, t1[2] * PS, t1[3] * PS);
        pa[c][2] = fa8_pack4(t2[0] * PS, t2[1] * PS, t3[0] * PS, t3[1] * PS);
        pa[c][3] = fa8_pack4(t2[2] * PS, t2[3] * PS, t3[2] * PS, t3[3] * PS);
      }
#pragma unroll
      for (int dt = 0; dt < 16; dt++) {
        int d = dt * 8 + (lane & 7);
        int ch = lane >> 3;
        unsigned b0, b1, b2, b3;
        fa8_ldsm4(b0, b1, b2, b3, fa8_smem(cV + d * 64 + ((ch ^ ((d >> 1) & 3)) << 4)));
        if (ACC16) {
          unsigned oh[2] = {0u, 0u};
          fa8_mma_e4m3_h(oh, pa[0], b0, b1);
          fa8_mma_e4m3_h(oh, pa[1], b2, b3);
          float2 a = fa8_h2f(oh[0]), c = fa8_h2f(oh[1]);
          o[dt][0] += a.x;
          o[dt][1] += a.y;
          o[dt][2] += c.x;
          o[dt][3] += c.y;
        } else {
          float ob[4] = {0.f, 0.f, 0.f, 0.f};
          fa8_mma_e4m3(ob, pa[0], b0, b1);
          fa8_mma_e4m3(ob, pa[1], b2, b3);
          o[dt][0] += ob[0];
          o[dt][1] += ob[1];
          o[dt][2] += ob[2];
          o[dt][3] += ob[3];
        }
      }
    } else {
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        unsigned pa[4];
        pa[0] = fa8_pack2(s[2 * kk][0], s[2 * kk][1]);
        pa[1] = fa8_pack2(s[2 * kk][2], s[2 * kk][3]);
        pa[2] = fa8_pack2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
        pa[3] = fa8_pack2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
        for (int dp = 0; dp < 8; dp++) {
          int r = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
          int ch = dp * 2 + (lane >> 4);
          unsigned b0, b1, b2, b3;
          fa8_ldsm4t(b0, b1, b2, b3, fa8_smem(cV + r * 256 + ((ch ^ (r & 7)) << 4)));
          fa8_mma_bf16(o[2 * dp], pa, b0, b1);
          fa8_mma_bf16(o[2 * dp + 1], pa, b2, b3);
        }
      }
    }
  }

  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  float il0 = l0 > 0.f ? 1.f / l0 : 0.f, il1 = l1 > 0.f ? 1.f / l1 : 0.f;
  fa8_t* ob0 = out + (((size_t)b * Sq + row0) * H + h) * FA8_D;
  fa8_t* ob1 = out + (((size_t)b * Sq + row1) * H + h) * FA8_D;
#pragma unroll
  for (int dt = 0; dt < 16; dt++) {
    int d = dt * 8 + t * 2;
    float v0 = 1.f, v1 = 1.f;
    if (PV8) {
      v0 = vsc[(size_t)bh * FA8_D + d];
      v1 = vsc[(size_t)bh * FA8_D + d + 1];
    }
    if (row0 < Sq) *reinterpret_cast<unsigned*>(ob0 + d) = fa8_pack2(o[dt][0] * il0 * v0, o[dt][1] * il0 * v1);
    if (row1 < Sq) *reinterpret_cast<unsigned*>(ob1 + d) = fa8_pack2(o[dt][2] * il1 * v0, o[dt][3] * il1 * v1);
  }
}

extern "C" __global__ void __launch_bounds__(128) flash_fp8_qk_hd128_bshd(
    const unsigned char* qq, const float* qs, const unsigned char* kq, const float* ks,
    const fa8_t* v, const float* vsc, fa8_t* out, int B, int H, int Sq, int Skv, int Spad) {
  fa8_main_impl<0, 0, 0>(qq, qs, kq, ks, v, vsc, out, B, H, Sq, Skv, Spad, nullptr, nullptr, 0);
}
extern "C" __global__ void __launch_bounds__(128) flash_fp8_full_hd128_bshd(
    const unsigned char* qq, const float* qs, const unsigned char* kq, const float* ks,
    const unsigned char* vt, const float* vsc, fa8_t* out, int B, int H, int Sq, int Skv, int Spad) {
  fa8_main_impl<1, 0, 0>(qq, qs, kq, ks, vt, vsc, out, B, H, Sq, Skv, Spad, nullptr, nullptr, 0);
}
extern "C" __global__ void __launch_bounds__(128) flash_fp8_a16_hd128_bshd(
    const unsigned char* qq, const float* qs, const unsigned char* kq, const float* ks,
    const unsigned char* vt, const float* vsc, fa8_t* out, int B, int H, int Sq, int Skv, int Spad) {
  fa8_main_impl<1, 1, 0>(qq, qs, kq, ks, vt, vsc, out, B, H, Sq, Skv, Spad, nullptr, nullptr, 0);
}
extern "C" __global__ void __launch_bounds__(128) flash_fp8_full_sparse_hd128_bshd(
    const unsigned char* qq, const float* qs, const unsigned char* kq, const float* ks,
    const unsigned char* vt, const float* vsc, fa8_t* out, int B, int H, int Sq, int Skv, int Spad,
    const unsigned short* blk_idx, const int* blk_cnt, int n_kblk_stride) {
  fa8_main_impl<1, 0, 1>(qq, qs, kq, ks, vt, vsc, out, B, H, Sq, Skv, Spad, blk_idx, blk_cnt, n_kblk_stride);
}
extern "C" __global__ void __launch_bounds__(128) flash_fp8_a16_sparse_hd128_bshd(
    const unsigned char* qq, const float* qs, const unsigned char* kq, const float* ks,
    const unsigned char* vt, const float* vsc, fa8_t* out, int B, int H, int Sq, int Skv, int Spad,
    const unsigned short* blk_idx, const int* blk_cnt, int n_kblk_stride) {
  fa8_main_impl<1, 1, 1>(qq, qs, kq, ks, vt, vsc, out, B, H, Sq, Skv, Spad, blk_idx, blk_cnt, n_kblk_stride);
}
