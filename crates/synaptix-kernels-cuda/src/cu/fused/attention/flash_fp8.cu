#include <cuda_bf16.h>
#include <cuda_fp8.h>

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
__device__ __forceinline__ void fa8_mma_bf16(float* d, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
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
__device__ __forceinline__ unsigned fa8_bf16x2(float lo, float hi) {
  unsigned u;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(u) : "f"(hi), "f"(lo));
  return u;
}

// ── Вспомогательные ядра ──────────────────────────────────────────────────

// По каналу (b,h,d): MODE 0 — среднее по токенам, 1 — amax. Блок = 128 нитей
// (по d), грид (B·H, SPLIT): частичные результаты атомиками в out (обнулён /
// для amax — 0). Среднее делится на S в главном квант-ядре.
template <int MODE>
__device__ __forceinline__ void fa8_col_reduce_impl(const __nv_bfloat16* x, float* out, int S, int H) {
  int bh = blockIdx.x, d = threadIdx.x;
  int b = bh / H, h = bh % H;
  int split = gridDim.y, per = (S + split - 1) / split;
  int s0 = blockIdx.y * per, s1 = min(S, s0 + per);
  float acc = 0.f;
  const __nv_bfloat16* p = x + ((size_t)b * S * H + h) * FA8_D + d;
  for (int s = s0; s < s1; s++) {
    float v = __bfloat162float(p[(size_t)s * H * FA8_D]);
    if (MODE == 0) acc += v; else acc = fmaxf(acc, fabsf(v));
  }
  if (MODE == 0) atomicAdd(out + (size_t)bh * FA8_D + d, acc);
  else atomicMax(reinterpret_cast<int*>(out + (size_t)bh * FA8_D + d), __float_as_int(acc));
}
extern "C" __global__ void fa8_col_sum(const __nv_bfloat16* x, float* out, int S, int H) { fa8_col_reduce_impl<0>(x, out, S, H); }
extern "C" __global__ void fa8_col_amax(const __nv_bfloat16* x, float* out, int S, int H) { fa8_col_reduce_impl<1>(x, out, S, H); }

// Строки [B,S,H,128] → e4m3 той же раскладки + шкала [B,H,S] (·mul).
// sum != null — вычесть среднее канала (sum/S). Варп = строка.
extern "C" __global__ void fa8_quant_rows(const __nv_bfloat16* x, const float* sum, unsigned char* out,
                                          float* sc, int B, int S, int H, float mul) {
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  long long r = (long long)blockIdx.x * 8 + warp;
  if (r >= (long long)B * S * H) return;
  int h = (int)(r % H);
  int s = (int)((r / H) % S);
  int b = (int)(r / ((long long)S * H));
  const __nv_bfloat16* p = x + r * FA8_D + lane * 4;
  float v[4];
  uint2 raw = *reinterpret_cast<const uint2*>(p);
  __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162*>(&raw.x), hi = *reinterpret_cast<__nv_bfloat162*>(&raw.y);
  v[0] = __low2float(lo); v[1] = __high2float(lo); v[2] = __low2float(hi); v[3] = __high2float(hi);
  if (sum) {
    const float* m = sum + ((size_t)b * H + h) * FA8_D + lane * 4;
    float inv_s = 1.0f / (float)S;
    for (int i = 0; i < 4; i++) v[i] -= m[i] * inv_s;
  }
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  float scale = fmaxf(amax / 448.0f, 1e-12f), inv = 1.0f / scale;
  *reinterpret_cast<unsigned*>(out + r * FA8_D + lane * 4) = fa8_pack4(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv);
  if (lane == 0) sc[((size_t)b * H + h) * S + s] = scale * mul;
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
extern "C" __global__ void fa8_quant_vt(const __nv_bfloat16* v, const float* amax, unsigned char* vt,
                                        float* vsc, int S, int H, int Spad) {
  __shared__ __nv_bfloat16 tile[FA8_BN][FA8_D + 8];
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
  float scale = fmaxf(a / 448.0f, 1e-12f), inv = 1.0f / scale;
  if (blockIdx.x == 0) vsc[(size_t)bh * FA8_D + d] = scale / 448.0f;
  unsigned w[16];
  for (int k = 0; k < 16; k++) {
    float x[4];
    for (int i = 0; i < 4; i++) x[i] = __bfloat162float(tile[fa8_perm(k * 4 + i)][d]) * inv;
    w[k] = fa8_pack4(x[0], x[1], x[2], x[3]);
  }
  uint4* dst = reinterpret_cast<uint4*>(vt + ((size_t)bh * FA8_D + d) * Spad + s0);
  for (int k = 0; k < 4; k++) dst[k] = make_uint4(w[4 * k], w[4 * k + 1], w[4 * k + 2], w[4 * k + 3]);
}

// ── Главное ядро ──────────────────────────────────────────────────────────
template <int PV8>
__device__ __forceinline__ void fa8_main_impl(
    const unsigned char* __restrict__ qq, const float* __restrict__ qs,
    const unsigned char* __restrict__ kq, const float* __restrict__ ks,
    const void* __restrict__ vptr, const float* __restrict__ vsc,
    __nv_bfloat16* __restrict__ out, int B, int H, int Sq, int Skv, int Spad) {
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* sQ = smem;
  unsigned char* sK = smem + 8192;
  unsigned char* sV = smem + 8192 + 16384;
  const int VBUF = PV8 ? 8192 : 16384;

  int bh = blockIdx.y, b = bh / H, h = bh % H;
  int q0 = blockIdx.x * FA8_BM;
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  size_t rs = (size_t)H * FA8_D;
  const unsigned char* qbase = qq + ((size_t)b * Sq * H + h) * FA8_D;
  const unsigned char* kbase = kq + ((size_t)b * Skv * H + h) * FA8_D;

  for (int c = tid; c < 512; c += 128) {
    int r = c >> 3, ch = c & 7, s = q0 + r;
    fa8_cp16(fa8_smem(sQ + r * 128 + ((ch ^ (r & 7)) << 4)), qbase + (size_t)min(s, Sq - 1) * rs + ch * 16, s < Sq);
  }
  auto load_kv = [&](int j, int buf) {
    int k0 = j * FA8_BN;
    unsigned char* dk = sK + buf * 8192;
    for (int c = tid; c < 512; c += 128) {
      int r = c >> 3, ch = c & 7, s = k0 + r;
      fa8_cp16(fa8_smem(dk + r * 128 + ((ch ^ (r & 7)) << 4)), kbase + (size_t)min(s, Skv - 1) * rs + ch * 16, s < Skv);
    }
    unsigned char* dv = sV + buf * VBUF;
    if (PV8) {
      const unsigned char* vt = (const unsigned char*)vptr + (size_t)bh * FA8_D * Spad + k0;
      for (int c = tid; c < 512; c += 128) {
        int d = c >> 2, ch = c & 3;
        fa8_cp16(fa8_smem(dv + d * 64 + ((ch ^ ((d >> 1) & 3)) << 4)), vt + (size_t)d * Spad + ch * 16, true);
      }
    } else {
      const __nv_bfloat16* vb = (const __nv_bfloat16*)vptr + ((size_t)b * Skv * H + h) * FA8_D;
      for (int c = tid; c < 1024; c += 128) {
        int r = c >> 4, ch = c & 15, s = k0 + r;
        fa8_cp16(fa8_smem(dv + r * 256 + ((ch ^ (r & 7)) << 4)), vb + (size_t)min(s, Skv - 1) * rs + ch * 8, s < Skv);
      }
    }
  };
  load_kv(0, 0);
  fa8_commit();

  int row0 = q0 + warp * 16 + g, row1 = row0 + 8;
  float qsc0 = row0 < Sq ? qs[(size_t)bh * Sq + row0] : 0.f;
  float qsc1 = row1 < Sq ? qs[(size_t)bh * Sq + row1] : 0.f;

  unsigned qf[4][4];
  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = FA8_NEG_INF, m1 = FA8_NEG_INF, l0 = 0.f, l1 = 0.f;

  int nblk = (Skv + FA8_BN - 1) / FA8_BN;
  for (int j = 0; j < nblk; j++) {
    fa8_wait0();
    __syncthreads();
    if (j == 0) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        int r = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        int ch = k * 2 + (lane >> 4);
        fa8_ldsm4(qf[k][0], qf[k][1], qf[k][2], qf[k][3], fa8_smem(sQ + r * 128 + ((ch ^ (r & 7)) << 4)));
      }
    }
    if (j + 1 < nblk) {
      load_kv(j + 1, (j + 1) & 1);
      fa8_commit();
    }
    int buf = j & 1;
    const unsigned char* cK = sK + buf * 8192;
    const unsigned char* cV = sV + buf * VBUF;

    float s[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
      for (int kp = 0; kp < 2; kp++) {
        int r = nt * 8 + (lane & 7);
        int ch = kp * 4 + (lane >> 3);
        unsigned b0, b1, b2, b3;
        fa8_ldsm4(b0, b1, b2, b3, fa8_smem(cK + r * 128 + ((ch ^ (r & 7)) << 4)));
        fa8_mma_e4m3(s[nt], qf[kp * 2], b0, b1);
        fa8_mma_e4m3(s[nt], qf[kp * 2 + 1], b2, b3);
      }
    }
    // Шкалы ключей и маска хвоста.
    float mx0 = FA8_NEG_INF, mx1 = FA8_NEG_INF;
    int kb = j * FA8_BN;
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
#pragma unroll
      for (int e = 0; e < 2; e++) {
        int col = kb + nt * 8 + t * 2 + e;
        if (col < Skv) {
          float kv = ks[(size_t)bh * Skv + col];
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
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      s[nt][0] = fa8_exp2(s[nt][0] - sh0);
      s[nt][1] = fa8_exp2(s[nt][1] - sh0);
      s[nt][2] = fa8_exp2(s[nt][2] - sh1);
      s[nt][3] = fa8_exp2(s[nt][3] - sh1);
      l0 += s[nt][0] + s[nt][1];
      l1 += s[nt][2] + s[nt][3];
    }
#pragma unroll
    for (int dt = 0; dt < 16; dt++) {
      o[dt][0] *= al0;
      o[dt][1] *= al0;
      o[dt][2] *= al1;
      o[dt][3] *= al1;
    }

    if (PV8) {
      unsigned pa[2][4];
#pragma unroll
      for (int c = 0; c < 2; c++) {
        const float* t0 = s[c * 4], *t1 = s[c * 4 + 1], *t2 = s[c * 4 + 2], *t3 = s[c * 4 + 3];
        pa[c][0] = fa8_pack4(t0[0] * 448.f, t0[1] * 448.f, t1[0] * 448.f, t1[1] * 448.f);
        pa[c][1] = fa8_pack4(t0[2] * 448.f, t0[3] * 448.f, t1[2] * 448.f, t1[3] * 448.f);
        pa[c][2] = fa8_pack4(t2[0] * 448.f, t2[1] * 448.f, t3[0] * 448.f, t3[1] * 448.f);
        pa[c][3] = fa8_pack4(t2[2] * 448.f, t2[3] * 448.f, t3[2] * 448.f, t3[3] * 448.f);
      }
#pragma unroll
      for (int dt = 0; dt < 16; dt++) {
        int d = dt * 8 + (lane & 7);
        int ch = lane >> 3;
        unsigned b0, b1, b2, b3;
        fa8_ldsm4(b0, b1, b2, b3, fa8_smem(cV + d * 64 + ((ch ^ ((d >> 1) & 3)) << 4)));
        float ob[4] = {0.f, 0.f, 0.f, 0.f};
        fa8_mma_e4m3(ob, pa[0], b0, b1);
        fa8_mma_e4m3(ob, pa[1], b2, b3);
        o[dt][0] += ob[0];
        o[dt][1] += ob[1];
        o[dt][2] += ob[2];
        o[dt][3] += ob[3];
      }
    } else {
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        unsigned pa[4];
        pa[0] = fa8_bf16x2(s[2 * kk][0], s[2 * kk][1]);
        pa[1] = fa8_bf16x2(s[2 * kk][2], s[2 * kk][3]);
        pa[2] = fa8_bf16x2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
        pa[3] = fa8_bf16x2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
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
  __nv_bfloat16* ob0 = out + (((size_t)b * Sq + row0) * H + h) * FA8_D;
  __nv_bfloat16* ob1 = out + (((size_t)b * Sq + row1) * H + h) * FA8_D;
#pragma unroll
  for (int dt = 0; dt < 16; dt++) {
    int d = dt * 8 + t * 2;
    float v0 = 1.f, v1 = 1.f;
    if (PV8) {
      v0 = vsc[(size_t)bh * FA8_D + d];
      v1 = vsc[(size_t)bh * FA8_D + d + 1];
    }
    if (row0 < Sq) *reinterpret_cast<unsigned*>(ob0 + d) = fa8_bf16x2(o[dt][0] * il0 * v0, o[dt][1] * il0 * v1);
    if (row1 < Sq) *reinterpret_cast<unsigned*>(ob1 + d) = fa8_bf16x2(o[dt][2] * il1 * v0, o[dt][3] * il1 * v1);
  }
}

extern "C" __global__ void __launch_bounds__(128) flash_fp8_qk_hd128_bshd(
    const unsigned char* qq, const float* qs, const unsigned char* kq, const float* ks,
    const __nv_bfloat16* v, const float* vsc, __nv_bfloat16* out, int B, int H, int Sq, int Skv, int Spad) {
  fa8_main_impl<0>(qq, qs, kq, ks, v, vsc, out, B, H, Sq, Skv, Spad);
}
extern "C" __global__ void __launch_bounds__(128) flash_fp8_full_hd128_bshd(
    const unsigned char* qq, const float* qs, const unsigned char* kq, const float* ks,
    const unsigned char* vt, const float* vsc, __nv_bfloat16* out, int B, int H, int Sq, int Skv, int Spad) {
  fa8_main_impl<1>(qq, qs, kq, ks, vt, vsc, out, B, H, Sq, Skv, Spad);
}
