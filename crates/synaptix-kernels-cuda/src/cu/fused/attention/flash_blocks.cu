#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Attention по таблице блоков KV (разреженное внимание QSA).
//
// Каждый запрос смотрит на свой набор блоков по `ratio` подряд идущих позиций
// плюс хвост — токены после последнего полного блока. Раньше эти позиции
// собирались гатером в отдельный буфер, и KV проходил через память трижды
// (чтение исходного, запись собранного, чтение собранного ядром); здесь ядро
// читает KV прямо по таблице.
//
//   q     (B, NH, D)      row-major, по одному запросу на строку
//   k/v   (NKV, CAP, D)   общий KV-буфер слоя
//   table (B, NB)         индексы блоков
//   tail  (B) from + (B) len
//   out   (B, NH, D)
//
// Блок CUDA считает все `NH/NKV` q-голов одной kv-головы разом: KV тогда
// читается один раз на группу, а не по разу на голову. Позицию берёт варп
// целиком — строка K/V ложится на 32 лейна подряд, то есть читается
// коалесцированно; у каждого варпа свой онлайн-софтмакс, в конце они
// сливаются через shared.

#define FB_NEG_INF (__int_as_float(0xFF800000))
#define FB_MAX_REP 8
#define FB_MAX_VEC 8
#define FB_WARPS 4

__device__ __forceinline__ bool fb_is_finite(float x) {
  return (__float_as_int(x) & 0x7F800000) != 0x7F800000;
}

__device__ __forceinline__ float fb_ld(const float* p) { return *p; }
__device__ __forceinline__ float fb_ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float fb_ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void fb_st(float* p, float v) { *p = v; }
__device__ __forceinline__ void fb_st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void fb_st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

__device__ __forceinline__ float fb_warp_sum(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off, 32);
  return v;
}

// Позиция `t`-го элемента набора запроса: сперва выбранные блоки, потом хвост.
__device__ __forceinline__ int fb_pos(
    const unsigned int* __restrict__ table, int nb, int ratio, int tail_from, int t) {
  int in_blocks = nb * ratio;
  if (t < in_blocks) {
    return (int)table[t / ratio] * ratio + (t % ratio);
  }
  return tail_from + (t - in_blocks);
}

// VEC/REP как параметры шаблона: тогда циклы по ним разворачиваются, а q и
// аккумулятор остаются в регистрах. Динамический вариант (VEC=0) считает те же
// формулы, но с рантайм-границами — он для нетипичных форм.
template <typename T, int VEC, int REP>
__device__ __forceinline__ void flash_blocks_impl(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    const unsigned int* __restrict__ table,
    const unsigned int* __restrict__ tail_from, const unsigned int* __restrict__ tail_len,
    T* __restrict__ out,
    int B, int NH, int NKV, int CAP, int D, int NB, int ratio, float scale) {
  int row = blockIdx.x;
  int bi = row / NKV;
  int h_kv = row % NKV;
  if (bi >= B) return;

  // Головы одной kv-головы берутся пачками: их состояние живёт в регистрах, и
  // на голове 256 все двенадцать разом туда не помещаются.
  int rep_tile = (REP > 0) ? REP : 4;
  int group = blockIdx.y;
  int total_rep = NH / NKV;
  int r_base = group * rep_tile;
  if (r_base >= total_rep) return;
  int n_rep = total_rep - r_base;
  if (n_rep > rep_tile) n_rep = rep_tile;
  int vec = (VEC > 0) ? VEC : D / 32;   // сколько элементов строки держит лейн
  int tid = threadIdx.x;
  int warp = tid >> 5, lane = tid & 31;
  int from = (int)tail_from[bi];
  int total = NB * ratio + (int)tail_len[bi];

  const unsigned int* row_table = table + (long)bi * NB;

  // Регистровое состояние варпа: q и аккумулятор распределены по лейнам,
  // максимум и нормировка — общие на варп.
  float q_reg[FB_MAX_REP][FB_MAX_VEC];
  float acc[FB_MAX_REP][FB_MAX_VEC];
  float run_m[FB_MAX_REP], run_l[FB_MAX_REP];
  #pragma unroll
  for (int r = 0; r < n_rep; r++) {
    int h = h_kv * total_rep + r_base + r;
    const T* q_row = q + ((long)bi * NH + h) * D;
    #pragma unroll
    for (int i = 0; i < vec; i++) {
      q_reg[r][i] = fb_ld(q_row + lane * vec + i);
      acc[r][i] = 0.0f;
    }
    run_m[r] = FB_NEG_INF;
    run_l[r] = 0.0f;
  }

  for (int t = warp; t < total; t += FB_WARPS) {
    int pos = fb_pos(row_table, NB, ratio, from, t);
    const T* k_row = k + ((long)h_kv * CAP + pos) * D;
    float kv_reg[FB_MAX_VEC];
    #pragma unroll
    for (int i = 0; i < vec; i++) kv_reg[i] = fb_ld(k_row + lane * vec + i);

    const T* v_row = v + ((long)h_kv * CAP + pos) * D;
    float vv_reg[FB_MAX_VEC];
    #pragma unroll
    for (int i = 0; i < vec; i++) vv_reg[i] = fb_ld(v_row + lane * vec + i);

    #pragma unroll
    for (int r = 0; r < n_rep; r++) {
      float part = 0.0f;
      #pragma unroll
      for (int i = 0; i < vec; i++) part += q_reg[r][i] * kv_reg[i];
      float s = fb_warp_sum(part) * scale;
      s = __shfl_sync(0xFFFFFFFFu, s, 0, 32);

      float m_new = fmaxf(run_m[r], s);
      float alpha = fb_is_finite(run_m[r]) ? __expf(run_m[r] - m_new) : 0.0f;
      float p = fb_is_finite(m_new) ? __expf(s - m_new) : 0.0f;
      run_m[r] = m_new;
      run_l[r] = run_l[r] * alpha + p;
      #pragma unroll
      for (int i = 0; i < vec; i++) acc[r][i] = acc[r][i] * alpha + p * vv_reg[i];
    }
  }

  // Слияние варпов: каждый отдал свой (m, l, acc), результат нормируется.
  extern __shared__ float smem[];
  float* sh_m = smem;                                  // [FB_WARPS, n_rep]
  float* sh_l = sh_m + FB_WARPS * n_rep;               // [FB_WARPS, n_rep]
  float* sh_acc = sh_l + FB_WARPS * n_rep;             // [FB_WARPS, n_rep, D]

  #pragma unroll
  for (int r = 0; r < n_rep; r++) {
    if (lane == 0) {
      sh_m[warp * n_rep + r] = run_m[r];
      sh_l[warp * n_rep + r] = run_l[r];
    }
    #pragma unroll
    for (int i = 0; i < vec; i++) {
      sh_acc[((long)warp * n_rep + r) * D + lane * vec + i] = acc[r][i];
    }
  }
  __syncthreads();

  for (int r = warp; r < n_rep; r += FB_WARPS) {
    float m = FB_NEG_INF;
    for (int w = 0; w < FB_WARPS; w++) m = fmaxf(m, sh_m[w * n_rep + r]);
    float l = 0.0f;
    float w_scale[FB_WARPS];
    for (int w = 0; w < FB_WARPS; w++) {
      float mw = sh_m[w * n_rep + r];
      float f = (fb_is_finite(mw) && fb_is_finite(m)) ? __expf(mw - m) : 0.0f;
      w_scale[w] = f;
      l += sh_l[w * n_rep + r] * f;
    }
    int h = h_kv * total_rep + r_base + r;
    #pragma unroll
    for (int i = 0; i < vec; i++) {
      int d = lane * vec + i;
      float a = 0.0f;
      for (int w = 0; w < FB_WARPS; w++) {
        a += sh_acc[((long)w * n_rep + r) * D + d] * w_scale[w];
      }
      fb_st(out + ((long)bi * NH + h) * D + d, (l > 0.0f) ? a / l : 0.0f);
    }
  }
}

#define FB_ENTRY(name, T, VEC, REP)                                                        \
  extern "C" __global__ void name(                                                         \
      const T* q, const T* k, const T* v,                                                  \
      const unsigned int* table, const unsigned int* tail_from,                            \
      const unsigned int* tail_len, T* out,                                                \
      int B, int NH, int NKV, int CAP, int D, int NB, int ratio, float scale) {             \
    flash_blocks_impl<T, VEC, REP>(q, k, v, table, tail_from, tail_len, out,               \
                                   B, NH, NKV, CAP, D, NB, ratio, scale);                  \
  }

FB_ENTRY(flash_blocks_f32, float, 0, 0)
FB_ENTRY(flash_blocks_f16, __half, 0, 0)
FB_ENTRY(flash_blocks_bf16, __nv_bfloat16, 0, 0)
FB_ENTRY(flash_blocks_f32_h128_r8, float, 4, 8)
FB_ENTRY(flash_blocks_f16_h128_r8, __half, 4, 8)
FB_ENTRY(flash_blocks_bf16_h128_r8, __nv_bfloat16, 4, 8)
FB_ENTRY(flash_blocks_f32_h256_r6, float, 8, 6)
FB_ENTRY(flash_blocks_f16_h256_r6, __half, 8, 6)
FB_ENTRY(flash_blocks_bf16_h256_r6, __nv_bfloat16, 8, 6)
