#include <cuda_fp16.h>
#include <cuda_fp8.h>

#ifdef SYN_OUT_BF16
#include <cuda_bf16.h>
typedef __nv_bfloat16 syn_out_t;
#define SYN_TO_OUT(v) __float2bfloat16(v)
#else
typedef __half syn_out_t;
#define SYN_TO_OUT(v) __float2half(v)
#endif

// MXFP8 GEMV (decode, M=1): y[N] = W[N,K] @ x[K]. W/x = E4M3 bytes (natural
// [.,K]) + E8M0 per-32-block scales (natural [., K/32]). Аналог gemv_nvfp4, но
// decode memory-bound (читаем весь W раз) → SIMT-dequant вместо block-scale MMA
// (на M=1 DRAM-saturated, tensor-core throughput не нужен; проще и без тонкой
// MMA-раскладки). E8M0-дек: scale = 2^(byte-127) = float_from_bits(byte<<23).
// Один warp = одна строка N; 32 нити делят K-блоки.
__device__ __forceinline__ float gemv_mxfp8_row(const __nv_fp8_e4m3 *__restrict__ w,
                                                const unsigned char *__restrict__ sw,
                                                const __nv_fp8_e4m3 *__restrict__ x,
                                                const unsigned char *__restrict__ sx,
                                                int row, int K, int lane) {
  const int kb = K / 32;

  float acc = 0.f;
  for (int b = lane; b < kb; b += 32) {
    float sw_v = __uint_as_float((unsigned)sw[(long)row * kb + b] << 23);
    float sx_v = __uint_as_float((unsigned)sx[b] << 23);
    // векторные лоады (2×uint4 = 32B на блок): байтовые wp[i] давали
    // 32×LDG.U8 → 200GB/s (3.4× от floor, M=1 82мкс). Конверсия и порядок
    // суммирования те же — бит-в-бит.
    const uint4 *wp = reinterpret_cast<const uint4 *>(w + (long)row * K + (long)b * 32);
    const uint4 *xp = reinterpret_cast<const uint4 *>(x + (long)b * 32);
    uint4 wv0 = wp[0], wv1 = wp[1];
    uint4 xv0 = xp[0], xv1 = xp[1];
    const __nv_fp8_e4m3 *wb0 = reinterpret_cast<const __nv_fp8_e4m3 *>(&wv0);
    const __nv_fp8_e4m3 *wb1 = reinterpret_cast<const __nv_fp8_e4m3 *>(&wv1);
    const __nv_fp8_e4m3 *xb0 = reinterpret_cast<const __nv_fp8_e4m3 *>(&xv0);
    const __nv_fp8_e4m3 *xb1 = reinterpret_cast<const __nv_fp8_e4m3 *>(&xv1);
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++)
      dot += float(wb0[i]) * float(xb0[i]);
#pragma unroll
    for (int i = 0; i < 16; i++)
      dot += float(wb1[i]) * float(xb1[i]);
    acc += sw_v * sx_v * dot;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;
}

extern "C" __global__ void gemv_mxfp8_e4m3(const __nv_fp8_e4m3 *__restrict__ w,
                                           const unsigned char *__restrict__ sw,
                                           const __nv_fp8_e4m3 *__restrict__ x,
                                           const unsigned char *__restrict__ sx,
                                           syn_out_t *__restrict__ out, int N, int K) {
  const int warps = blockDim.x >> 5;
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  if (row >= N)
    return;
  const int lane = threadIdx.x & 31;
  float acc = gemv_mxfp8_row(w, sw, x, sx, row, K, lane);
  if (lane == 0)
    out[row] = SYN_TO_OUT(acc);
}

// Групповой вариант: до трёх матриц одной K с общей активацией одним
// запуском (q/k/v одного слоя). Варп — строка; строки идут подряд
// [n0 | n1 | n2], варп сам выбирает матрицу по диапазону.
extern "C" __global__ void gemv_mxfp8_e4m3_grouped(
    unsigned long long w0, unsigned long long sw0, unsigned long long o0, int n0,
    unsigned long long w1, unsigned long long sw1, unsigned long long o1, int n1,
    unsigned long long w2, unsigned long long sw2, unsigned long long o2, int n2,
    const __nv_fp8_e4m3 *__restrict__ x, const unsigned char *__restrict__ sx, int K) {
  const int warps = blockDim.x >> 5;
  int grow = blockIdx.x * warps + (threadIdx.x >> 5);
  unsigned long long wu, su, ou; int n;
  if (grow < n0) { wu = w0; su = sw0; ou = o0; n = n0; }
  else if (grow < n0 + n1) { grow -= n0; wu = w1; su = sw1; ou = o1; n = n1; }
  else { grow -= n0 + n1; wu = w2; su = sw2; ou = o2; n = n2; }
  if (grow >= n)
    return;
  const __nv_fp8_e4m3 *w = (const __nv_fp8_e4m3 *)(size_t)wu;
  const unsigned char *sw = (const unsigned char *)(size_t)su;
  syn_out_t *out = (syn_out_t *)(size_t)ou;
  const int lane = threadIdx.x & 31;
  float acc = gemv_mxfp8_row(w, sw, x, sx, grow, K, lane);
  if (lane == 0)
    out[grow] = SYN_TO_OUT(acc);
}
