#include <cuda_fp16.h>
#include <cuda_fp8.h>

// MXFP8 GEMV (decode, M=1): y[N] = W[N,K] @ x[K]. W/x = E4M3 bytes (natural
// [.,K]) + E8M0 per-32-block scales (natural [., K/32]). Аналог gemv_nvfp4, но
// decode memory-bound (читаем весь W раз) → SIMT-dequant вместо block-scale MMA
// (на M=1 DRAM-saturated, tensor-core throughput не нужен; проще и без тонкой
// MMA-раскладки). E8M0-дек: scale = 2^(byte-127) = float_from_bits(byte<<23).
// Один warp = одна строка N; 32 нити делят K-блоки.
extern "C" __global__ void gemv_mxfp8_e4m3(const __nv_fp8_e4m3 *__restrict__ w,
                                           const unsigned char *__restrict__ sw,
                                           const __nv_fp8_e4m3 *__restrict__ x,
                                           const unsigned char *__restrict__ sx,
                                           __half *__restrict__ out, int N, int K) {
  const int warps = blockDim.x >> 5;
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  if (row >= N)
    return;
  const int lane = threadIdx.x & 31;
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
  if (lane == 0)
    out[row] = __float2half(acc);
}
