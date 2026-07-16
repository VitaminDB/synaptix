#include <cuda_fp16.h>
#include <cuda_fp8.h>

// MXFP8 (Blackwell-нативный block-scale FP8) квант: per-32-block E8M0 scale + E4M3.
// natural layout [rows, K/32] (gemv decode, хранение веса, v1-GEMM cp.async-путь).
//
// MXFP8 natural квант: per-32-block E8M0 + e4m3, scales в natural layout
// [rows, K/32] (для gemv decode + хранения веса). Один поток = (row, kblock).
extern "C" __global__ void mxfp8_quant_natural(const __half *__restrict__ in,
                                               __nv_fp8_e4m3 *__restrict__ out_fp8,
                                               unsigned char *__restrict__ out_scales,
                                               int rows, int K) {
  const int kb_total = K / 32;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)rows * kb_total)
    return;
  const int row = (int)(idx / kb_total);
  const int kblock = (int)(idx % kb_total);
  const __half *xp = in + (long)row * K + (long)kblock * 32;
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; i++)
    amax = fmaxf(amax, fabsf(__half2float(xp[i])));
  unsigned char sbyte = (unsigned char)((__float_as_uint(__uint_as_float(__float_as_uint(amax) & 0x7F800000u) / 256.0f)) >> 23);
  float sv = fmaxf(__uint_as_float(((unsigned)sbyte) << 23), 1e-12f);
  __nv_fp8_e4m3 *op = out_fp8 + (long)row * K + (long)kblock * 32;
#pragma unroll
  for (int i = 0; i < 32; i++)
    op[i] = __nv_fp8_e4m3(fminf(fmaxf(__half2float(xp[i]) / sv, -448.0f), 448.0f));
  out_scales[(long)row * kb_total + kblock] = sbyte;
}

// Векторизованный вариант (та же арифметика бит-в-бит): uint4-лоады/сторы вместо
// 32 скалярных half с варп-страйдом 64B (32 сектора/инструкцию → квант был 37%
// времени linear_quant на LTX-формах). Требует 16B-выровненного in (гейт в обёртке).
extern "C" __global__ void mxfp8_quant_natural_fast(const __half *__restrict__ in,
                                                    __nv_fp8_e4m3 *__restrict__ out_fp8,
                                                    unsigned char *__restrict__ out_scales,
                                                    int rows, int K) {
  const int kb_total = K / 32;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)rows * kb_total)
    return;
  const int row = (int)(idx / kb_total);
  const int kblock = (int)(idx % kb_total);
  const uint4 *src = reinterpret_cast<const uint4 *>(in + (long)row * K + (long)kblock * 32);
  uint4 p[4];
  p[0] = src[0]; p[1] = src[1]; p[2] = src[2]; p[3] = src[3];
  __half2 h[16];
  *reinterpret_cast<uint4 *>(&h[0]) = p[0];
  *reinterpret_cast<uint4 *>(&h[4]) = p[1];
  *reinterpret_cast<uint4 *>(&h[8]) = p[2];
  *reinterpret_cast<uint4 *>(&h[12]) = p[3];
  float v[32];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    float2 f = __half22float2(h[i]);
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
    amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
  }
  unsigned char sbyte = (unsigned char)((__float_as_uint(__uint_as_float(__float_as_uint(amax) & 0x7F800000u) / 256.0f)) >> 23);
  float sv = fmaxf(__uint_as_float(((unsigned)sbyte) << 23), 1e-12f);
  unsigned char ob[32];
#pragma unroll
  for (int i = 0; i < 32; i++)
    ob[i] = __nv_fp8_e4m3(fminf(fmaxf(v[i] / sv, -448.0f), 448.0f)).__x;
  uint4 *dst = reinterpret_cast<uint4 *>(out_fp8 + (long)row * K + (long)kblock * 32);
  dst[0] = *reinterpret_cast<uint4 *>(&ob[0]);
  dst[1] = *reinterpret_cast<uint4 *>(&ob[16]);
  out_scales[(long)row * kb_total + kblock] = sbyte;
}

// ×4-параллельный вариант (та же арифметика бит-в-бит): поток = 8 элементов
// (1×uint4-лоад), amax k32-блока собирается shfl_xor-бабочкой по четвёрке
// потоков (fmax коммутативен → bit-same scale). На малых M fast-вариант
// (поток = блок) давал 128 CTA / Waves 0.39 — латентность не спрятана
// (квант M=256 был 2.4× от DRAM-floor).
extern "C" __global__ void mxfp8_quant_natural_fast4(const __half *__restrict__ in,
                                                     __nv_fp8_e4m3 *__restrict__ out_fp8,
                                                     unsigned char *__restrict__ out_scales,
                                                     int rows, int K) {
  const int kb_total = K / 32;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)rows * kb_total * 4)
    return;
  const long blk = idx >> 2;
  const int sub = (int)(idx & 3);
  const int row = (int)(blk / kb_total);
  const int kblock = (int)(blk % kb_total);
  const uint4 p = *reinterpret_cast<const uint4 *>(in + (long)row * K + (long)kblock * 32 + sub * 8);
  __half2 h[4];
  *reinterpret_cast<uint4 *>(&h[0]) = p;
  float v[8];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 f = __half22float2(h[i]);
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
    amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  unsigned char sbyte = (unsigned char)((__float_as_uint(__uint_as_float(__float_as_uint(amax) & 0x7F800000u) / 256.0f)) >> 23);
  float sv = fmaxf(__uint_as_float(((unsigned)sbyte) << 23), 1e-12f);
  unsigned char ob[8];
#pragma unroll
  for (int i = 0; i < 8; i++)
    ob[i] = __nv_fp8_e4m3(fminf(fmaxf(v[i] / sv, -448.0f), 448.0f)).__x;
  *reinterpret_cast<uint2 *>(out_fp8 + (long)row * K + (long)kblock * 32 + sub * 8) =
      *reinterpret_cast<uint2 *>(&ob[0]);
  if (sub == 0)
    out_scales[(long)row * kb_total + kblock] = sbyte;
}

// MXFP8 dequant → f16: out[row,k] = e4m3(packed) * 2^(E8M0[row,k/32]-127).
// Для prefill (M>1): деквант веса в f16, дальше обычный f16 TN-linear.
extern "C" __global__ void mxfp8_dequant_f16(const __nv_fp8_e4m3 *__restrict__ packed,
                                             const unsigned char *__restrict__ scales,
                                             __half *__restrict__ out, int rows, int K) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)rows * K)
    return;
  const int row = (int)(idx / K);
  const int kcol = (int)(idx % K);
  float sv = __uint_as_float(((unsigned)scales[(long)row * (K / 32) + (kcol / 32)]) << 23);
  out[idx] = __float2half(float(packed[idx]) * sv);
}
