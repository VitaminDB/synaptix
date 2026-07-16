#include <cuda_fp16.h>

// NVFP4 quantize / dequant kernels (foundation).
//
// Формат — NVIDIA Blackwell NVFP4: 4-битные элементы (E2M1) с per-block
// FP8 E4M3 scale, block размер = 16 элементов. Используется как:
//   1) operand A/B в cuBLASLt nvfp4 GEMM (`mma.kind::nvf4` на sm_120).
//   2) operand A/B в native warp-level `mma.sync.kind::mxf4nvf4.block_scale`
//      (sm_120a, кастомные kernels).
//
// Scale tensor — tile-major layout по cuBLASLt 13 spec §3.1.4.4.2:
//   tile  = 128 outer × 4 inner blocks (= 512 байт), покрывает 128×64 data.
//   внутри tile: off = (outer%32)*16 + (outer/32)*4 + inner.
//   tiles: tile_base = (tile_col*4 + tile_row*sf_inner_dim) * 128.
//   sf_inner_dim = roundup(inner_dim, 64) / 16  (multiple of 4).
//
// Data tensor (packed nibbles) — линейный row-major по (outer, inner):
//   element (outer_idx, inner_idx) → byte (outer_idx * inner_dim + inner_idx) / 2,
//   low nibble = even inner_idx, high nibble = odd inner_idx.
//
// File компилируется на sm_80 baseline — kernels не используют sm_120-specific
// PTX. Для native mma path (sm_120a) есть отдельный nvfp4_mma.cu.

extern "C" {

__device__ __forceinline__ float decode_e4m3(unsigned char byte) {
    bool sign = (byte & 0x80) != 0;
    int exp_bits = (byte >> 3) & 0x0F;
    int mantissa = byte & 0x07;
    if (exp_bits == 15 && mantissa == 7) {
        return __int_as_float(0x7FC00000);
    }
    float val;
    if (exp_bits == 0) {
        val = mantissa * 0.001953125f;
    } else {
        int exp_raw = exp_bits - 7;
        float frac = 1.0f + mantissa * 0.125f;
        val = frac * exp2f((float)exp_raw);
    }
    return sign ? -val : val;
}

__device__ __forceinline__ unsigned char encode_e4m3(float x) {
    if (isnan(x)) return 0x7F;
    float v = fminf(fmaxf(x, -448.0f), 448.0f);
    unsigned int sign = signbit(v) ? 1 : 0;
    float abs_v = fabsf(v);
    if (abs_v == 0.0f) return (unsigned char)(sign << 7);
    int exp_raw = (int)floorf(log2f(abs_v));
    int exp_biased = exp_raw + 7;
    unsigned int mantissa;
    if (exp_biased < 1) {
        int m = (int)nearbyintf(abs_v * 512.0f);
        m = max(0, min(7, m));
        return (unsigned char)((sign << 7) | (unsigned int)m);
    }
    if (exp_biased > 15) {
        return (unsigned char)((sign << 7) | 0x7E);
    }
    float pow2 = exp2f((float)exp_raw);
    int m = (int)nearbyintf(((abs_v / pow2) - 1.0f) * 8.0f);
    if (m == 8) {
        m = 0;
        exp_biased += 1;
        if (exp_biased > 15) return (unsigned char)((sign << 7) | 0x7E);
    }
    if (exp_biased == 15 && m == 7) m = 6;
    mantissa = (unsigned int)(m & 0x07);
    return (unsigned char)((sign << 7) | ((unsigned int)exp_biased << 3) | mantissa);
}

__device__ __forceinline__ float decode_e2m1(unsigned char nib) {
    unsigned int idx = nib & 0x07u;
    bool sign = (nib & 0x08u) != 0u;
    float mag;
    switch (idx) {
        case 0: mag = 0.0f; break;
        case 1: mag = 0.5f; break;
        case 2: mag = 1.0f; break;
        case 3: mag = 1.5f; break;
        case 4: mag = 2.0f; break;
        case 5: mag = 3.0f; break;
        case 6: mag = 4.0f; break;
        default: mag = 6.0f; break;
    }
    return sign ? -mag : mag;
}

__device__ __forceinline__ unsigned char encode_e2m1_rtne(float x) {
    unsigned int sign = signbit(x) ? 0x08 : 0x00;
    float abs_x = fabsf(x);
    unsigned int idx;
    if (abs_x >= 5.0f) idx = 7;
    else if (abs_x >= 3.5f) idx = 6;
    else if (abs_x >= 2.5f) idx = 5;
    else if (abs_x >= 1.75f) idx = 4;
    else if (abs_x >= 1.25f) idx = 3;
    else if (abs_x >= 0.75f) idx = 2;
    else if (abs_x >= 0.25f) idx = 1;
    else idx = 0;
    return (unsigned char)(sign | idx);
}

__device__ __forceinline__ unsigned int tile_scale_offset(
    unsigned int outer,
    unsigned int block_col,
    unsigned int sf_inner_dim
) {
    unsigned int tile_row = outer >> 7;
    unsigned int tile_col = block_col >> 2;
    unsigned int local_outer = outer & 127u;
    unsigned int local_inner = block_col & 3u;
    unsigned int tile_base = (tile_col * 4u + tile_row * sf_inner_dim) * 128u;
    unsigned int off_in_tile = (local_outer & 31u) * 16u
                             + (local_outer >> 5) * 4u
                             + local_inner;
    return tile_base + off_in_tile;
}

__global__ void quantize_f16_to_nvfp4(
    const __half* __restrict__ x,
    unsigned char* __restrict__ packed,
    unsigned char* __restrict__ scales_e4m3,
    unsigned int outer_dim,
    unsigned int inner_dim,
    unsigned int sf_inner_dim,
    unsigned int outer_offset
) {
    extern __shared__ unsigned char shm[];
    float* amax_sm = (float*)shm;
    unsigned char* nib_sm = shm + 16 * 4;

    unsigned int block_col = blockIdx.x;
    unsigned int outer = blockIdx.y + outer_offset;
    unsigned int tid = threadIdx.x;

    unsigned int inner = block_col * 16u + tid;
    unsigned int data_idx = outer * inner_dim + inner;

    float val = 0.0f;
    if (outer < outer_dim && inner < inner_dim) {
        val = __half2float(x[data_idx]);
    }
    amax_sm[tid] = fabsf(val);
    __syncthreads();
    for (unsigned int s = 8u; s > 0u; s >>= 1) {
        if (tid < s) {
            float a = amax_sm[tid];
            float b = amax_sm[tid + s];
            amax_sm[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    float amax = amax_sm[0];

    float scale_raw = (amax > 0.0f) ? (amax / 6.0f) : 1e-9f;
    unsigned char scale_byte = encode_e4m3(scale_raw);
    float scale_q = decode_e4m3(scale_byte);
    if (scale_q == 0.0f) scale_q = 1e-9f;

    if (tid == 0) {
        unsigned int off = tile_scale_offset(outer, block_col, sf_inner_dim);
        scales_e4m3[off] = scale_byte;
    }

#ifdef SYN_E2M1_THRESHOLD
    unsigned char nibble;
    {
        float av = fabsf(val);
        unsigned int idx = (unsigned int)(av >= 0.25f * scale_q) + (av >= 0.75f * scale_q)
                         + (av >= 1.25f * scale_q) + (av >= 1.75f * scale_q)
                         + (av >= 2.5f * scale_q) + (av >= 3.5f * scale_q)
                         + (av >= 5.0f * scale_q);
        nibble = (unsigned char)((signbit(val) ? 8u : 0u) | idx);
    }
#else
    unsigned char nibble = encode_e2m1_rtne(val / scale_q);
#endif
    nib_sm[tid] = nibble;
    __syncthreads();

    if ((tid & 1u) == 0u) {
        unsigned int low_inner = block_col * 16u + tid;
        unsigned int high_inner = low_inner + 1u;
        unsigned char low = (outer < outer_dim && low_inner < inner_dim) ? nib_sm[tid] : 0;
        unsigned char high = (outer < outer_dim && high_inner < inner_dim) ? nib_sm[tid + 1] : 0;
        unsigned int byte_idx = (outer * inner_dim + low_inner) >> 1;
        packed[byte_idx] = (low & 0x0F) | ((high & 0x0F) << 4);
    }
}

// Быстрый квантизатор: 1 поток = 1 группа из 16 элементов (2×uint4-лоада,
// без shared/syncthreads; старое ядро = блок 16 потоков на группу → 6.8M
// микроблоков на LTX-формах, 76% времени linear_quant). Арифметика бит-в-бит
// та же (тот же amax — max порядконезависим, тот же encode/decode и деление).
// Требует 16B-выровненного x (гейт в обёртке).
// outer_cov >= outer_dim: строки [outer_dim, outer_cov) пишут scale=0 (хвост
// 128-тайла scale-раскладки) — буфер скейлов полностью определён БЕЗ CE-memset
// со стороны вызывающего (тот аллоцирует uninit).
__global__ void quantize_f16_to_nvfp4_fast(
    const __half* __restrict__ x,
    unsigned char* __restrict__ packed,
    unsigned char* __restrict__ scales_e4m3,
    unsigned int outer_dim,
    unsigned int inner_dim,
    unsigned int sf_inner_dim,
    unsigned int outer_cov
) {
    unsigned int groups_per_row = inner_dim >> 4;
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)outer_cov * groups_per_row;
    if (g >= total) return;
    unsigned int outer = (unsigned int)(g / groups_per_row);
    unsigned int block_col = (unsigned int)(g % groups_per_row);
    if (outer >= outer_dim) {
        scales_e4m3[tile_scale_offset(outer, block_col, sf_inner_dim)] = 0;
        return;
    }

    const uint4* src = reinterpret_cast<const uint4*>(
        x + (size_t)outer * inner_dim + (size_t)block_col * 16u);
    uint4 p0 = src[0];
    uint4 p1 = src[1];
    __half2 h[8];
    *reinterpret_cast<uint4*>(&h[0]) = p0;
    *reinterpret_cast<uint4*>(&h[4]) = p1;

    float v[16];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float2 f = __half22float2(h[i]);
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }

    float scale_raw = (amax > 0.0f) ? (amax / 6.0f) : 1e-9f;
    unsigned char scale_byte = encode_e4m3(scale_raw);
    float scale_q = decode_e4m3(scale_byte);
    if (scale_q == 0.0f) scale_q = 1e-9f;
    scales_e4m3[tile_scale_offset(outer, block_col, sf_inner_dim)] = scale_byte;

    unsigned long long out8;
    unsigned char* ob = reinterpret_cast<unsigned char*>(&out8);
#ifdef SYN_E2M1_THRESHOLD
    // Пороговый e2m1 (опт-ин -DSYN_E2M1_THRESHOLD): |x| против ТОЧНЫХ кратных
    // scale (мантисса e4m3 — 3 бита → 3s/5s/7s представимы в f32 точно) —
    // математически точный квант без округления деления; биты отличаются от
    // div-пути только в 0.5ulp-окнах div.rn у порогов.
    float t1 = 0.25f * scale_q, t2 = 0.75f * scale_q, t3 = 1.25f * scale_q;
    float t4 = 1.75f * scale_q, t5 = 2.5f * scale_q, t6 = 3.5f * scale_q;
    float t7 = 5.0f * scale_q;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float a0 = fabsf(v[2 * i]), a1 = fabsf(v[2 * i + 1]);
        unsigned int lo = (unsigned int)(a0 >= t1) + (a0 >= t2) + (a0 >= t3)
                        + (a0 >= t4) + (a0 >= t5) + (a0 >= t6) + (a0 >= t7);
        unsigned int hi = (unsigned int)(a1 >= t1) + (a1 >= t2) + (a1 >= t3)
                        + (a1 >= t4) + (a1 >= t5) + (a1 >= t6) + (a1 >= t7);
        lo |= signbit(v[2 * i]) ? 8u : 0u;
        hi |= signbit(v[2 * i + 1]) ? 8u : 0u;
        ob[i] = (unsigned char)((lo & 0x0F) | ((hi & 0x0F) << 4));
    }
#else
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned char lo = encode_e2m1_rtne(v[2 * i] / scale_q);
        unsigned char hi = encode_e2m1_rtne(v[2 * i + 1] / scale_q);
        ob[i] = (unsigned char)((lo & 0x0F) | ((hi & 0x0F) << 4));
    }
#endif
    *reinterpret_cast<unsigned long long*>(
        packed + ((size_t)outer * inner_dim + (size_t)block_col * 16u) / 2) = out8;
}

// Дексквантизация NVFP4 → F16. Используется в roundtrip-валидации (на CUDA)
// и как fallback path там где cuBLASLt nvfp4 GEMM недоступен.
// Grid: 2D (gridDim.x = inner_dim/16, gridDim.y = outer_dim), block = 16
// thread'ов (= один scale-block).
__global__ void nvfp4_dequant_f16(
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scales_e4m3,
    __half* __restrict__ out,
    unsigned int outer_dim,
    unsigned int inner_dim,
    unsigned int sf_inner_dim,
    unsigned int outer_offset
) {
    unsigned int block_col = blockIdx.x;
    unsigned int outer = blockIdx.y + outer_offset;
    unsigned int tid = threadIdx.x;

    if (outer >= outer_dim) return;
    unsigned int inner = block_col * 16u + tid;
    if (inner >= inner_dim) return;

    unsigned int scale_off = tile_scale_offset(outer, block_col, sf_inner_dim);
    float scale_q = decode_e4m3(scales_e4m3[scale_off]);

    unsigned int byte_idx = (outer * inner_dim + inner) >> 1;
    unsigned char byte = packed[byte_idx];
    unsigned char nib = (inner & 1u) ? (byte >> 4) : (byte & 0x0F);
    float v = decode_e2m1(nib) * scale_q;

    out[outer * inner_dim + inner] = __float2half(v);
}

}
