// Декодеры под-блоков (32 значения) одноблобных форматов + NVFP4/MXFP8 движка.
// Общий заголовок для blockq_dequant.cu и blockq_gemv.cu: Rust-сторона склеивает
// ggml_tables.cuh + этот файл + ядра. Контракт — бит в бит с CPU-эталоном
// synaptix_core::quant (порядок операций, модуль с --fmad=false).
#include <cuda_fp16.h>

#define QK_K 256
#define IQ1S_DELTA 0.125f

__device__ __forceinline__ float rd_f16(const uint8_t *b, int off) {
    uint16_t v = (uint16_t)b[off] | ((uint16_t)b[off + 1] << 8);
    return __half2float(__ushort_as_half(v));
}
__device__ __forceinline__ uint16_t rd_u16(const uint8_t *b, int off) {
    return (uint16_t)b[off] | ((uint16_t)b[off + 1] << 8);
}
__device__ __forceinline__ uint32_t rd_u32(const uint8_t *b, int off) {
    return (uint32_t)b[off] | ((uint32_t)b[off + 1] << 8) | ((uint32_t)b[off + 2] << 16) |
           ((uint32_t)b[off + 3] << 24);
}
__device__ __forceinline__ float e8m0_half(uint8_t x) {
    uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t)(x - 1) << 23);
    return __uint_as_float(bits);
}
__device__ __forceinline__ float ue4m3_half(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int exp = (x >> 3) & 0xF;
    float man = (float)(x & 0x7);
    float raw = exp == 0 ? man * ldexpf(1.0f, -9) : (1.0f + man / 8.0f) * ldexpf(1.0f, exp - 7);
    return raw * 0.5f;
}
__device__ __forceinline__ float sgn(uint8_t signs, int j) {
    return (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
}

// ---------------------------------------------------------------- SQ<bits>
// Биты 0..7 байта → нибблы 0..7 слова (бит i → бит 4i): три маскированных сдвига.
__device__ __forceinline__ uint32_t spread8_nib(uint32_t b) {
    b = (b | (b << 12)) & 0x000F000Fu;
    b = (b | (b << 6)) & 0x03030303u;
    return (b | (b << 3)) & 0x11111111u;
}
template <int BITS>
__device__ __forceinline__ void deq_sq(const uint8_t *blk, int s, float *y) {
    float d = rd_f16(blk, 0);
    float dmin = rd_f16(blk, 2);
    float dl = d * (float)blk[4 + s];
    float ml = dmin * (float)blk[12 + s];
    uint32_t planes[8];
    const uint8_t *pp = blk + 20 + s * BITS * 4;
    // Блоб из 148/…/276-байтных блоков выровнен на 4 (размер блока кратен 4),
    // проверка — на случай среза с нечётным смещением.
    if ((reinterpret_cast<unsigned long long>(pp) & 3ull) == 0) {
#pragma unroll
        for (int j = 0; j < BITS; ++j) planes[j] = reinterpret_cast<const uint32_t *>(pp)[j];
    } else {
#pragma unroll
        for (int j = 0; j < BITS; ++j) planes[j] = rd_u32(pp, j * 4);
    }
    if (BITS <= 4) {
        // Плоскости → нибблы по 8 значений за раз (≈3 оп. на вес вместо 3·BITS),
        // ниббл → float через магию 2²³ (точно: целое < 16).
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            uint32_t qn = 0;
#pragma unroll
            for (int j = 0; j < BITS; ++j) qn |= spread8_nib((planes[j] >> (8 * g)) & 0xFFu) << j;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float q = __uint_as_float(0x4B000000u | ((qn >> (4 * e)) & 0xFu)) - 8388608.0f;
                y[g * 8 + e] = dl * q - ml;
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            uint32_t q = 0;
#pragma unroll
            for (int j = 0; j < BITS; ++j) q |= ((planes[j] >> i) & 1u) << j;
            y[i] = dl * (float)q - ml;
        }
    }
}

// ---------------------------------------------------------------- 32-блоки
__device__ __forceinline__ void deq_q4_0(const uint8_t *b, int, float *y) {
    float d = rd_f16(b, 0);
    const uint8_t *qs = b + 2;
    for (int j = 0; j < 16; ++j) {
        y[j] = (float)((int)(qs[j] & 0x0F) - 8) * d;
        y[j + 16] = (float)((int)(qs[j] >> 4) - 8) * d;
    }
}
__device__ __forceinline__ void deq_q4_1(const uint8_t *b, int, float *y) {
    float d = rd_f16(b, 0), m = rd_f16(b, 2);
    const uint8_t *qs = b + 4;
    for (int j = 0; j < 16; ++j) {
        y[j] = (float)(qs[j] & 0x0F) * d + m;
        y[j + 16] = (float)(qs[j] >> 4) * d + m;
    }
}
__device__ __forceinline__ void deq_q5_0(const uint8_t *b, int, float *y) {
    float d = rd_f16(b, 0);
    uint32_t qh = rd_u32(b, 2);
    const uint8_t *qs = b + 6;
    for (int j = 0; j < 16; ++j) {
        uint8_t xh0 = (uint8_t)(((qh >> j) << 4) & 0x10);
        uint8_t xh1 = (uint8_t)((qh >> (j + 12)) & 0x10);
        y[j] = (float)((int)((qs[j] & 0x0F) | xh0) - 16) * d;
        y[j + 16] = (float)((int)((qs[j] >> 4) | xh1) - 16) * d;
    }
}
__device__ __forceinline__ void deq_q5_1(const uint8_t *b, int, float *y) {
    float d = rd_f16(b, 0), m = rd_f16(b, 2);
    uint32_t qh = rd_u32(b, 4);
    const uint8_t *qs = b + 8;
    for (int j = 0; j < 16; ++j) {
        uint8_t xh0 = (uint8_t)(((qh >> j) << 4) & 0x10);
        uint8_t xh1 = (uint8_t)((qh >> (j + 12)) & 0x10);
        y[j] = (float)((qs[j] & 0x0F) | xh0) * d + m;
        y[j + 16] = (float)((qs[j] >> 4) | xh1) * d + m;
    }
}
__device__ __forceinline__ void deq_q8_0(const uint8_t *b, int, float *y) {
    float d = rd_f16(b, 0);
    for (int j = 0; j < 32; ++j) y[j] = (float)(int8_t)b[2 + j] * d;
}
__device__ __forceinline__ void deq_q8_1(const uint8_t *b, int, float *y) {
    float d = rd_f16(b, 0);
    for (int j = 0; j < 32; ++j) y[j] = (float)(int8_t)b[4 + j] * d;
}
__device__ __forceinline__ void deq_q8_k(const uint8_t *b, int s, float *y) {
    float d = __uint_as_float(rd_u32(b, 0));
    for (int j = 0; j < 32; ++j) y[j] = d * (float)(int8_t)b[4 + s * 32 + j];
}
__device__ __forceinline__ void deq_q1_0(const uint8_t *b, int s, float *y) {
    float d = rd_f16(b, 0), neg_d = -d;
    for (int i = 0; i < 32; ++i) {
        int j = s * 32 + i;
        uint8_t bit = (b[2 + j / 8] >> (j % 8)) & 1;
        y[i] = bit ? d : neg_d;
    }
}
__device__ __forceinline__ void deq_q2_0(const uint8_t *b, int s, float *y) {
    float d = rd_f16(b, 0);
    for (int i = 0; i < 32; ++i) {
        int j = s * 32 + i;
        uint8_t q = (b[2 + j / 4] >> ((j % 4) * 2)) & 3;
        y[i] = (float)((int)q - 1) * d;
    }
}
__device__ __forceinline__ void deq_mxfp4(const uint8_t *b, int, float *y) {
    float d = e8m0_half(b[0]);
    const uint8_t *qs = b + 1;
    for (int j = 0; j < 16; ++j) {
        y[j] = (float)kvalues_fp4[qs[j] & 0x0F] * d;
        y[j + 16] = (float)kvalues_fp4[qs[j] >> 4] * d;
    }
}
__device__ __forceinline__ void deq_nvfp4(const uint8_t *b, int s2, float *y) {
    for (int t = 0; t < 2; ++t) {
        int s = s2 * 2 + t;
        float d = ue4m3_half(b[s]);
        const uint8_t *qs = b + 4 + s * 8;
        for (int j = 0; j < 8; ++j) {
            y[t * 16 + j] = (float)kvalues_fp4[qs[j] & 0x0F] * d;
            y[t * 16 + j + 8] = (float)kvalues_fp4[qs[j] >> 4] * d;
        }
    }
}
__device__ __forceinline__ void deq_iq4_nl(const uint8_t *b, int, float *y) {
    float d = rd_f16(b, 0);
    const uint8_t *qs = b + 2;
    for (int j = 0; j < 16; ++j) {
        y[j] = d * (float)kvalues_iq4nl[qs[j] & 0xF];
        y[j + 16] = d * (float)kvalues_iq4nl[qs[j] >> 4];
    }
}

// ---------------------------------------------------------------- K-кванты
__device__ __forceinline__ void deq_q2_k(const uint8_t *b, int s, float *y) {
    const uint8_t *scales = b;
    const uint8_t *qs = b + 16;
    float d = rd_f16(b, 80), dmin = rd_f16(b, 82);
    int n = (s / 4) * 128, shift = (s % 4) * 2;
    const uint8_t *q = qs + n / 4;
    uint8_t sc1 = scales[2 * s], sc2 = scales[2 * s + 1];
    float dl1 = d * (float)(sc1 & 0xF), ml1 = dmin * (float)(sc1 >> 4);
    float dl2 = d * (float)(sc2 & 0xF), ml2 = dmin * (float)(sc2 >> 4);
    for (int l = 0; l < 16; ++l) {
        y[l] = dl1 * (float)((q[l] >> shift) & 3) - ml1;
        y[16 + l] = dl2 * (float)((q[l + 16] >> shift) & 3) - ml2;
    }
}
__device__ __forceinline__ void deq_q3_k(const uint8_t *b, int s, float *y) {
    const uint32_t KMASK1 = 0x03030303u, KMASK2 = 0x0f0f0f0fu;
    const uint8_t *hmask = b;
    const uint8_t *qs = b + 32;
    float d_all = rd_f16(b, 108);
    uint32_t aux[4];
    aux[0] = rd_u32(b, 96);
    aux[1] = rd_u32(b, 100);
    aux[2] = rd_u32(b, 104);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    const int8_t *scales = (const int8_t *)aux;
    int n = (s / 4) * 128, shift = (s % 4) * 2;
    uint8_t m = (uint8_t)(1u << s); // бит hmask сквозной по всем 8 под-блокам
    const uint8_t *q = qs + n / 4;
    float dl1 = d_all * (float)((int)scales[2 * s] - 32);
    float dl2 = d_all * (float)((int)scales[2 * s + 1] - 32);
    for (int l = 0; l < 16; ++l) {
        int hi1 = (hmask[l] & m) ? 0 : 4;
        int hi2 = (hmask[l + 16] & m) ? 0 : 4;
        y[l] = dl1 * (float)((int)((q[l] >> shift) & 3) - hi1);
        y[16 + l] = dl2 * (float)((int)((q[l + 16] >> shift) & 3) - hi2);
    }
}
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}
__device__ __forceinline__ void deq_q4_k(const uint8_t *b, int s, float *y) {
    float d = rd_f16(b, 0), dmin = rd_f16(b, 2);
    const uint8_t *scales = b + 4;
    const uint8_t *q = b + 16 + (s / 2) * 32;
    uint8_t sc, m;
    scale_min_k4(s, scales, &sc, &m);
    float dl = d * (float)sc, ml = dmin * (float)m;
    if (s % 2 == 0) {
        for (int l = 0; l < 32; ++l) y[l] = dl * (float)(q[l] & 0xF) - ml;
    } else {
        for (int l = 0; l < 32; ++l) y[l] = dl * (float)(q[l] >> 4) - ml;
    }
}
__device__ __forceinline__ void deq_q5_k(const uint8_t *b, int s, float *y) {
    float d = rd_f16(b, 0), dmin = rd_f16(b, 2);
    const uint8_t *scales = b + 4;
    const uint8_t *qh = b + 16;
    const uint8_t *ql = b + 48 + (s / 2) * 32;
    uint8_t sc, m;
    scale_min_k4(s, scales, &sc, &m);
    float dl = d * (float)sc, ml = dmin * (float)m;
    uint8_t u = (uint8_t)(((s % 2 == 0) ? 1u : 2u) << (2 * (s / 2)));
    if (s % 2 == 0) {
        for (int l = 0; l < 32; ++l) {
            int hi = (qh[l] & u) ? 16 : 0;
            y[l] = dl * (float)((int)(ql[l] & 0xF) + hi) - ml;
        }
    } else {
        for (int l = 0; l < 32; ++l) {
            int hi = (qh[l] & u) ? 16 : 0;
            y[l] = dl * (float)((int)(ql[l] >> 4) + hi) - ml;
        }
    }
}
__device__ __forceinline__ void deq_q6_k(const uint8_t *b, int s, float *y) {
    float d = rd_f16(b, 208);
    int n = s / 4, t = s % 4;
    const uint8_t *ql = b + n * 64;
    const uint8_t *qh = b + 128 + n * 32;
    const uint8_t *sc = b + 192 + n * 8;
    for (int l = 0; l < 32; ++l) {
        int is = l / 16;
        int q;
        int8_t scv;
        if (t == 0) {
            q = (int)((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
            scv = (int8_t)sc[is];
        } else if (t == 1) {
            q = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            scv = (int8_t)sc[is + 2];
        } else if (t == 2) {
            q = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            scv = (int8_t)sc[is + 4];
        } else {
            q = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            scv = (int8_t)sc[is + 6];
        }
        y[l] = d * (float)scv * (float)q;
    }
}
__device__ __forceinline__ void deq_iq4_xs(const uint8_t *b, int ib, float *y) {
    float d = rd_f16(b, 0);
    uint16_t h = rd_u16(b, 2);
    const uint8_t *scales_l = b + 4;
    const uint8_t *q = b + 8 + ib * 16;
    int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((h >> (2 * ib)) & 3) << 4);
    float dl = d * (float)(ls - 32);
    for (int j = 0; j < 16; ++j) {
        y[j] = dl * (float)kvalues_iq4nl[q[j] & 0xF];
        y[j + 16] = dl * (float)kvalues_iq4nl[q[j] >> 4];
    }
}

// ---------------------------------------------------------------- IQ
__device__ __forceinline__ void deq_iq2_xxs(const uint8_t *b, int ib32, float *y) {
    float d = rd_f16(b, 0);
    const uint8_t *qs = b + 2;
    uint32_t aux0 = rd_u32(qs, 8 * ib32), aux1 = rd_u32(qs, 8 * ib32 + 4);
    float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
    for (int l = 0; l < 4; ++l) {
        const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + ((aux0 >> (8 * l)) & 0xFF));
        uint8_t signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
        for (int j = 0; j < 8; ++j) y[l * 8 + j] = db * (float)grid[j] * sgn(signs, j);
    }
}
__device__ __forceinline__ void deq_iq2_xs(const uint8_t *b, int ib32, float *y) {
    float d = rd_f16(b, 0);
    const uint8_t *qs = b + 2;
    const uint8_t *scales = b + 66;
    float db0 = d * (0.5f + (float)(scales[ib32] & 0xF)) * 0.25f;
    float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;
    for (int l = 0; l < 4; ++l) {
        uint16_t q = rd_u16(qs, 2 * (4 * ib32 + l));
        const uint8_t *grid = (const uint8_t *)(iq2xs_grid + (q & 511));
        uint8_t signs = ksigns_iq2xs[q >> 9];
        float db = (l / 2 == 0) ? db0 : db1;
        for (int j = 0; j < 8; ++j) y[l * 8 + j] = db * (float)grid[j] * sgn(signs, j);
    }
}
__device__ __forceinline__ void deq_iq2_s(const uint8_t *b, int ib32, float *y) {
    float d = rd_f16(b, 0);
    const uint8_t *qs = b + 2;
    const uint8_t *qh = b + 66;
    const uint8_t *scales = b + 74;
    const uint8_t *signs = qs + QK_K / 8;
    float db0 = d * (0.5f + (float)(scales[ib32] & 0xF)) * 0.25f;
    float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;
    for (int l = 0; l < 4; ++l) {
        float dl = (l / 2 == 0) ? db0 : db1;
        int idx = (int)qs[4 * ib32 + l] | ((((int)qh[ib32]) << (8 - 2 * l)) & 0x300);
        const uint8_t *grid = (const uint8_t *)(iq2s_grid + idx);
        uint8_t sg = signs[4 * ib32 + l];
        for (int j = 0; j < 8; ++j) y[l * 8 + j] = dl * (float)grid[j] * sgn(sg, j);
    }
}
__device__ __forceinline__ void deq_iq3_xxs(const uint8_t *b, int ib32, float *y) {
    float d = rd_f16(b, 0);
    const uint8_t *qs = b + 2;
    const uint8_t *scales_and_signs = qs + QK_K / 4;
    uint32_t aux32 = rd_u32(scales_and_signs, 4 * ib32);
    float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    for (int l = 0; l < 4; ++l) {
        uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
        const uint8_t *grid1 = (const uint8_t *)(iq3xxs_grid + qs[8 * ib32 + 2 * l]);
        const uint8_t *grid2 = (const uint8_t *)(iq3xxs_grid + qs[8 * ib32 + 2 * l + 1]);
        for (int j = 0; j < 4; ++j) {
            y[l * 8 + j] = db * (float)grid1[j] * sgn(signs, j);
            y[l * 8 + j + 4] = db * (float)grid2[j] * sgn(signs, j + 4);
        }
    }
}
__device__ __forceinline__ void deq_iq3_s(const uint8_t *b, int ib32, float *y) {
    float d = rd_f16(b, 0);
    const uint8_t *qs = b + 2;
    const uint8_t *qh = b + 66;
    const uint8_t *signs = b + 74;
    const uint8_t *scales = b + 106;
    int pair = ib32 / 2, half = ib32 % 2;
    float db = d * (float)(1 + 2 * (int)((scales[pair] >> (4 * half)) & 0xF));
    const uint8_t *q = qs + pair * 16 + half * 8;
    const uint8_t *sg = signs + pair * 8 + half * 4;
    int h = qh[pair * 2 + half];
    for (int l = 0; l < 4; ++l) {
        int g1 = (int)q[2 * l] | ((h << (8 - 2 * l)) & 256);
        int g2 = (int)q[2 * l + 1] | ((h << (7 - 2 * l)) & 256);
        const uint8_t *grid1 = (const uint8_t *)(iq3s_grid + g1);
        const uint8_t *grid2 = (const uint8_t *)(iq3s_grid + g2);
        for (int j = 0; j < 4; ++j) {
            y[l * 8 + j] = db * (float)grid1[j] * sgn(sg[l], j);
            y[l * 8 + j + 4] = db * (float)grid2[j] * sgn(sg[l], j + 4);
        }
    }
}
__device__ __forceinline__ void deq_iq1_s(const uint8_t *b, int ib, float *y) {
    float d = rd_f16(b, 0);
    const uint8_t *qs = b + 2;
    uint16_t h = rd_u16(b + 34, 2 * ib);
    float dl = d * (float)(2 * (int)((h >> 12) & 7) + 1);
    float delta = (h & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
    for (int l = 0; l < 4; ++l) {
        int idx = (int)qs[4 * ib + l] | ((int)((h >> (3 * l)) & 7) << 8);
        const int8_t *grid = (const int8_t *)(iq1s_grid + idx);
        for (int j = 0; j < 8; ++j) y[l * 8 + j] = dl * ((float)grid[j] + delta);
    }
}
__device__ __forceinline__ void deq_iq1_m(const uint8_t *b, int ib, float *y) {
    const uint8_t *qs = b;
    const uint8_t *qh = b + 32;
    uint16_t sc[4] = {rd_u16(b, 48), rd_u16(b, 50), rd_u16(b, 52), rd_u16(b, 54)};
    uint16_t bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    float d = __half2float(__ushort_as_half(bits));
    float dl1 = d * (float)(2 * (int)((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) + 1);
    float dl2 = d * (float)(2 * (int)((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
    const uint8_t *q = qs + 4 * ib;
    const uint8_t *h = qh + 2 * ib;
    int idx[4] = {(int)q[0] | (((int)h[0] << 8) & 0x700), (int)q[1] | (((int)h[0] << 4) & 0x700),
                  (int)q[2] | (((int)h[1] << 8) & 0x700), (int)q[3] | (((int)h[1] << 4) & 0x700)};
    float delta[4] = {(h[0] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA, (h[0] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA,
                      (h[1] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA, (h[1] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA};
    for (int l = 0; l < 4; ++l) {
        float dl = l < 2 ? dl1 : dl2;
        const int8_t *grid = (const int8_t *)(iq1s_grid + idx[l]);
        for (int j = 0; j < 8; ++j) y[l * 8 + j] = dl * ((float)grid[j] + delta[l]);
    }
}

// ---------------------------------------------------------------- тернарные
__device__ __forceinline__ float tq_val(uint8_t byte, uint8_t pow, float d) {
    uint8_t q = (uint8_t)(byte * pow);
    int xi = (int)(((uint16_t)q * 3) >> 8);
    return (float)(xi - 1) * d;
}
__device__ __forceinline__ void deq_tq1_0(const uint8_t *b, int s, float *y) {
    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    const uint8_t *qs = b;
    const uint8_t *qh = b + 48;
    float d = rd_f16(b, 52);
    if (s < 5) {
        // 160 значений: n = s, m = 0..32 из первых 32 байт.
        for (int m = 0; m < 32; ++m) y[m] = tq_val(qs[m], pow3[s], d);
    } else if (s < 7) {
        // 80 значений из байт 32..48: под-блок s покрывает n = 2(s-5), 2(s-5)+1 по 16.
        int n0 = 2 * (s - 5);
        for (int m = 0; m < 16; ++m) {
            y[m] = tq_val(qs[32 + m], pow3[n0], d);
            y[16 + m] = tq_val(qs[32 + m], pow3[n0 + 1], d);
        }
    } else {
        // n = 4 из байт 32..48 (16 значений), затем qh: n = 0..4 × 4 байта.
        for (int m = 0; m < 16; ++m) y[m] = tq_val(qs[32 + m], pow3[4], d);
        for (int n = 0; n < 4; ++n)
            for (int j = 0; j < 4; ++j) y[16 + n * 4 + j] = tq_val(qh[j], pow3[n], d);
    }
}
__device__ __forceinline__ void deq_tq2_0(const uint8_t *b, int s, float *y) {
    const uint8_t *qs = b + (s / 4) * 32;
    float d = rd_f16(b, 64);
    int l = s % 4;
    for (int m = 0; m < 32; ++m) {
        int q = (qs[m] >> (l * 2)) & 3;
        y[m] = (float)(q - 1) * d;
    }
}


// ---------------------------------------------------------------- NVFP4/MXFP8 движка
// NVFP4 synaptix: packed [N, K/2] линейно (младший ниббл — чётный индекс),
// шкалы E4M3 тайл-мажорно (см. nvfp4_quant.cu: tile_scale_offset).
// E4M3 → f32 без ветвлений: байт раскладывается в биты f16 (экспонента
// сдвинута на 15 − 7 = 8, субнормали f16 покрывают субнормали E4M3), затем
// ×2⁸. NaN (S.1111.111) отдельно — в весах его не бывает, но декод честный.
__device__ __forceinline__ float syn_decode_e4m3(uint8_t byte) {
    if ((byte & 0x7F) == 0x7F) return __int_as_float(0x7FC00000);
    const unsigned short hb = (unsigned short)(((byte & 0x80u) << 8) | ((byte & 0x7Fu) << 7));
    return __half2float(__ushort_as_half(hb)) * 256.0f;
}
// E2M1 → f32: удвоенные модули {0,1,2,3,4,6,8,12} — нибблы одной константы;
// локальный массив-таблица уходил в local memory и тормозил GEMV в разы.
__device__ __forceinline__ float syn_decode_e2m1(uint8_t nib) {
    const float mag = (float)((0xC8643210u >> ((nib & 7u) * 4u)) & 0xFu) * 0.5f;
    return __int_as_float(__float_as_int(mag) | ((unsigned)(nib & 8u) << 28));
}
__device__ __forceinline__ unsigned syn_nvfp4_scale_off(unsigned outer, unsigned block_col, unsigned sf_inner_dim) {
    unsigned tile_row = outer >> 7, tile_col = block_col >> 2;
    unsigned local_outer = outer & 127u, local_inner = block_col & 3u;
    unsigned tile_base = (tile_col * 4u + tile_row * sf_inner_dim) * 128u;
    return tile_base + (local_outer & 31u) * 16u + (local_outer >> 5) * 4u + local_inner;
}
/// Под-блок `s` (32 значения = два блока по 16) строки `row` веса [N, K].
/// 16 байт веса одной векторной загрузкой (адрес кратен 16: K кратно 32).
__device__ __forceinline__ void deq_nvfp4_syn(const uint8_t *packed, const uint8_t *scales, unsigned row,
                                              unsigned K, unsigned s, float *y) {
    unsigned sf_inner = ((K + 63u) / 64u) * 4u;
    const uint4 v = *reinterpret_cast<const uint4 *>(packed + ((size_t)row * K + (size_t)s * 32u) / 2u);
    const unsigned words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        float sc = syn_decode_e4m3(scales[syn_nvfp4_scale_off(row, s * 2u + h, sf_inner)]);
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            unsigned w = words[h * 2 + q];
#pragma unroll
            for (int j = 0; j < 8; ++j) y[h * 16 + q * 8 + j] = syn_decode_e2m1((w >> (4 * j)) & 0xFu) * sc;
        }
    }
}
// MXFP8 synaptix: e4m3 [N, K] натурально + E8M0 [N, K/32].
__device__ __forceinline__ float syn_e4m3_to_f32(uint8_t byte) { return syn_decode_e4m3(byte); }
__device__ __forceinline__ void deq_mxfp8_syn(const uint8_t *packed, const uint8_t *scales, unsigned row,
                                              unsigned K, unsigned s, float *y) {
    // E8M0 × 2⁸ (сдвиг экспоненты E4M3 → f16) одним множителем.
    const float sc_raw = __uint_as_float((unsigned)scales[(size_t)row * (K / 32u) + s] << 23);
    const float sc = sc_raw * 256.0f;
    (void)sc;
    (void)sc_raw;
    const uint4 *p = reinterpret_cast<const uint4 *>(packed + (size_t)row * K + (size_t)s * 32u);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const uint4 v = p[q];
        const unsigned words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            unsigned w = words[t];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
            // Ada+: аппаратный cvt пары E4M3 → f16x2 (масштаб без ×2⁸).
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                unsigned h2;
                unsigned short pair = (unsigned short)(w >> (16 * j));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(pair));
                float2 f = __half22float2(*reinterpret_cast<__half2 *>(&h2));
                y[q * 16 + t * 4 + 2 * j] = f.x * sc_raw;
                y[q * 16 + t * 4 + 2 * j + 1] = f.y * sc_raw;
            }
#else
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                unsigned b = (w >> (8 * j)) & 0xFFu;
                unsigned short hb = (unsigned short)(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
                y[q * 16 + t * 4 + j] = __half2float(__ushort_as_half(hb)) * sc;
            }
#endif
        }
    }
}
