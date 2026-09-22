// Энкодер SQ (synaptix quant, b = 1..8 бит): f16/bf16-матрица [rows, K] →
// блоб супер-блоков по 256 (см. synaptix_core::quant::sq). Один поток —
// под-блок из 32 значений: min/max → RTN-кандидат → 21 шкала вокруг него с
// пересчётом (шкала, минимум) методом наименьших квадратов; восемь потоков
// супер-блока сводят максимумы шаффлом. Порядок операций и --fmad=false
// повторяют CPU-эталон `fit_sub_block`/`quant_super_block` бит в бит.
#include <cuda_fp16.h>
#ifdef SYN_IN_BF16
#include <cuda_bf16.h>
typedef __nv_bfloat16 syn_in_t;
#define SYN_IN_TO_F(v) __bfloat162float(v)
#else
typedef __half syn_in_t;
#define SYN_IN_TO_F(v) __half2float(v)
#endif

typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;

#define SQ_SUPER 256
#define SQ_SUB 32
#define SQ_SUBS 8
#define SQ_HEADER 20
#define SQ_STEPS 21

__device__ __forceinline__ float sq_clamp(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }

// (scale, m): w = scale·q − m, m ≥ 0. См. `fit_sub_block` в sq.rs.
__device__ __forceinline__ void sq_fit(const float *v, float qmax, float &scale_out, float &m_out) {
    float mn = v[0], mx = v[0];
#pragma unroll
    for (int i = 0; i < SQ_SUB; ++i) { mn = fminf(mn, v[i]); mx = fmaxf(mx, v[i]); }
    mn = fminf(mn, 0.f);
    if (!(mx > mn)) { scale_out = 0.f; m_out = 0.f; return; }
    float s0 = (mx - mn) / qmax;
    float m0 = -mn;
    float best_err = 0.f;
    for (int i = 0; i < SQ_SUB; ++i) {
        float q = sq_clamp(roundf((v[i] + m0) / s0), 0.f, qmax);
        float diff = s0 * q - m0 - v[i];
        best_err += diff * diff;
    }
    float bs = s0, bm = m0;
    for (int is = 0; is < SQ_STEPS; ++is) {
        float s = (mx - mn) / (qmax + (-1.0f + 0.1f * (float)is));
        if (!(s > 0.f)) continue;
        float sum_l = 0.f, sum_l2 = 0.f, sum_x = 0.f, sum_xl = 0.f;
        for (int i = 0; i < SQ_SUB; ++i) {
            float l = sq_clamp(roundf((v[i] - mn) / s), 0.f, qmax);
            sum_l += l;
            sum_l2 += l * l;
            sum_x += v[i];
            sum_xl += v[i] * l;
        }
        const float n = (float)SQ_SUB;
        float det = n * sum_l2 - sum_l * sum_l;
        float a, b;
        if (det > 0.f) {
            a = (n * sum_xl - sum_x * sum_l) / det;
            b = (sum_l2 * sum_x - sum_l * sum_xl) / det;
            if (b > 0.f) { a = (sum_l2 > 0.f) ? (sum_xl / sum_l2) : 0.f; b = 0.f; }
        } else if (sum_l2 > 0.f) {
            a = sum_xl / sum_l2; b = 0.f;
        } else {
            continue;
        }
        if (!(a > 0.f)) continue;
        float err = 0.f;
        for (int i = 0; i < SQ_SUB; ++i) {
            float l = sq_clamp(roundf((v[i] - mn) / s), 0.f, qmax);
            float diff = a * l + b - v[i];
            err += diff * diff;
        }
        if (err < best_err) { best_err = err; bs = a; bm = -b; }
    }
    scale_out = bs; m_out = bm;
}

// Поток = под-блок; blockDim кратен 8, восемь соседних потоков = супер-блок.
extern "C" __global__ void sq_quant(const syn_in_t *__restrict__ in, uint8_t *__restrict__ out,
                                    unsigned rows, unsigned k, unsigned bits) {
    const unsigned supers = (k + SQ_SUPER - 1) / SQ_SUPER;
    const unsigned total = rows * supers * SQ_SUBS;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = tid < total;
    // Неактивные потоки участвуют в шаффле нулями.
    unsigned row = 0, sb = 0, s = 0;
    if (active) { row = tid / (supers * SQ_SUBS); sb = (tid / SQ_SUBS) % supers; s = tid % SQ_SUBS; }
    const unsigned off = sb * SQ_SUPER + s * SQ_SUB;
    const float qmax = (float)((1u << bits) - 1u);
    float v[SQ_SUB];
    if (active && off < k) {
        const syn_in_t *p = in + (size_t)row * k + off;
#pragma unroll
        for (int i = 0; i < SQ_SUB; ++i) v[i] = SYN_IN_TO_F(p[i]);
    } else {
#pragma unroll
        for (int i = 0; i < SQ_SUB; ++i) v[i] = 0.f;
    }
    float scale, m;
    sq_fit(v, qmax, scale, m);
    // Максимумы по восьми под-блокам супер-блока (ширина шаффла 8).
    float smax = scale, mmax = m;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
        smax = fmaxf(smax, __shfl_xor_sync(0xffffffffu, smax, o, 8));
        mmax = fmaxf(mmax, __shfl_xor_sync(0xffffffffu, mmax, o, 8));
    }
    if (!active) return;
    const __half d16 = __float2half_rn(smax / 255.0f);
    const __half dmin16 = __float2half_rn(mmax / 255.0f);
    const float d = __half2float(d16);
    const float dmin = __half2float(dmin16);
    const unsigned sbb = SQ_HEADER + SQ_SUBS * bits * 4;
    const unsigned rb = supers * sbb;
    uint8_t *blk = out + (size_t)row * rb + (size_t)sb * sbb;
    if (s == 0) {
        const uint16_t dh = __half_as_ushort(d16);
        const uint16_t mh = __half_as_ushort(dmin16);
        blk[0] = (uint8_t)(dh & 0xff); blk[1] = (uint8_t)(dh >> 8);
        blk[2] = (uint8_t)(mh & 0xff); blk[3] = (uint8_t)(mh >> 8);
    }
    const uint8_t sc = (d > 0.f) ? (uint8_t)sq_clamp(roundf(scale / d), 0.f, 255.f) : 0;
    const uint8_t mi = (dmin > 0.f) ? (uint8_t)sq_clamp(roundf(m / dmin), 0.f, 255.f) : 0;
    blk[4 + s] = sc;
    blk[12 + s] = mi;
    const float dl = d * (float)sc;
    const float ml = dmin * (float)mi;
    uint32_t planes[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) planes[j] = 0u;
#pragma unroll
    for (int i = 0; i < SQ_SUB; ++i) {
        uint32_t q = (dl > 0.f) ? (uint32_t)sq_clamp(roundf((v[i] + ml) / dl), 0.f, qmax) : 0u;
#pragma unroll
        for (int j = 0; j < 8; ++j) planes[j] |= ((q >> j) & 1u) << i;
    }
    uint32_t *pp = reinterpret_cast<uint32_t *>(blk + SQ_HEADER + s * bits * 4);
    for (unsigned j = 0; j < bits; ++j) pp[j] = planes[j];
}
