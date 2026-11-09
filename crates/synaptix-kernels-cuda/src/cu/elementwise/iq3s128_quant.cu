// Кодировщик IQ3_S-128 на карте: [rows, k] f16/bf16 → блоб (56 байт на 128).
// Повторяет CPU-эталон synaptix_core::quant::ggml::quantize_iq3_s128_block
// бит в бит (модуль с --fmad=false, тот же порядок операций): блок CUDA —
// один 128-блок строки, восемь варпов — восемь групп по 4 значения под-блока;
// варп ищет ближайшую точку сетки (лейн — 16 точек, затем сведение
// «минимум, при равенстве меньший индекс»).
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifdef SYN_IN_BF16
typedef __nv_bfloat16 in_t;
__device__ __forceinline__ float in_f(in_t v) { return __bfloat162float(v); }
#else
typedef __half in_t;
__device__ __forceinline__ float in_f(in_t v) { return __half2float(v); }
#endif

// Модульная таблица — ggml_tables.cuh подключается перед этим файлом.

struct Iq3Smem {
    float x[128];
    float ax[128];
    float w[128];
    float grid[512 * 4];
    float gerr[8];
    unsigned short gidx[8];
};

// Ближайшая точка сетки для группы `gi` под-блока `sb` при масштабе `s`.
// Возвращает (через smem) индекс и ошибку группы в исходных единицах.
__device__ __forceinline__ void iq3_group(Iq3Smem &m, int sb, int gi, float s) {
    const int lane = threadIdx.x & 31;
    const int base = 32 * sb + 4 * gi;
    float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
    float a0 = m.ax[base] * inv, a1 = m.ax[base + 1] * inv, a2 = m.ax[base + 2] * inv, a3 = m.ax[base + 3] * inv;
    float w0 = m.w[base], w1 = m.w[base + 1], w2 = m.w[base + 2], w3 = m.w[base + 3];
    float be = __int_as_float(0x7f800000);
    int bi = 0;
    for (int t = 0; t < 16; ++t) {
        int i = lane + 32 * t;
        const float *p = &m.grid[4 * i];
        float e = w0 * (a0 - p[0]) * (a0 - p[0]) + w1 * (a1 - p[1]) * (a1 - p[1]) +
                  w2 * (a2 - p[2]) * (a2 - p[2]) + w3 * (a3 - p[3]) * (a3 - p[3]);
        if (e < be) { be = e; bi = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float oe = __shfl_xor_sync(0xffffffffu, be, o);
        int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (oe < be || (oe == be && oi < bi)) { be = oe; bi = oi; }
    }
    if (lane == 0) {
        const float *p = &m.grid[4 * bi];
        float err = 0.0f;
        for (int j = 0; j < 4; ++j) {
            float d = m.ax[base + j] - s * p[j];
            err += m.w[base + j] * d * d;
        }
        m.gerr[gi] = err;
        m.gidx[gi] = (unsigned short)bi;
    }
}

// Ошибка под-блока при масштабе s (сумма групп в порядке 0..7, как на CPU);
// коды — в m.gidx. Все нити блока должны войти.
__device__ __forceinline__ float iq3_codes(Iq3Smem &m, int sb, float s) {
    const int warp = threadIdx.x >> 5;
    __syncthreads();
    iq3_group(m, sb, warp, s);
    __syncthreads();
    float err = 0.0f;
    for (int gi = 0; gi < 8; ++gi) err += m.gerr[gi];
    return err;
}

extern "C" __global__ void iq3s128_quant(const in_t *__restrict__ in, const float *__restrict__ imp,
                                         unsigned char *__restrict__ out, unsigned rows, unsigned k) {
    __shared__ Iq3Smem m;
    const unsigned blocks_per_row = k / 128;
    const unsigned row = blockIdx.x / blocks_per_row;
    const unsigned blk = blockIdx.x % blocks_per_row;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    for (int i = tid; i < 512; i += blockDim.x) {
        uint32_t v = iq3s_grid[i];
        m.grid[4 * i] = (float)(v & 0xff);
        m.grid[4 * i + 1] = (float)((v >> 8) & 0xff);
        m.grid[4 * i + 2] = (float)((v >> 16) & 0xff);
        m.grid[4 * i + 3] = (float)(v >> 24);
    }
    const in_t *src = in + (size_t)row * k + (size_t)blk * 128;
    if (tid < 128) {
        float v = in_f(src[tid]);
        m.x[tid] = v;
        m.ax[tid] = fabsf(v);
    }
    __syncthreads();
    // σ² и веса ошибки — последовательно, как на CPU.
    __shared__ float sigma2_s;
    if (tid == 0) {
        float s2 = 0.0f;
        for (int i = 0; i < 128; ++i) s2 += m.x[i] * m.x[i];
        sigma2_s = s2 / 128.0f;
    }
    __syncthreads();
    if (tid < 128) {
        float h = imp ? imp[(size_t)blk * 128 + tid] : 1.0f;
        m.w[tid] = h * sqrtf(sigma2_s + m.x[tid] * m.x[tid]);
    }
    __syncthreads();

    // 1) лучший вещественный масштаб под-блоков.
    float best_s[4] = {0.f, 0.f, 0.f, 0.f};
    for (int sb = 0; sb < 4; ++sb) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, m.ax[32 * sb + i]);
        if (amax == 0.0f) continue;
        float be = __int_as_float(0x7f800000);
        for (int c = 0; c < 17; ++c) {
            float s = amax / 15.0f * (0.6f + 0.05f * (float)c);
            float e = iq3_codes(m, sb, s);
            if (e < be) { be = e; best_s[sb] = s; }
        }
    }
    float smax = fmaxf(fmaxf(best_s[0], best_s[1]), fmaxf(best_s[2], best_s[3]));
    unsigned char *dst = out + ((size_t)row * blocks_per_row + blk) * 56;
    if (smax == 0.0f) {
        for (int i = tid; i < 56; i += blockDim.x) dst[i] = 0;
        return;
    }
    // 2) общий d (f16) и 4-битные шкалы.
    float best_tot = __int_as_float(0x7f800000), best_d = 0.0f;
    unsigned char best_sc[4] = {0, 0, 0, 0};
    unsigned short best_codes[4][8];
    bool have = false;
    for (int c = 0; c < 9; ++c) {
        float d = smax / 31.0f * (0.9f + 0.025f * (float)c);
        d = __half2float(__float2half_rn(d));
        if (d <= 0.0f) continue;
        unsigned char sc[4];
        unsigned short codes[4][8];
        float tot = 0.0f;
        for (int sb = 0; sb < 4; ++sb) {
            float qf = roundf((best_s[sb] / d - 1.0f) / 2.0f);
            qf = fminf(fmaxf(qf, 0.0f), 15.0f);
            int q = (int)qf;
            int cand[3] = {q > 0 ? q - 1 : 0, q, q < 15 ? q + 1 : 15};
            float be = __int_as_float(0x7f800000);
            for (int t = 0; t < 3; ++t) {
                float s = d * (float)(1 + 2 * cand[t]);
                float e = iq3_codes(m, sb, s);
                if (e < be) {
                    be = e;
                    sc[sb] = (unsigned char)cand[t];
                    for (int gi = 0; gi < 8; ++gi) codes[sb][gi] = m.gidx[gi];
                }
            }
            tot += be;
        }
        if (!have || tot < best_tot) {
            have = true;
            best_tot = tot;
            best_d = d;
            for (int sb = 0; sb < 4; ++sb) {
                best_sc[sb] = sc[sb];
                for (int gi = 0; gi < 8; ++gi) best_codes[sb][gi] = codes[sb][gi];
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        for (int i = 0; i < 56; ++i) dst[i] = 0;
        __half dh = __float2half_rn(best_d);
        unsigned short db = *reinterpret_cast<unsigned short *>(&dh);
        dst[0] = (unsigned char)(db & 0xff);
        dst[1] = (unsigned char)(db >> 8);
        for (int sb = 0; sb < 4; ++sb) {
            for (int gi = 0; gi < 8; ++gi) {
                int g = best_codes[sb][gi];
                int group = sb * 8 + gi;
                dst[2 + group] = (unsigned char)(g & 0xff);
                if (g & 0x100) dst[34 + group / 8] |= (unsigned char)(1 << (group % 8));
            }
            dst[54 + sb / 2] |= (unsigned char)(best_sc[sb] << (4 * (sb % 2)));
        }
        for (int i = 0; i < 128; ++i)
            if (m.x[i] < 0.0f) dst[38 + i / 8] |= (unsigned char)(1 << (i % 8));
    }
}
