// Портируемый GEMV по квантованным весам: out[m, N] = x[m, K] · W[N, K]ᵀ,
// M ≤ 8 строк активации (декод, спекуляция, малый батч). Веса любого
// одноблобного формата (SQ, типы ggml) и NVFP4/MXFP8 движка декодируются в
// регистрах (blockq_decode.cuh), активация — f16 или bf16 (модуль под
// -DSYN_ACT_BF16), накопление f32. Варп считает одну строку N: лейны делят
// под-блоки по 32 вдоль K (чтение веса по варпу непрерывное), редукция
// шаффлом. Это то, что делают dequantize_mul_mat_vec llama.cpp; dp4a-вариант
// (int8-активация) — отдельным ядром позже.
//
// Батчевый вариант (эксперты MoE): таблицы указателей на карте, blockIdx.y —
// эксперт, у каждого своя строка активации и строка выхода, M = 1.
#ifdef SYN_ACT_BF16
#include <cuda_bf16.h>
typedef __nv_bfloat16 act_t;
typedef __nv_bfloat162 act2_t;
__device__ __forceinline__ float2 act2_to_f2(act2_t v) { return __bfloat1622float2(v); }
__device__ __forceinline__ act_t to_act(float v) { return __float2bfloat16_rn(v); }
#else
typedef __half act_t;
typedef __half2 act2_t;
__device__ __forceinline__ float2 act2_to_f2(act2_t v) { return __half22float2(v); }
__device__ __forceinline__ act_t to_act(float v) { return __float2half_rn(v); }
#endif

#define GEMV_MAX_M 8
#define GEMV_WARPS 4

// Скалярное произведение 32 декодированных весов с 32 значениями активации
// (64 байта, четыре uint4).
__device__ __forceinline__ float dot32(const float *y, const act_t *x) {
    const uint4 *xp = reinterpret_cast<const uint4 *>(x);
    float acc = 0.f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        uint4 v = xp[q];
        const act2_t *h = reinterpret_cast<const act2_t *>(&v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f = act2_to_f2(h[i]);
            acc += y[q * 8 + 2 * i] * f.x;
            acc += y[q * 8 + 2 * i + 1] * f.y;
        }
    }
    return acc;
}

// LOAD32(row, s, y): декодировать под-блок s строки row в y[32].
#define GEMV_BODY(LOAD32)                                                                  \
    const int lane = threadIdx.x & 31;                                                     \
    const unsigned subs = K / 32;                                                          \
    float acc[GEMV_MAX_M];                                                                 \
    _Pragma("unroll") for (int m = 0; m < GEMV_MAX_M; ++m) acc[m] = 0.f;                   \
    for (unsigned s = lane; s < subs; s += 32) {                                           \
        float y[32];                                                                       \
        LOAD32(row, s, y);                                                                 \
        const act_t *xs = x + (size_t)s * 32;                                              \
        for (int m = 0; m < M; ++m) acc[m] += dot32(y, xs + (size_t)m * x_stride);         \
    }                                                                                      \
    _Pragma("unroll") for (int m = 0; m < GEMV_MAX_M; ++m) {                               \
        float v = acc[m];                                                                  \
        _Pragma("unroll") for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o); \
        acc[m] = v;                                                                        \
    }                                                                                      \
    if (lane == 0) {                                                                       \
        for (int m = 0; m < M; ++m) out[(size_t)m * out_stride + row] = to_act(acc[m]);    \
    }

// Одноблобные форматы: BB байт на блок, SUBS_PER_BLK под-блоков в блоке.
#define GEMV_BLOB(NAME, FN, BB, SUBS_PER_BLK)                                                      \
    extern "C" __global__ void NAME(const uint8_t *__restrict__ w, const uint8_t *__restrict__ sw, \
                                    const act_t *__restrict__ x, act_t *__restrict__ out,         \
                                    unsigned N, unsigned K, int M, unsigned x_stride,             \
                                    unsigned out_stride, unsigned row_bytes) {                    \
        const unsigned row = blockIdx.x * GEMV_WARPS + (threadIdx.x >> 5);                        \
        if (row >= N) return;                                                                     \
        const uint8_t *wrow = w + (size_t)row * row_bytes;                                        \
        (void)sw;                                                                                 \
        _Pragma("unroll") for (int m = 0; m < GEMV_MAX_M; ++m) {}                                 \
        GEMV_BODY(([&](unsigned r, unsigned s, float *y) {                                        \
            (void)r;                                                                              \
            FN(wrow + (size_t)(s / SUBS_PER_BLK) * (BB), (int)(s % SUBS_PER_BLK), y);             \
        }))                                                                                       \
    }                                                                                             \
    extern "C" __global__ void NAME##_batched(                                                    \
        const unsigned long long *__restrict__ w_ptrs, const unsigned long long *__restrict__ sw_ptrs, \
        const unsigned long long *__restrict__ x_ptrs, const unsigned long long *__restrict__ out_ptrs, \
        unsigned N, unsigned K, unsigned row_bytes) {                                             \
        const unsigned e = blockIdx.y;                                                            \
        const uint8_t *w = (const uint8_t *)w_ptrs[e];                                            \
        const act_t *x = (const act_t *)x_ptrs[e];                                                \
        act_t *out = (act_t *)out_ptrs[e];                                                        \
        (void)sw_ptrs;                                                                            \
        const int M = 1;                                                                          \
        const unsigned x_stride = 0, out_stride = 0;                                              \
        const unsigned row = blockIdx.x * GEMV_WARPS + (threadIdx.x >> 5);                        \
        if (row >= N) return;                                                                     \
        const uint8_t *wrow = w + (size_t)row * row_bytes;                                        \
        GEMV_BODY(([&](unsigned r, unsigned s, float *y) {                                        \
            (void)r;                                                                              \
            FN(wrow + (size_t)(s / SUBS_PER_BLK) * (BB), (int)(s % SUBS_PER_BLK), y);             \
        }))                                                                                       \
    }

// NVFP4/MXFP8 движка: декодер сам считает адрес по (row, K, s).
#define GEMV_SYN(NAME, FN)                                                                         \
    extern "C" __global__ void NAME(const uint8_t *__restrict__ w, const uint8_t *__restrict__ sw, \
                                    const act_t *__restrict__ x, act_t *__restrict__ out,         \
                                    unsigned N, unsigned K, int M, unsigned x_stride,             \
                                    unsigned out_stride, unsigned row_bytes) {                    \
        const unsigned row = blockIdx.x * GEMV_WARPS + (threadIdx.x >> 5);                        \
        if (row >= N) return;                                                                     \
        (void)row_bytes;                                                                          \
        GEMV_BODY(([&](unsigned r, unsigned s, float *y) { FN(w, sw, r, K, s, y); }))             \
    }                                                                                             \
    extern "C" __global__ void NAME##_batched(                                                    \
        const unsigned long long *__restrict__ w_ptrs, const unsigned long long *__restrict__ sw_ptrs, \
        const unsigned long long *__restrict__ x_ptrs, const unsigned long long *__restrict__ out_ptrs, \
        unsigned N, unsigned K, unsigned row_bytes) {                                             \
        const unsigned e = blockIdx.y;                                                            \
        const uint8_t *w = (const uint8_t *)w_ptrs[e];                                            \
        const uint8_t *sw = (const uint8_t *)sw_ptrs[e];                                          \
        const act_t *x = (const act_t *)x_ptrs[e];                                                \
        act_t *out = (act_t *)out_ptrs[e];                                                        \
        (void)row_bytes;                                                                          \
        const int M = 1;                                                                          \
        const unsigned x_stride = 0, out_stride = 0;                                              \
        const unsigned row = blockIdx.x * GEMV_WARPS + (threadIdx.x >> 5);                        \
        if (row >= N) return;                                                                     \
        GEMV_BODY(([&](unsigned r, unsigned s, float *y) { FN(w, sw, r, K, s, y); }))             \
    }

GEMV_BLOB(gemv_sq1, deq_sq<1>, 52, 8)
GEMV_BLOB(gemv_sq2, deq_sq<2>, 84, 8)
GEMV_BLOB(gemv_sq3, deq_sq<3>, 116, 8)
GEMV_BLOB(gemv_sq4, deq_sq<4>, 148, 8)
GEMV_BLOB(gemv_sq5, deq_sq<5>, 180, 8)
GEMV_BLOB(gemv_sq6, deq_sq<6>, 212, 8)
GEMV_BLOB(gemv_sq7, deq_sq<7>, 244, 8)
GEMV_BLOB(gemv_sq8, deq_sq<8>, 276, 8)

GEMV_BLOB(gemv_q4_0, deq_q4_0, 18, 1)
GEMV_BLOB(gemv_q4_1, deq_q4_1, 20, 1)
GEMV_BLOB(gemv_q5_0, deq_q5_0, 22, 1)
GEMV_BLOB(gemv_q5_1, deq_q5_1, 24, 1)
GEMV_BLOB(gemv_q8_0, deq_q8_0, 34, 1)
GEMV_BLOB(gemv_q1_0, deq_q1_0, 18, 4)
GEMV_BLOB(gemv_q2_0, deq_q2_0, 18, 2)
GEMV_BLOB(gemv_mxfp4, deq_mxfp4, 17, 1)
GEMV_BLOB(gemv_nvfp4, deq_nvfp4, 36, 2)
GEMV_BLOB(gemv_iq4_nl, deq_iq4_nl, 18, 1)
GEMV_BLOB(gemv_q2_k, deq_q2_k, 84, 8)
GEMV_BLOB(gemv_q3_k, deq_q3_k, 110, 8)
GEMV_BLOB(gemv_q4_k, deq_q4_k, 144, 8)
GEMV_BLOB(gemv_q5_k, deq_q5_k, 176, 8)
GEMV_BLOB(gemv_q6_k, deq_q6_k, 210, 8)
GEMV_BLOB(gemv_iq4_xs, deq_iq4_xs, 136, 8)
GEMV_BLOB(gemv_iq2_xxs, deq_iq2_xxs, 66, 8)
GEMV_BLOB(gemv_iq2_xs, deq_iq2_xs, 74, 8)
GEMV_BLOB(gemv_iq2_s, deq_iq2_s, 82, 8)
GEMV_BLOB(gemv_iq3_xxs, deq_iq3_xxs, 98, 8)
GEMV_BLOB(gemv_iq3_s, deq_iq3_s, 110, 8)
GEMV_BLOB(gemv_iq1_s, deq_iq1_s, 50, 8)
GEMV_BLOB(gemv_iq1_m, deq_iq1_m, 56, 8)
GEMV_BLOB(gemv_tq1_0, deq_tq1_0, 54, 8)
GEMV_BLOB(gemv_tq2_0, deq_tq2_0, 66, 8)

GEMV_SYN(gemv_nvfp4_syn, deq_nvfp4_syn)
GEMV_SYN(gemv_mxfp8_syn, deq_mxfp8_syn)
