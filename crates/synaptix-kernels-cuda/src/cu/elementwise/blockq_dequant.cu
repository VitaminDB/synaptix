// Точки входа деквантования: [rows, k] из блоба формата в f16/bf16.
// Перед файлом склеены ggml_tables.cuh и blockq_decode.cuh (см. elementwise/blockq.rs).
#ifdef SYN_OUT_BF16
#include <cuda_bf16.h>
typedef __nv_bfloat16 out_t;
__device__ __forceinline__ out_t to_out(float v) { return __float2bfloat16_rn(v); }
#else
typedef __half out_t;
__device__ __forceinline__ out_t to_out(float v) { return __float2half_rn(v); }
#endif

// ---------------------------------------------------------------- точки входа
// src: [rows][row_bytes], out: [rows][k]; поток = под-блок из 32 значений.
#define DEQ_ENTRY(NAME, FN, BB, SUBS_PER_BLK)                                                    \
    extern "C" __global__ void NAME(const uint8_t *__restrict__ src, out_t *__restrict__ out,   \
                                    unsigned rows, unsigned k, unsigned row_bytes) {           \
        unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;                                  \
        unsigned subs_per_row = k / 32;                                                        \
        if (tid >= rows * subs_per_row) return;                                                \
        unsigned row = tid / subs_per_row;                                                     \
        unsigned s = tid % subs_per_row;                                                       \
        const uint8_t *blk = src + (size_t)row * row_bytes + (size_t)(s / SUBS_PER_BLK) * (BB);\
        float y[32];                                                                           \
        FN(blk, (int)(s % SUBS_PER_BLK), y);                                                   \
        out_t *o = out + (size_t)row * k + (size_t)s * 32;                                     \
        _Pragma("unroll") for (int i = 0; i < 32; ++i) o[i] = to_out(y[i]);                    \
    }

// Gather эмбеддингов: строки таблицы [vocab][row_bytes] по индексам ids[n_ids]
// → out [n_ids][k]. Индекс вне таблицы даёт строку нулей (как embed_gather).
#define DEQG_ENTRY(NAME, FN, BB, SUBS_PER_BLK)                                                  \
    extern "C" __global__ void NAME(const uint8_t *__restrict__ src,                          \
                                    const unsigned *__restrict__ ids, out_t *__restrict__ out, \
                                    unsigned n_ids, unsigned vocab, unsigned k,               \
                                    unsigned row_bytes) {                                      \
        unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;                                  \
        unsigned subs_per_row = k / 32;                                                        \
        if (tid >= n_ids * subs_per_row) return;                                               \
        unsigned r = tid / subs_per_row;                                                       \
        unsigned s = tid % subs_per_row;                                                       \
        unsigned row = ids[r];                                                                 \
        out_t *o = out + (size_t)r * k + (size_t)s * 32;                                       \
        if (row >= vocab) {                                                                    \
            _Pragma("unroll") for (int i = 0; i < 32; ++i) o[i] = to_out(0.f);                 \
            return;                                                                            \
        }                                                                                      \
        const uint8_t *blk = src + (size_t)row * row_bytes + (size_t)(s / SUBS_PER_BLK) * (BB);\
        float y[32];                                                                           \
        FN(blk, (int)(s % SUBS_PER_BLK), y);                                                   \
        _Pragma("unroll") for (int i = 0; i < 32; ++i) o[i] = to_out(y[i]);                    \
    }

DEQ_ENTRY(deq_sq1, deq_sq<1>, 52, 8)
DEQ_ENTRY(deq_sq2, deq_sq<2>, 84, 8)
DEQ_ENTRY(deq_sq3, deq_sq<3>, 116, 8)
DEQ_ENTRY(deq_sq4, deq_sq<4>, 148, 8)
DEQ_ENTRY(deq_sq5, deq_sq<5>, 180, 8)
DEQ_ENTRY(deq_sq6, deq_sq<6>, 212, 8)
DEQ_ENTRY(deq_sq7, deq_sq<7>, 244, 8)
DEQ_ENTRY(deq_sq8, deq_sq<8>, 276, 8)

DEQ_ENTRY(deq_q4_0, deq_q4_0, 18, 1)
DEQ_ENTRY(deq_q4_1, deq_q4_1, 20, 1)
DEQ_ENTRY(deq_q5_0, deq_q5_0, 22, 1)
DEQ_ENTRY(deq_q5_1, deq_q5_1, 24, 1)
DEQ_ENTRY(deq_q8_0, deq_q8_0, 34, 1)
DEQ_ENTRY(deq_q8_1, deq_q8_1, 36, 1)
DEQ_ENTRY(deq_q8_k, deq_q8_k, 292, 8)
DEQ_ENTRY(deq_q1_0, deq_q1_0, 18, 4)
DEQ_ENTRY(deq_q2_0, deq_q2_0, 18, 2)
DEQ_ENTRY(deq_mxfp4, deq_mxfp4, 17, 1)
DEQ_ENTRY(deq_nvfp4, deq_nvfp4, 36, 2)
DEQ_ENTRY(deq_iq4_nl, deq_iq4_nl, 18, 1)
DEQ_ENTRY(deq_q2_k, deq_q2_k, 84, 8)
DEQ_ENTRY(deq_q3_k, deq_q3_k, 110, 8)
DEQ_ENTRY(deq_q4_k, deq_q4_k, 144, 8)
DEQ_ENTRY(deq_q5_k, deq_q5_k, 176, 8)
DEQ_ENTRY(deq_q6_k, deq_q6_k, 210, 8)
DEQ_ENTRY(deq_iq4_xs, deq_iq4_xs, 136, 8)
DEQ_ENTRY(deq_iq2_xxs, deq_iq2_xxs, 66, 8)
DEQ_ENTRY(deq_iq2_xs, deq_iq2_xs, 74, 8)
DEQ_ENTRY(deq_iq2_s, deq_iq2_s, 82, 8)
DEQ_ENTRY(deq_iq3_xxs, deq_iq3_xxs, 98, 8)
DEQ_ENTRY(deq_iq3_s, deq_iq3_s, 110, 8)
DEQ_ENTRY(deq_iq1_s, deq_iq1_s, 50, 8)
DEQ_ENTRY(deq_iq1_m, deq_iq1_m, 56, 8)
DEQ_ENTRY(deq_tq1_0, deq_tq1_0, 54, 8)
DEQ_ENTRY(deq_tq2_0, deq_tq2_0, 66, 8)

// Те же форматы — gather по индексам.
DEQG_ENTRY(deqg_sq1, deq_sq<1>, 52, 8)
DEQG_ENTRY(deqg_sq2, deq_sq<2>, 84, 8)
DEQG_ENTRY(deqg_sq3, deq_sq<3>, 116, 8)
DEQG_ENTRY(deqg_sq4, deq_sq<4>, 148, 8)
DEQG_ENTRY(deqg_sq5, deq_sq<5>, 180, 8)
DEQG_ENTRY(deqg_sq6, deq_sq<6>, 212, 8)
DEQG_ENTRY(deqg_sq7, deq_sq<7>, 244, 8)
DEQG_ENTRY(deqg_sq8, deq_sq<8>, 276, 8)
DEQG_ENTRY(deqg_q4_0, deq_q4_0, 18, 1)
DEQG_ENTRY(deqg_q4_1, deq_q4_1, 20, 1)
DEQG_ENTRY(deqg_q5_0, deq_q5_0, 22, 1)
DEQG_ENTRY(deqg_q5_1, deq_q5_1, 24, 1)
DEQG_ENTRY(deqg_q8_0, deq_q8_0, 34, 1)
DEQG_ENTRY(deqg_q8_1, deq_q8_1, 36, 1)
DEQG_ENTRY(deqg_q8_k, deq_q8_k, 292, 8)
DEQG_ENTRY(deqg_q1_0, deq_q1_0, 18, 4)
DEQG_ENTRY(deqg_q2_0, deq_q2_0, 18, 2)
DEQG_ENTRY(deqg_mxfp4, deq_mxfp4, 17, 1)
DEQG_ENTRY(deqg_nvfp4, deq_nvfp4, 36, 2)
DEQG_ENTRY(deqg_iq4_nl, deq_iq4_nl, 18, 1)
DEQG_ENTRY(deqg_q2_k, deq_q2_k, 84, 8)
DEQG_ENTRY(deqg_q3_k, deq_q3_k, 110, 8)
DEQG_ENTRY(deqg_q4_k, deq_q4_k, 144, 8)
DEQG_ENTRY(deqg_q5_k, deq_q5_k, 176, 8)
DEQG_ENTRY(deqg_q6_k, deq_q6_k, 210, 8)
DEQG_ENTRY(deqg_iq4_xs, deq_iq4_xs, 136, 8)
DEQG_ENTRY(deqg_iq2_xxs, deq_iq2_xxs, 66, 8)
DEQG_ENTRY(deqg_iq2_xs, deq_iq2_xs, 74, 8)
DEQG_ENTRY(deqg_iq2_s, deq_iq2_s, 82, 8)
DEQG_ENTRY(deqg_iq3_xxs, deq_iq3_xxs, 98, 8)
DEQG_ENTRY(deqg_iq3_s, deq_iq3_s, 110, 8)
DEQG_ENTRY(deqg_iq1_s, deq_iq1_s, 50, 8)
DEQG_ENTRY(deqg_iq1_m, deq_iq1_m, 56, 8)
DEQG_ENTRY(deqg_tq1_0, deq_tq1_0, 54, 8)
DEQG_ENTRY(deqg_tq2_0, deq_tq2_0, 66, 8)

// NVFP4/MXFP8 движка: полоса строк [row_off, row_off+rows) веса [N, K] → out [rows, K].
#define DEQ_ENTRY_SYN(NAME, FN)                                                                 \
    extern "C" __global__ void NAME(const uint8_t *__restrict__ w, const uint8_t *__restrict__ sw, \
                                    out_t *__restrict__ out, unsigned rows, unsigned k,         \
                                    unsigned row_off) {                                        \
        unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;                                  \
        unsigned subs_per_row = k / 32;                                                        \
        if (tid >= rows * subs_per_row) return;                                                \
        unsigned row = tid / subs_per_row;                                                     \
        unsigned s = tid % subs_per_row;                                                       \
        float y[32];                                                                           \
        FN(w, sw, row + row_off, k, s, y);                                                     \
        out_t *o = out + (size_t)row * k + (size_t)s * 32;                                     \
        _Pragma("unroll") for (int i = 0; i < 32; ++i) o[i] = to_out(y[i]);                    \
    }
DEQ_ENTRY_SYN(deq_nvfp4_syn, deq_nvfp4_syn)
DEQ_ENTRY_SYN(deq_mxfp8_syn, deq_mxfp8_syn)
