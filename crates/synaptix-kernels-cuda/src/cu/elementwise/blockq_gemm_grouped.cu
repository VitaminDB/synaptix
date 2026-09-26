// Групповой GEMM экспертов MoE по квантованным весам (префилл на картах без
// FP4 MMA, sm_80+): строки отсортированы по экспертам, каждому сегменту —
// свой вес. `Y[r, N] = X[src(r), K] · W_e[N, K]ᵀ` для строк r сегмента e;
// `src(r) = x_rows[r]` (сбор строк токенов прямо в загрузке тайла, без
// копии активации на пары) или `r`, если `x_rows` нулевой.
//
// Тайл 64×64×64, 4 варпа (2×2, у варпа 32×32). Вес деквантуется прямо в smem
// декодерами blockq_decode.cuh (поток — один под-блок из 32 значений), без
// промежуточной плотной копии в глобальной памяти; умножение — mma.sync
// m16n8k16 (bf16/f16 → f32) с ldmatrix. Тайлы строк (эксперт, начало, конец)
// считает хост: на сегмент ⌈строк/64⌉ тайлов, grid = (⌈N/64⌉, тайлы).
//
// Модуль собирается на один формат: хост дописывает строку
// GG_BLOB(...)/GG_SYN(...) под нужный вес (см. elementwise/blockq.rs).
#ifdef SYN_ACT_BF16
#include <cuda_bf16.h>
typedef __nv_bfloat16 act_t;
typedef __nv_bfloat162 act2_t;
__device__ __forceinline__ act2_t to_act2(float a, float b) { return __floats2bfloat162_rn(a, b); }
#define GG_MMA "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
#else
typedef __half act_t;
typedef __half2 act2_t;
__device__ __forceinline__ act2_t to_act2(float a, float b) { return __floats2half2_rn(a, b); }
#define GG_MMA "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
#endif

#define GG_BM 64
#define GG_BN 64
#define GG_BK 64
#define GG_LD (GG_BK + 8)  // +16 байт на строку: ldmatrix без конфликтов банков
#define GG_THREADS 128

__device__ __forceinline__ unsigned smem_u32(const void *p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

// 32 значения под-блока → 64 байта строки B в smem.
__device__ __forceinline__ void gg_store_row32(act_t *dst, const float *y) {
    uint4 *d4 = reinterpret_cast<uint4 *>(dst);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        act2_t h[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) h[i] = to_act2(y[q * 8 + 2 * i], y[q * 8 + 2 * i + 1]);
        d4[q] = *reinterpret_cast<const uint4 *>(h);
    }
}

// LOADB(nn, s, y): декодировать под-блок s строки веса nn (nn < N, s < K/32).
#define GG_BODY(LOADB)                                                                          \
    __shared__ __align__(16) act_t As[GG_BM][GG_LD];                                           \
    __shared__ __align__(16) act_t Bs[GG_BN][GG_LD];                                           \
    const uint4 tile = tiles[blockIdx.y];                                                      \
    const unsigned e = tile.x, r0 = tile.y, r1 = tile.z;                                       \
    const unsigned n0 = blockIdx.x * GG_BN;                                                    \
    const unsigned tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;                        \
    const unsigned wm = warp >> 1, wn = warp & 1;                                              \
    (void)e;                                                                                   \
    float acc[2][4][4];                                                                        \
    _Pragma("unroll") for (int a = 0; a < 2; ++a)                                              \
    _Pragma("unroll") for (int b = 0; b < 4; ++b)                                              \
    _Pragma("unroll") for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;                          \
    for (unsigned k0 = 0; k0 < K; k0 += GG_BK) {                                               \
        /* A: 64 строки × 64 столбца, по 16 байт на поток за проход. */                        \
        _Pragma("unroll") for (int i = 0; i < 4; ++i) {                                        \
            unsigned idx = tid + i * GG_THREADS;                                               \
            unsigned r = idx >> 3, c = (idx & 7) * 8;                                          \
            unsigned row = r0 + r;                                                             \
            uint4 v = make_uint4(0, 0, 0, 0);                                                  \
            if (row < r1 && k0 + c < K) {                                                      \
                const unsigned src = x_rows ? x_rows[row] : row;                               \
                v = *reinterpret_cast<const uint4 *>(X + (size_t)src * K + k0 + c);            \
            }                                                                                  \
            *reinterpret_cast<uint4 *>(&As[r][c]) = v;                                         \
        }                                                                                      \
        /* B: поток — строка веса n и половина BK (под-блок из 32). */                         \
        {                                                                                      \
            unsigned n = tid >> 1, half = tid & 1;                                             \
            unsigned nn = n0 + n, kk = k0 + half * 32;                                         \
            float y[32];                                                                       \
            if (nn < N && kk < K) {                                                            \
                LOADB(nn, kk / 32, y);                                                         \
            } else {                                                                           \
                _Pragma("unroll") for (int i = 0; i < 32; ++i) y[i] = 0.f;                     \
            }                                                                                  \
            gg_store_row32(&Bs[n][half * 32], y);                                              \
        }                                                                                      \
        __syncthreads();                                                                       \
        _Pragma("unroll") for (int kk = 0; kk < GG_BK; kk += 16) {                             \
            unsigned a[2][4], b[4][2];                                                         \
            _Pragma("unroll") for (int mi = 0; mi < 2; ++mi) {                                 \
                const act_t *p = &As[wm * 32 + mi * 16 + (lane & 15)][kk + (lane >> 4) * 8];   \
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"  \
                             : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3])  \
                             : "r"(smem_u32(p)));                                              \
            }                                                                                  \
            _Pragma("unroll") for (int ni = 0; ni < 4; ++ni) {                                 \
                const act_t *p = &Bs[wn * 32 + ni * 8 + (lane & 7)][kk + ((lane >> 3) & 1) * 8]; \
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"         \
                             : "=r"(b[ni][0]), "=r"(b[ni][1])                                  \
                             : "r"(smem_u32(p)));                                              \
            }                                                                                  \
            _Pragma("unroll") for (int mi = 0; mi < 2; ++mi)                                   \
            _Pragma("unroll") for (int ni = 0; ni < 4; ++ni)                                   \
                asm volatile(GG_MMA " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"  \
                             : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),                     \
                               "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])                      \
                             : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]),     \
                               "r"(b[ni][0]), "r"(b[ni][1]));                                  \
        }                                                                                      \
        __syncthreads();                                                                       \
    }                                                                                          \
    /* Выход: у m16n8 поток держит (g, 2c..2c+1) и (g+8, 2c..2c+1). */                         \
    _Pragma("unroll") for (int mi = 0; mi < 2; ++mi)                                           \
    _Pragma("unroll") for (int ni = 0; ni < 4; ++ni) {                                         \
        unsigned col = n0 + wn * 32 + ni * 8 + (lane & 3) * 2;                                 \
        if (col >= N) continue;                                                                \
        _Pragma("unroll") for (int h = 0; h < 2; ++h) {                                        \
            unsigned row = r0 + wm * 32 + mi * 16 + (lane >> 2) + h * 8;                       \
            if (row < r1)                                                                      \
                *reinterpret_cast<act2_t *>(Y + (size_t)row * N + col) =                       \
                    to_act2(acc[mi][ni][2 * h], acc[mi][ni][2 * h + 1]);                       \
        }                                                                                      \
    }

#define GG_ARGS                                                                                 \
    const unsigned long long *__restrict__ w_table, const unsigned long long *__restrict__ s_table, \
    const uint4 *__restrict__ tiles, const act_t *__restrict__ X,                              \
    const unsigned *__restrict__ x_rows, act_t *__restrict__ Y,                                \
    unsigned N, unsigned K, unsigned row_bytes

// Одноблобные форматы: BB байт на блок, SUBS_PER_BLK под-блоков в блоке.
#define GG_BLOB(NAME, FN, BB, SUBS_PER_BLK)                                                     \
    extern "C" __global__ void __launch_bounds__(GG_THREADS) NAME(GG_ARGS) {                   \
        const uint8_t *w = (const uint8_t *)w_table[tiles[blockIdx.y].x];                      \
        (void)s_table;                                                                         \
        GG_BODY(([&](unsigned nn, unsigned s, float *y) {                                      \
            const uint8_t *wrow = w + (size_t)nn * row_bytes;                                  \
            FN(wrow + (size_t)(s / SUBS_PER_BLK) * (BB), (int)(s % SUBS_PER_BLK), y);          \
        }))                                                                                    \
    }

// NVFP4/MXFP8 движка: декодер сам считает адрес по (row, K, s).
#define GG_SYN(NAME, FN)                                                                        \
    extern "C" __global__ void __launch_bounds__(GG_THREADS) NAME(GG_ARGS) {                   \
        const uint8_t *w = (const uint8_t *)w_table[tiles[blockIdx.y].x];                      \
        const uint8_t *sw = (const uint8_t *)s_table[tiles[blockIdx.y].x];                     \
        (void)row_bytes;                                                                       \
        GG_BODY(([&](unsigned nn, unsigned s, float *y) { FN(w, sw, nn, K, s, y); }))          \
    }
