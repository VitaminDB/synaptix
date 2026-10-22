// Обратный групповой GEMM экспертов MoE по квантованным весам (обучение):
// `DX[r, Kw] = DY[r, Nw] · W_e[Nw, Kw]` для строк r сегмента эксперта e —
// градиент по входу проекции `y = x · W_eᵀ`. Свёртка идёт по строкам веса,
// выход — по его столбцам, поэтому вес не транспонируется в памяти: строка
// веса деквантуется в smem как есть (`Bs[строка][столбец]`), а фрагмент B
// для mma берёт `ldmatrix.trans`.
//
// Тайл 64×64×64, 4 варпа (2×2, у варпа 32×32), mma.sync m16n8k16
// bf16 → f32. Деквант — декодерами blockq_decode.cuh (поток — под-блок из 32
// значений строки веса). Выход в F32: градиент копится в полной точности.
// Тайлы строк (эксперт, начало, конец) считает хост, grid = (⌈Kw/64⌉, тайлы).
//
// Модуль собирается на один формат: хост дописывает строку
// GGN_BLOB(...)/GGN_SYN(...) под нужный вес (см. train.rs).
#include <cuda_bf16.h>
typedef __nv_bfloat16 act_t;
typedef __nv_bfloat162 act2_t;
__device__ __forceinline__ act2_t to_act2(float a, float b) { return __floats2bfloat162_rn(a, b); }

#define GGN_BM 64
#define GGN_BN 64
#define GGN_BK 64
#define GGN_LD (GGN_BK + 8)
#define GGN_THREADS 128

__device__ __forceinline__ unsigned ggn_smem_u32(const void *p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void ggn_store_row32(act_t *dst, const float *y) {
    uint4 *d4 = reinterpret_cast<uint4 *>(dst);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        act2_t h[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) h[i] = to_act2(y[q * 8 + 2 * i], y[q * 8 + 2 * i + 1]);
        d4[q] = *reinterpret_cast<const uint4 *>(h);
    }
}

// LOADB(nn, s, y): под-блок s (столбцы 32·s…) строки веса nn.
#define GGN_BODY(LOADB)                                                                         \
    __shared__ __align__(16) act_t As[GGN_BM][GGN_LD];                                         \
    __shared__ __align__(16) act_t Bs[GGN_BK][GGN_LD];                                         \
    const uint4 tile = tiles[blockIdx.y];                                                      \
    const unsigned r0 = tile.y, r1 = tile.z;                                                   \
    const unsigned c0 = blockIdx.x * GGN_BN;                                                   \
    const unsigned tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;                        \
    const unsigned wm = warp >> 1, wn = warp & 1;                                              \
    float acc[2][4][4];                                                                        \
    _Pragma("unroll") for (int a = 0; a < 2; ++a)                                              \
    _Pragma("unroll") for (int b = 0; b < 4; ++b)                                              \
    _Pragma("unroll") for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;                          \
    for (unsigned k0 = 0; k0 < Nw; k0 += GGN_BK) {                                             \
        /* A: 64 строки dY × 64 по свёртке, 16 байт на поток за проход. */                     \
        _Pragma("unroll") for (int i = 0; i < 4; ++i) {                                        \
            unsigned idx = tid + i * GGN_THREADS;                                              \
            unsigned r = idx >> 3, c = (idx & 7) * 8;                                          \
            unsigned row = r0 + r;                                                             \
            uint4 v = make_uint4(0, 0, 0, 0);                                                  \
            if (row < r1 && k0 + c < Nw)                                                       \
                v = *reinterpret_cast<const uint4 *>(DY + (size_t)row * Nw + k0 + c);          \
            *reinterpret_cast<uint4 *>(&As[r][c]) = v;                                         \
        }                                                                                      \
        /* B: поток — строка веса (по свёртке) и половина столбцов тайла. */                  \
        {                                                                                      \
            unsigned n = tid >> 1, half = tid & 1;                                             \
            unsigned nn = k0 + n, kk = c0 + half * 32;                                         \
            float y[32];                                                                       \
            if (nn < Nw && kk < Kw) {                                                          \
                LOADB(nn, kk / 32, y);                                                         \
            } else {                                                                           \
                _Pragma("unroll") for (int i = 0; i < 32; ++i) y[i] = 0.f;                     \
            }                                                                                  \
            ggn_store_row32(&Bs[n][half * 32], y);                                             \
        }                                                                                      \
        __syncthreads();                                                                       \
        _Pragma("unroll") for (int kk = 0; kk < GGN_BK; kk += 16) {                            \
            unsigned a[2][4], b[4][2];                                                         \
            _Pragma("unroll") for (int mi = 0; mi < 2; ++mi) {                                 \
                const act_t *p = &As[wm * 32 + mi * 16 + (lane & 15)][kk + (lane >> 4) * 8];   \
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"  \
                             : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3])  \
                             : "r"(ggn_smem_u32(p)));                                          \
            }                                                                                  \
            _Pragma("unroll") for (int ni = 0; ni < 4; ++ni) {                                 \
                const act_t *p = &Bs[kk + (lane & 15)][wn * 32 + ni * 8];                      \
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"   \
                             : "=r"(b[ni][0]), "=r"(b[ni][1])                                  \
                             : "r"(ggn_smem_u32(p)));                                          \
            }                                                                                  \
            _Pragma("unroll") for (int mi = 0; mi < 2; ++mi)                                   \
            _Pragma("unroll") for (int ni = 0; ni < 4; ++ni)                                   \
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "            \
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"           \
                             : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),                     \
                               "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])                      \
                             : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]),     \
                               "r"(b[ni][0]), "r"(b[ni][1]));                                  \
        }                                                                                      \
        __syncthreads();                                                                       \
    }                                                                                          \
    _Pragma("unroll") for (int mi = 0; mi < 2; ++mi)                                           \
    _Pragma("unroll") for (int ni = 0; ni < 4; ++ni) {                                         \
        unsigned col = c0 + wn * 32 + ni * 8 + (lane & 3) * 2;                                 \
        if (col >= Kw) continue;                                                               \
        _Pragma("unroll") for (int h = 0; h < 2; ++h) {                                        \
            unsigned row = r0 + wm * 32 + mi * 16 + (lane >> 2) + h * 8;                       \
            if (row < r1)                                                                      \
                *reinterpret_cast<float2 *>(DX + (size_t)row * Kw + col) =                     \
                    make_float2(acc[mi][ni][2 * h], acc[mi][ni][2 * h + 1]);                   \
        }                                                                                      \
    }

#define GGN_ARGS                                                                                \
    const unsigned long long *__restrict__ w_table, const unsigned long long *__restrict__ s_table, \
    const uint4 *__restrict__ tiles, const act_t *__restrict__ DY, float *__restrict__ DX,     \
    unsigned Nw, unsigned Kw, unsigned row_bytes

// Одноблобные форматы: BB байт на блок, SUBS_PER_BLK под-блоков в блоке.
#define GGN_BLOB(NAME, FN, BB, SUBS_PER_BLK)                                                    \
    extern "C" __global__ void __launch_bounds__(GGN_THREADS) NAME(GGN_ARGS) {                 \
        const uint8_t *w = (const uint8_t *)w_table[tiles[blockIdx.y].x];                      \
        (void)s_table;                                                                         \
        GGN_BODY(([&](unsigned nn, unsigned s, float *y) {                                     \
            const uint8_t *wrow = w + (size_t)nn * row_bytes;                                  \
            FN(wrow + (size_t)(s / SUBS_PER_BLK) * (BB), (int)(s % SUBS_PER_BLK), y);          \
        }))                                                                                    \
    }

// NVFP4/MXFP8 движка: декодер сам считает адрес по (строка, K, s).
#define GGN_SYN(NAME, FN)                                                                       \
    extern "C" __global__ void __launch_bounds__(GGN_THREADS) NAME(GGN_ARGS) {                 \
        const uint8_t *w = (const uint8_t *)w_table[tiles[blockIdx.y].x];                      \
        const uint8_t *sw = (const uint8_t *)s_table[tiles[blockIdx.y].x];                     \
        (void)row_bytes;                                                                       \
        GGN_BODY(([&](unsigned nn, unsigned s, float *y) { FN(w, sw, nn, Kw, s, y); }))        \
    }
