// Групповой GEMM экспертов на FP8 MMA (Ada/Hopper/Blackwell, sm_89+):
// `Y[r, N] = X[src(r), K] · W_e[N, K]ᵀ`, обе стороны в MXFP8 — E4M3 и масштаб
// E8M0 на каждые 32 значения по K. У Ada нет блочно-масштабной MMA, но
// `mma.sync m16n8k32 e4m3` берёт по K ровно один MX-блок: частичная сумма
// блока считается с нулевым аккумулятором и добавляется в f32-сумму с
// множителем `scale_A[строка] · scale_B[столбец]` (степени двойки — точно).
//
// A — активация, заранее квантованная в MXFP8 (natural: байты [rows, K],
// масштабы [rows, K/32]); строки собираются по `x_rows` прямо в загрузке тайла.
// B — вес эксперта: MXFP8 копируется как есть, прочие форматы декодируются
// (blockq_decode.cuh) и переквантуются в MXFP8 в регистрах потока — у потока
// ровно один под-блок из 32 значений, тем же правилом масштаба, что у
// mxfp8_quant_natural.
//
// Тайл 128×128×64 (два MX-блока), 8 варпов 2×4, у варпа 64×32, двойной буфер. Модуль собирается
// на один формат веса: хост дописывает строку GG8_BLOB/GG8_SYN/GG8_MX.
#include <cuda_fp16.h>
#ifdef SYN_ACT_BF16
#include <cuda_bf16.h>
typedef __nv_bfloat16 act_t;
typedef __nv_bfloat162 act2_t;
__device__ __forceinline__ act2_t to_act2(float a, float b) { return __floats2bfloat162_rn(a, b); }
#else
typedef __half act_t;
typedef __half2 act2_t;
__device__ __forceinline__ act2_t to_act2(float a, float b) { return __floats2half2_rn(a, b); }
#endif

#ifndef G8_BM
#define G8_BM 128  // 64 или 128; у варпа (G8_BM/2)×32
#endif
#define G8_MI (G8_BM / 32)
#define G8_BN 128
#define G8_BK 64
#define G8_LD (G8_BK + 16)  // байт на строку smem: ldmatrix без конфликтов банков
#define G8_THREADS 256
#define G8_ST 3             // стадий конвейера: шаг t считается, t+1 и t+2 в пути
#define G8_A_BYTES (G8_BM * G8_LD)
#define G8_B_BYTES (G8_BN * G8_LD)
// Динамическая smem (> 48 КБ, хост выставляет лимит): A[ST], B[ST], масштабы.
#define G8_SMEM_BYTES (G8_ST * (G8_A_BYTES + G8_B_BYTES) + G8_ST * (G8_BM + G8_BN) * 2 * 4)

__device__ __forceinline__ unsigned g8_smem(const void *p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ float g8_e8m0(unsigned char b) { return __uint_as_float((unsigned)b << 23); }

// 16 байт global → smem в обход регистров (sm_80+); valid = false — нули.
__device__ __forceinline__ void g8_cp16(void *dst, const void *src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(g8_smem(dst)), "l"(src),
                 "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void g8_cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void g8_cp_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }

// Пара float → два байта E4M3 (младший — `lo`), RN с насыщением.
__device__ __forceinline__ unsigned short g8_pack2(float lo, float hi) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}

// 32 значения → 32 байта E4M3 в dst + множитель блока (как mxfp8_quant_natural:
// масштаб 2^(⌊log2 amax⌋ − 8), значения насыщаются на ±448).
__device__ __forceinline__ float g8_quant32(const float *y, unsigned char *dst) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(y[i]));
    const unsigned sbyte = __float_as_uint(__uint_as_float(__float_as_uint(amax) & 0x7F800000u) / 256.0f) >> 23;
    const float sv = fmaxf(__uint_as_float(sbyte << 23), 1e-12f);
    const float inv = 1.0f / sv;
    uint4 w[2];
    unsigned *wp = reinterpret_cast<unsigned *>(w);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned lo = g8_pack2(y[4 * i] * inv, y[4 * i + 1] * inv);
        unsigned hi = g8_pack2(y[4 * i + 2] * inv, y[4 * i + 3] * inv);
        wp[i] = lo | (hi << 16);
    }
    uint4 *d = reinterpret_cast<uint4 *>(dst);
    d[0] = w[0];
    d[1] = w[1];
    return sbyte ? sv : 0.f;
}

extern __shared__ __align__(16) unsigned char g8_dsmem[];

// Конвейер на G8_ST стадий. На шаге t: запрос шага t+2 (cp.async байтов A и,
// у MXFP8, B; сырые байты масштабов/NVFP4 — в регистры), счёт шага t, затем
// досчёт шага t+1 (декод/переквант веса и масштабы из регистров, полученных
// шагом раньше) — к счёту t+1 всё на месте, латентность памяти спрятана за
// двумя шагами счёта. Один барьер на шаг.
//
// FETCH(nn, kb, dst, raw) — запрос, FINISH(nn, kb, dst, raw, sc) — досчёт.
#define G8_BODY_T(G8_RAW, FETCH, FINISH)                                                        \
    unsigned char (*As)[G8_BM][G8_LD] = reinterpret_cast<unsigned char (*)[G8_BM][G8_LD]>(g8_dsmem); \
    unsigned char (*Bs)[G8_BN][G8_LD] =                                                        \
        reinterpret_cast<unsigned char (*)[G8_BN][G8_LD]>(g8_dsmem + G8_ST * G8_A_BYTES);      \
    float (*sA)[G8_BM][2] =                                                                    \
        reinterpret_cast<float (*)[G8_BM][2]>(g8_dsmem + G8_ST * (G8_A_BYTES + G8_B_BYTES));   \
    float (*sB)[G8_BN][2] = reinterpret_cast<float (*)[G8_BN][2]>(                             \
        g8_dsmem + G8_ST * (G8_A_BYTES + G8_B_BYTES) + G8_ST * G8_BM * 2 * 4);                 \
    const uint4 tile = tiles[blockIdx.y];                                                      \
    const unsigned r0 = tile.y, r1 = tile.z;                                                   \
    const unsigned n0 = blockIdx.x * G8_BN;                                                    \
    const unsigned tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;                        \
    const unsigned wm = warp >> 2, wn = warp & 3;                                              \
    const unsigned kb_total = K / 32;                                                          \
    const unsigned steps = (K + G8_BK - 1) / G8_BK;                                            \
    /* A: у потока строки tid/4 и tid/4 + 64, кусок 16 байт; масштаб — строка     */          \
    /* tid/2, блок tid%2.                                                          */          \
    const unsigned a_r = tid >> 2, a_c = (tid & 3) * 16;                                       \
    const bool a_ok0 = r0 + a_r < r1, a_ok1 = G8_BM > 64 && r0 + a_r + 64 < r1;                \
    const unsigned char *a_ptr0 =                                                              \
        xq + (size_t)(a_ok0 ? (x_rows ? x_rows[r0 + a_r] : r0 + a_r) : 0) * K + a_c;           \
    const unsigned char *a_ptr1 =                                                              \
        xq + (size_t)(a_ok1 ? (x_rows ? x_rows[r0 + a_r + 64] : r0 + a_r + 64) : 0) * K + a_c; \
    const unsigned s_r = (tid >> 1) % G8_BM, s_blk = tid & 1;                                  \
    const bool s_own = (tid >> 1) < G8_BM; /* при G8_BM = 64 масштабы — у половины потоков */  \
    const bool s_ok = s_own && r0 + s_r < r1;                                                  \
    const unsigned char *s_ptr =                                                               \
        xs + (size_t)(s_ok ? (x_rows ? x_rows[r0 + s_r] : r0 + s_r) : 0) * kb_total;           \
    /* B: поток — строка веса tid/2 и MX-блок tid%2 (G8_BN·2 == G8_THREADS). */                \
    const unsigned b_n = tid >> 1, b_blk = tid & 1, b_nn = n0 + b_n;                           \
    auto issue = [&](int st, unsigned t, unsigned char &ra, G8_RAW &rb, bool &okb) {           \
        const unsigned k0 = t * G8_BK;                                                         \
        g8_cp16(&As[st][a_r][a_c], a_ptr0 + k0, a_ok0 && k0 + a_c < K);                        \
        if (G8_BM > 64) g8_cp16(&As[st][(a_r + 64) % G8_BM][a_c], a_ptr1 + k0, a_ok1 && k0 + a_c < K); \
        const unsigned kb = k0 / 32 + s_blk;                                                   \
        ra = (s_ok && kb < kb_total) ? s_ptr[kb] : 0;                                          \
        const unsigned kbb = k0 / 32 + b_blk;                                                  \
        okb = b_nn < N && kbb < kb_total;                                                      \
        if (okb) FETCH(b_nn, kbb, &Bs[st][b_n][b_blk * 32], rb);                               \
    };                                                                                         \
    auto finish = [&](int st, unsigned t, unsigned char ra, G8_RAW &rb, bool okb) {            \
        if (s_own) sA[st][s_r][s_blk] = g8_e8m0(ra);                                           \
        unsigned char *dst = &Bs[st][b_n][b_blk * 32];                                         \
        float sc = 0.f;                                                                        \
        if (okb) {                                                                             \
            FINISH(b_nn, t * G8_BK / 32 + b_blk, dst, rb, sc);                                 \
        } else {                                                                               \
            uint4 *d = reinterpret_cast<uint4 *>(dst);                                         \
            d[0] = make_uint4(0, 0, 0, 0);                                                     \
            d[1] = make_uint4(0, 0, 0, 0);                                                     \
        }                                                                                      \
        sB[st][b_n][b_blk] = sc;                                                               \
    };                                                                                         \
    float acc[G8_MI][4][4];                                                                        \
    _Pragma("unroll") for (int a = 0; a < G8_MI; ++a)                                              \
    _Pragma("unroll") for (int b = 0; b < 4; ++b)                                              \
    _Pragma("unroll") for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;                          \
    unsigned char ra0 = 0, ra1 = 0, ra2 = 0;                                                   \
    G8_RAW rb0, rb1, rb2;                                                                      \
    bool ok0 = false, ok1 = false, ok2 = false;                                                \
    issue(0, 0, ra0, rb0, ok0);                                                                \
    g8_cp_commit();                                                                            \
    if (steps > 1) issue(1, 1, ra1, rb1, ok1);                                                 \
    g8_cp_commit();                                                                            \
    finish(0, 0, ra0, rb0, ok0);                                                               \
    g8_cp_wait1();                                                                             \
    __syncthreads();                                                                           \
    int cur = 0, nx1 = 1, nx2 = 2;                                                             \
    for (unsigned t = 0; t < steps; ++t) {                                                     \
        if (t + 2 < steps) issue(nx2, t + 2, ra2, rb2, ok2);                                   \
        g8_cp_commit();                                                                        \
        _Pragma("unroll") for (int j = 0; j < 2; ++j) {                                        \
            unsigned a[G8_MI][4], b[4][2];                                                         \
            float sa[G8_MI][2], sb[4][2];                                                          \
            _Pragma("unroll") for (int mi = 0; mi < G8_MI; ++mi) {                                 \
                const unsigned char *p = &As[cur][wm * (G8_BM / 2) + mi * 16 + (lane & 15)][j * 32 + (lane >> 4) * 16]; \
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"  \
                             : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3])  \
                             : "r"(g8_smem(p)));                                               \
                sa[mi][0] = sA[cur][wm * (G8_BM / 2) + mi * 16 + (lane >> 2)][j];                       \
                sa[mi][1] = sA[cur][wm * (G8_BM / 2) + mi * 16 + (lane >> 2) + 8][j];                   \
            }                                                                                  \
            _Pragma("unroll") for (int ni = 0; ni < 4; ++ni) {                                 \
                const unsigned char *p = &Bs[cur][wn * 32 + ni * 8 + (lane & 7)][j * 32 + ((lane >> 3) & 1) * 16]; \
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"         \
                             : "=r"(b[ni][0]), "=r"(b[ni][1])                                  \
                             : "r"(g8_smem(p)));                                               \
                sb[ni][0] = sB[cur][wn * 32 + ni * 8 + (lane & 3) * 2][j];                     \
                sb[ni][1] = sB[cur][wn * 32 + ni * 8 + (lane & 3) * 2 + 1][j];                 \
            }                                                                                  \
            /* Сначала MMA строки варпа, потом масштабы: FMA не ждёт каждую MMA. */           \
            _Pragma("unroll") for (int mi = 0; mi < G8_MI; ++mi) {                                 \
                float tt[4][4];                                                                \
                _Pragma("unroll") for (int ni = 0; ni < 4; ++ni)                               \
                    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "                 \
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"             \
                        : "=f"(tt[ni][0]), "=f"(tt[ni][1]), "=f"(tt[ni][2]), "=f"(tt[ni][3])   \
                        : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]),          \
                          "r"(b[ni][0]), "r"(b[ni][1]), "f"(0.f));                             \
                _Pragma("unroll") for (int ni = 0; ni < 4; ++ni) {                             \
                    acc[mi][ni][0] += tt[ni][0] * (sa[mi][0] * sb[ni][0]);                     \
                    acc[mi][ni][1] += tt[ni][1] * (sa[mi][0] * sb[ni][1]);                     \
                    acc[mi][ni][2] += tt[ni][2] * (sa[mi][1] * sb[ni][0]);                     \
                    acc[mi][ni][3] += tt[ni][3] * (sa[mi][1] * sb[ni][1]);                     \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
        if (t + 1 < steps) finish(nx1, t + 1, ra1, rb1, ok1);                                  \
        ra1 = ra2;                                                                             \
        rb1 = rb2;                                                                             \
        ok1 = ok2;                                                                             \
        g8_cp_wait1();                                                                         \
        __syncthreads();                                                                       \
        const int old = cur;                                                                   \
        cur = nx1;                                                                             \
        nx1 = nx2;                                                                             \
        nx2 = old;                                                                             \
    }                                                                                          \
    _Pragma("unroll") for (int mi = 0; mi < G8_MI; ++mi)                                           \
    _Pragma("unroll") for (int ni = 0; ni < 4; ++ni) {                                         \
        unsigned col = n0 + wn * 32 + ni * 8 + (lane & 3) * 2;                                 \
        if (col >= N) continue;                                                                \
        _Pragma("unroll") for (int h = 0; h < 2; ++h) {                                        \
            unsigned row = r0 + wm * (G8_BM / 2) + mi * 16 + (lane >> 2) + h * 8;                       \
            if (row < r1)                                                                      \
                *reinterpret_cast<act2_t *>(Y + (size_t)row * N + col) =                       \
                    to_act2(acc[mi][ni][2 * h], acc[mi][ni][2 * h + 1]);                       \
        }                                                                                      \
    }

#define G8_ARGS                                                                                 \
    const unsigned long long *__restrict__ w_table, const unsigned long long *__restrict__ s_table, \
    const uint4 *__restrict__ tiles, const unsigned char *__restrict__ xq,                     \
    const unsigned char *__restrict__ xs, const unsigned *__restrict__ x_rows,                 \
    act_t *__restrict__ Y, unsigned N, unsigned K, unsigned row_bytes

// Форматы: FETCH(nn, kb, dst, raw) — до счёта шага (сырые байты в регистры или
// cp.async в smem), FINISH(nn, kb, dst, raw, sc) — после (декод, переквант,
// запись). G8_RAW — регистры между ними.

// Одноблобные форматы: весь декод в FINISH (раскладка блока у форматов разная).
#define GG8_BLOB(NAME, FN, BB, SUBS_PER_BLK)                                                    \
    struct NAME##_raw {};                                                                      \
    extern "C" __global__ void __launch_bounds__(G8_THREADS) NAME(G8_ARGS) {                   \
        const uint8_t *w = (const uint8_t *)w_table[tiles[blockIdx.y].x];                      \
        (void)s_table;                                                                         \
        typedef NAME##_raw G8_RAW_T;                                                           \
        G8_BODY_T(G8_RAW_T,                                                                    \
            ([&](unsigned, unsigned, unsigned char *, G8_RAW_T &) {}),                        \
            ([&](unsigned nn, unsigned s, unsigned char *dst, G8_RAW_T &, float &sc) {         \
                const uint8_t *wrow = w + (size_t)nn * row_bytes;                              \
                float y[32];                                                                   \
                FN(wrow + (size_t)(s / SUBS_PER_BLK) * (BB), (int)(s % SUBS_PER_BLK), y);      \
                sc = g8_quant32(y, dst);                                                       \
            }))                                                                                \
    }

// NVFP4 движка: 16 байт нибблов и два масштаба E4M3 — в регистры до счёта.
struct g8_nvfp4_raw {
    uint4 v;
    unsigned char s0, s1;
};
#define GG8_SYN(NAME, FN)                                                                       \
    extern "C" __global__ void __launch_bounds__(G8_THREADS) NAME(G8_ARGS) {                   \
        const uint8_t *w = (const uint8_t *)w_table[tiles[blockIdx.y].x];                      \
        const uint8_t *sw = (const uint8_t *)s_table[tiles[blockIdx.y].x];                     \
        (void)row_bytes;                                                                       \
        const unsigned sf_inner = ((K + 63u) / 64u) * 4u;                                      \
        G8_BODY_T(g8_nvfp4_raw,                                                                \
            ([&](unsigned nn, unsigned s, unsigned char *, g8_nvfp4_raw &r) {                  \
                r.v = *reinterpret_cast<const uint4 *>(w + ((size_t)nn * K + (size_t)s * 32u) / 2u); \
                r.s0 = sw[syn_nvfp4_scale_off(nn, s * 2u, sf_inner)];                          \
                r.s1 = sw[syn_nvfp4_scale_off(nn, s * 2u + 1u, sf_inner)];                     \
            }),                                                                                \
            ([&](unsigned, unsigned, unsigned char *dst, g8_nvfp4_raw &r, float &sc) {         \
                const unsigned words[4] = {r.v.x, r.v.y, r.v.z, r.v.w};                        \
                const float s0 = syn_decode_e4m3(r.s0), s1 = syn_decode_e4m3(r.s1);            \
                float y[32];                                                                   \
                _Pragma("unroll") for (int q = 0; q < 4; ++q)                                  \
                _Pragma("unroll") for (int j = 0; j < 8; ++j)                                  \
                    y[q * 8 + j] = syn_decode_e2m1((words[q] >> (4 * j)) & 0xFu) * (q < 2 ? s0 : s1); \
                sc = g8_quant32(y, dst);                                                       \
            }))                                                                                \
    }

// MXFP8 движка: байты — cp.async прямо в smem, масштаб — в регистр.
#define GG8_MX(NAME)                                                                            \
    extern "C" __global__ void __launch_bounds__(G8_THREADS) NAME(G8_ARGS) {                   \
        const uint8_t *w = (const uint8_t *)w_table[tiles[blockIdx.y].x];                      \
        const uint8_t *sw = (const uint8_t *)s_table[tiles[blockIdx.y].x];                     \
        (void)row_bytes;                                                                       \
        G8_BODY_T(unsigned char,                                                               \
            ([&](unsigned nn, unsigned s, unsigned char *dst, unsigned char &r) {              \
                const unsigned char *p = w + (size_t)nn * K + (size_t)s * 32;                  \
                g8_cp16(dst, p, true);                                                         \
                g8_cp16(dst + 16, p + 16, true);                                               \
                r = sw[(size_t)nn * (K / 32) + s];                                             \
            }),                                                                                \
            ([&](unsigned, unsigned, unsigned char *, unsigned char &r, float &sc) { sc = g8_e8m0(r); })) \
    }
