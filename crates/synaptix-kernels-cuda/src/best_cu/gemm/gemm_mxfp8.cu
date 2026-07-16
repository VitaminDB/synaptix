#include <cuda_fp16.h>
#include <cuda_fp8.h>

// КОРРЕКТНОЕ MXFP8 GEMM (sm_120a), порт gau-nernst/learn-cuda 09a_block_scaled_mm_sm120 v1.
// Простой cp.async-конвейер (БЕЗ TMA/warp-spec) — проверен cos=0.999999 на outlier-данных,
// в отличие от нашего TMA-warp-spec gemm_mxfp8.cu (баг на широком scale-разбросе: TMA-swizzle
// данных рассинхронен с ldmatrix-read). Здесь producer-swizzle == consumer-swizzle по построению.
// C[M,N] f16 = A[M,K] @ B[N,K]ᵀ, e4m3 + натуральные E8M0 per-32 scale [rows,K/32], F32-аккум.

constexpr int WARP_SIZE = 32;
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 32;
__device__ __host__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template <int STRIDE> __device__ inline int swizzle(int row, int col) {
    if constexpr (STRIDE > 16) col ^= (row % 8) / (128 / STRIDE > 1 ? 128 / STRIDE : 1);
    return row * STRIDE + col * 16;
}

template <int num> __device__ inline void ldmatrix(int *r, int addr) {
    if constexpr (num == 4)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// mma m16n8k32 e4m3 block_scale ue8m0 (как эталон: БЕЗ scale_vec::1X — дефолт 1X).
__device__ inline void mma_mxfp8(int A[4], int B[2], float C[4], int SFA, short bidA,
                                 short tidA, int SFB, short bidB, short tidB) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(SFA), "h"(bidA), "h"(tidA), "r"(SFB), "h"(bidB), "h"(tidB));
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline void gmem_to_smem(int dst, const char *src, int src_stride, int tid) {
    constexpr int ne = 16;
    constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * ne);
    auto load = [&](int idx) {
        const int row = idx / WIDTH, col = idx % WIDTH;
        const int da = dst + swizzle<WIDTH>(row, col / ne);
        const char *sa = src + (row * src_stride + col);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(da), "l"(sa));
    };
    for (int i = 0; i < num_iters; i++) load((i * TB_SIZE + tid) * ne);
    if constexpr ((HEIGHT * WIDTH) % (TB_SIZE * ne) != 0) {
        const int idx = (num_iters * TB_SIZE + tid) * ne;
        if (idx < HEIGHT * WIDTH) load(idx);
    }
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline void load_scales(int dst, const char *src, int src_stride, int tid) {
    constexpr int cp_size = WIDTH; // sizeof(e8m0)=1
    auto load_row = [&](int row) {
        const int da = dst + row * WIDTH;
        const char *sa = src + row * src_stride;
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::"r"(da), "l"(sa), "n"(cp_size));
    };
    for (int i = 0; i < HEIGHT / TB_SIZE; i++) load_row(i * TB_SIZE + tid);
    if constexpr (HEIGHT % TB_SIZE != 0) {
        const int row = HEIGHT / TB_SIZE * TB_SIZE + tid;
        if (row < HEIGHT) load_row(row);
    }
}

template <int BM, int BN, int BK, int NWM, int NWN, int NSTAGES>
__device__ void mxfp8_device(const char *A_ptr, const char *B_ptr, const char *SFA_ptr,
                                const char *SFB_ptr, __half *C_ptr, int M, int N, int K) {
    constexpr int TB = NWM * NWN * WARP_SIZE;
    constexpr int WARP_M = BM / NWM, WARP_N = BN / NWN;
    const int bid = blockIdx.x;
    const int bid_m = bid / cdiv(N, BN), bid_n = bid % cdiv(N, BN);
    const int tid = threadIdx.x, warp_id = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    const int wm = warp_id / NWN, wn = warp_id % NWN;
    const int off_m = bid_m * BM, off_n = bid_n * BN;
    A_ptr += off_m * K;
    B_ptr += off_n * K;
    SFA_ptr += off_m * (K / 32);
    SFB_ptr += off_n * (K / 32);
    C_ptr += (off_m + wm * WARP_M) * N + (off_n + wn * WARP_N);

    extern __shared__ char smem_ptr[];
    const int A_smem = __cvta_generic_to_shared(smem_ptr);
    const int B_smem = A_smem + BM * BK;
    const int SFA_smem = B_smem + BN * BK;
    const int SFB_smem = SFA_smem + BM * (BK / 32);
    constexpr int STAGE_SIZE = (BM + BN) * (BK + BK / 32);

    const int A_addr = A_smem + swizzle<BK>(wm * WARP_M + (lane % 16), lane / 16);
    const int B_addr = B_smem + swizzle<BK>(wn * WARP_N + (lane % 8), lane / 8);
    const int SFA_addr = SFA_smem + (wm * WARP_M + (lane % 4) * 8 + (lane / 4)) * 4;
    const int SFB_addr = SFB_smem + (wn * WARP_N + (lane % 4) * 8 + (lane / 4)) * 4;

    int A_rmem[WARP_M / MMA_M][BK / MMA_K][4];
    int B_rmem[WARP_N / MMA_N][BK / MMA_K][2];
    int SFA_rmem[WARP_M / 32], SFB_rmem[WARP_N / 32];
    float acc[WARP_M / MMA_M][WARP_N / MMA_N][4] = {};

    auto load = [&](int s) {
        gmem_to_smem<BM, BK, TB>(A_smem + s * STAGE_SIZE, A_ptr, K, tid);
        gmem_to_smem<BN, BK, TB>(B_smem + s * STAGE_SIZE, B_ptr, K, tid);
        load_scales<BM, BK / 32, TB>(SFA_smem + s * STAGE_SIZE, SFA_ptr, K / 32, tid);
        load_scales<BN, BK / 32, TB>(SFB_smem + s * STAGE_SIZE, SFB_ptr, K / 32, tid);
        A_ptr += BK;
        B_ptr += BK;
        SFA_ptr += BK / 32;
        SFB_ptr += BK / 32;
        asm volatile("cp.async.commit_group;");
    };
    auto compute = [&](int s) {
        for (int k = 0; k < BK / MMA_K; k++)
            for (int m = 0; m < WARP_M / MMA_M; m++)
                ldmatrix<4>(A_rmem[m][k], (A_addr + s * STAGE_SIZE + m * MMA_M * BK) ^ (k * 32));
        for (int k = 0; k < BK / MMA_K; k += 2)
            for (int n = 0; n < WARP_N / MMA_N; n++)
                ldmatrix<4>(B_rmem[n][k], (B_addr + s * STAGE_SIZE + n * MMA_N * BK) ^ (k * 32));
        for (int r = 0; r < WARP_M / 32; r++)
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(SFA_rmem[r])
                         : "r"(SFA_addr + s * STAGE_SIZE + r * 32 * 4));
        for (int r = 0; r < WARP_N / 32; r++)
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(SFB_rmem[r])
                         : "r"(SFB_addr + s * STAGE_SIZE + r * 32 * 4));
        for (int k = 0; k < BK / MMA_K; k++)
            for (int m = 0; m < WARP_M / MMA_M; m++)
                for (int n = 0; n < WARP_N / MMA_N; n++)
                    mma_mxfp8(A_rmem[m][k], B_rmem[n][k], acc[m][n], SFA_rmem[m / 2], k, m % 2,
                              SFB_rmem[n / 4], k, n % 4);
    };

    int nk = cdiv(K, BK);
    for (int s = 0; s < NSTAGES - 1; s++) load(s);
    for (int it = 0; it < nk - (NSTAGES - 1); it++) {
        __syncthreads();
        load((it + NSTAGES - 1) % NSTAGES);
        asm volatile("cp.async.wait_group %0;" ::"n"(NSTAGES - 1));
        __syncthreads();
        compute(it % NSTAGES);
    }
    for (int it = nk - (NSTAGES - 1); it < nk; it++) {
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group %0;" ::"n"(NSTAGES - 1));
        __syncthreads();
        compute(it % NSTAGES);
    }
    for (int m = 0; m < WARP_M / MMA_M; m++)
        for (int n = 0; n < WARP_N / MMA_N; n++) {
            const int row = m * MMA_M + lane / 4, col = n * MMA_N + (lane % 4) * 2;
            float *rg = acc[m][n];
            reinterpret_cast<__half2 *>(C_ptr + (row + 0) * N + col)[0] = __float22half2_rn({rg[0], rg[1]});
            reinterpret_cast<__half2 *>(C_ptr + (row + 8) * N + col)[0] = __float22half2_rn({rg[2], rg[3]});
        }
}

extern "C" __global__ __launch_bounds__(128) void gn_mxfp8_128x128(
    const char *A, const char *B, const char *SFA, const char *SFB, __half *C, int M, int N, int K) {
    mxfp8_device<128, 128, 128, 2, 2, 3>(A, B, SFA, SFB, C, M, N, K);
}

// ===== ROT: порт k-конвейера CUTLASS sm120 (рецепт nvfp4, коммит abb04654) =====
// TMA(::cta!)+mbarrier+fused-producer 256 потоков + ротация по ПАРАМ k32-блоков
// (B-фрагмент x4-ldmatrix покрывает 2 суб-блока → пара ≡ KCH=2 структуре nvfp4):
// double-buffer фрагментов, ранний release стадии, wait перед последней gemm-пачкой,
// staged-эпилог. Данные/скейлы natural (как gn_mxfp8_128x128), bit-exact к нему
// (k-порядок mma не меняется). SWIZZLE_128B TMA == swizzle<128> прод-ядра.

__device__ __forceinline__ unsigned int swz128(unsigned int off) {
    return off ^ (((off >> 7) & 7u) << 4);
}

template <int NREG> __device__ __forceinline__ void mx_setmaxnreg_dec() {
    if constexpr (NREG > 0)
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(NREG));
}
template <int NREG> __device__ __forceinline__ void mx_setmaxnreg_inc() {
    if constexpr (NREG > 0)
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NREG));
}

template <unsigned int BM, unsigned int BN, unsigned int WM, unsigned int WN,
          unsigned int STAGES, unsigned int PROD_W = 0u, int RDEC = 0, int RINC = 0,
          bool SPLITK = false>
__device__ __forceinline__ void mxfp8_rot_device(
    const void* __restrict__ a_desc,
    const void* __restrict__ b_desc,
    const void* __restrict__ sfa_desc,
    const void* __restrict__ sfb_desc,
    __half*     __restrict__ out,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int raster_gr,
    const void* __restrict__ out_desc = nullptr,
    float*       __restrict__ ws = nullptr,
    unsigned int kt0 = 0u,
    unsigned int kt_len_in = 0u)
{
    constexpr unsigned int BK = 128;
    constexpr unsigned int THREADS = WM * WN * 32u;
    constexpr unsigned int WARP_M = BM / WM;
    constexpr unsigned int WARP_N = BN / WN;
    constexpr unsigned int MA = WARP_M / 16u;
    constexpr unsigned int NB = WARP_N / 8u;
    constexpr unsigned int A_SZ = BM * BK;
    constexpr unsigned int B_SZ = BN * BK;
    // SF-бокс TMA = 16Б/строку (минимум inner-dim TMA; 4Б стадии — внутри окна
    // из 4 стадий, координата floor(g/4)*16, чтение со смещением (c%4)*4).
    constexpr unsigned int SFA_SZ = BM * 16u;
    constexpr unsigned int SFB_SZ = BN * 16u;
    constexpr unsigned int STAGE = A_SZ + B_SZ + SFA_SZ + SFB_SZ;
    constexpr unsigned int TX = STAGE;
    constexpr unsigned int B_OFF   = A_SZ;
    constexpr unsigned int SFA_OFF = A_SZ + B_SZ;
    constexpr unsigned int SFB_OFF = SFA_OFF + SFA_SZ;
    constexpr unsigned int BAR_OFF = STAGES * STAGE;

    extern __shared__ __align__(128) unsigned char smem[];
    unsigned int sbase = (unsigned int)__cvta_generic_to_shared(smem);
    #define MFULL_A(b)  (sbase + BAR_OFF + (b) * 8u)
    #define MEMPTY_A(b) (sbase + BAR_OFF + STAGES * 8u + (b) * 8u)

    unsigned int tid  = threadIdx.x;
    unsigned int warp = tid >> 5;
    unsigned int lane = tid & 31u;
    unsigned int num_kt = SPLITK ? kt_len_in : K / BK;
    constexpr unsigned int CONS_WARPS = WM * WN;
    bool is_prod = PROD_W != 0u && warp >= CONS_WARPS;

    // L2-растр (урок bf16 #1): 1D-грид → компактное окно волны raster_gr N-тайлов
    // × M-полоса (веса L2-резидентны на вес-bound формах вроде ff_up 64MB).
    unsigned int tiles_m = (M + BM - 1u) / BM;
    unsigned int bid = blockIdx.x;
    unsigned int sup = bid / (tiles_m * raster_gr);
    unsigned int rem = bid % (tiles_m * raster_gr);
    unsigned int block_m0 = (rem / raster_gr) * BM;
    unsigned int block_n0 = (sup * raster_gr + rem % raster_gr) * BN;

    if (tid == 0) {
        #pragma unroll
        for (unsigned int s = 0; s < STAGES; s++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(MFULL_A(s)));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                         :: "r"(MEMPTY_A(s)), "r"(THREADS));
        }
    }
    __syncthreads();

    auto issue_chunk = [&](unsigned int g) {
        unsigned int buf = g % STAGES;
        unsigned int pass = g / STAGES;
        unsigned int fa = MFULL_A(buf);
        if (pass > 0) {
            unsigned int ph = (pass - 1u) & 1u;
            asm volatile(
              "{\n.reg .pred p;\nMWE_%=:\n"
              "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
              "@!p bra MWE_%=;\n}\n" :: "r"(MEMPTY_A(buf)), "r"(ph) : "memory");
        }
        unsigned int kb = (kt0 + g) * BK;
        unsigned int ks = ((kt0 + g) & ~3u) * (BK / 32u);
        unsigned long long st;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n"
                     : "=l"(st) : "r"(fa), "r"(TX));
        asm volatile(
          "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
          " [%0], [%1, {%2, %3}], [%4];\n"
          :: "r"(sbase + buf * STAGE), "l"((unsigned long long)a_desc),
             "r"(kb), "r"(block_m0), "r"(fa) : "memory");
        asm volatile(
          "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
          " [%0], [%1, {%2, %3}], [%4];\n"
          :: "r"(sbase + buf * STAGE + B_OFF), "l"((unsigned long long)b_desc),
             "r"(kb), "r"(block_n0), "r"(fa) : "memory");
        asm volatile(
          "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
          " [%0], [%1, {%2, %3}], [%4];\n"
          :: "r"(sbase + buf * STAGE + SFA_OFF), "l"((unsigned long long)sfa_desc),
             "r"(ks), "r"(block_m0), "r"(fa) : "memory");
        asm volatile(
          "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
          " [%0], [%1, {%2, %3}], [%4];\n"
          :: "r"(sbase + buf * STAGE + SFB_OFF), "l"((unsigned long long)sfb_desc),
             "r"(ks), "r"(block_n0), "r"(fa) : "memory");
    };

    if constexpr (PROD_W != 0u) {
        if (is_prod) {
            mx_setmaxnreg_dec<RDEC>();
            if (warp == CONS_WARPS && lane == 0) {
                for (unsigned int g = 0; g < num_kt; g++)
                    issue_chunk(g);
            }
            return;
        }
        mx_setmaxnreg_inc<RINC>();
    } else {
        if (tid == 0) {
            unsigned int pre = (STAGES - 1u < num_kt) ? STAGES - 1u : num_kt;
            for (unsigned int g = 0; g < pre; g++)
                issue_chunk(g);
        }
    }

    unsigned int wm = warp / WN;
    unsigned int wn = warp % WN;
    // ldmatrix-адреса в терминах прод-swizzle<128>: row*128 + col16*16, XOR бит 4-6.
    unsigned int a_lane = swz128((wm * WARP_M + (lane & 15u)) * BK + (lane >> 4) * 16u);
    unsigned int b_lane = swz128((wn * WARP_N + (lane & 7u)) * BK + (lane >> 3) * 16u);
    unsigned int sfa_lane = (wm * WARP_M + (lane & 3u) * 8u + (lane >> 2)) * 16u;
    unsigned int sfb_lane = (wn * WARP_N + (lane & 3u) * 8u + (lane >> 2)) * 16u;

    unsigned int aR[2][MA][2][4];
    unsigned int bR[2][NB][2][2];
    unsigned int sfaR[2][WARP_M / 32u];
    unsigned int sfbR[2][WARP_N / 32u];
    float acc[MA][NB][4];
    #pragma unroll
    for (unsigned int m = 0; m < MA; m++)
        #pragma unroll
        for (unsigned int n = 0; n < NB; n++) {
            acc[m][n][0] = 0.f; acc[m][n][1] = 0.f; acc[m][n][2] = 0.f; acc[m][n][3] = 0.f;
        }

    // pp = пара k32-блоков (kk = 2*pp, 2*pp+1). SF читаются в слот ВМЕСТЕ с
    // фрагментами пары (значения одинаковы для обеих пар стадии — перечитка
    // 3 LDS.u32 дешевле динамической индексации слота → local memory).
    auto load_pair = [&](unsigned int buf, unsigned int pp, unsigned int sfo,
                         unsigned int (&a)[MA][2][4], unsigned int (&b)[NB][2][2],
                         unsigned int (&sfa)[WARP_M / 32u], unsigned int (&sfb)[WARP_N / 32u]) {
        unsigned int abase = sbase + buf * STAGE;
        unsigned int bbase = abase + B_OFF;
        #pragma unroll
        for (unsigned int r = 0; r < WARP_M / 32u; r++)
            asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(sfa[r])
                : "r"(abase + SFA_OFF + sfa_lane + sfo + r * 512u));
        #pragma unroll
        for (unsigned int r = 0; r < WARP_N / 32u; r++)
            asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(sfb[r])
                : "r"(abase + SFB_OFF + sfb_lane + sfo + r * 512u));
        #pragma unroll
        for (unsigned int m = 0; m < MA; m++) {
            #pragma unroll
            for (unsigned int d = 0; d < 2; d++) {
                unsigned int kk = pp * 2u + d;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a[m][d][0]), "=r"(a[m][d][1]), "=r"(a[m][d][2]), "=r"(a[m][d][3])
                    : "r"((abase + m * (16u * BK) + a_lane) ^ (kk * 32u)));
            }
        }
        #pragma unroll
        for (unsigned int n = 0; n < NB; n++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(b[n][0][0]), "=r"(b[n][0][1]), "=r"(b[n][1][0]), "=r"(b[n][1][1])
                : "r"((bbase + n * (8u * BK) + b_lane) ^ (pp * 64u)));
        }
    };
    auto gemm_pair = [&](unsigned int pp,
                         const unsigned int (&a)[MA][2][4], const unsigned int (&b)[NB][2][2],
                         const unsigned int (&sfa)[WARP_M / 32u],
                         const unsigned int (&sfb)[WARP_N / 32u]) {
        #pragma unroll
        for (unsigned int d = 0; d < 2; d++) {
            unsigned int kk = pp * 2u + d;
            #pragma unroll
            for (unsigned int m = 0; m < MA; m++) {
                #pragma unroll
                for (unsigned int n = 0; n < NB; n++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.f32.e4m3.e4m3.f32.ue8m0 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};"
                        : "+f"(acc[m][n][0]), "+f"(acc[m][n][1]), "+f"(acc[m][n][2]), "+f"(acc[m][n][3])
                        : "r"(a[m][d][0]), "r"(a[m][d][1]), "r"(a[m][d][2]), "r"(a[m][d][3]),
                          "r"(b[n][d][0]), "r"(b[n][d][1]),
                          "r"(sfa[m / 2u]), "h"((short)kk), "h"((short)(m & 1u)),
                          "r"(sfb[n / 4u]), "h"((short)kk), "h"((short)(n & 3u)));
                }
            }
        }
    };

    asm volatile(
      "{\n.reg .pred p;\nMWF0_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra MWF0_%=;\n}\n" :: "r"(MFULL_A(0)), "r"(0u) : "memory");
    load_pair(0u, 0u, 0u, aR[0], bR[0], sfaR[0], sfbR[0]);

    unsigned int c = 0;
    for (; c + 1u < num_kt; c++) {
        unsigned int buf  = c % STAGES;
        unsigned int nbuf = (c + 1u) % STAGES;
        unsigned int nph  = ((c + 1u) / STAGES) & 1u;
        unsigned int sfo  = (c & 3u) * 4u;
        unsigned int nsfo = ((c + 1u) & 3u) * 4u;
        load_pair(buf, 1u, sfo, aR[1], bR[1], sfaR[1], sfbR[1]);
        gemm_pair(0u, aR[0], bR[0], sfaR[0], sfbR[0]);
        unsigned long long st;
        asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
                     : "=l"(st) : "r"(MEMPTY_A(buf)) : "memory");
        if constexpr (PROD_W == 0u) {
            if (tid == 0) {
                unsigned int g = c + STAGES - 1u;
                if (g < num_kt)
                    issue_chunk(g);
            }
        }
        asm volatile(
          "{\n.reg .pred p;\nMWF_%=:\n"
          "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
          "@!p bra MWF_%=;\n}\n" :: "r"(MFULL_A(nbuf)), "r"(nph) : "memory");
        load_pair(nbuf, 0u, nsfo, aR[0], bR[0], sfaR[0], sfbR[0]);
        gemm_pair(1u, aR[1], bR[1], sfaR[1], sfbR[1]);
    }
    {
        unsigned int buf = c % STAGES;
        unsigned int sfo = (c & 3u) * 4u;
        load_pair(buf, 1u, sfo, aR[1], bR[1], sfaR[1], sfbR[1]);
        gemm_pair(0u, aR[0], bR[0], sfaR[0], sfbR[0]);
        unsigned long long st;
        asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
                     : "=l"(st) : "r"(MEMPTY_A(buf)) : "memory");
        gemm_pair(1u, aR[1], bR[1], sfaR[1], sfbR[1]);
    }

    if constexpr (SPLITK) {
        // split-K: f32-партиалы из acc прямо в ws[z*M*N] (без smem-stage);
        // фикс-порядок суммирования в reduce → детерминизм.
        float* wsz = ws + (size_t)blockIdx.y * (size_t)M * (size_t)N;
        #pragma unroll
        for (unsigned int m = 0; m < MA; m++) {
            unsigned int row = wm * WARP_M + m * 16u + (lane >> 2);
            #pragma unroll
            for (unsigned int n = 0; n < NB; n++) {
                unsigned int col = wn * WARP_N + n * 8u + (lane & 3u) * 2u;
                if (block_m0 + row < M) {
                    float* d = wsz + (size_t)(block_m0 + row) * N + block_n0 + col;
                    asm volatile("st.global.cs.v2.f32 [%0], {%1,%2};\n"
                        :: "l"(d), "f"(acc[m][n][0]), "f"(acc[m][n][1]) : "memory");
                }
                if (block_m0 + row + 8u < M) {
                    float* d = wsz + (size_t)(block_m0 + row + 8u) * N + block_n0 + col;
                    asm volatile("st.global.cs.v2.f32 [%0], {%1,%2};\n"
                        :: "l"(d), "f"(acc[m][n][2]), "f"(acc[m][n][3]) : "memory");
                }
            }
        }
        return;
    }
    // TMA-store эпилог (рецепт nvfp4 0384f96d): один bar (стадии дочитаны),
    // варп пишет СВОЙ регион [WARP_M × WARP_N] half row-major в приватный слот
    // stage-smem через stmatrix.x4 (БЕЗ .trans: C-строка = mma-строка, fragment
    // (lane/4, lane%4) — ровно m8n8-лейаут), fence.proxy.async → lane0 один
    // TMA-store; OOB-строки M-хвоста клипает дескриптор (гард бесплатно).
    constexpr unsigned int WSLOT = WARP_M * WARP_N * 2u;
    static_assert(NB % 2u == 0, "mxfp8 ROT: stmatrix.x4 берёт пары n");
    static_assert(CONS_WARPS * WSLOT <= STAGES * STAGE,
                  "mxfp8 ROT: эпилог-слоты не влезают в smem стадий");
    asm volatile("bar.sync 7, %0;\n" :: "r"(THREADS) : "memory");
    unsigned int slot = sbase + warp * WSLOT;
    unsigned int oct  = lane >> 3;
    unsigned int srow = lane & 7u;
    #pragma unroll
    for (unsigned int m = 0; m < MA; m++) {
        #pragma unroll
        for (unsigned int n = 0; n < NB; n += 2u) {
            // x4: матрицы (rows0-7,n),(rows8-15,n),(rows0-7,n+1),(rows8-15,n+1).
            unsigned int t_n    = n + (oct >> 1);
            unsigned int t_half = oct & 1u;
            // Свизл слота по ширине строки (как nvfp4): 128Б → Swizzle<3,4,3>,
            // 64Б → Swizzle<2,4,3>; без него stmatrix бьётся в банк-конфликт.
            unsigned int soff = (m * 16u + t_half * 8u + srow) * (WARP_N * 2u)
                              + t_n * 16u;
            if constexpr (WARP_N * 2u == 128u) {
                soff ^= ((soff >> 7u) & 7u) << 4u;
            } else if constexpr (WARP_N * 2u == 64u) {
                soff ^= ((soff >> 7u) & 3u) << 4u;
            }
            __half2 p0 = __float22half2_rn({acc[m][n][0], acc[m][n][1]});
            __half2 p1 = __float22half2_rn({acc[m][n][2], acc[m][n][3]});
            __half2 p2 = __float22half2_rn({acc[m][n + 1][0], acc[m][n + 1][1]});
            __half2 p3 = __float22half2_rn({acc[m][n + 1][2], acc[m][n + 1][3]});
            asm volatile(
                "stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(slot + soff), "r"(*(unsigned int*)&p0), "r"(*(unsigned int*)&p1),
                   "r"(*(unsigned int*)&p2), "r"(*(unsigned int*)&p3) : "memory");
        }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if (lane == 0) {
        unsigned int gx = (block_n0 + wn * WARP_N) * 2u;
        unsigned int gy = block_m0 + wm * WARP_M;
        asm volatile(
          "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
          " [%0, {%1, %2}], [%3];\n"
          :: "l"((unsigned long long)out_desc), "r"(gx), "r"(gy), "r"(slot) : "memory");
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
    }
    (void)out;
    #undef MFULL_A
    #undef MEMPTY_A
}

extern "C" __global__ __launch_bounds__(256, 1) void gn_mxfp8_rot_128x128_s2(
    const void* a_desc, const void* b_desc, const void* sfa_desc, const void* sfb_desc,
    __half* out, unsigned int M, unsigned int N, unsigned int K, unsigned int raster_gr,
    const void* out_desc) {
    mxfp8_rot_device<128, 128, 2, 4, 2>(a_desc, b_desc, sfa_desc, sfb_desc, out, M, N, K, raster_gr, out_desc);
}

// Выделенный producer-warpgroup (384 потока, setmaxnreg 240/24 — работает после
// ::cta): при S=2 продьюсер заполняет released-буфер мгновенно (fused-tid0 голодал:
// long_sb 18.6% + wait 21.7%).
extern "C" __global__ __launch_bounds__(384, 1) void gn_mxfp8_drot_128x128_s2(
    const void* a_desc, const void* b_desc, const void* sfa_desc, const void* sfb_desc,
    __half* out, unsigned int M, unsigned int N, unsigned int K, unsigned int raster_gr,
    const void* out_desc) {
    mxfp8_rot_device<128, 128, 2, 4, 2, 4u, 24, 240>(a_desc, b_desc, sfa_desc, sfb_desc, out, M, N, K, raster_gr, out_desc);
}

// 64×128 (M-узкий): attn средние M — 128-тайл даёт 64 CTA на M=256 (полмашины);
// 64-тайл удваивает грид. s3 — глубже конвейер (стадия 27.6KB → 3×=83KB ok).
extern "C" __global__ __launch_bounds__(384, 1) void gn_mxfp8_drot_128x128_s2_sk(
    const void* a_desc, const void* b_desc, const void* sfa_desc, const void* sfb_desc,
    float* ws, unsigned int M, unsigned int N, unsigned int K, unsigned int raster_gr,
    unsigned int kt_chunk) {
    unsigned int kt0 = blockIdx.y * kt_chunk;
    unsigned int total = K / 128u;
    unsigned int kt_len = (kt0 + kt_chunk <= total) ? kt_chunk : (total - kt0);
    mxfp8_rot_device<128, 128, 2, 4, 2, 4u, 24, 240, true>(
        a_desc, b_desc, sfa_desc, sfb_desc, nullptr, M, N, K, raster_gr, nullptr, ws, kt0, kt_len);
}

// Редьюс split-K: фикс-порядок суммирования (детерминизм), f32 → f16.
extern "C" __global__ void mxfp8_sk_reduce(const float* __restrict__ ws,
                                           __half* __restrict__ out,
                                           long long mn, int splits) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < mn; i += stride) {
        float v = 0.f;
        for (int s = 0; s < splits; s++)
            v += ws[(size_t)s * mn + i];
        out[i] = __float2half(v);
    }
}

extern "C" __global__ __launch_bounds__(384, 1) void gn_mxfp8_drot_64x128_s2(
    const void* a_desc, const void* b_desc, const void* sfa_desc, const void* sfb_desc,
    __half* out, unsigned int M, unsigned int N, unsigned int K, unsigned int raster_gr,
    const void* out_desc) {
    mxfp8_rot_device<64, 128, 2, 4, 2, 4u, 24, 240>(a_desc, b_desc, sfa_desc, sfb_desc, out, M, N, K, raster_gr, out_desc);
}
extern "C" __global__ __launch_bounds__(384, 1) void gn_mxfp8_drot_64x128_s3(
    const void* a_desc, const void* b_desc, const void* sfa_desc, const void* sfb_desc,
    __half* out, unsigned int M, unsigned int N, unsigned int K, unsigned int raster_gr,
    const void* out_desc) {
    mxfp8_rot_device<64, 128, 2, 4, 3, 4u, 24, 240>(a_desc, b_desc, sfa_desc, sfb_desc, out, M, N, K, raster_gr, out_desc);
}
// 64×256 (N-широкий, bf16-урок b256t): меньше A-перечтений на широком N;
// стадия (64+256)×144 = 46.1KB → 2 стадии 92.2KB (лимит 99KB).
extern "C" __global__ __launch_bounds__(384, 1) void gn_mxfp8_drot_64x256_s2(
    const void* a_desc, const void* b_desc, const void* sfa_desc, const void* sfb_desc,
    __half* out, unsigned int M, unsigned int N, unsigned int K, unsigned int raster_gr,
    const void* out_desc) {
    mxfp8_rot_device<64, 256, 2, 4, 2, 4u, 24, 240>(a_desc, b_desc, sfa_desc, sfb_desc, out, M, N, K, raster_gr, out_desc);
}
