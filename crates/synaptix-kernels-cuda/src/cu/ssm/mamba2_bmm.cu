#include <cuda_bf16.h>

// Mamba2 chunked-SSD helper: batched BF16 matmul с F32 accumulator.
//
// Semantically computes C[b, m, n] = sum_k A[b, m, k] * B[b, n, k].
// Layout:
//   A  [batch, M, K]  BF16 row-major
//   B  [batch, N, K]  BF16 row-major   (т.е. B физически хранится как (N, K))
//   C  [batch, M, N]  F32  row-major
// Для standard `C = A @ B_logical` где B_logical (K, N) row-major — caller
// должен подать на вход (N, K) row-major (= transposed). Для нашего use case
// это удобно — половина Mamba2 chunked SSD bmm-ов уже имеет нужный layout
// без транспонирования.
//
// Используется `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` —
// идентичный pattern из `cu/mma_gemm_bf16.cu`. ldmatrix.x4 для A,
// manual pack для B (вторая половина 16x16 → m16n8 tile).
//
// Ограничения: M%16==0, N%8==0, K%16==0. Проверяется в Rust-обёртке.
// На границе тайла OOB-страница нулей через bounds-check в gmem-load.

template <unsigned int WARPS_M, unsigned int WARPS_N>
__device__ __forceinline__ void mamba2_bmm_bf16_f32acc_impl(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float*               __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int batch_count)
{
    constexpr unsigned int THREADS      = WARPS_M * WARPS_N * 32u;
    constexpr unsigned int A_SMEM_ELEMS = WARPS_M * 16u * 16u; // (WARPS_M*16) × 16 BF16
    constexpr unsigned int B_SMEM_ELEMS = WARPS_N * 8u  * 16u; // (WARPS_N*8)  × 16 BF16

    extern __shared__ __nv_bfloat16 smem_bf[];
    __nv_bfloat16* smem_A = smem_bf;
    __nv_bfloat16* smem_B = smem_bf + A_SMEM_ELEMS;

    unsigned int tid  = threadIdx.x;
    unsigned int warp = tid >> 5;
    unsigned int lane = tid & 31u;
    unsigned int warp_m_idx = warp / WARPS_N;
    unsigned int warp_n_idx = warp % WARPS_N;

    unsigned int b_idx       = blockIdx.z;
    unsigned int m_warp_base = blockIdx.x * (WARPS_M * 16u) + warp_m_idx * 16u;
    unsigned int n_warp_base = blockIdx.y * (WARPS_N * 8u)  + warp_n_idx * 8u;

    if (b_idx >= batch_count) return;

    const __nv_bfloat16* Ab = A + (size_t)b_idx * (size_t)M * (size_t)K;
    const __nv_bfloat16* Bb = B + (size_t)b_idx * (size_t)N * (size_t)K;
    float*               Cb = C + (size_t)b_idx * (size_t)M * (size_t)N;

    // Manual pack индексы для A (PTX m16n8k16 row.col convention).
    // Per-lane (g = lane/4, t = lane%4) для A operand (m16k16 row):
    //   a0 = packed A[g,   2t..2t+1]
    //   a1 = packed A[g+8, 2t..2t+1]
    //   a2 = packed A[g,   2t+8..2t+9]
    //   a3 = packed A[g+8, 2t+8..2t+9]
    // smem_A хранится как (M_warp_tile=16, K=16) row-major.
    unsigned int a_row_lo = lane >> 2;       // 0..7
    unsigned int a_row_hi = a_row_lo + 8u;   // 8..15
    unsigned int a_col_lo = (lane & 3u) << 1; // 0..7 (низкая K-пара)
    unsigned int a_col_hi = a_col_lo + 8u;   // 8..15 (высокая K-пара)

    // Manual pack индексы для B (PTX m16n8k16 row.col convention, образец flash_mxfp8_prefill.cu).
    // Per-lane (g = lane/4, t = lane%4):
    //   b0 packs B[n=g, k=2t]   и B[n=g, k=2t+1]
    //   b1 packs B[n=g, k=2t+8] и B[n=g, k=2t+9]
    // smem_B хранится как (N=8, K=16) row-major внутри warp_n тайла.
    unsigned int b_n_idx = lane >> 2;       // 0..7 — какой из 8 N-столбцов
    unsigned int b_k_lo  = (lane & 3u) << 1; // 0..7 — низкая K-пара

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    unsigned int num_chunks_k = K >> 4; // K % 16 проверено caller'ом.

    for (unsigned int chunk_k = 0; chunk_k < num_chunks_k; chunk_k++) {
        // ── Load A chunk: WARPS_M tiles по 16×16 BF16 (32 uint4 per tile).
        {
            unsigned int total_uint4 = WARPS_M * 32u;
            for (unsigned int i = tid; i < total_uint4; i += THREADS) {
                unsigned int wm        = i / 32u;
                unsigned int idx_in_wm = i & 31u;
                unsigned int row       = idx_in_wm >> 1;
                unsigned int col_half  = idx_in_wm & 1u;
                unsigned int gmem_m    = blockIdx.x * (WARPS_M * 16u) + wm * 16u + row;
                unsigned int gmem_k    = chunk_k * 16u + col_half * 8u;
                __nv_bfloat16* dst     = smem_A + wm * 256u + row * 16u + col_half * 8u;
                if (gmem_m < M) {
                    const __nv_bfloat16* src = Ab + (size_t)gmem_m * K + gmem_k;
                    *((uint4*)dst) = *((const uint4*)src);
                } else {
                    uint4 zero = {0u, 0u, 0u, 0u};
                    *((uint4*)dst) = zero;
                }
            }
        }
        // ── Load B chunk: WARPS_N tiles по 8×16 BF16 (16 uint4 per tile).
        {
            unsigned int total_uint4 = WARPS_N * 16u;
            for (unsigned int i = tid; i < total_uint4; i += THREADS) {
                unsigned int wn        = i / 16u;
                unsigned int idx_in_wn = i & 15u;
                unsigned int row       = idx_in_wn >> 1;
                unsigned int col_half  = idx_in_wn & 1u;
                unsigned int gmem_n    = blockIdx.y * (WARPS_N * 8u) + wn * 8u + row;
                unsigned int gmem_k    = chunk_k * 16u + col_half * 8u;
                __nv_bfloat16* dst     = smem_B + wn * 128u + row * 16u + col_half * 8u;
                if (gmem_n < N) {
                    const __nv_bfloat16* src = Bb + (size_t)gmem_n * K + gmem_k;
                    *((uint4*)dst) = *((const uint4*)src);
                } else {
                    uint4 zero = {0u, 0u, 0u, 0u};
                    *((uint4*)dst) = zero;
                }
            }
        }
        __syncthreads();

        // ── Manual pack для A (16×16 BF16 → 4 packed uint32).
        const __nv_bfloat16* A_warp = smem_A + warp_m_idx * 256u;
        __nv_bfloat16 a_lo_lo_0 = A_warp[a_row_lo * 16u + a_col_lo];
        __nv_bfloat16 a_lo_lo_1 = A_warp[a_row_lo * 16u + a_col_lo + 1u];
        __nv_bfloat16 a_hi_lo_0 = A_warp[a_row_hi * 16u + a_col_lo];
        __nv_bfloat16 a_hi_lo_1 = A_warp[a_row_hi * 16u + a_col_lo + 1u];
        __nv_bfloat16 a_lo_hi_0 = A_warp[a_row_lo * 16u + a_col_hi];
        __nv_bfloat16 a_lo_hi_1 = A_warp[a_row_lo * 16u + a_col_hi + 1u];
        __nv_bfloat16 a_hi_hi_0 = A_warp[a_row_hi * 16u + a_col_hi];
        __nv_bfloat16 a_hi_hi_1 = A_warp[a_row_hi * 16u + a_col_hi + 1u];
        unsigned int a0 = ((unsigned int)__bfloat16_as_ushort(a_lo_lo_0))
                        | ((unsigned int)__bfloat16_as_ushort(a_lo_lo_1) << 16);
        unsigned int a1 = ((unsigned int)__bfloat16_as_ushort(a_hi_lo_0))
                        | ((unsigned int)__bfloat16_as_ushort(a_hi_lo_1) << 16);
        unsigned int a2 = ((unsigned int)__bfloat16_as_ushort(a_lo_hi_0))
                        | ((unsigned int)__bfloat16_as_ushort(a_lo_hi_1) << 16);
        unsigned int a3 = ((unsigned int)__bfloat16_as_ushort(a_hi_hi_0))
                        | ((unsigned int)__bfloat16_as_ushort(a_hi_hi_1) << 16);

        // ── Manual pack для B (8×16 BF16, 2 uint32 per lane).
        // smem_B[warp_n_idx * 128 + n_row * 16 + k_col].
        const __nv_bfloat16* B_warp = smem_B + warp_n_idx * 128u + b_n_idx * 16u;
        __nv_bfloat16 b00 = B_warp[b_k_lo];
        __nv_bfloat16 b01 = B_warp[b_k_lo + 1u];
        __nv_bfloat16 b10 = B_warp[b_k_lo + 8u];
        __nv_bfloat16 b11 = B_warp[b_k_lo + 9u];
        unsigned int b0 = ((unsigned int)__bfloat16_as_ushort(b00))
                        | ((unsigned int)__bfloat16_as_ushort(b01) << 16);
        unsigned int b1 = ((unsigned int)__bfloat16_as_ushort(b10))
                        | ((unsigned int)__bfloat16_as_ushort(b11) << 16);

        // ── mma.sync m16n8k16 BF16 → F32 acc.
        float n0, n1, n2, n3;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(n0), "=f"(n1), "=f"(n2), "=f"(n3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(d0), "f"(d1), "f"(d2), "f"(d3)
        );
        d0 = n0; d1 = n1; d2 = n2; d3 = n3;
        __syncthreads();
    }

    // ── Store F32 output в (batch, M, N) row-major.
    // Fragment layout (per lane) для m16n8 acc:
    //   d0 → C[m_warp_base + lane/4,     n_warp_base + (lane%4)*2 + 0]
    //   d1 → C[m_warp_base + lane/4,     n_warp_base + (lane%4)*2 + 1]
    //   d2 → C[m_warp_base + lane/4 + 8, n_warp_base + (lane%4)*2 + 0]
    //   d3 → C[m_warp_base + lane/4 + 8, n_warp_base + (lane%4)*2 + 1]
    unsigned int m_row_top = m_warp_base + (lane >> 2);
    unsigned int m_row_bot = m_row_top + 8u;
    unsigned int n_col_lo  = n_warp_base + ((lane & 3u) << 1);
    unsigned int n_col_hi  = n_col_lo + 1u;

    if (m_row_top < M) {
        if (n_col_lo < N) Cb[(size_t)m_row_top * N + n_col_lo] = d0;
        if (n_col_hi < N) Cb[(size_t)m_row_top * N + n_col_hi] = d1;
    }
    if (m_row_bot < M) {
        if (n_col_lo < N) Cb[(size_t)m_row_bot * N + n_col_lo] = d2;
        if (n_col_hi < N) Cb[(size_t)m_row_bot * N + n_col_hi] = d3;
    }
}

extern "C" __global__ void mamba2_bmm_bf16_f32acc_1x4(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float*               __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int batch_count)
{
    mamba2_bmm_bf16_f32acc_impl<1, 4>(A, B, C, M, N, K, batch_count);
}

extern "C" __global__ void mamba2_bmm_bf16_f32acc_2x2(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float*               __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int batch_count)
{
    mamba2_bmm_bf16_f32acc_impl<2, 2>(A, B, C, M, N, K, batch_count);
}

extern "C" __global__ void mamba2_bmm_bf16_f32acc_2x4(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float*               __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int batch_count)
{
    mamba2_bmm_bf16_f32acc_impl<2, 4>(A, B, C, M, N, K, batch_count);
}

extern "C" __global__ void mamba2_bmm_bf16_f32acc_4x2(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float*               __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int batch_count)
{
    mamba2_bmm_bf16_f32acc_impl<4, 2>(A, B, C, M, N, K, batch_count);
}

extern "C" __global__ void mamba2_bmm_bf16_f32acc_4x4(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float*               __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int batch_count)
{
    mamba2_bmm_bf16_f32acc_impl<4, 4>(A, B, C, M, N, K, batch_count);
}
