#include <cuda_fp16.h>

#ifdef SYN_OUT_BF16
#include <cuda_bf16.h>
typedef __nv_bfloat16 syn_out_t;
#define SYN_TO_OUT(v) __float2bfloat16(v)
#else
typedef __half syn_out_t;
#define SYN_TO_OUT(v) __float2half(v)
#endif

template <unsigned int WARPS>
__device__ __forceinline__ void mma_gemm_shuf_impl(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N,
    unsigned int K,
    unsigned int sf_inner_dim_w,
    unsigned int sf_inner_dim_x)
{
    constexpr unsigned int M_TILE = WARPS * 16;
    constexpr unsigned int THREADS = WARPS * 32;

    extern __shared__ unsigned char smem[];
    unsigned char* smem_x = smem;
    unsigned int k_half = K >> 1;

    unsigned int tid  = threadIdx.x;
    unsigned int warp = tid >> 5;
    unsigned int lane = tid & 31u;
    unsigned int batch_row = blockIdx.y;

    unsigned int x_u32_count = k_half >> 2;
    unsigned int*       smem_x_u32 = (unsigned int*)smem_x;
    const unsigned int* gmem_x_u32 =
        (const unsigned int*)(packed_x + batch_row * k_half);
    for (unsigned int i = tid; i < x_u32_count; i += THREADS) {
        smem_x_u32[i] = gmem_x_u32[i];
    }
    __syncthreads();

    unsigned int m_warp_base = blockIdx.x * M_TILE + warp * 16u;
    if (m_warp_base >= N) return;

    unsigned int m_t = lane & 3u;
    unsigned int k_t = lane >> 2;
    unsigned int s_a = lane & 1u;
    unsigned int s_c = lane >> 2;
    unsigned int m_for_sfa = m_warp_base + s_a * 8u + s_c;

    unsigned int m_block_warp = m_warp_base >> 4;
    unsigned int block_base = m_block_warp * (K >> 6) * 512u;
    unsigned int top_off = k_t * 32u + m_t * 4u;
    unsigned int bot_off = (k_t + 8u) * 32u + m_t * 4u;

    unsigned int k_lo_off    = m_t * 4u;
    unsigned int k_hi_off    = k_lo_off + 16u;

    unsigned int tile_row_w   = m_for_sfa >> 7;
    unsigned int local_outer_w  = m_for_sfa & 127u;
    unsigned int off_in_tile_w  = (local_outer_w & 31u) * 16u + (local_outer_w >> 5) * 4u;
    unsigned int sfa_row_base = tile_row_w * sf_inner_dim_w * 128u + off_in_tile_w;

    unsigned int tile_row_x   = batch_row >> 7;
    unsigned int local_outer_x  = batch_row & 127u;
    unsigned int off_in_tile_x  = (local_outer_x & 31u) * 16u + (local_outer_x >> 5) * 4u;
    unsigned int sfb_row_base = tile_row_x * sf_inner_dim_x * 128u + off_in_tile_x;

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    unsigned int num_chunks = K >> 6;

    for (unsigned int chunk = 0; chunk < num_chunks; chunk++) {
        unsigned int chunk_base = block_base + chunk * 512u;
        unsigned int k_chunk_b  = chunk << 5;

        unsigned int a0 = *(const unsigned int*)(packed_w + chunk_base + top_off);
        unsigned int a1 = *(const unsigned int*)(packed_w + chunk_base + bot_off);
        unsigned int a2 = *(const unsigned int*)(packed_w + chunk_base + top_off + 16u);
        unsigned int a3 = *(const unsigned int*)(packed_w + chunk_base + bot_off + 16u);

        unsigned int b0 = *(const unsigned int*)(smem_x + k_chunk_b + k_lo_off);
        unsigned int b1 = *(const unsigned int*)(smem_x + k_chunk_b + k_hi_off);

        unsigned int sfa0 = *(const unsigned int*)(scales_w + sfa_row_base + chunk * 512u);
        unsigned int sfb0 = *(const unsigned int*)(scales_x + sfb_row_base + chunk * 512u);

        constexpr unsigned short tidA = 0, bidA = 0, tidB = 0, bidB = 0;
        float n0, n1, n2, n3;
        asm volatile(
          "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
          "{%0, %1, %2, %3},"
          "{%4, %5, %6, %7},"
          "{%8, %9},"
          "{%10, %11, %12, %13},"
          "{%14},"
          "{%15, %16},"
          "{%17},"
          "{%18, %19};\n"
          : "=f"(n0), "=f"(n1), "=f"(n2), "=f"(n3)
          : "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
            "r"(b0),  "r"(b1),
            "f"(d0),  "f"(d1),  "f"(d2),  "f"(d3),
            "r"(sfa0), "h"(bidA), "h"(tidA),
            "r"(sfb0), "h"(bidB), "h"(tidB));
        d0 = n0; d1 = n1; d2 = n2; d3 = n3;
    }

    if ((lane & 3u) == 0u) {
        unsigned int row_top = lane >> 2;
        unsigned int m_top_g = m_warp_base + row_top;
        unsigned int m_bot_g = m_top_g + 8u;
        syn_out_t* out_row = out + batch_row * N;
        if (m_top_g < N) out_row[m_top_g] = SYN_TO_OUT(d0);
        if (m_bot_g < N) out_row[m_bot_g] = SYN_TO_OUT(d2);
    }
}

extern "C" __global__ void nvfp4_mma_gemm_shuf_f16_w4(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N, unsigned int K,
    unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x)
{
    mma_gemm_shuf_impl<4>(packed_w, scales_w, packed_x, scales_x, out,
                          N, K, sf_inner_dim_w, sf_inner_dim_x);
}

extern "C" __global__ void nvfp4_mma_gemm_shuf_f16_w8(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N, unsigned int K,
    unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x)
{
    mma_gemm_shuf_impl<8>(packed_w, scales_w, packed_x, scales_x, out,
                          N, K, sf_inner_dim_w, sf_inner_dim_x);
}

template <unsigned int WARPS>
__device__ __forceinline__ void mma_gemm_shuf_n8_impl(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N,
    unsigned int K,
    unsigned int sf_inner_dim_w,
    unsigned int sf_inner_dim_x)
{
    constexpr unsigned int M_TILE = WARPS * 16;
    constexpr unsigned int BATCH_TILE = 8;

    unsigned int k_half = K >> 1;
    unsigned int batch_tile_base = blockIdx.y * BATCH_TILE;

    unsigned int tid  = threadIdx.x;
    unsigned int warp = tid >> 5;
    unsigned int lane = tid & 31u;

    const unsigned char* x_base = packed_x + batch_tile_base * k_half;

    unsigned int m_warp_base = blockIdx.x * M_TILE + warp * 16u;
    if (m_warp_base >= N) return;

    unsigned int m_t = lane & 3u;
    unsigned int n_t = lane >> 2;

    unsigned int s_a = lane & 1u;
    unsigned int s_c = lane >> 2;
    unsigned int m_for_sfa = m_warp_base + s_a * 8u + s_c;
    unsigned int tile_row_w   = m_for_sfa >> 7;
    unsigned int local_outer_w  = m_for_sfa & 127u;
    unsigned int off_in_tile_w  = (local_outer_w & 31u) * 16u + (local_outer_w >> 5) * 4u;
    unsigned int sfa_row_base = tile_row_w * sf_inner_dim_w * 128u + off_in_tile_w;

    // ФИКС: B-scale (sfb) batch-row маппинг ДОЛЖЕН совпадать с B-data (n_t=lane>>2),
    // НЕ lane&7. Нить lane держит X-данные для batch-row n_t=lane/4 (x_row_off ниже),
    // поэтому её scale обязан быть для той же строки. (lane&7) брал scale ЧУЖОЙ строки
    // → per-row ошибка ~7.7 (vs квант 0.12). Broadcast верен т.к. 1 строка/блок.
    unsigned int sfb_row_global = batch_tile_base + (lane >> 2);
    unsigned int tile_row_x   = sfb_row_global >> 7;
    unsigned int local_outer_x  = sfb_row_global & 127u;
    unsigned int off_in_tile_x  = (local_outer_x & 31u) * 16u + (local_outer_x >> 5) * 4u;
    unsigned int sfb_row_base = tile_row_x * sf_inner_dim_x * 128u + off_in_tile_x;

    unsigned int m_block_warp = m_warp_base >> 4;
    unsigned int block_base = m_block_warp * (K >> 6) * 512u;
    unsigned int k_t = lane >> 2;
    unsigned int top_off = k_t * 32u + m_t * 4u;
    unsigned int bot_off = (k_t + 8u) * 32u + m_t * 4u;

    unsigned int x_row_off = n_t * k_half;
    unsigned int k_lo_off  = m_t * 4u;
    unsigned int k_hi_off  = k_lo_off + 16u;

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    unsigned int num_chunks = K >> 6;

    for (unsigned int chunk = 0; chunk < num_chunks; chunk++) {
        unsigned int chunk_base = block_base + chunk * 512u;
        unsigned int k_chunk_b  = chunk << 5;

        unsigned int a0 = *(const unsigned int*)(packed_w + chunk_base + top_off);
        unsigned int a1 = *(const unsigned int*)(packed_w + chunk_base + bot_off);
        unsigned int a2 = *(const unsigned int*)(packed_w + chunk_base + top_off + 16u);
        unsigned int a3 = *(const unsigned int*)(packed_w + chunk_base + bot_off + 16u);

        unsigned int b0 = *(const unsigned int*)(x_base + x_row_off + k_chunk_b + k_lo_off);
        unsigned int b1 = *(const unsigned int*)(x_base + x_row_off + k_chunk_b + k_hi_off);

        unsigned int sfa0 = *(const unsigned int*)(scales_w + sfa_row_base + chunk * 512u);
        unsigned int sfb0 = *(const unsigned int*)(scales_x + sfb_row_base + chunk * 512u);

        constexpr unsigned short tidA = 0, bidA = 0, tidB = 0, bidB = 0;
        float n0, n1, n2, n3;
        asm volatile(
          "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
          "{%0, %1, %2, %3},"
          "{%4, %5, %6, %7},"
          "{%8, %9},"
          "{%10, %11, %12, %13},"
          "{%14},"
          "{%15, %16},"
          "{%17},"
          "{%18, %19};\n"
          : "=f"(n0), "=f"(n1), "=f"(n2), "=f"(n3)
          : "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
            "r"(b0),  "r"(b1),
            "f"(d0),  "f"(d1),  "f"(d2),  "f"(d3),
            "r"(sfa0), "h"(bidA), "h"(tidA),
            "r"(sfb0), "h"(bidB), "h"(tidB));
        d0 = n0; d1 = n1; d2 = n2; d3 = n3;
    }

    unsigned int m_row_top = m_warp_base + (lane >> 2);
    unsigned int m_row_bot = m_row_top + 8u;
    unsigned int batch_col0 = batch_tile_base + ((lane & 3u) << 1);
    unsigned int batch_col1 = batch_col0 + 1u;
    if (m_row_top < N) {
        __stcs(out + batch_col0 * N + m_row_top, SYN_TO_OUT(d0));
        __stcs(out + batch_col1 * N + m_row_top, SYN_TO_OUT(d1));
    }
    if (m_row_bot < N) {
        __stcs(out + batch_col0 * N + m_row_bot, SYN_TO_OUT(d2));
        __stcs(out + batch_col1 * N + m_row_bot, SYN_TO_OUT(d3));
    }
}

extern "C" __global__ void nvfp4_mma_gemm_shuf_n8_f16_w4(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N, unsigned int K,
    unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x)
{
    mma_gemm_shuf_n8_impl<4>(packed_w, scales_w, packed_x, scales_x, out,
                             N, K, sf_inner_dim_w, sf_inner_dim_x);
}

extern "C" __global__ void nvfp4_mma_gemm_shuf_n8_f16_w8(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N, unsigned int K,
    unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x)
{
    mma_gemm_shuf_n8_impl<8>(packed_w, scales_w, packed_x, scales_x, out,
                             N, K, sf_inner_dim_w, sf_inner_dim_x);
}

template <unsigned int WARPS_M, unsigned int WARPS_N>
__device__ __forceinline__ void mma_gemm_shuf_2d_impl(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N,
    unsigned int K,
    unsigned int sf_inner_dim_w,
    unsigned int sf_inner_dim_x)
{
    constexpr unsigned int BLOCK_M = WARPS_M * 16;
    constexpr unsigned int BLOCK_N = WARPS_N * 8;
    constexpr unsigned int WARPS_TOTAL = WARPS_M * WARPS_N;
    (void)BLOCK_M; (void)WARPS_TOTAL;

    unsigned int k_half = K >> 1;

    unsigned int tid  = threadIdx.x;
    unsigned int warp = tid >> 5;
    unsigned int lane = tid & 31u;

    unsigned int warp_m_idx = warp / WARPS_N;
    unsigned int warp_n_idx = warp % WARPS_N;

    unsigned int m_warp_base = blockIdx.x * (WARPS_M * 16u) + warp_m_idx * 16u;
    unsigned int batch_warp_base = blockIdx.y * (WARPS_N * 8u) + warp_n_idx * 8u;

    if (m_warp_base >= N) return;

    unsigned int m_t = lane & 3u;
    unsigned int n_t = lane >> 2;

    unsigned int s_a = lane & 1u;
    unsigned int s_c = lane >> 2;
    unsigned int m_for_sfa = m_warp_base + s_a * 8u + s_c;
    unsigned int tile_row_w   = m_for_sfa >> 7;
    unsigned int local_outer_w  = m_for_sfa & 127u;
    unsigned int off_in_tile_w  = (local_outer_w & 31u) * 16u + (local_outer_w >> 5) * 4u;
    unsigned int sfa_row_base = tile_row_w * sf_inner_dim_w * 128u + off_in_tile_w;

    // ФИКС: B-scale (sfb) batch-row маппинг = n_t (lane>>2), совпадает с B-data ниже
    // (x_row_off = (batch_warp_base + n_t)). Было (lane&7) → scale чужой строки → баг.
    unsigned int sfb_row_global = batch_warp_base + (lane >> 2);
    unsigned int tile_row_x   = sfb_row_global >> 7;
    unsigned int local_outer_x  = sfb_row_global & 127u;
    unsigned int off_in_tile_x  = (local_outer_x & 31u) * 16u + (local_outer_x >> 5) * 4u;
    unsigned int sfb_row_base = tile_row_x * sf_inner_dim_x * 128u + off_in_tile_x;

    unsigned int m_block_warp = m_warp_base >> 4;
    unsigned int block_base = m_block_warp * (K >> 6) * 512u;
    unsigned int k_t = lane >> 2;
    unsigned int top_off = k_t * 32u + m_t * 4u;
    unsigned int bot_off = (k_t + 8u) * 32u + m_t * 4u;

    unsigned int x_row_off = (batch_warp_base + n_t) * k_half;
    unsigned int k_lo_off  = m_t * 4u;
    unsigned int k_hi_off  = k_lo_off + 16u;

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    unsigned int num_chunks = K >> 6;

    for (unsigned int chunk = 0; chunk < num_chunks; chunk++) {
        unsigned int chunk_base = block_base + chunk * 512u;
        unsigned int k_chunk_b  = chunk << 5;

        unsigned int a0 = *(const unsigned int*)(packed_w + chunk_base + top_off);
        unsigned int a1 = *(const unsigned int*)(packed_w + chunk_base + bot_off);
        unsigned int a2 = *(const unsigned int*)(packed_w + chunk_base + top_off + 16u);
        unsigned int a3 = *(const unsigned int*)(packed_w + chunk_base + bot_off + 16u);

        unsigned int b0 = *(const unsigned int*)(packed_x + x_row_off + k_chunk_b + k_lo_off);
        unsigned int b1 = *(const unsigned int*)(packed_x + x_row_off + k_chunk_b + k_hi_off);

        unsigned int sfa0 = *(const unsigned int*)(scales_w + sfa_row_base + chunk * 512u);
        unsigned int sfb0 = *(const unsigned int*)(scales_x + sfb_row_base + chunk * 512u);

        constexpr unsigned short tidA = 0, bidA = 0, tidB = 0, bidB = 0;
        float n0, n1, n2, n3;
        asm volatile(
          "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
          "{%0, %1, %2, %3},"
          "{%4, %5, %6, %7},"
          "{%8, %9},"
          "{%10, %11, %12, %13},"
          "{%14},"
          "{%15, %16},"
          "{%17},"
          "{%18, %19};\n"
          : "=f"(n0), "=f"(n1), "=f"(n2), "=f"(n3)
          : "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
            "r"(b0),  "r"(b1),
            "f"(d0),  "f"(d1),  "f"(d2),  "f"(d3),
            "r"(sfa0), "h"(bidA), "h"(tidA),
            "r"(sfb0), "h"(bidB), "h"(tidB));
        d0 = n0; d1 = n1; d2 = n2; d3 = n3;
    }

    unsigned int m_row_top = m_warp_base + (lane >> 2);
    unsigned int m_row_bot = m_row_top + 8u;
    unsigned int batch_col0 = batch_warp_base + ((lane & 3u) << 1);
    unsigned int batch_col1 = batch_col0 + 1u;
    if (m_row_top < N) {
        __stcs(out + batch_col0 * N + m_row_top, SYN_TO_OUT(d0));
        __stcs(out + batch_col1 * N + m_row_top, SYN_TO_OUT(d1));
    }
    if (m_row_bot < N) {
        __stcs(out + batch_col0 * N + m_row_bot, SYN_TO_OUT(d2));
        __stcs(out + batch_col1 * N + m_row_bot, SYN_TO_OUT(d3));
    }
}

extern "C" __global__ void nvfp4_mma_gemm_shuf_2d_f16_2x2(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N, unsigned int K,
    unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x)
{
    mma_gemm_shuf_2d_impl<2, 2>(packed_w, scales_w, packed_x, scales_x, out,
                                N, K, sf_inner_dim_w, sf_inner_dim_x);
}

extern "C" __global__ void nvfp4_mma_gemm_shuf_2d_f16_4x2(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N, unsigned int K,
    unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x)
{
    mma_gemm_shuf_2d_impl<4, 2>(packed_w, scales_w, packed_x, scales_x, out,
                                N, K, sf_inner_dim_w, sf_inner_dim_x);
}

extern "C" __global__ void nvfp4_mma_gemm_shuf_2d_f16_4x4(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N, unsigned int K,
    unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x)
{
    mma_gemm_shuf_2d_impl<4, 4>(packed_w, scales_w, packed_x, scales_x, out,
                                N, K, sf_inner_dim_w, sf_inner_dim_x);
}

extern "C" __global__ void nvfp4_mma_gemm_shuf_2d_f16_8x4(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N, unsigned int K,
    unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x)
{
    mma_gemm_shuf_2d_impl<8, 4>(packed_w, scales_w, packed_x, scales_x, out,
                                N, K, sf_inner_dim_w, sf_inner_dim_x);
}

extern "C" __global__ void nvfp4_mma_gemm_shuf_2d_f16_4x8(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N, unsigned int K,
    unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x)
{
    mma_gemm_shuf_2d_impl<4, 8>(packed_w, scales_w, packed_x, scales_x, out,
                                N, K, sf_inner_dim_w, sf_inner_dim_x);
}

__device__ __forceinline__ unsigned int sf_off_in_tile(unsigned int row) {
    unsigned int lo = row & 127u;
    return (lo & 31u) * 16u + (lo >> 5) * 4u;
}

template <unsigned int WARPS_M, unsigned int WARPS_N, unsigned int MU, unsigned int NU>
__device__ __forceinline__ void mma_gemm_shuf_2dr_impl(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N,
    unsigned int K,
    unsigned int sf_inner_dim_w,
    unsigned int sf_inner_dim_x)
{
    constexpr unsigned int BLOCK_M = WARPS_M * MU * 16u;
    constexpr unsigned int BLOCK_N = WARPS_N * NU * 8u;
    (void)BLOCK_M; (void)BLOCK_N;

    unsigned int k_half = K >> 1;
    unsigned int num_chunks = K >> 6;

    unsigned int tid  = threadIdx.x;
    unsigned int warp = tid >> 5;
    unsigned int lane = tid & 31u;

    unsigned int warp_m_idx = warp / WARPS_N;
    unsigned int warp_n_idx = warp % WARPS_N;

    unsigned int base_m = blockIdx.x * (WARPS_M * MU * 16u) + warp_m_idx * (MU * 16u);
    unsigned int base_n = blockIdx.y * (WARPS_N * NU * 8u) + warp_n_idx * (NU * 8u);

    unsigned int m_t = lane & 3u;
    unsigned int n_t = lane >> 2;
    unsigned int k_t = lane >> 2;
    unsigned int s_a = lane & 1u;
    unsigned int s_c = lane >> 2;

    unsigned int top_off = k_t * 32u + m_t * 4u;
    unsigned int bot_off = (k_t + 8u) * 32u + m_t * 4u;
    unsigned int k_lo_off = m_t * 4u;
    unsigned int k_hi_off = k_lo_off + 16u;

    unsigned int w_block_base[MU];
    unsigned int sfa_row_base[MU];
    #pragma unroll
    for (unsigned int mu = 0; mu < MU; mu++) {
        unsigned int m_sub = base_m + mu * 16u;
        w_block_base[mu] = (m_sub >> 4) * num_chunks * 512u;
        unsigned int m_for_sfa = m_sub + s_a * 8u + s_c;
        sfa_row_base[mu] = (m_for_sfa >> 7) * sf_inner_dim_w * 128u + sf_off_in_tile(m_for_sfa);
    }

    unsigned int x_row_off[NU];
    unsigned int sfb_row_base[NU];
    #pragma unroll
    for (unsigned int nu = 0; nu < NU; nu++) {
        unsigned int n_sub = base_n + nu * 8u;
        x_row_off[nu] = (n_sub + n_t) * k_half;
        // ФИКС: sfb batch-row = n_t (lane>>2), совпадает с B-data x_row_off выше. Было (lane&7).
        unsigned int sfb_row_global = n_sub + n_t;
        sfb_row_base[nu] = (sfb_row_global >> 7) * sf_inner_dim_x * 128u + sf_off_in_tile(sfb_row_global);
    }

    float d[MU][NU][4];
    #pragma unroll
    for (unsigned int mu = 0; mu < MU; mu++)
        #pragma unroll
        for (unsigned int nu = 0; nu < NU; nu++) {
            d[mu][nu][0] = 0.f; d[mu][nu][1] = 0.f; d[mu][nu][2] = 0.f; d[mu][nu][3] = 0.f;
        }

    for (unsigned int chunk = 0; chunk < num_chunks; chunk++) {
        unsigned int chunk_off = chunk * 512u;
        unsigned int k_chunk_b = chunk << 5;

        unsigned int a[MU][4];
        unsigned int sfa[MU];
        #pragma unroll
        for (unsigned int mu = 0; mu < MU; mu++) {
            const unsigned char* wp = packed_w + w_block_base[mu] + chunk_off;
            a[mu][0] = *(const unsigned int*)(wp + top_off);
            a[mu][1] = *(const unsigned int*)(wp + bot_off);
            a[mu][2] = *(const unsigned int*)(wp + top_off + 16u);
            a[mu][3] = *(const unsigned int*)(wp + bot_off + 16u);
            sfa[mu] = *(const unsigned int*)(scales_w + sfa_row_base[mu] + chunk_off);
        }
        unsigned int b[NU][2];
        unsigned int sfb[NU];
        #pragma unroll
        for (unsigned int nu = 0; nu < NU; nu++) {
            const unsigned char* xp = packed_x + x_row_off[nu] + k_chunk_b;
            b[nu][0] = *(const unsigned int*)(xp + k_lo_off);
            b[nu][1] = *(const unsigned int*)(xp + k_hi_off);
            sfb[nu] = *(const unsigned int*)(scales_x + sfb_row_base[nu] + chunk_off);
        }

        constexpr unsigned short tidA = 0, bidA = 0, tidB = 0, bidB = 0;
        #pragma unroll
        for (unsigned int mu = 0; mu < MU; mu++) {
            #pragma unroll
            for (unsigned int nu = 0; nu < NU; nu++) {
                float n0, n1, n2, n3;
                asm volatile(
                  "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                  "{%0, %1, %2, %3},"
                  "{%4, %5, %6, %7},"
                  "{%8, %9},"
                  "{%10, %11, %12, %13},"
                  "{%14},"
                  "{%15, %16},"
                  "{%17},"
                  "{%18, %19};\n"
                  : "=f"(n0), "=f"(n1), "=f"(n2), "=f"(n3)
                  : "r"(a[mu][0]), "r"(a[mu][1]), "r"(a[mu][2]), "r"(a[mu][3]),
                    "r"(b[nu][0]), "r"(b[nu][1]),
                    "f"(d[mu][nu][0]), "f"(d[mu][nu][1]), "f"(d[mu][nu][2]), "f"(d[mu][nu][3]),
                    "r"(sfa[mu]), "h"(bidA), "h"(tidA),
                    "r"(sfb[nu]), "h"(bidB), "h"(tidB));
                d[mu][nu][0] = n0; d[mu][nu][1] = n1; d[mu][nu][2] = n2; d[mu][nu][3] = n3;
            }
        }
    }

    #pragma unroll
    for (unsigned int mu = 0; mu < MU; mu++) {
        unsigned int m_row_top = base_m + mu * 16u + (lane >> 2);
        unsigned int m_row_bot = m_row_top + 8u;
        #pragma unroll
        for (unsigned int nu = 0; nu < NU; nu++) {
            unsigned int batch_col0 = base_n + nu * 8u + ((lane & 3u) << 1);
            unsigned int batch_col1 = batch_col0 + 1u;
            if (m_row_top < N) {
                __stcs(out + batch_col0 * N + m_row_top, SYN_TO_OUT(d[mu][nu][0]));
                __stcs(out + batch_col1 * N + m_row_top, SYN_TO_OUT(d[mu][nu][1]));
            }
            if (m_row_bot < N) {
                __stcs(out + batch_col0 * N + m_row_bot, SYN_TO_OUT(d[mu][nu][2]));
                __stcs(out + batch_col1 * N + m_row_bot, SYN_TO_OUT(d[mu][nu][3]));
            }
        }
    }
}

extern "C" __global__ void nvfp4_mma_gemm_shuf_2dr_f16_4x4_m2n2(
    const unsigned char* __restrict__ packed_w, const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x, const unsigned char* __restrict__ scales_x,
    syn_out_t* __restrict__ out, unsigned int N, unsigned int K,
    unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x)
{
    mma_gemm_shuf_2dr_impl<4, 4, 2, 2>(packed_w, scales_w, packed_x, scales_x, out,
                                       N, K, sf_inner_dim_w, sf_inner_dim_x);
}

extern "C" __global__ void nvfp4_mma_gemm_shuf_2dr_f16_4x2_m2n4(
    const unsigned char* __restrict__ packed_w, const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x, const unsigned char* __restrict__ scales_x,
    syn_out_t* __restrict__ out, unsigned int N, unsigned int K,
    unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x)
{
    mma_gemm_shuf_2dr_impl<4, 2, 2, 4>(packed_w, scales_w, packed_x, scales_x, out,
                                       N, K, sf_inner_dim_w, sf_inner_dim_x);
}

extern "C" __global__ void nvfp4_mma_gemm_shuf_2dr_f16_2x2_m2n2(
    const unsigned char* __restrict__ packed_w, const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x, const unsigned char* __restrict__ scales_x,
    syn_out_t* __restrict__ out, unsigned int N, unsigned int K,
    unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x)
{
    mma_gemm_shuf_2dr_impl<2, 2, 2, 2>(packed_w, scales_w, packed_x, scales_x, out,
                                       N, K, sf_inner_dim_w, sf_inner_dim_x);
}

extern "C" __global__ void nvfp4_mma_gemm_shuf_2dr_f16_2x2_m4n4(
    const unsigned char* __restrict__ packed_w, const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x, const unsigned char* __restrict__ scales_x,
    syn_out_t* __restrict__ out, unsigned int N, unsigned int K,
    unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x)
{
    mma_gemm_shuf_2dr_impl<2, 2, 4, 4>(packed_w, scales_w, packed_x, scales_x, out,
                                       N, K, sf_inner_dim_w, sf_inner_dim_x);
}

extern "C" __global__ void nvfp4_mma_gemm_shuf_2dr_f16_4x2_m2n8(
    const unsigned char* __restrict__ packed_w, const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x, const unsigned char* __restrict__ scales_x,
    syn_out_t* __restrict__ out, unsigned int N, unsigned int K,
    unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x)
{
    mma_gemm_shuf_2dr_impl<4, 2, 2, 8>(packed_w, scales_w, packed_x, scales_x, out,
                                       N, K, sf_inner_dim_w, sf_inner_dim_x);
}

extern "C" __global__ void nvfp4_mma_gemm_shuf_2dr_f16_2x2_m4n8(
    const unsigned char* __restrict__ packed_w, const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x, const unsigned char* __restrict__ scales_x,
    syn_out_t* __restrict__ out, unsigned int N, unsigned int K,
    unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x)
{
    mma_gemm_shuf_2dr_impl<2, 2, 4, 8>(packed_w, scales_w, packed_x, scales_x, out,
                                       N, K, sf_inner_dim_w, sf_inner_dim_x);
}









#undef FULL_A
#undef EMPTY_A

__device__ __forceinline__ unsigned int pack_f2h2(float lo, float hi) {
#ifdef SYN_OUT_BF16
    __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
#else
    __half2 h = __floats2half2_rn(lo, hi);
#endif
    return *reinterpret_cast<unsigned int*>(&h);
}

template <unsigned int SWZ>
__device__ __forceinline__ unsigned int swz_tile_off(unsigned int off) {
    // cute Swizzle<2,4,3> для SWIZZLE_64B: XOR бит[5:4] (16B-атом) с бит[8:7].
    if constexpr (SWZ == 64u) return off ^ (((off >> 7) & 3u) << 4u);
    else                      return off;
}

// N=0 → no-op (даёт инстансам без warp-spec-перераспределения чистый ptxas-бюджет:
// присутствие setmaxnreg в ядре заставляет ptxas жать регистры).
template <int N> __device__ __forceinline__ void setmaxnreg_dec() {
    if constexpr (N > 0)
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(N));
}
template <int N> __device__ __forceinline__ void setmaxnreg_inc() {
    if constexpr (N > 0)
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(N));
}

template <unsigned int WARPS_M, unsigned int WARPS_N, unsigned int MU, unsigned int NU,
          unsigned int STAGES, unsigned int KCH, int RDEC, int RINC, unsigned int SWZ = 0u,
          unsigned int PROD_W = 4u, unsigned int ROT = 0u>
__device__ __forceinline__ void matmul_nvfp4_full_device(
    const void* __restrict__ w_desc,
    const void* __restrict__ x_desc,
    const void* __restrict__ sfa_desc,
    const void* __restrict__ sfb_desc,
    syn_out_t* __restrict__ out,
    unsigned int N,
    unsigned int K,
    unsigned int sf_inner_dim_w,
    unsigned int sf_inner_dim_x,
    unsigned int BATCH,
    const void* __restrict__ out_desc)
{
    constexpr unsigned int CONS_WARPS   = WARPS_M * WARPS_N;
    constexpr unsigned int PROD_WARPS   = PROD_W;
    constexpr unsigned int CONS_THREADS = CONS_WARPS * 32u;
    constexpr unsigned int BLOCK_M = WARPS_M * MU * 16u;
    constexpr unsigned int BLOCK_N = WARPS_N * NU * 8u;
    constexpr unsigned int ROWB    = 32u * KCH;
    constexpr unsigned int W_SZ    = BLOCK_M * ROWB;
    constexpr unsigned int X_SZ    = BLOCK_N * ROWB;
    constexpr unsigned int NUM_W_TILES = (BLOCK_M + 127u) / 128u;
    constexpr unsigned int NUM_X_TILES = (BLOCK_N + 127u) / 128u;
    constexpr unsigned int SF_SZ   = KCH * 512u;
    constexpr unsigned int SFA_SZ  = NUM_W_TILES * SF_SZ;
    constexpr unsigned int SFB_SZ  = NUM_X_TILES * SF_SZ;
    constexpr unsigned int TX = W_SZ + X_SZ + SFA_SZ + SFB_SZ;
    constexpr unsigned int SF_BOX_ROWS = KCH * 2u;

    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* sW_base   = smem;
    unsigned char* sX_base   = sW_base + STAGES * W_SZ;
    unsigned char* sSFA_base = sX_base + STAGES * X_SZ;
    unsigned char* sSFB_base = sSFA_base + STAGES * SFA_SZ;
    unsigned long long* full  = (unsigned long long*)(sSFB_base + STAGES * SFB_SZ);
    unsigned long long* empty = full + STAGES;

    unsigned int sw_base_a   = (unsigned int)__cvta_generic_to_shared(sW_base);
    unsigned int sx_base_a   = (unsigned int)__cvta_generic_to_shared(sX_base);
    unsigned int ssfa_base_a = (unsigned int)__cvta_generic_to_shared(sSFA_base);
    unsigned int ssfb_base_a = (unsigned int)__cvta_generic_to_shared(sSFB_base);
    unsigned int full_base_a  = (unsigned int)__cvta_generic_to_shared(full);
    unsigned int empty_base_a = (unsigned int)__cvta_generic_to_shared(empty);
    #define FULL_A(b)  (full_base_a + (b) * 8u)
    #define EMPTY_A(b) (empty_base_a + (b) * 8u)

    unsigned int tid  = threadIdx.x;
    unsigned int warp = tid >> 5;
    unsigned int lane = tid & 31u;
    bool is_prod = warp >= CONS_WARPS;
    unsigned int num_kt = (K >> 6) / KCH;

    unsigned int block_m0 = blockIdx.x * BLOCK_M;
    unsigned int block_n0 = blockIdx.y * BLOCK_N;
    unsigned int sf_row_w = (block_m0 >> 7) * (sf_inner_dim_w >> 1);
    unsigned int sf_row_x = (block_n0 >> 7) * (sf_inner_dim_x >> 1);

    // fused-режим (PROD_WARPS==0): выпуск размазан по 3 варпам (W+SFA / X / SFB,
    // expect_tx суммируется) — TMA-латентность не блокирует один math-варп
    // (SASS-дуэль: у CUTLASS выпуск в выделенной producer-warpgroup).
    constexpr unsigned int FULL_CNT = (PROD_WARPS == 0u) ? 3u : 1u;
    if (tid == 0) {
        #pragma unroll
        for (unsigned int s = 0; s < STAGES; s++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                         :: "r"(FULL_A(s)), "r"(FULL_CNT));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                         :: "r"(EMPTY_A(s)), "r"(CONS_THREADS));
        }
    }
    __syncthreads();

    // Выпуск TMA-чанка g. role: 0 = W+SFA, 1 = X, 2 = SFB (fused-режим);
    // выделенный producer (drot) зовёт все роли одним lane'ом.
    auto issue_chunk = [&](unsigned int g, unsigned int role) {
        unsigned int buf = g % STAGES;
        unsigned int pass = g / STAGES;
        unsigned int fa  = FULL_A(buf);
        if (pass > 0) {
            unsigned int ph = (pass - 1u) & 1u;
            asm volatile(
              "{\n.reg .pred p;\nWEF_%=:\n"
              "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
              "@!p bra WEF_%=;\n}\n" :: "r"(EMPTY_A(buf)), "r"(ph) : "memory");
        }
        unsigned long long st;
        if (role == 0u) {
            unsigned int w_sub = g * (KCH * 2u);
            unsigned int w_rowblk = block_m0 >> 4;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n"
                         : "=l"(st) : "r"(fa), "r"(W_SZ + SFA_SZ));
            asm volatile(
              "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
              " [%0], [%1, {%2, %3, %4}], [%5];\n"
              :: "r"(sw_base_a + buf * W_SZ), "l"((unsigned long long)w_desc),
                 "r"(0u), "r"(w_sub), "r"(w_rowblk), "r"(fa) : "memory");
            #pragma unroll
            for (unsigned int t = 0; t < NUM_W_TILES; t++) {
                unsigned int sfa_row = sf_row_w + t * (sf_inner_dim_w >> 1) + g * SF_BOX_ROWS;
                asm volatile(
                  "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                  " [%0], [%1, {%2, %3}], [%4];\n"
                  :: "r"(ssfa_base_a + buf * SFA_SZ + t * SF_SZ),
                     "l"((unsigned long long)sfa_desc), "r"(0u), "r"(sfa_row), "r"(fa) : "memory");
            }
        } else if (role == 1u) {
            unsigned int kb = g * ROWB;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n"
                         : "=l"(st) : "r"(fa), "r"(X_SZ));
            asm volatile(
              "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
              " [%0], [%1, {%2, %3}], [%4];\n"
              :: "r"(sx_base_a + buf * X_SZ), "l"((unsigned long long)x_desc),
                 "r"(kb), "r"(block_n0), "r"(fa) : "memory");
        } else {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n"
                         : "=l"(st) : "r"(fa), "r"(SFB_SZ));
            #pragma unroll
            for (unsigned int t = 0; t < NUM_X_TILES; t++) {
                unsigned int sfb_row = sf_row_x + t * (sf_inner_dim_x >> 1) + g * SF_BOX_ROWS;
                asm volatile(
                  "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                  " [%0], [%1, {%2, %3}], [%4];\n"
                  :: "r"(ssfb_base_a + buf * SFB_SZ + t * SF_SZ),
                     "l"((unsigned long long)sfb_desc),
                     "r"(0u), "r"(sfb_row), "r"(fa) : "memory");
            }
        }
    };

    if constexpr (PROD_WARPS == 0) {
        // fused: пролог префетча (S-1 чанков) выпускают tid 0/32/64 (свои роли).
        if (tid < 96u && (tid & 31u) == 0u) {
            unsigned int role = tid >> 5;
            unsigned int pre = (STAGES - 1u < num_kt) ? STAGES - 1u : num_kt;
            for (unsigned int g = 0; g < pre; g++)
                issue_chunk(g, role);
        }
    }

    if (is_prod) {
        setmaxnreg_dec<RDEC>();
        if (warp == CONS_WARPS && lane == 0) {
            for (unsigned int c = 0; c < num_kt; c++) {
                unsigned int buf = c % STAGES;
                unsigned int kk  = c / STAGES;
                unsigned int fa  = FULL_A(buf);
                if (kk > 0) {
                    unsigned int ph = (kk - 1u) & 1u;
                    asm volatile(
                      "{\n.reg .pred p;\nWE_%=:\n"
                      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                      "@!p bra WE_%=;\n}\n" :: "r"(EMPTY_A(buf)), "r"(ph) : "memory");
                }
                unsigned int kb = c * ROWB;
                unsigned int w_sub = c * (KCH * 2u);
                unsigned int w_rowblk = block_m0 >> 4;
                unsigned long long st;
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n"
                             : "=l"(st) : "r"(fa), "r"(TX));
                asm volatile(
                  "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                  " [%0], [%1, {%2, %3, %4}], [%5];\n"
                  :: "r"(sw_base_a + buf * W_SZ), "l"((unsigned long long)w_desc),
                     "r"(0u), "r"(w_sub), "r"(w_rowblk), "r"(fa) : "memory");
                asm volatile(
                  "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                  " [%0], [%1, {%2, %3}], [%4];\n"
                  :: "r"(sx_base_a + buf * X_SZ), "l"((unsigned long long)x_desc),
                     "r"(kb), "r"(block_n0), "r"(fa) : "memory");
                #pragma unroll
                for (unsigned int t = 0; t < NUM_W_TILES; t++) {
                    unsigned int sfa_row = sf_row_w + t * (sf_inner_dim_w >> 1) + c * SF_BOX_ROWS;
                    asm volatile(
                      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                      " [%0], [%1, {%2, %3}], [%4];\n"
                      :: "r"(ssfa_base_a + buf * SFA_SZ + t * SF_SZ),
                         "l"((unsigned long long)sfa_desc), "r"(0u), "r"(sfa_row), "r"(fa) : "memory");
                }
                #pragma unroll
                for (unsigned int t = 0; t < NUM_X_TILES; t++) {
                    unsigned int sfb_row = sf_row_x + t * (sf_inner_dim_x >> 1) + c * SF_BOX_ROWS;
                    asm volatile(
                      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                      " [%0], [%1, {%2, %3}], [%4];\n"
                      :: "r"(ssfb_base_a + buf * SFB_SZ + t * SF_SZ),
                         "l"((unsigned long long)sfb_desc),
                         "r"(0u), "r"(sfb_row), "r"(fa) : "memory");
                }
            }
        }
        return;
    }

    setmaxnreg_inc<RINC>();
    unsigned int warp_m_idx = warp / WARPS_N;
    unsigned int warp_n_idx = warp % WARPS_N;
    unsigned int base_m = block_m0 + warp_m_idx * (MU * 16u);
    unsigned int base_n = block_n0 + warp_n_idx * (NU * 8u);
    unsigned int s_a = lane & 1u;
    unsigned int s_c = lane >> 2;
    unsigned int a_lane_off = (lane & 15u) * 32u + (lane >> 4) * 16u;
    unsigned int b_lane_off = (lane & 7u) * ROWB + ((lane & 8u) ? 16u : 0u);

    unsigned int off_in_tile_w[MU];
    #pragma unroll
    for (unsigned int mu = 0; mu < MU; mu++) {
        unsigned int m_for_sfa = base_m + mu * 16u + s_a * 8u + s_c;
        unsigned int tw = (m_for_sfa >> 7) - (block_m0 >> 7);
        off_in_tile_w[mu] = tw * SF_SZ + sf_off_in_tile(m_for_sfa);
    }
    unsigned int off_in_tile_x[NU];
    #pragma unroll
    for (unsigned int nu = 0; nu < NU; nu++) {
        unsigned int sfb_row_global = base_n + nu * 8u + (lane >> 2);
        unsigned int tx = (sfb_row_global >> 7) - (block_n0 >> 7);
        off_in_tile_x[nu] = tx * SF_SZ + sf_off_in_tile(sfb_row_global);
    }

    float d[MU][NU][4];
    #pragma unroll
    for (unsigned int mu = 0; mu < MU; mu++)
        #pragma unroll
        for (unsigned int nu = 0; nu < NU; nu++) {
            d[mu][nu][0] = 0.f; d[mu][nu][1] = 0.f; d[mu][nu][2] = 0.f; d[mu][nu][3] = 0.f;
        }

    if constexpr (ROT != 0) {
        // Порт шедулинга CUTLASS sm120_blockscaled_mma_tma.hpp:814-901 (k64-конвейер):
        // (1) регистровый double-buffer фрагментов — ldsm k64-блока kk+1 выдаётся ДО
        //     mma-пачки блока kk (ldsm-латентность прячется за mma);
        // (2) release стадии сразу после последнего ldsm (на пол-тайла раньше);
        // (3) wait следующей стадии ПЕРЕД последней gemm-пачкой тайла — после
        //     разблокировки у варпа сразу готова регистровая пачка mma.
        // Порядок mma по k не меняется → бит-в-бит с не-ROT.
        static_assert(KCH == 2, "ROT: k64 double-buffer рассчитан на KCH=2");
        static_assert(WARPS_M * MU * 16u <= 128u, "ROT: SF-база рассчитана на 1 W-тайл");
        static_assert(MU == 2 || MU == 4, "ROT: имм-формула SFA");
        static_assert(NU <= 8 && (NU * 8u) % 32u == 0, "ROT: имм-формула SFB");
        // Рег-диета: все smem-адреса от ОДНОЙ базы sw_base_a + constexpr-офсеты
        // (sx/ssfa/ssfb/full/empty-базы мертвы в консьюмере); SF-оффсеты — база+имм
        // (атом (32,4):(16,4) периодичен по mu/nu). Бюджет drot384 = 240.
        constexpr unsigned int RX_OFF   = STAGES * W_SZ;
        constexpr unsigned int RSFA_OFF = STAGES * (W_SZ + X_SZ);
        constexpr unsigned int RSFB_OFF = RSFA_OFF + STAGES * SFA_SZ;
        constexpr unsigned int RBAR_OFF = RSFB_OFF + STAGES * SFB_SZ;
        unsigned int a_base = sw_base_a + warp_m_idx * (MU * KCH * 512u) + a_lane_off;
        unsigned int b_base = sw_base_a + RX_OFF
                            + swz_tile_off<SWZ>(warp_n_idx * (NU * 8u) * ROWB + b_lane_off);
        unsigned int sfa_base = sw_base_a + RSFA_OFF
                              + ((lane & 1u) * 8u + (lane >> 2)) * 16u
                              + ((base_m & 127u) >> 5) * 4u;
        unsigned int sfb_base = sw_base_a + RSFB_OFF
                              + ((warp_n_idx * (NU * 8u)) >> 7) * SF_SZ
                              + (lane >> 2) * 16u + ((base_n & 127u) >> 5) * 4u;
        #define RFULL_A(b)  (sw_base_a + RBAR_OFF + (b) * 8u)
        #define REMPTY_A(b) (sw_base_a + RBAR_OFF + STAGES * 8u + (b) * 8u)
        unsigned int aR[2][MU][4];
        unsigned int bR[2][NU][2];
        unsigned int sfaR[2][MU];
        unsigned int sfbR[2][NU];

        auto load_frag = [&](unsigned int buf, unsigned int kk,
                             unsigned int (&a)[MU][4], unsigned int (&sfa)[MU],
                             unsigned int (&b)[NU][2], unsigned int (&sfb)[NU]) {
            unsigned int a_addr0 = a_base + buf * W_SZ + kk * 512u;
            unsigned int b_addr0 = (b_base + buf * X_SZ) ^ (kk * 32u);
            unsigned int sfa_addr = sfa_base + buf * SFA_SZ + kk * 512u;
            unsigned int sfb_addr = sfb_base + buf * SFB_SZ + kk * 512u;
            #pragma unroll
            for (unsigned int mu = 0; mu < MU; mu++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a[mu][0]), "=r"(a[mu][1]), "=r"(a[mu][2]), "=r"(a[mu][3])
                    : "r"(a_addr0 + mu * (KCH * 512u)));
            }
            // SF-пары смежны по 4Б при warp-страйде 64 (база 8Б-выровнена) → LDS.64.
            if constexpr (MU == 4 && (WARPS_M * MU * 16u) % 64u == 0) {
                asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n"
                    : "=r"(sfa[0]), "=r"(sfa[2]) : "r"(sfa_addr));
                asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n"
                    : "=r"(sfa[1]), "=r"(sfa[3]) : "r"(sfa_addr + 256u));
            } else {
                #pragma unroll
                for (unsigned int mu = 0; mu < MU; mu++)
                    asm volatile("ld.shared.u32 %0, [%1];\n"
                        : "=r"(sfa[mu]) : "r"(sfa_addr + (mu & 1u) * 256u + (mu >> 1) * 4u));
            }
            // B: x4-ldmatrix кроет ДВА nu за инструкцию (лейны 0-15 → nu, 16-31 →
            // nu+1; +8 строк = +512Б — бит 9, swizzle-XOR (бит 8:7→4:5) инвариантен).
            // SASS-дуэль vs CUTLASS: их 16 LDSM/64MMA = ровно x4; наш x2 давал 24.
            if constexpr (NU % 2u == 0) {
                unsigned int b_addr4 = b_addr0 + ((lane & 16u) ? (8u * ROWB) : 0u);
                #pragma unroll
                for (unsigned int nu = 0; nu < NU; nu += 2u) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(b[nu][0]), "=r"(b[nu][1]), "=r"(b[nu + 1][0]), "=r"(b[nu + 1][1])
                        : "r"(b_addr4 + nu * (8u * ROWB)));
                }
            } else {
                #pragma unroll
                for (unsigned int nu = 0; nu < NU; nu++) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];\n"
                        : "=r"(b[nu][0]), "=r"(b[nu][1]) : "r"(b_addr0 + nu * (8u * ROWB)));
                }
            }
            if constexpr (NU == 8 && (WARPS_N * NU * 8u) % 64u == 0) {
                #pragma unroll
                for (unsigned int nu = 0; nu < 4; nu++)
                    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n"
                        : "=r"(sfb[nu]), "=r"(sfb[nu + 4]) : "r"(sfb_addr + nu * 128u));
            } else {
                #pragma unroll
                for (unsigned int nu = 0; nu < NU; nu++)
                    asm volatile("ld.shared.u32 %0, [%1];\n"
                        : "=r"(sfb[nu]) : "r"(sfb_addr + (nu & 3u) * 128u + (nu >> 2) * 4u));
            }
        };
        auto gemm_frag = [&](const unsigned int (&a)[MU][4], const unsigned int (&sfa)[MU],
                             const unsigned int (&b)[NU][2], const unsigned int (&sfb)[NU]) {
            constexpr unsigned short tidA = 0, bidA = 0, tidB = 0, bidB = 0;
            #pragma unroll
            for (unsigned int mu = 0; mu < MU; mu++) {
                #pragma unroll
                for (unsigned int nu = 0; nu < NU; nu++) {
                    // In-out "+f" на аккумулятор: D==C в одних регистрах — без
                    // копий n→d после asm (SASS-дуэль: 47 MOV/64MMA vs 6 у CUTLASS).
                    asm volatile(
                      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                      "{%0, %1, %2, %3},"
                      "{%4, %5, %6, %7},"
                      "{%8, %9},"
                      "{%0, %1, %2, %3},"
                      "{%10},"
                      "{%11, %12},"
                      "{%13},"
                      "{%14, %15};\n"
                      : "+f"(d[mu][nu][0]), "+f"(d[mu][nu][1]), "+f"(d[mu][nu][2]), "+f"(d[mu][nu][3])
                      : "r"(a[mu][0]), "r"(a[mu][1]), "r"(a[mu][2]), "r"(a[mu][3]),
                        "r"(b[nu][0]), "r"(b[nu][1]),
                        "r"(sfa[mu]), "h"(bidA), "h"(tidA),
                        "r"(sfb[nu]), "h"(bidB), "h"(tidB));
                }
            }
        };

        asm volatile(
          "{\n.reg .pred p;\nWFR0_%=:\n"
          "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
          "@!p bra WFR0_%=;\n}\n" :: "r"(RFULL_A(0)), "r"(0u) : "memory");
        load_frag(0u, 0u, aR[0], sfaR[0], bR[0], sfbR[0]);

        auto body = [&](unsigned int cc, unsigned int buf, unsigned int nbuf,
                        unsigned int nph) {
            load_frag(buf, 1u, aR[1], sfaR[1], bR[1], sfbR[1]);
            gemm_frag(aR[0], sfaR[0], bR[0], sfbR[0]);
            unsigned long long st;
            asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
                         : "=l"(st) : "r"(REMPTY_A(buf)) : "memory");
            if constexpr (PROD_WARPS == 0) {
                // fused-issue ПОСЛЕ выпуска gemm-пачки: спины tid 0/32/64 (роли
                // W+SFA / X / SFB) прячутся за уже выданными OMMA своих варпов.
                if (tid < 96u && (tid & 31u) == 0u) {
                    unsigned int g = cc + STAGES - 1u;
                    if (g < num_kt)
                        issue_chunk(g, tid >> 5);
                }
            }
            asm volatile(
              "{\n.reg .pred p;\nWFR_%=:\n"
              "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
              "@!p bra WFR_%=;\n}\n" :: "r"(RFULL_A(nbuf)), "r"(nph) : "memory");
            load_frag(nbuf, 0u, aR[0], sfaR[0], bR[0], sfbR[0]);
            gemm_frag(aR[1], sfaR[1], bR[1], sfbR[1]);
        };
        unsigned int c = 0;
        // Stage-unroll: c в заголовке блока кратно STAGES → buf/nbuf compile-time,
        // smem-адреса складываются в имм-офсеты ldmatrix/lds (SASS-дуэль vs CUTLASS:
        // 82 адресных IADD/IMAD/MOV в теле против их 29 — балласт тела цикла).
        for (; c + STAGES < num_kt;) {
            unsigned int q0 = (c / STAGES) & 1u;
            unsigned int q1 = q0 ^ 1u;
            #pragma unroll
            for (unsigned int s = 0; s < STAGES; s++, c++) {
                body(c, s, (s + 1u == STAGES) ? 0u : s + 1u,
                     (s + 1u == STAGES) ? q1 : q0);
            }
        }
        for (; c + 1u < num_kt; c++) {
            body(c, c % STAGES, (c + 1u) % STAGES, ((c + 1u) / STAGES) & 1u);
        }
        {
            unsigned int buf = c % STAGES;
            load_frag(buf, 1u, aR[1], sfaR[1], bR[1], sfbR[1]);
            gemm_frag(aR[0], sfaR[0], bR[0], sfbR[0]);
            unsigned long long st;
            asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
                         : "=l"(st) : "r"(REMPTY_A(buf)) : "memory");
            gemm_frag(aR[1], sfaR[1], bR[1], sfbR[1]);
        }

        // Субтайл-эпилог TMA-store (рецепт CUTLASS Sm120 epilogue): один bar
        // (стадии дочитаны всеми консьюмерами), варп пишет СВОЙ регион
        // [WROWS батч-строк × WCOLS колонок] f16 row-major в приватный слот
        // stage-smem через stmatrix.x4.trans (фрагмент mma ложится строчно без
        // LDS-обмена), fence.proxy.async → lane0 один TMA-store; OOB-строки
        // клипает дескриптор (BATCH-гард бесплатно). Срез vs 3-фазного
        // (bar+64×STS.u16+bar+LDS+STG): фикс-цена запуска была ~10µs при любом
        // M (K-регрессия), у qutlass ~5µs — эпилог-цепочка главная статья.
        constexpr unsigned int WROWS = NU * 8u;
        constexpr unsigned int WCOLS = MU * 16u;
        constexpr unsigned int WSLOT = WROWS * WCOLS * 2u;
        static_assert(NU % 2u == 0, "ROT: stmatrix.x4 берёт пары nu");
        static_assert(CONS_WARPS * WSLOT <= STAGES * (W_SZ + X_SZ + SFA_SZ + SFB_SZ),
                      "ROT: эпилог-слоты не влезают в smem стадий");
        asm volatile("bar.sync 7, %0;\n" :: "r"(CONS_THREADS) : "memory");
        unsigned int slot = sw_base_a + warp * WSLOT;
        unsigned int oct  = lane >> 3;
        unsigned int srow = lane & 7u;
        #pragma unroll
        for (unsigned int mu = 0; mu < MU; mu++) {
            #pragma unroll
            for (unsigned int nu = 0; nu < NU; nu += 2u) {
                // x4: октеты → матрицы (nu,col-half0),(nu,half1),(nu+1,half0),(nu+1,half1);
                // .trans кладёт mma-колонку (батч-строку) строкой слота.
                unsigned int t_nu   = nu + (oct >> 1);
                unsigned int t_half = oct & 1u;
                // Свизл слота по ширине строки — без него строки фикс-страйда
                // = 8-way банк-конфликт stmatrix (425 vs 438 TF). 128Б-строки:
                // Swizzle<3,4,3> (бит[6:4]^=бит[9:7], дескриптор SWIZZLE_128B);
                // 64Б-строки: Swizzle<2,4,3> (= swz_tile_off<64>, SWIZZLE_64B).
                unsigned int soff = (t_nu * 8u + srow) * (WCOLS * 2u)
                                  + mu * 32u + t_half * 16u;
                if constexpr (WCOLS * 2u == 128u) {
                    soff ^= ((soff >> 7u) & 7u) << 4u;
                } else if constexpr (WCOLS * 2u == 64u) {
                    soff ^= ((soff >> 7u) & 3u) << 4u;
                }
                unsigned int addr = slot + soff;
                unsigned int v0 = pack_f2h2(d[mu][nu][0], d[mu][nu][1]);
                unsigned int v1 = pack_f2h2(d[mu][nu][2], d[mu][nu][3]);
                unsigned int v2 = pack_f2h2(d[mu][nu + 1][0], d[mu][nu + 1][1]);
                unsigned int v3 = pack_f2h2(d[mu][nu + 1][2], d[mu][nu + 1][3]);
                asm volatile(
                    "stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
            }
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        if (lane == 0) {
            unsigned int gx = (block_m0 + warp_m_idx * WCOLS) * 2u;
            unsigned int gy = block_n0 + warp_n_idx * WROWS;
            asm volatile(
              "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
              " [%0, {%1, %2}], [%3];\n"
              :: "l"((unsigned long long)out_desc), "r"(gx), "r"(gy), "r"(slot) : "memory");
            asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
            asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        }
        return;
    } else {

    for (unsigned int c = 0; c < num_kt; c++) {
        if constexpr (PROD_WARPS == 0) {
            // fused-producer: tid 0/32/64 выпускают чанк c+S-1 по ролям (empty-wait
            // внутри issue_chunk ждёт освобождения буфера консьюмерами c-1).
            if (tid < 96u && (tid & 31u) == 0u) {
                unsigned int g = c + STAGES - 1u;
                if (g < num_kt)
                    issue_chunk(g, tid >> 5);
            }
        }
        unsigned int buf = c % STAGES;
        unsigned int ph  = (c / STAGES) & 1u;
        asm volatile(
          "{\n.reg .pred p;\nWF_%=:\n"
          "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
          "@!p bra WF_%=;\n}\n" :: "r"(FULL_A(buf)), "r"(ph) : "memory");

        unsigned int smem_w_addr = sw_base_a + buf * W_SZ;
        unsigned int smem_x_addr = sx_base_a + buf * X_SZ;
        const unsigned char* sfa_smem = sSFA_base + buf * SFA_SZ;
        const unsigned char* sfb_smem = sSFB_base + buf * SFB_SZ;
        constexpr unsigned short tidA = 0, bidA = 0, tidB = 0, bidB = 0;
        #pragma unroll
        for (unsigned int kk = 0; kk < KCH; kk++) {
            unsigned int koff  = kk * 32u;
            unsigned int sf_kk = kk * 512u;
            unsigned int a[MU][4];
            unsigned int sfa[MU];
            #pragma unroll
            for (unsigned int mu = 0; mu < MU; mu++) {
                unsigned int a_in_tile = (warp_m_idx * MU + mu) * (KCH * 512u) + kk * 512u + a_lane_off;
                unsigned int a_addr = smem_w_addr + a_in_tile;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a[mu][0]), "=r"(a[mu][1]), "=r"(a[mu][2]), "=r"(a[mu][3]) : "r"(a_addr));
                sfa[mu] = *(const unsigned int*)(sfa_smem + sf_kk + off_in_tile_w[mu]);
            }
            unsigned int b[NU][2];
            unsigned int sfb[NU];
            #pragma unroll
            for (unsigned int nu = 0; nu < NU; nu++) {
                unsigned int b_in_tile = (warp_n_idx * (NU * 8u) + nu * 8u) * ROWB + b_lane_off + koff;
                unsigned int b_addr = smem_x_addr + swz_tile_off<SWZ>(b_in_tile);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];\n"
                    : "=r"(b[nu][0]), "=r"(b[nu][1]) : "r"(b_addr));
                sfb[nu] = *(const unsigned int*)(sfb_smem + sf_kk + off_in_tile_x[nu]);
            }
            #pragma unroll
            for (unsigned int mu = 0; mu < MU; mu++) {
                #pragma unroll
                for (unsigned int nu = 0; nu < NU; nu++) {
                    float n0, n1, n2, n3;
                    asm volatile(
                      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                      "{%0, %1, %2, %3},"
                      "{%4, %5, %6, %7},"
                      "{%8, %9},"
                      "{%10, %11, %12, %13},"
                      "{%14},"
                      "{%15, %16},"
                      "{%17},"
                      "{%18, %19};\n"
                      : "=f"(n0), "=f"(n1), "=f"(n2), "=f"(n3)
                      : "r"(a[mu][0]), "r"(a[mu][1]), "r"(a[mu][2]), "r"(a[mu][3]),
                        "r"(b[nu][0]), "r"(b[nu][1]),
                        "f"(d[mu][nu][0]), "f"(d[mu][nu][1]), "f"(d[mu][nu][2]), "f"(d[mu][nu][3]),
                        "r"(sfa[mu]), "h"(bidA), "h"(tidA),
                        "r"(sfb[nu]), "h"(bidB), "h"(tidB));
                    d[mu][nu][0] = n0; d[mu][nu][1] = n1; d[mu][nu][2] = n2; d[mu][nu][3] = n3;
                }
            }
        }
        unsigned long long st;
        asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
                     : "=l"(st) : "r"(EMPTY_A(buf)) : "memory");
    }
    }

    #pragma unroll
    for (unsigned int mu = 0; mu < MU; mu++) {
        unsigned int m_row_top = base_m + mu * 16u + (lane >> 2);
        unsigned int m_row_bot = m_row_top + 8u;
        #pragma unroll
        for (unsigned int nu = 0; nu < NU; nu++) {
            unsigned int batch_col0 = base_n + nu * 8u + ((lane & 3u) << 1);
            unsigned int batch_col1 = batch_col0 + 1u;
            bool b0 = batch_col0 < BATCH, b1 = batch_col1 < BATCH;
            if (m_row_top < N) {
                if (b0) __stcs(out + batch_col0 * N + m_row_top, SYN_TO_OUT(d[mu][nu][0]));
                if (b1) __stcs(out + batch_col1 * N + m_row_top, SYN_TO_OUT(d[mu][nu][1]));
            }
            if (m_row_bot < N) {
                if (b0) __stcs(out + batch_col0 * N + m_row_bot, SYN_TO_OUT(d[mu][nu][2]));
                if (b1) __stcs(out + batch_col1 * N + m_row_bot, SYN_TO_OUT(d[mu][nu][3]));
            }
        }
    }
}
#undef FULL_A
#undef EMPTY_A

template <unsigned int WARPS_M, unsigned int WARPS_N, unsigned int MU, unsigned int NU,
          unsigned int STAGES, unsigned int KCH, int RDEC, int RINC, unsigned int SWZ = 0u>
__device__ __forceinline__ void matmul_nvfp4_full_persistent_device(
    const void* __restrict__ w_desc,
    const void* __restrict__ x_desc,
    const void* __restrict__ sfa_desc,
    const void* __restrict__ sfb_desc,
    syn_out_t* __restrict__ out,
    unsigned int N,
    unsigned int K,
    unsigned int batch,
    unsigned int sf_inner_dim_w,
    unsigned int sf_inner_dim_x)
{
    constexpr unsigned int CONS_WARPS   = WARPS_M * WARPS_N;
    constexpr unsigned int CONS_THREADS = CONS_WARPS * 32u;
    constexpr unsigned int BLOCK_M = WARPS_M * MU * 16u;
    constexpr unsigned int BLOCK_N = WARPS_N * NU * 8u;
    constexpr unsigned int ROWB    = 32u * KCH;
    constexpr unsigned int W_SZ    = BLOCK_M * ROWB;
    constexpr unsigned int X_SZ    = BLOCK_N * ROWB;
    constexpr unsigned int NUM_W_TILES = (BLOCK_M + 127u) / 128u;
    constexpr unsigned int NUM_X_TILES = (BLOCK_N + 127u) / 128u;
    constexpr unsigned int SF_SZ   = KCH * 512u;
    constexpr unsigned int SFA_SZ  = NUM_W_TILES * SF_SZ;
    constexpr unsigned int SFB_SZ  = NUM_X_TILES * SF_SZ;
    constexpr unsigned int TX = W_SZ + X_SZ + SFA_SZ + SFB_SZ;
    constexpr unsigned int SF_BOX_ROWS = KCH * 2u;

    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* sW_base   = smem;
    unsigned char* sX_base   = sW_base + STAGES * W_SZ;
    unsigned char* sSFA_base = sX_base + STAGES * X_SZ;
    unsigned char* sSFB_base = sSFA_base + STAGES * SFA_SZ;
    unsigned long long* full  = (unsigned long long*)(sSFB_base + STAGES * SFB_SZ);
    unsigned long long* empty = full + STAGES;

    unsigned int sw_base_a   = (unsigned int)__cvta_generic_to_shared(sW_base);
    unsigned int sx_base_a   = (unsigned int)__cvta_generic_to_shared(sX_base);
    unsigned int ssfa_base_a = (unsigned int)__cvta_generic_to_shared(sSFA_base);
    unsigned int ssfb_base_a = (unsigned int)__cvta_generic_to_shared(sSFB_base);
    unsigned int full_base_a  = (unsigned int)__cvta_generic_to_shared(full);
    unsigned int empty_base_a = (unsigned int)__cvta_generic_to_shared(empty);
    #define FULL_A(b)  (full_base_a + (b) * 8u)
    #define EMPTY_A(b) (empty_base_a + (b) * 8u)

    unsigned int tid  = threadIdx.x;
    unsigned int warp = tid >> 5;
    unsigned int lane = tid & 31u;
    bool is_prod = warp >= CONS_WARPS;
    unsigned int num_kt = (K >> 6) / KCH;

    unsigned int num_tiles_m = N / BLOCK_M;
    unsigned int num_tiles_n = batch / BLOCK_N;
    unsigned int total_tiles = num_tiles_m * num_tiles_n;

    if (tid == 0) {
        #pragma unroll
        for (unsigned int s = 0; s < STAGES; s++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(FULL_A(s)));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                         :: "r"(EMPTY_A(s)), "r"(CONS_THREADS));
        }
    }
    __syncthreads();

    if (is_prod) {
        setmaxnreg_dec<RDEC>();
        if (warp == CONS_WARPS && lane == 0) {

            unsigned int gc = 0;
            for (unsigned int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {

                unsigned int tile_n = tile % num_tiles_n;
                unsigned int tile_m = tile / num_tiles_n;
                unsigned int block_m0 = tile_m * BLOCK_M;
                unsigned int block_n0 = tile_n * BLOCK_N;
                unsigned int sf_row_w = (block_m0 >> 7) * (sf_inner_dim_w >> 1);
                unsigned int sf_row_x = (block_n0 >> 7) * (sf_inner_dim_x >> 1);
                for (unsigned int c = 0; c < num_kt; c++, gc++) {
                    unsigned int buf = gc % STAGES;
                    unsigned int pass = gc / STAGES;
                    unsigned int fa  = FULL_A(buf);
                    if (pass > 0) {
                        unsigned int ph = (pass - 1u) & 1u;
                        asm volatile(
                          "{\n.reg .pred p;\nWE_%=:\n"
                          "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                          "@!p bra WE_%=;\n}\n" :: "r"(EMPTY_A(buf)), "r"(ph) : "memory");
                    }
                    unsigned int kb = c * ROWB;
                    unsigned int w_sub = c * (KCH * 2u);
                    unsigned int w_rowblk = block_m0 >> 4;
                    unsigned long long st;
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n"
                                 : "=l"(st) : "r"(fa), "r"(TX));
                    asm volatile(
                      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                      " [%0], [%1, {%2, %3, %4}], [%5];\n"
                      :: "r"(sw_base_a + buf * W_SZ), "l"((unsigned long long)w_desc),
                         "r"(0u), "r"(w_sub), "r"(w_rowblk), "r"(fa) : "memory");
                    asm volatile(
                      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                      " [%0], [%1, {%2, %3}], [%4];\n"
                      :: "r"(sx_base_a + buf * X_SZ), "l"((unsigned long long)x_desc),
                         "r"(kb), "r"(block_n0), "r"(fa) : "memory");
                    #pragma unroll
                    for (unsigned int t = 0; t < NUM_W_TILES; t++) {
                        unsigned int sfa_row = sf_row_w + t * (sf_inner_dim_w >> 1) + c * SF_BOX_ROWS;
                        asm volatile(
                          "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                          " [%0], [%1, {%2, %3}], [%4];\n"
                          :: "r"(ssfa_base_a + buf * SFA_SZ + t * SF_SZ),
                             "l"((unsigned long long)sfa_desc), "r"(0u), "r"(sfa_row), "r"(fa) : "memory");
                    }
                    #pragma unroll
                    for (unsigned int t = 0; t < NUM_X_TILES; t++) {
                        unsigned int sfb_row = sf_row_x + t * (sf_inner_dim_x >> 1) + c * SF_BOX_ROWS;
                        asm volatile(
                          "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                          " [%0], [%1, {%2, %3}], [%4];\n"
                          :: "r"(ssfb_base_a + buf * SFB_SZ + t * SF_SZ),
                             "l"((unsigned long long)sfb_desc),
                             "r"(0u), "r"(sfb_row), "r"(fa) : "memory");
                    }
                }
            }
        }
        return;
    }

    setmaxnreg_inc<RINC>();
    unsigned int warp_m_idx = warp / WARPS_N;
    unsigned int warp_n_idx = warp % WARPS_N;
    unsigned int s_a = lane & 1u;
    unsigned int s_c = lane >> 2;
    unsigned int a_lane_off = (lane & 15u) * 32u + (lane >> 4) * 16u;
    unsigned int b_lane_off = (lane & 7u) * ROWB + ((lane & 8u) ? 16u : 0u);

    unsigned int gc = 0;
    for (unsigned int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        unsigned int tile_n = tile % num_tiles_n;
        unsigned int tile_m = tile / num_tiles_n;
        unsigned int block_m0 = tile_m * BLOCK_M;
        unsigned int block_n0 = tile_n * BLOCK_N;
        unsigned int base_m = block_m0 + warp_m_idx * (MU * 16u);
        unsigned int base_n = block_n0 + warp_n_idx * (NU * 8u);

        unsigned int off_in_tile_w[MU];
        #pragma unroll
        for (unsigned int mu = 0; mu < MU; mu++) {
            unsigned int m_for_sfa = base_m + mu * 16u + s_a * 8u + s_c;
            unsigned int tw = (m_for_sfa >> 7) - (block_m0 >> 7);
            off_in_tile_w[mu] = tw * SF_SZ + sf_off_in_tile(m_for_sfa);
        }
        unsigned int off_in_tile_x[NU];
        #pragma unroll
        for (unsigned int nu = 0; nu < NU; nu++) {
            // ldmatrix грузит B по b_lane_off=lane&7, но MMA B-фрагмент n = lane>>2 → sfb=lane>>2.
            unsigned int sfb_row_global = base_n + nu * 8u + (lane >> 2);
            unsigned int tx = (sfb_row_global >> 7) - (block_n0 >> 7);
            off_in_tile_x[nu] = tx * SF_SZ + sf_off_in_tile(sfb_row_global);
        }

        float d[MU][NU][4];
        #pragma unroll
        for (unsigned int mu = 0; mu < MU; mu++)
            #pragma unroll
            for (unsigned int nu = 0; nu < NU; nu++) {
                d[mu][nu][0] = 0.f; d[mu][nu][1] = 0.f; d[mu][nu][2] = 0.f; d[mu][nu][3] = 0.f;
            }

        for (unsigned int c = 0; c < num_kt; c++, gc++) {
            unsigned int buf = gc % STAGES;
            unsigned int ph  = (gc / STAGES) & 1u;
            asm volatile(
              "{\n.reg .pred p;\nWF_%=:\n"
              "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
              "@!p bra WF_%=;\n}\n" :: "r"(FULL_A(buf)), "r"(ph) : "memory");

            unsigned int smem_w_addr = sw_base_a + buf * W_SZ;
            unsigned int smem_x_addr = sx_base_a + buf * X_SZ;
            const unsigned char* sfa_smem = sSFA_base + buf * SFA_SZ;
            const unsigned char* sfb_smem = sSFB_base + buf * SFB_SZ;
            constexpr unsigned short tidA = 0, bidA = 0, tidB = 0, bidB = 0;
            #pragma unroll
            for (unsigned int kk = 0; kk < KCH; kk++) {
                unsigned int koff  = kk * 32u;
                unsigned int sf_kk = kk * 512u;
                unsigned int a[MU][4];
                unsigned int sfa[MU];
                #pragma unroll
                for (unsigned int mu = 0; mu < MU; mu++) {
                    unsigned int a_in_tile = (warp_m_idx * MU + mu) * (KCH * 512u) + kk * 512u + a_lane_off;
                    unsigned int a_addr = smem_w_addr + a_in_tile;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(a[mu][0]), "=r"(a[mu][1]), "=r"(a[mu][2]), "=r"(a[mu][3]) : "r"(a_addr));
                    sfa[mu] = *(const unsigned int*)(sfa_smem + sf_kk + off_in_tile_w[mu]);
                }
                unsigned int b[NU][2];
                unsigned int sfb[NU];
                #pragma unroll
                for (unsigned int nu = 0; nu < NU; nu++) {
                    unsigned int b_in_tile = (warp_n_idx * (NU * 8u) + nu * 8u) * ROWB + b_lane_off + koff;
                    unsigned int b_addr = smem_x_addr + swz_tile_off<SWZ>(b_in_tile);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];\n"
                        : "=r"(b[nu][0]), "=r"(b[nu][1]) : "r"(b_addr));
                    sfb[nu] = *(const unsigned int*)(sfb_smem + sf_kk + off_in_tile_x[nu]);
                }
                #pragma unroll
                for (unsigned int mu = 0; mu < MU; mu++) {
                    #pragma unroll
                    for (unsigned int nu = 0; nu < NU; nu++) {
                        float n0, n1, n2, n3;
                        asm volatile(
                          "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                          "{%0, %1, %2, %3},"
                          "{%4, %5, %6, %7},"
                          "{%8, %9},"
                          "{%10, %11, %12, %13},"
                          "{%14},"
                          "{%15, %16},"
                          "{%17},"
                          "{%18, %19};\n"
                          : "=f"(n0), "=f"(n1), "=f"(n2), "=f"(n3)
                          : "r"(a[mu][0]), "r"(a[mu][1]), "r"(a[mu][2]), "r"(a[mu][3]),
                            "r"(b[nu][0]), "r"(b[nu][1]),
                            "f"(d[mu][nu][0]), "f"(d[mu][nu][1]), "f"(d[mu][nu][2]), "f"(d[mu][nu][3]),
                            "r"(sfa[mu]), "h"(bidA), "h"(tidA),
                            "r"(sfb[nu]), "h"(bidB), "h"(tidB));
                        d[mu][nu][0] = n0; d[mu][nu][1] = n1; d[mu][nu][2] = n2; d[mu][nu][3] = n3;
                    }
                }
            }
            unsigned long long st;
            asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
                         : "=l"(st) : "r"(EMPTY_A(buf)) : "memory");
        }

        #pragma unroll
        for (unsigned int mu = 0; mu < MU; mu++) {
            unsigned int m_row_top = base_m + mu * 16u + (lane >> 2);
            unsigned int m_row_bot = m_row_top + 8u;
            #pragma unroll
            for (unsigned int nu = 0; nu < NU; nu++) {
                unsigned int batch_col0 = base_n + nu * 8u + ((lane & 3u) << 1);
                unsigned int batch_col1 = batch_col0 + 1u;
                if (m_row_top < N) {
                    __stcs(out + batch_col0 * N + m_row_top, SYN_TO_OUT(d[mu][nu][0]));
                    __stcs(out + batch_col1 * N + m_row_top, SYN_TO_OUT(d[mu][nu][1]));
                }
                if (m_row_bot < N) {
                    __stcs(out + batch_col0 * N + m_row_bot, SYN_TO_OUT(d[mu][nu][2]));
                    __stcs(out + batch_col1 * N + m_row_bot, SYN_TO_OUT(d[mu][nu][3]));
                }
            }
        }
    }
}
#undef FULL_A
#undef EMPTY_A

#define GN_NVFP4_FULL(NAME, WM, WN, MU, NU, ST, KCH, RDEC, RINC)                            \
    extern "C" __global__ __launch_bounds__((WM * WN + 4) * 32u, 1) void NAME(              \
        const void* w_desc, const void* x_desc, const void* sfa_desc, const void* sfb_desc, \
        syn_out_t* out, unsigned int N, unsigned int K,                                        \
        unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x, unsigned int BATCH,  \
        const void* out_desc) {                                                             \
        matmul_nvfp4_full_device<WM, WN, MU, NU, ST, KCH, RDEC, RINC, 0u>(                  \
            w_desc, x_desc, sfa_desc, sfb_desc, out, N, K, sf_inner_dim_w, sf_inner_dim_x,  \
            BATCH, out_desc);                                                                         \
    }

#define GN_NVFP4_FULL_SWZ(NAME, WM, WN, MU, NU, ST, KCH, RDEC, RINC, SWZ)                   \
    extern "C" __global__ __launch_bounds__((WM * WN + 4) * 32u, 1) void NAME(              \
        const void* w_desc, const void* x_desc, const void* sfa_desc, const void* sfb_desc, \
        syn_out_t* out, unsigned int N, unsigned int K,                                        \
        unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x, unsigned int BATCH,  \
        const void* out_desc) {                                                             \
        matmul_nvfp4_full_device<WM, WN, MU, NU, ST, KCH, RDEC, RINC, SWZ>(                 \
            w_desc, x_desc, sfa_desc, sfb_desc, out, N, K, sf_inner_dim_w, sf_inner_dim_x,  \
            BATCH, out_desc);                                                                         \
    }

GN_NVFP4_FULL(gn_nvfp4_full_128x128_s2, 2, 2, 4, 8, 2, 2, 24, 240)
GN_NVFP4_FULL(gn_nvfp4_full_128x128_c256_s4, 4, 2, 2, 8, 4, 2, 40, 232)

GN_NVFP4_FULL_SWZ(gn_nvfp4_full_128x128_c256_s4_swz, 4, 2, 2, 8, 4, 2, 40, 232, 64)
GN_NVFP4_FULL_SWZ(gn_nvfp4_full_128x64_s3_swz, 4, 2, 2, 4, 3, 2, 40, 232, 64)
GN_NVFP4_FULL_SWZ(gn_nvfp4_full_128x64_s4_swz, 4, 2, 2, 4, 4, 2, 40, 232, 64)
GN_NVFP4_FULL_SWZ(gn_nvfp4_full_128x128_c256_s3_swz, 4, 2, 2, 8, 3, 2, 40, 232, 64)

// Большие тайлы (урок bf16-b256): меньше L2-перечиток операндов на FLOP —
// attn/26520 у 128×128 упирался в L2 (66.5% на базовом клоке → сатурация на бусте).
// 128×256 (batch-256): W-трафик ×0.5; 256×128 (features-256): A-трафик ×0.5.
// P1: один producer-варп вместо 4 (3 были балластом — setmaxnreg на sm_120 не
// перераспределяет) → 288 потоков → ptxas-бюджет 227 рег → d[4][8][4]=128 не спиллит.
// FUSED: без выделенных producer-варпов (TMA выпускает tid 0 в консьюмер-цикле) —
// 256 потоков ровно → ptxas-бюджет 256 рег (квант 128 потоков резал 288→168+спиллы).
#define GN_NVFP4_FULL_SWZ_FUSED(NAME, WM, WN, MU, NU, ST, KCH, SWZ)                         \
    extern "C" __global__ __launch_bounds__(WM * WN * 32u, 1) void NAME(                    \
        const void* w_desc, const void* x_desc, const void* sfa_desc, const void* sfb_desc, \
        syn_out_t* out, unsigned int N, unsigned int K,                                        \
        unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x, unsigned int BATCH,  \
        const void* out_desc) {                                                             \
        matmul_nvfp4_full_device<WM, WN, MU, NU, ST, KCH, 0, 0, SWZ, 0u>(                   \
            w_desc, x_desc, sfa_desc, sfb_desc, out, N, K, sf_inner_dim_w, sf_inner_dim_x,  \
            BATCH, out_desc);                                                                         \
    }

GN_NVFP4_FULL_SWZ_FUSED(gn_nvfp4_full_128x256_s3_swz, 2, 4, 4, 8, 3, 2, 64)
GN_NVFP4_FULL_SWZ(gn_nvfp4_full_256x128_s3_swz, 4, 2, 4, 8, 3, 2, 40, 232, 64)

// ROT=1: k64-конвейер по схеме CUTLASS (double-buffer фрагментов + ранний release +
// wait перед последней gemm-пачкой). _rot = fused-producer, _drot = 4 producer-варпа.
#define GN_NVFP4_FULL_SWZ_FUSED_ROT(NAME, WM, WN, MU, NU, ST, KCH, SWZ)                     \
    extern "C" __global__ __launch_bounds__(WM * WN * 32u, 1) void NAME(                    \
        const void* w_desc, const void* x_desc, const void* sfa_desc, const void* sfb_desc, \
        syn_out_t* out, unsigned int N, unsigned int K,                                        \
        unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x, unsigned int BATCH,  \
        const void* out_desc) {                                                             \
        matmul_nvfp4_full_device<WM, WN, MU, NU, ST, KCH, 0, 0, SWZ, 0u, 1u>(               \
            w_desc, x_desc, sfa_desc, sfb_desc, out, N, K, sf_inner_dim_w, sf_inner_dim_x,  \
            BATCH, out_desc);                                                                         \
    }

#define GN_NVFP4_FULL_SWZ_DROT(NAME, WM, WN, MU, NU, ST, KCH, RDEC, RINC, SWZ)              \
    extern "C" __global__ __launch_bounds__((WM * WN + 4) * 32u, 1) void NAME(              \
        const void* w_desc, const void* x_desc, const void* sfa_desc, const void* sfb_desc, \
        syn_out_t* out, unsigned int N, unsigned int K,                                        \
        unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x, unsigned int BATCH,  \
        const void* out_desc) {                                                             \
        matmul_nvfp4_full_device<WM, WN, MU, NU, ST, KCH, RDEC, RINC, SWZ, 4u, 1u>(         \
            w_desc, x_desc, sfa_desc, sfb_desc, out, N, K, sf_inner_dim_w, sf_inner_dim_x,  \
            BATCH, out_desc);                                                                         \
    }

GN_NVFP4_FULL_SWZ_FUSED_ROT(gn_nvfp4_full_128x256_s3_swz_rot, 2, 4, 4, 8, 3, 2, 64)
GN_NVFP4_FULL_SWZ_DROT(gn_nvfp4_full_128x128_c256_s4_swz_drot, 4, 2, 2, 8, 4, 2, 40, 232, 64)
// Структура qutlass: выделенный producer-warpgroup (384 потока), per-role регистры —
// РАБОТАЮТ после ::cta-фикса TMA (::cluster глушил setmaxnreg, ptxas C7506).
// 240/32 = пул РОВНО (256·240+128·32=65536) → флаки-дедлок setmaxnreg.inc
// (тот же класс, что mxfp8 RDEC=32: пул 65536 ровно виснет); RDEC=24 спиллил
// ПРОДЬЮСЕРА (local_ld 240/поток → TMA-выпуск тормозил конвейер).
// Лайвлок-матрица (e2e-репродюсер, prof 250 запусков может молчать):
// 32/240 ВИСНЕТ, 32/232 ВИСНЕТ (пул 63488 — арифметика пула НЕ корень),
// 24/240 стабилен но продьюсер спиллит (339.7 e2e < rot 351.6),
// 40/232 стабилен (= пара рабочего c256_s4_drot): RDEC=32 — проклятое
// значение на sm_120 (лайвлок setmaxnreg-аллокатора; gdb attach выталкивает).
GN_NVFP4_FULL_SWZ_DROT(gn_nvfp4_full_128x256_s3_swz_drot, 2, 4, 4, 8, 3, 2, 40, 232, 64)
#define GN_NVFP4_FULL_PERSIST_SWZ(NAME, WM, WN, MU, NU, ST, KCH, RDEC, RINC, SWZ)            \
    extern "C" __global__ __launch_bounds__((WM * WN + 4) * 32u, 1) void NAME(              \
        const void* w_desc, const void* x_desc, const void* sfa_desc, const void* sfb_desc, \
        syn_out_t* out, unsigned int N, unsigned int K, unsigned int batch,                    \
        unsigned int sf_inner_dim_w, unsigned int sf_inner_dim_x) {                         \
        matmul_nvfp4_full_persistent_device<WM, WN, MU, NU, ST, KCH, RDEC, RINC, SWZ>(      \
            w_desc, x_desc, sfa_desc, sfb_desc, out, N, K, batch,                           \
            sf_inner_dim_w, sf_inner_dim_x);                                                \
    }

GN_NVFP4_FULL_PERSIST_SWZ(gn_nvfp4_full_persist_c256_s4_swz, 4, 2, 2, 8, 4, 2, 40, 232, 64)
GN_NVFP4_FULL_PERSIST_SWZ(gn_nvfp4_full_persist_c256_s3_swz, 4, 2, 2, 8, 3, 2, 40, 232, 64)
