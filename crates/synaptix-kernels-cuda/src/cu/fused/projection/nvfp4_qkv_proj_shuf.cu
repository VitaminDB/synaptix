#include <cuda_fp16.h>

// Fused NVFP4 GEMV для Q/K/V одной триадой launch'ей в одном kernel.
//
// Идея: на decode (M=1) Q/K/V проекции читают один и тот же X. Если делать 3
// отдельных GEMV-launch'а, X читается из gmem 3 раза. Здесь X грузится в smem
// один раз, и используется тремя последовательными mma-секциями. K у всех трёх
// одинаковый (hidden_size), sf_inner_dim_w тоже одинаковый.
//
// Block size: WARPS=4 → M_TILE=64 threads=128 (один из вариантов backbone).
// Grid X = ceil(max(N_q, N_k, N_v) / M_TILE).
// Каждый warp проверяет m_warp_base < N_{q,k,v} → skip если не попал.

template <unsigned int WARPS>
__device__ __forceinline__ void mma_qkv_section(
    const unsigned char* __restrict__ packed_w,   // shuffled (N/16, K/64, 16, 32)
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ smem_x,     // X в smem
    const unsigned char* __restrict__ scales_x,
    __half*              __restrict__ out,
    unsigned int N,
    unsigned int K,
    unsigned int sf_inner_dim_w,
    unsigned int warp,
    unsigned int lane,
    unsigned int tile_base)
{
    unsigned int m_warp_base = tile_base + warp * 16u;
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

    unsigned int k_lo_off = m_t * 4u;
    unsigned int k_hi_off = k_lo_off + 16u;

    unsigned int tile_row_w   = m_for_sfa >> 7;
    unsigned int local_outer  = m_for_sfa & 127u;
    unsigned int off_in_tile  = (local_outer & 31u) * 16u + (local_outer >> 5) * 4u;
    unsigned int sfa_row_base = tile_row_w * sf_inner_dim_w * 128u + off_in_tile;

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
        unsigned int sfb0 = *(const unsigned int*)(scales_x + chunk * 512u);

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
        if (m_top_g < N) out[m_top_g] = __float2half(d0);
        if (m_bot_g < N) out[m_bot_g] = __float2half(d2);
    }
}

template <unsigned int WARPS>
__device__ __forceinline__ void qkv_proj_shuf_impl(
    const unsigned char* __restrict__ packed_w_q,
    const unsigned char* __restrict__ scales_w_q,
    const unsigned char* __restrict__ packed_w_k,
    const unsigned char* __restrict__ scales_w_k,
    const unsigned char* __restrict__ packed_w_v,
    const unsigned char* __restrict__ scales_w_v,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    __half*              __restrict__ out_q,
    __half*              __restrict__ out_k,
    __half*              __restrict__ out_v,
    unsigned int N_q,
    unsigned int N_k,
    unsigned int N_v,
    unsigned int K,
    unsigned int sf_inner_dim_w)
{
    constexpr unsigned int M_TILE = WARPS * 16;
    constexpr unsigned int THREADS = WARPS * 32;

    extern __shared__ unsigned char smem[];
    unsigned char* smem_x = smem;

    unsigned int tid  = threadIdx.x;
    unsigned int warp = tid >> 5;
    unsigned int lane = tid & 31u;

    unsigned int x_u32_count = (K >> 1) >> 2;
    unsigned int*       smem_x_u32 = (unsigned int*)smem_x;
    const unsigned int* gmem_x_u32 = (const unsigned int*)packed_x;
    for (unsigned int i = tid; i < x_u32_count; i += THREADS) {
        smem_x_u32[i] = gmem_x_u32[i];
    }
    __syncthreads();

    unsigned int tile_base = blockIdx.x * M_TILE;

    mma_qkv_section<WARPS>(packed_w_q, scales_w_q, smem_x, scales_x, out_q,
                           N_q, K, sf_inner_dim_w, warp, lane, tile_base);
    mma_qkv_section<WARPS>(packed_w_k, scales_w_k, smem_x, scales_x, out_k,
                           N_k, K, sf_inner_dim_w, warp, lane, tile_base);
    mma_qkv_section<WARPS>(packed_w_v, scales_w_v, smem_x, scales_x, out_v,
                           N_v, K, sf_inner_dim_w, warp, lane, tile_base);
}

extern "C" __global__ void nvfp4_qkv_proj_shuf_f16_w4(
    const unsigned char* __restrict__ packed_w_q,
    const unsigned char* __restrict__ scales_w_q,
    const unsigned char* __restrict__ packed_w_k,
    const unsigned char* __restrict__ scales_w_k,
    const unsigned char* __restrict__ packed_w_v,
    const unsigned char* __restrict__ scales_w_v,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    __half*              __restrict__ out_q,
    __half*              __restrict__ out_k,
    __half*              __restrict__ out_v,
    unsigned int N_q,
    unsigned int N_k,
    unsigned int N_v,
    unsigned int K,
    unsigned int sf_inner_dim_w)
{
    qkv_proj_shuf_impl<4>(packed_w_q, scales_w_q, packed_w_k, scales_w_k,
                          packed_w_v, scales_w_v, packed_x, scales_x,
                          out_q, out_k, out_v, N_q, N_k, N_v, K, sf_inner_dim_w);
}

extern "C" __global__ void nvfp4_qkv_proj_shuf_f16_w8(
    const unsigned char* __restrict__ packed_w_q,
    const unsigned char* __restrict__ scales_w_q,
    const unsigned char* __restrict__ packed_w_k,
    const unsigned char* __restrict__ scales_w_k,
    const unsigned char* __restrict__ packed_w_v,
    const unsigned char* __restrict__ scales_w_v,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    __half*              __restrict__ out_q,
    __half*              __restrict__ out_k,
    __half*              __restrict__ out_v,
    unsigned int N_q,
    unsigned int N_k,
    unsigned int N_v,
    unsigned int K,
    unsigned int sf_inner_dim_w)
{
    qkv_proj_shuf_impl<8>(packed_w_q, scales_w_q, packed_w_k, scales_w_k,
                          packed_w_v, scales_w_v, packed_x, scales_x,
                          out_q, out_k, out_v, N_q, N_k, N_v, K, sf_inner_dim_w);
}
