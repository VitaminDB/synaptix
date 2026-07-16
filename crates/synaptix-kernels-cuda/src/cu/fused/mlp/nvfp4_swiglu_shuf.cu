#include <cuda_fp16.h>

// Fused NVFP4 SwiGLU FFN на shuffled W layout.
//
// Computes: out = silu(W_gate @ x) * (W_up @ x)
// W_gate, W_up — (N, K) packed NVFP4 в pre-shuffled layout (см. nvfp4_mma_gemv_shuf.cu).
// X — (K,) packed NVFP4. K общий, N общий (intermediate_size).
// Block обрабатывает один M_TILE rows: считает gate-row и up-row на тех же
// X-warp-fragments, перемножает silu(gate)*up и пишет финальный результат.

__device__ __forceinline__ float silu(float v) {
    return v / (1.f + __expf(-v));
}

template <unsigned int WARPS>
__device__ __forceinline__ void mma_tile_acc(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ smem_x,
    const unsigned char* __restrict__ scales_x,
    unsigned int K,
    unsigned int sf_inner_dim_w,
    unsigned int warp,
    unsigned int lane,
    unsigned int m_warp_base,
    float &d0_out,
    float &d2_out)
{
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
    d0_out = d0;
    d2_out = d2;
}

template <unsigned int WARPS>
__device__ __forceinline__ void swiglu_shuf_impl(
    const unsigned char* __restrict__ packed_w_gate,
    const unsigned char* __restrict__ scales_w_gate,
    const unsigned char* __restrict__ packed_w_up,
    const unsigned char* __restrict__ scales_w_up,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    __half*              __restrict__ out,
    unsigned int N,
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

    unsigned int m_warp_base = blockIdx.x * M_TILE + warp * 16u;
    if (m_warp_base >= N) return;

    float g_top = 0.f, g_bot = 0.f;
    float u_top = 0.f, u_bot = 0.f;
    mma_tile_acc<WARPS>(packed_w_gate, scales_w_gate, smem_x, scales_x,
                        K, sf_inner_dim_w, warp, lane, m_warp_base, g_top, g_bot);
    mma_tile_acc<WARPS>(packed_w_up, scales_w_up, smem_x, scales_x,
                        K, sf_inner_dim_w, warp, lane, m_warp_base, u_top, u_bot);

    if ((lane & 3u) == 0u) {
        unsigned int row_top = lane >> 2;
        unsigned int m_top_g = m_warp_base + row_top;
        unsigned int m_bot_g = m_top_g + 8u;
        if (m_top_g < N) out[m_top_g] = __float2half(silu(g_top) * u_top);
        if (m_bot_g < N) out[m_bot_g] = __float2half(silu(g_bot) * u_bot);
    }
}

extern "C" __global__ void nvfp4_swiglu_shuf_f16_w4(
    const unsigned char* __restrict__ packed_w_gate,
    const unsigned char* __restrict__ scales_w_gate,
    const unsigned char* __restrict__ packed_w_up,
    const unsigned char* __restrict__ scales_w_up,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    __half*              __restrict__ out,
    unsigned int N,
    unsigned int K,
    unsigned int sf_inner_dim_w)
{
    swiglu_shuf_impl<4>(packed_w_gate, scales_w_gate, packed_w_up, scales_w_up,
                        packed_x, scales_x, out, N, K, sf_inner_dim_w);
}

extern "C" __global__ void nvfp4_swiglu_shuf_f16_w8(
    const unsigned char* __restrict__ packed_w_gate,
    const unsigned char* __restrict__ scales_w_gate,
    const unsigned char* __restrict__ packed_w_up,
    const unsigned char* __restrict__ scales_w_up,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    __half*              __restrict__ out,
    unsigned int N,
    unsigned int K,
    unsigned int sf_inner_dim_w)
{
    swiglu_shuf_impl<8>(packed_w_gate, scales_w_gate, packed_w_up, scales_w_up,
                        packed_x, scales_x, out, N, K, sf_inner_dim_w);
}
