#include <cuda_fp16.h>

#ifdef SYN_OUT_BF16
#include <cuda_bf16.h>
typedef __nv_bfloat16 syn_out_t;
#define SYN_TO_OUT(v) __float2bfloat16(v)
#else
typedef __half syn_out_t;
#define SYN_TO_OUT(v) __float2half(v)
#endif

extern "C" __global__ void nvfp4_w_repack(
    const unsigned char* __restrict__ packed_w_in,
    unsigned char* __restrict__ packed_w_out,
    unsigned int N,
    unsigned int K)
{

    unsigned int m_block  = blockIdx.x;
    unsigned int k_chunk  = blockIdx.y;
    unsigned int tid      = threadIdx.x;

    unsigned int row_in_block = tid >> 3;
    unsigned int u32_idx_in_chunk = tid & 7u;
    unsigned int byte_in_chunk = u32_idx_in_chunk * 4u;

    unsigned int row = m_block * 16u + row_in_block;
    if (row >= N) return;
    unsigned int src_byte = row * (K >> 1) + k_chunk * 32u + byte_in_chunk;
    unsigned int val = *(const unsigned int*)(packed_w_in + src_byte);

    unsigned int dst_byte = m_block * (K >> 6) * 512u
                          + k_chunk * 512u
                          + row_in_block * 32u
                          + byte_in_chunk;
    *(unsigned int*)(packed_w_out + dst_byte) = val;
}

// Один K-chunk: load весов (4×U32) + активации (2×U32) + scales → MMA, аккумулируя
// в d0..d3. Вынесено, чтобы гонять НЕСКОЛЬКО независимых аккумуляторов (ILP):
// серийная цепочка `d=mma(d)` ограничивала латентность; 2 цепочки скрывают её.
__device__ __forceinline__ void nvfp4_gemv_mma_chunk(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ smem_x,
    const unsigned char* __restrict__ scales_x,
    unsigned int block_base,
    unsigned int chunk,
    unsigned int top_off,
    unsigned int bot_off,
    unsigned int k_lo_off,
    unsigned int k_hi_off,
    unsigned int sfa_row_base,
    unsigned int x_sf_off,
    float& d0, float& d1, float& d2, float& d3)
{
    unsigned int chunk_base = block_base + chunk * 512u;
    unsigned int k_chunk_b  = chunk << 5;
    unsigned int a0 = *(const unsigned int*)(packed_w + chunk_base + top_off);
    unsigned int a1 = *(const unsigned int*)(packed_w + chunk_base + bot_off);
    unsigned int a2 = *(const unsigned int*)(packed_w + chunk_base + top_off + 16u);
    unsigned int a3 = *(const unsigned int*)(packed_w + chunk_base + bot_off + 16u);
    unsigned int b0 = *(const unsigned int*)(smem_x + k_chunk_b + k_lo_off);
    unsigned int b1 = *(const unsigned int*)(smem_x + k_chunk_b + k_hi_off);
    unsigned int sfa0 = *(const unsigned int*)(scales_w + sfa_row_base + chunk * 512u);
    unsigned int sfb0 = *(const unsigned int*)(scales_x + chunk * 512u + x_sf_off);
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

// `x_sf_off` — смещение строки активации внутри tile масштабов
// (`(outer%32)*16 + (outer/32)*4`). Для одиночного GEMV активация всегда одна
// строка и смещение нулевое; батчу оно нужно, чтобы читать свою строку из
// общего кванта, посчитанного разом для всех экспертов.
template <unsigned int WARPS>
__device__ __forceinline__ void mma_gemv_shuf_impl(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N,
    unsigned int K,
    unsigned int sf_inner_dim_w,
    unsigned int x_sf_off)
{
    constexpr unsigned int M_TILE = WARPS * 16;
    constexpr unsigned int THREADS = WARPS * 32;

    extern __shared__ unsigned char smem[];
    unsigned char* smem_x = smem;
    unsigned int k_half = K >> 1;

    unsigned int tid  = threadIdx.x;
    unsigned int warp = tid >> 5;
    unsigned int lane = tid & 31u;

    unsigned int x_u32_count = k_half >> 2;
    unsigned int*       smem_x_u32 = (unsigned int*)smem_x;
    const unsigned int* gmem_x_u32 = (const unsigned int*)packed_x;
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
    unsigned int local_outer  = m_for_sfa & 127u;
    unsigned int off_in_tile  = (local_outer & 31u) * 16u + (local_outer >> 5) * 4u;
    unsigned int sfa_row_base = tile_row_w * sf_inner_dim_w * 128u + off_in_tile;

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    unsigned int num_chunks = K >> 6;
    for (unsigned int chunk = 0; chunk < num_chunks; chunk++) {
        nvfp4_gemv_mma_chunk(packed_w, scales_w, smem_x, scales_x, block_base, chunk,
                             top_off, bot_off, k_lo_off, k_hi_off, sfa_row_base, x_sf_off,
                             d0, d1, d2, d3);
    }

    if ((lane & 3u) == 0u) {
        unsigned int row_top = lane >> 2;
        unsigned int m_top_g = m_warp_base + row_top;
        unsigned int m_bot_g = m_top_g + 8u;
        if (m_top_g < N) out[m_top_g] = SYN_TO_OUT(d0);
        if (m_bot_g < N) out[m_bot_g] = SYN_TO_OUT(d2);
    }
}

extern "C" __global__ void nvfp4_mma_gemv_shuf_f16_w4(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N, unsigned int K, unsigned int sf_inner_dim_w)
{
    mma_gemv_shuf_impl<4>(packed_w, scales_w, packed_x, scales_x, out, N, K, sf_inner_dim_w, 0u);
}

extern "C" __global__ void nvfp4_mma_gemv_shuf_f16_w8(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N, unsigned int K, unsigned int sf_inner_dim_w)
{
    mma_gemv_shuf_impl<8>(packed_w, scales_w, packed_x, scales_x, out, N, K, sf_inner_dim_w, 0u);
}

// Батч GEMV по списку весов: blockIdx.z выбирает эксперта, указатели на его
// packed/scales и на его активацию берутся из массивов. Нужен MoE-декоду, где
// на слой приходится десяток матриц по одной строке каждая: отдельными
// запусками они упираются в launch overhead, а не в вычисления.
extern "C" __global__ void nvfp4_mma_gemv_shuf_f16_w8_batched(
    const unsigned long long* __restrict__ w_ptrs,
    const unsigned long long* __restrict__ sw_ptrs,
    const unsigned long long* __restrict__ xp_ptrs,
    const unsigned long long* __restrict__ xs_ptrs,
    const unsigned int*       __restrict__ x_sf_offs,
    syn_out_t*           __restrict__ out,
    unsigned int N, unsigned int K, unsigned int sf_inner_dim_w)
{
    unsigned int e = blockIdx.z;
    const unsigned char* pw = (const unsigned char*)(size_t)w_ptrs[e];
    const unsigned char* sw = (const unsigned char*)(size_t)sw_ptrs[e];
    const unsigned char* px = (const unsigned char*)(size_t)xp_ptrs[e];
    const unsigned char* sx = (const unsigned char*)(size_t)xs_ptrs[e];
    unsigned int x_sf_off = x_sf_offs == nullptr ? 0u : x_sf_offs[e];
    mma_gemv_shuf_impl<8>(pw, sw, px, sx, out + (size_t)e * (size_t)N, N, K, sf_inner_dim_w,
                          x_sf_off);
}

extern "C" __global__ void nvfp4_mma_gemv_shuf_f16_w8_persistent(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    syn_out_t*           __restrict__ out,
    unsigned int N,
    unsigned int K,
    unsigned int sf_inner_dim_w)
{
    constexpr unsigned int WARPS = 8;
    constexpr unsigned int M_TILE = WARPS * 16;
    constexpr unsigned int THREADS = WARPS * 32;

    extern __shared__ unsigned char smem[];
    unsigned char* smem_x = smem;
    unsigned int k_half = K >> 1;

    unsigned int tid  = threadIdx.x;
    unsigned int warp = tid >> 5;
    unsigned int lane = tid & 31u;

    unsigned int x_u32_count = k_half >> 2;
    unsigned int*       smem_x_u32 = (unsigned int*)smem_x;
    const unsigned int* gmem_x_u32 = (const unsigned int*)packed_x;
    for (unsigned int i = tid; i < x_u32_count; i += THREADS) {
        smem_x_u32[i] = gmem_x_u32[i];
    }
    __syncthreads();

    unsigned int m_t = lane & 3u;
    unsigned int k_t = lane >> 2;
    unsigned int s_a = lane & 1u;
    unsigned int s_c = lane >> 2;
    unsigned int k_lo_off    = m_t * 4u;
    unsigned int k_hi_off    = k_lo_off + 16u;
    unsigned int top_off     = k_t * 32u + m_t * 4u;
    unsigned int bot_off     = (k_t + 8u) * 32u + m_t * 4u;
    unsigned int num_chunks  = K >> 6;
    unsigned int num_tiles   = N / M_TILE;

    for (unsigned int tile_id = blockIdx.x; tile_id < num_tiles; tile_id += gridDim.x) {
        unsigned int m_warp_base = tile_id * M_TILE + warp * 16u;
        unsigned int m_for_sfa = m_warp_base + s_a * 8u + s_c;
        unsigned int m_block_warp = m_warp_base >> 4;
        unsigned int block_base = m_block_warp * num_chunks * 512u;

        unsigned int tile_row_w   = m_for_sfa >> 7;
        unsigned int local_outer  = m_for_sfa & 127u;
        unsigned int off_in_tile  = (local_outer & 31u) * 16u + (local_outer >> 5) * 4u;
        unsigned int sfa_row_base = tile_row_w * sf_inner_dim_w * 128u + off_in_tile;

        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

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
            if (m_top_g < N) out[m_top_g] = SYN_TO_OUT(d0);
            if (m_bot_g < N) out[m_bot_g] = SYN_TO_OUT(d2);
        }
    }
}
