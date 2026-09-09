#include <cuda_fp16.h>
#include <cuda_bf16.h>

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


// ── Split-K: блок = KS варпов над ОДНИМ 16-строчным тайлом ─────────────
//
// Варп w берёт чанки w, w+KS, w+2KS, …, частичные суммы складываются через
// smem. У w4/w8 варп тянул полный K, и на N=2112 (плотный MLP Gemma-4)
// получалось 33 блока на 82 SM — веса читались на 370 ГБ/с из 900. Здесь грид
// N/16 блоков по 8 варпов: в разы больше загрузок в полёте при тех же байтах.
//
// Из d0..d3 столбец 0 (единственный при M=1) держат нити с lane&3==0: d0 —
// строка lane>>2, d2 — строка +8. `red` — smem [KS][8][2] float.
template <unsigned int KS>
__device__ __forceinline__ void mma_gemv_tile_splitk(
    const unsigned char* __restrict__ packed_w,
    const unsigned char* __restrict__ scales_w,
    const unsigned char* __restrict__ smem_x,
    const unsigned char* __restrict__ scales_x,
    unsigned int tile,
    unsigned int K,
    unsigned int sf_inner_dim_w,
    unsigned int x_sf_off,
    float* red,
    float& top,
    float& bot,
    unsigned int warp = threadIdx.x >> 5)
{
    unsigned int tid  = threadIdx.x;
    unsigned int lane = tid & 31u;

    unsigned int m_warp_base = tile * 16u;
    unsigned int m_t = lane & 3u;
    unsigned int k_t = lane >> 2;
    unsigned int s_a = lane & 1u;
    unsigned int s_c = lane >> 2;
    unsigned int m_for_sfa = m_warp_base + s_a * 8u + s_c;

    unsigned int block_base = tile * (K >> 6) * 512u;
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
    #pragma unroll 4
    for (unsigned int chunk = warp; chunk < num_chunks; chunk += KS) {
        nvfp4_gemv_mma_chunk(packed_w, scales_w, smem_x, scales_x, block_base, chunk,
                             top_off, bot_off, k_lo_off, k_hi_off, sfa_row_base, x_sf_off,
                             d0, d1, d2, d3);
    }
    if ((lane & 3u) == 0u) {
        red[(warp * 8u + (lane >> 2)) * 2u]      = d0;
        red[(warp * 8u + (lane >> 2)) * 2u + 1u] = d2;
    }
    __syncthreads();
    top = 0.f;
    bot = 0.f;
    if (warp == 0u && (lane & 3u) == 0u) {
        #pragma unroll
        for (unsigned int w = 0; w < KS; ++w) {
            top += red[(w * 8u + (lane >> 2)) * 2u];
            bot += red[(w * 8u + (lane >> 2)) * 2u + 1u];
        }
    }
}

#define SYN_SPLITK_WARPS 8u

__device__ __forceinline__ void load_x_smem(
    const unsigned char* __restrict__ packed_x, unsigned char* smem_x, unsigned int K)
{
    unsigned int x_u32_count = (K >> 1) >> 2;
    unsigned int*       dst = (unsigned int*)smem_x;
    const unsigned int* src = (const unsigned int*)packed_x;
    for (unsigned int i = threadIdx.x; i < x_u32_count; i += blockDim.x) dst[i] = src[i];
    __syncthreads();
}

// Групповой GEMV: до трёх весов одной K с общей квант-активацией одним
// запуском (q/k/v, gate/up). Блок = один 16-строчный тайл одной из матриц;
// тайлы идут подряд: [n0/16 | n1/16 | n2/16]. blockDim = 256 (8 варпов split-K).
// Строки роутера MoE (f32 [E, K] · bf16 x[K]) считаются хвостовыми блоками
// того же запуска: варп — строка, порядок fmaf как у mma_gemv_f32 (бит-в-бит
// с host-путём). Отдельным ядром эти 1.4 МБ упирались в латентность (16
// блоков, 11 мкс); среди GEMV-блоков их латентность спрятана.
__device__ __forceinline__ void router_row_f32(
    const float* __restrict__ w, const __nv_bfloat16* __restrict__ x, float* __restrict__ logits,
    unsigned int row, unsigned int K, unsigned int lane)
{
    const float* w_row = w + (size_t)row * K;
    const float4* w4 = reinterpret_cast<const float4*>(w_row);
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    float acc = 0.f;
    unsigned int k4 = K >> 2;
    #pragma unroll 4
    for (unsigned int i = lane; i < k4; i += 32u) {
        float4 wv = w4[i];
        float2 xa = __bfloat1622float2(x2[2 * i]);
        float2 xb = __bfloat1622float2(x2[2 * i + 1]);
        acc = fmaf(wv.x, xa.x, acc);
        acc = fmaf(wv.y, xa.y, acc);
        acc = fmaf(wv.z, xb.x, acc);
        acc = fmaf(wv.w, xb.y, acc);
    }
    for (unsigned int i = (k4 << 2) + lane; i < K; i += 32u)
        acc = fmaf(w_row[i], __bfloat162float(x[i]), acc);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) logits[row] = acc;
}

// Указатели групп приходят числами: у пустых групп они нулевые, а хосту так
// не нужно держать три изменяемые вьюхи разом. Хвост грида (r_rows > 0):
// строки роутера, по 8 на блок (варп — строка).
extern "C" __global__ void nvfp4_mma_gemv_shuf_grouped_splitk(
    unsigned long long w0, unsigned long long s0, unsigned long long o0, unsigned int n0,
    unsigned long long w1, unsigned long long s1, unsigned long long o1, unsigned int n1,
    unsigned long long w2, unsigned long long s2, unsigned long long o2, unsigned int n2,
    const unsigned char* __restrict__ packed_x,
    const unsigned char* __restrict__ scales_x,
    unsigned int K, unsigned int sf_inner_dim_w,
    unsigned long long r_w, unsigned long long r_x, unsigned long long r_logits, unsigned int r_rows)
{
    extern __shared__ unsigned char smem[];
    __shared__ float red[SYN_SPLITK_WARPS * 8u * 2u];
    unsigned int t = blockIdx.x;
    unsigned int t0 = n0 >> 4, t1 = n1 >> 4, t2 = n2 >> 4;
    if (t >= t0 + t1 + t2) {
        unsigned int row = (t - (t0 + t1 + t2)) * 8u + (threadIdx.x >> 5);
        if (row < r_rows) {
            router_row_f32((const float*)(size_t)r_w, (const __nv_bfloat16*)(size_t)r_x,
                           (float*)(size_t)r_logits, row, K, threadIdx.x & 31u);
        }
        return;
    }
    unsigned long long pw_u, sw_u, o_u; unsigned int tile;
    if (t < t0)           { pw_u = w0; sw_u = s0; o_u = o0; tile = t; }
    else if (t < t0 + t1) { pw_u = w1; sw_u = s1; o_u = o1; tile = t - t0; }
    else                  { pw_u = w2; sw_u = s2; o_u = o2; tile = t - t0 - t1; }
    const unsigned char* pw = (const unsigned char*)(size_t)pw_u;
    const unsigned char* sw = (const unsigned char*)(size_t)sw_u;
    syn_out_t* out = (syn_out_t*)(size_t)o_u;
    load_x_smem(packed_x, smem, K);
    float top, bot;
    mma_gemv_tile_splitk<SYN_SPLITK_WARPS>(pw, sw, smem, scales_x, tile, K, sf_inner_dim_w, 0u, red, top, bot);
    unsigned int lane = threadIdx.x & 31u;
    if (threadIdx.x < 32u && (lane & 3u) == 0u) {
        unsigned int m = tile * 16u + (lane >> 2);
        out[m] = SYN_TO_OUT(top);
        out[m + 8u] = SYN_TO_OUT(bot);
    }
}

// Вариант с MT тайлами на блок (KS = 8/MT варпов на тайл): меньше блоков и
// больше чанков на варп — для подбора под форму (K=704 у down-проекции
// экспертов делится на 8 варпов всего по 1–2 чанка).
template <unsigned int MT>
__device__ __forceinline__ void indexed_splitk_mt(
    const unsigned long long* __restrict__ w_table,
    const unsigned long long* __restrict__ s_table,
    const unsigned int* __restrict__ idx,
    const unsigned char* __restrict__ x_packed,
    const unsigned char* __restrict__ x_scales,
    unsigned int row_bytes, int rows_per_pair,
    unsigned long long out_u, unsigned long long acc_u, unsigned long long wts_u,
    unsigned int N, unsigned int K, unsigned int sf_inner_dim_w, unsigned int experts,
    unsigned char* smem, float* red)
{
    constexpr unsigned int KS = SYN_SPLITK_WARPS / MT;
    syn_out_t* out = (syn_out_t*)(size_t)out_u;
    float* acc = (float*)(size_t)acc_u;
    const float* wts = (const float*)(size_t)wts_u;
    unsigned int e = blockIdx.z;
    unsigned int ex = idx[e];
    if (ex >= experts) ex = 0u;
    const unsigned char* pw = (const unsigned char*)(size_t)w_table[ex];
    const unsigned char* sw = (const unsigned char*)(size_t)s_table[ex];
    unsigned int row = rows_per_pair ? e : 0u;
    const unsigned char* px = x_packed + (size_t)row * row_bytes;
    unsigned int x_sf_off = (row % 32u) * 16u + (row / 32u) * 4u;
    load_x_smem(px, smem, K);
    unsigned int warp = threadIdx.x >> 5;
    unsigned int tg = warp / KS;
    unsigned int wk = warp % KS;
    unsigned int tile = blockIdx.x * MT + tg;
    float top, bot;
    mma_gemv_tile_splitk<KS>(pw, sw, smem, x_scales, tile, K, sf_inner_dim_w, x_sf_off,
                             red + tg * (KS * 8u * 2u), top, bot, wk);
    unsigned int lane = threadIdx.x & 31u;
    if (wk == 0u && (lane & 3u) == 0u) {
        unsigned int m = tile * 16u + (lane >> 2);
        if (m + 8u < N || m < N) {
            if (acc) {
                float wv = wts[e];
                atomicAdd(acc + m, wv * top);
                atomicAdd(acc + m + 8u, wv * bot);
            } else {
                out[(size_t)e * N + m] = SYN_TO_OUT(top);
                out[(size_t)e * N + m + 8u] = SYN_TO_OUT(bot);
            }
        }
    }
}

extern "C" __global__ void nvfp4_mma_gemv_shuf_indexed_splitk_mt2(
    const unsigned long long* __restrict__ w_table,
    const unsigned long long* __restrict__ s_table,
    const unsigned int* __restrict__ idx,
    const unsigned char* __restrict__ x_packed,
    const unsigned char* __restrict__ x_scales,
    unsigned int row_bytes, int rows_per_pair,
    unsigned long long out_u, unsigned long long acc_u, unsigned long long wts_u,
    unsigned int N, unsigned int K, unsigned int sf_inner_dim_w, unsigned int experts)
{
    extern __shared__ unsigned char smem[];
    __shared__ float red[SYN_SPLITK_WARPS * 8u * 2u];
    indexed_splitk_mt<2>(w_table, s_table, idx, x_packed, x_scales, row_bytes, rows_per_pair,
                         out_u, acc_u, wts_u, N, K, sf_inner_dim_w, experts, smem, red);
}

extern "C" __global__ void nvfp4_mma_gemv_shuf_indexed_splitk_mt4(
    const unsigned long long* __restrict__ w_table,
    const unsigned long long* __restrict__ s_table,
    const unsigned int* __restrict__ idx,
    const unsigned char* __restrict__ x_packed,
    const unsigned char* __restrict__ x_scales,
    unsigned int row_bytes, int rows_per_pair,
    unsigned long long out_u, unsigned long long acc_u, unsigned long long wts_u,
    unsigned int N, unsigned int K, unsigned int sf_inner_dim_w, unsigned int experts)
{
    extern __shared__ unsigned char smem[];
    __shared__ float red[SYN_SPLITK_WARPS * 8u * 2u];
    indexed_splitk_mt<4>(w_table, s_table, idx, x_packed, x_scales, row_bytes, rows_per_pair,
                         out_u, acc_u, wts_u, N, K, sf_inner_dim_w, experts, smem, red);
}

// Индексный GEMV экспертов: blockIdx.z — пара (токен, слот), эксперт берётся
// из `idx` на карте, адреса — из фиксированных таблиц. Без ptr_gather и без
// скретчей указателей. Активация: rows_per_pair=0 — все пары читают строку 0,
// 1 — пара p читает строку p (выход первой проекции). Эпилог: либо строки
// `out[p, N]` (f16/bf16 по модулю), либо `acc[m] += wts[p]·y` в f32 атомиками —
// взвешенная сумма по экспертам без отдельных ядер взвешивания и редукции
// (`acc` обнуляет роутер).
extern "C" __global__ void nvfp4_mma_gemv_shuf_indexed_splitk(
    const unsigned long long* __restrict__ w_table,
    const unsigned long long* __restrict__ s_table,
    const unsigned int* __restrict__ idx,
    const unsigned char* __restrict__ x_packed,
    const unsigned char* __restrict__ x_scales,
    unsigned int row_bytes,
    int rows_per_pair,
    unsigned long long out_u,
    unsigned long long acc_u,
    unsigned long long wts_u,
    unsigned int N, unsigned int K, unsigned int sf_inner_dim_w, unsigned int experts)
{
    extern __shared__ unsigned char smem[];
    __shared__ float red[SYN_SPLITK_WARPS * 8u * 2u];
    syn_out_t* out = (syn_out_t*)(size_t)out_u;
    float* acc = (float*)(size_t)acc_u;
    const float* wts = (const float*)(size_t)wts_u;
    unsigned int e = blockIdx.z;
    unsigned int ex = idx[e];
    if (ex >= experts) ex = 0u;
    const unsigned char* pw = (const unsigned char*)(size_t)w_table[ex];
    const unsigned char* sw = (const unsigned char*)(size_t)s_table[ex];
    unsigned int row = rows_per_pair ? e : 0u;
    const unsigned char* px = x_packed + (size_t)row * row_bytes;
    unsigned int x_sf_off = (row % 32u) * 16u + (row / 32u) * 4u;
    load_x_smem(px, smem, K);
    float top, bot;
    mma_gemv_tile_splitk<SYN_SPLITK_WARPS>(pw, sw, smem, x_scales, blockIdx.x, K, sf_inner_dim_w, x_sf_off, red, top, bot);
    unsigned int lane = threadIdx.x & 31u;
    if (threadIdx.x < 32u && (lane & 3u) == 0u) {
        unsigned int m = blockIdx.x * 16u + (lane >> 2);
        if (acc) {
            float wv = wts[e];
            atomicAdd(acc + m, wv * top);
            atomicAdd(acc + m + 8u, wv * bot);
        } else {
            out[(size_t)e * N + m] = SYN_TO_OUT(top);
            out[(size_t)e * N + m + 8u] = SYN_TO_OUT(bot);
        }
    }
}

// ── Выбор эксперта индексом НА КАРТЕ ────────────────────────────────────────
//
// Пакетный GEMV читает четыре массива указателей (вес, масштабы веса,
// активация, масштабы активации) и массив смещений строки — раньше их собирал
// хост по выбору роутера. Под захватом CUDA-графа так нельзя: выбор приходит
// с карты. Ядро собирает те же массивы из ФИКСИРОВАННОЙ таблицы адресов
// экспертов по индексам `idx`.
//
// `rows_per_pair = 0` — все пары читают строку 0 (у первой проекции эксперта
// активация одна на токен); `1` — пара `p` читает строку `p` (у второй
// проекции активацией служит выход первой, по строке на пару).
extern "C" __global__ void nvfp4_expert_ptr_gather(
    const long long* __restrict__ w_table,
    const long long* __restrict__ s_table,
    const unsigned int* __restrict__ idx,
    long long* __restrict__ w_out,
    long long* __restrict__ s_out,
    long long* __restrict__ xp_out,
    long long* __restrict__ xs_out,
    unsigned int* __restrict__ off_out,
    long long xp_base,
    long long xs_base,
    int experts,
    int pairs,
    int rows_per_pair,
    int row_bytes) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) return;
    int e = (int)idx[p];
    if (e < 0 || e >= experts) e = 0;
    int row = rows_per_pair ? p : 0;
    w_out[p] = w_table[e];
    s_out[p] = s_table[e];
    xp_out[p] = xp_base + (long long)row * (long long)row_bytes;
    xs_out[p] = xs_base;
    // Та же раскладка tile'ов масштабов, что пишет квантователь активации.
    off_out[p] = (unsigned int)((row % 32) * 16 + (row / 32) * 4);
}
