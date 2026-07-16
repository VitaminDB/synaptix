// FlashAttention-2 forward kernel (F16 inputs, F32 accum, GQA inside kernel,
// online softmax + KV-tiling). Минимальная корректная реализация без
// TensorCore/CUTLASS.
//
// Каждый CUDA-block обрабатывает один кортеж (b, h, q_row) → один выходной
// row длины hd. Внутри block — online softmax по KV-измерению tile'ами
// BLOCK_KV. K/V принимаются в KV-форме (nkv heads); GQA-broadcast делается
// маппингом kv_head = h / n_rep — экономит ×n_rep на K/V памяти.
//
// Bit-эквивалентность стандартному softmax-attention при F32 accum (см.
// FlashAttention-2 paper), отклонения только за счёт F16 input/output.
//
// Параметры:
//   block_dim.x = BLOCK_D = 128 threads (фиксировано).
//   d_per_thread = hd / BLOCK_D — каждый thread обслуживает несколько
//   d-каналов с шагом BLOCK_D. Поддерживается hd ∈ {128, 256, 384, 512}
//   (любое кратное 128, ≤ 512).

#include <cuda_fp16.h>

// NVRTC не подгружает <math.h> с макро-константами — определяем сами.
#define FA_POS_INF (__int_as_float(0x7F800000))
#define FA_NEG_INF (__int_as_float(0xFF800000))

#define BLOCK_D            128
#define BLOCK_KV           64
#define D_PER_THREAD_MAX   4   // single-row kernel: sup hd = 512 (stride BLOCK_D=128)

// BLOCK_M tiled kernel параметризуется через -D BLOCK_M={32,64}; default 32.
#ifndef BLOCK_M
#define BLOCK_M 32
#endif
#define ROWS_PER_WARP        (BLOCK_M / 4)              // 8 для M=32, 16 для M=64
#define D_PER_LANE_MAX       8                          // tiled kernel: sup hd = 256 (lane stride 32)

__device__ __forceinline__ bool fa_is_finite(float x) {
    return (__float_as_int(x) & 0x7F800000) != 0x7F800000;
}

extern "C" {

__global__ void flash_attn2_fwd_f16(
    const __half* __restrict__ q,    // (B, nh,  T_chunk, hd) row-major
    const __half* __restrict__ k,    // (B, nkv, *T_stride*, hd) row-major (см. t_stride)
    const __half* __restrict__ v,    // (B, nkv, *T_stride*, hd) row-major
    __half* __restrict__       out,  // (B, nh,  T_chunk, hd)
    float scale,                      // = hd^-0.5
    unsigned int B, unsigned int nh, unsigned int nkv,
    unsigned int T_chunk, unsigned int T_cache, unsigned int hd,
    unsigned int n_rep,               // = nh / nkv (GQA expansion внутри kernel)
    unsigned int q_pos_base,          // global q-position offset (для causal)
    int causal,                       // 0/1
    unsigned int t_stride,            // physical stride dim_T в k/v (0 → = T_cache, backward compat).
                                      // Для preallocated KV-ring (Phase C): T_cache = effective seq_pos,
                                      // t_stride = max_seq_len. Loop limits остаются на T_cache.
    const unsigned int* __restrict__ T_cache_ptr  // Phase D: device-resident T_cache (NULL → immediate).
                                                  // Когда non-NULL, читается ОДИН раз в начале kernel'я
                                                  // через shared broadcast, переопределяет immediate T_cache.
                                                  // Используется CUDA-graph replay flow: captured graph
                                                  // хранит pointer фиксированно, значение обновляется через
                                                  // memcpy_htod перед каждым launch'ем.
) {
    // Block layout:
    //   grid = (B*nh, T_chunk, 1)
    //   block = (BLOCK_D, 1, 1) = 128 threads = 4 warps.
    unsigned int bh    = blockIdx.x;
    unsigned int b     = bh / nh;
    unsigned int h     = bh % nh;
    unsigned int q_row = blockIdx.y;
    unsigned int tid   = threadIdx.x;

    if (b >= B || q_row >= T_chunk) return;

    // Phase D: optional device-resident T_cache override.
    if (T_cache_ptr != nullptr) {
        __shared__ unsigned int T_cache_sh;
        if (tid == 0) T_cache_sh = *T_cache_ptr;
        __syncthreads();
        T_cache = T_cache_sh;
    }

    unsigned int kv_h = h / n_rep;

    // d_per_thread = hd / BLOCK_D (1 для hd=128, 2 для hd=256, ...).
    unsigned int d_per_thread = hd / BLOCK_D;
    unsigned int n_d_chunks   = hd / 32;   // = 4 для hd=128, 8 для hd=256.

    // Shared memory layout:
    //   q_sm[hd]               F16
    //   s_sm[BLOCK_KV]         F32  (scores и потом P)
    //   meta[3]                F32  [m_new, l_new, alpha]
    extern __shared__ unsigned char shm[];
    __half* q_sm = (__half*)shm;
    float*  s_sm = (float*)(q_sm + hd);
    float*  meta = s_sm + BLOCK_KV;

    // 1. Загрузить Q-row (каждый thread берёт d_per_thread каналов).
    #pragma unroll D_PER_THREAD_MAX
    for (int dp = 0; dp < D_PER_THREAD_MAX; ++dp) {
        if ((unsigned int)dp >= d_per_thread) break;
        unsigned int d = tid + (unsigned int)dp * BLOCK_D;
        if (d < hd) {
            q_sm[d] = q[((size_t)((b * nh) + h) * T_chunk + q_row) * hd + d];
        }
    }
    __syncthreads();

    // Per-thread state.
    float m_curr = FA_NEG_INF;
    float l_curr = 0.0f;
    float acc[D_PER_THREAD_MAX];
    #pragma unroll
    for (int i = 0; i < D_PER_THREAD_MAX; ++i) acc[i] = 0.0f;

    unsigned int q_pos_global = q_pos_base + q_row;
    int n_kv_blocks = (int)((T_cache + BLOCK_KV - 1) / BLOCK_KV);

    unsigned int t_stride_eff = (t_stride > 0) ? t_stride : T_cache;
    size_t kv_base_offset =
        ((size_t)b * nkv + kv_h) * (size_t)t_stride_eff * hd;

    int warp_id = (int)(tid >> 5);   // 0..3
    int lane    = (int)(tid & 31);   // 0..31

    for (int kv_block = 0; kv_block < n_kv_blocks; ++kv_block) {
        int kv_base = kv_block * BLOCK_KV;
        int rem = (int)T_cache - kv_base;
        int kv_count = rem < BLOCK_KV ? rem : BLOCK_KV;

        // ─── Stage 1: compute s_sm[j] = scale * (q · k[kv_base+j]) для j=0..63.
        // 4 warps × (BLOCK_KV/4 = 16) j-чанков = 64 j-значения.
        // Branch на kv_t < T_cache hoisted в kv_count (only-last-block check).
        #pragma unroll
        for (int j_local = 0; j_local < BLOCK_KV / 4; ++j_local) {
            int j = j_local * 4 + warp_id;
            int kv_t = kv_base + j;
            float partial = 0.0f;
            if (j < kv_count) {
                #pragma unroll 4
                for (int d_chunk = 0; d_chunk < 16; ++d_chunk) {
                    if ((unsigned int)d_chunk >= n_d_chunks) break;
                    int d = d_chunk * 32 + lane;
                    float qv = __half2float(q_sm[d]);
                    float kv = __half2float(__ldg(
                        &k[kv_base_offset + (size_t)kv_t * hd + d]));
                    partial += qv * kv;
                }
            }
            // Warp-reduce sum.
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xFFFFFFFFu, partial, off);
            }
            if (lane == 0) {
                float s = partial * scale;
                if (j >= kv_count) {
                    s = FA_NEG_INF;
                } else if (causal && kv_t > (int)q_pos_global) {
                    s = FA_NEG_INF;
                }
                s_sm[j] = s;
            }
        }
        __syncthreads();

        // ─── Stage 2: parallel online-softmax внутри warp 0
        // (BLOCK_KV=64 → 32 lanes × 2 j-значения per lane).
        if (warp_id == 0) {
            float s0 = s_sm[lane];
            float s1 = s_sm[lane + 32];

            float m_local = (s0 > s1) ? s0 : s1;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float other = __shfl_xor_sync(0xFFFFFFFFu, m_local, off);
                if (other > m_local) m_local = other;
            }
            float m_block = m_local;
            float m_new = (m_block > m_curr) ? m_block : m_curr;

            float alpha;
            if (!fa_is_finite(m_curr)) {
                alpha = 0.0f;
            } else if (!fa_is_finite(m_new)) {
                alpha = 1.0f;
            } else {
                alpha = expf(m_curr - m_new);
            }

            float p0, p1;
            if (!fa_is_finite(m_new)) {
                p0 = 0.0f; p1 = 0.0f;
            } else {
                p0 = (s0 == FA_NEG_INF) ? 0.0f : expf(s0 - m_new);
                p1 = (s1 == FA_NEG_INF) ? 0.0f : expf(s1 - m_new);
            }
            float row_sum = p0 + p1;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                row_sum += __shfl_xor_sync(0xFFFFFFFFu, row_sum, off);
            }

            s_sm[lane]      = p0;
            s_sm[lane + 32] = p1;

            float l_new = l_curr * alpha + row_sum;
            if (lane == 0) {
                meta[0] = m_new;
                meta[1] = l_new;
                meta[2] = alpha;
            }
        }
        __syncthreads();

        float m_new = meta[0];
        float l_new = meta[1];
        float alpha = meta[2];

        // ─── Stage 3: acc[d] = acc[d]*alpha + sum_j(P[j] * V[kv_base+j, d])
        // Branch на kv_t < T_cache hoisted в kv_count.
        #pragma unroll D_PER_THREAD_MAX
        for (int dp = 0; dp < D_PER_THREAD_MAX; ++dp) {
            if ((unsigned int)dp >= d_per_thread) break;
            unsigned int d = tid + (unsigned int)dp * BLOCK_D;
            acc[dp] *= alpha;
            #pragma unroll 8
            for (int j = 0; j < BLOCK_KV; ++j) {
                if (j < kv_count) {
                    int kv_t = kv_base + j;
                    float vv = __half2float(__ldg(
                        &v[kv_base_offset + (size_t)kv_t * hd + d]));
                    acc[dp] += s_sm[j] * vv;
                }
            }
        }

        m_curr = m_new;
        l_curr = l_new;
        __syncthreads();
    }

    // ─── Stage 4: out[b, h, q_row, d] = acc[d] / l_curr.
    #pragma unroll D_PER_THREAD_MAX
    for (int dp = 0; dp < D_PER_THREAD_MAX; ++dp) {
        if ((unsigned int)dp >= d_per_thread) break;
        unsigned int d = tid + (unsigned int)dp * BLOCK_D;
        if (d < hd) {
            float result = (l_curr > 0.0f) ? (acc[dp] / l_curr) : 0.0f;
            out[((size_t)((b * nh) + h) * T_chunk + q_row) * hd + d]
                = __float2half(result);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FlashAttention-2 forward с BLOCK_M tiling (q-измерение).
//
// Главный win vs single-row kernel'я: один CUDA-block обрабатывает BLOCK_M
// (32 или 64) q-rows подряд для одной (b, h) пары. K/V tile грузится через
// __ldg ОДИН раз и переиспользуется для всех BLOCK_M строк → ×BLOCK_M
// редукции K/V-трафика для prefill.
//
// Алгоритмически идентичен single-row kernel'ю (online softmax + KV-tiling,
// F32 accum), отличие только в layout работы:
//   - Grid: (B*nh, ceil(T_chunk / BLOCK_M), 1).
//   - 4 warps в block; warp_id ∈ [0,3] обслуживает ROWS_PER_WARP = BLOCK_M/4
//     своих строк (8 для M=32, 16 для M=64). Внутри warp лейн обслуживает
//     `d_per_lane = hd/32` d-каналов через stride 32.
//
// Поддержка hd ∈ {128, 256} (`d_per_lane ≤ D_PER_LANE_MAX = 8`). Для hd > 256
// диспатчер в Rust обёртке fallback'ает на single-row kernel.
__global__ void flash_attn2_fwd_f16_tiled(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__       out,
    float scale,
    unsigned int B, unsigned int nh, unsigned int nkv,
    unsigned int T_chunk, unsigned int T_cache, unsigned int hd,
    unsigned int n_rep,
    unsigned int q_pos_base,
    int causal,
    unsigned int t_stride,  // 0 → = T_cache (backward compat); см. flash_attn2_fwd_f16
    const unsigned int* __restrict__ T_cache_ptr  // Phase D: NULL → immediate.
) {
    unsigned int bh     = blockIdx.x;
    unsigned int b      = bh / nh;
    unsigned int h      = bh % nh;
    unsigned int q_tile = blockIdx.y;
    unsigned int tid    = threadIdx.x;
    int warp_id = (int)(tid >> 5);   // 0..3
    int lane    = (int)(tid & 31);   // 0..31

    if (b >= B) return;

    unsigned int q_base = q_tile * BLOCK_M;
    if (q_base >= T_chunk) return;

    if (T_cache_ptr != nullptr) {
        __shared__ unsigned int T_cache_sh;
        if (tid == 0) T_cache_sh = *T_cache_ptr;
        __syncthreads();
        T_cache = T_cache_sh;
    }
    unsigned int q_rem = T_chunk - q_base;
    unsigned int q_count = q_rem < BLOCK_M ? q_rem : BLOCK_M;

    unsigned int kv_h       = h / n_rep;
    unsigned int d_per_lane = hd / 32;   // 4 (hd=128), 8 (hd=256)

    // Shared memory layout:
    //   q_sm[BLOCK_M * hd]       F16
    //   s_sm[BLOCK_M * BLOCK_KV] F32  (scores и потом P)
    //   m_sm[BLOCK_M]            F32
    //   l_sm[BLOCK_M]            F32
    //   alpha_sm[BLOCK_M]        F32
    extern __shared__ unsigned char shm[];
    __half* q_sm    = (__half*)shm;
    float*  s_sm    = (float*)(q_sm + (size_t)BLOCK_M * hd);
    float*  m_sm    = s_sm + (size_t)BLOCK_M * BLOCK_KV;
    float*  l_sm    = m_sm + BLOCK_M;
    float*  alpha_sm = l_sm + BLOCK_M;

    // ─── Stage 0: cooperatively load q_sm[BLOCK_M, hd] ───
    {
        unsigned int q_total = BLOCK_M * hd;
        for (unsigned int i = tid; i < q_total; i += BLOCK_D) {
            unsigned int r = i / hd;
            unsigned int d = i - r * hd;
            if (r < q_count) {
                q_sm[i] = q[((size_t)((b * nh) + h) * T_chunk + (q_base + r)) * hd + d];
            } else {
                q_sm[i] = __float2half(0.0f);
            }
        }
    }
    // Init m/l per row.
    if (tid < BLOCK_M) {
        m_sm[tid] = FA_NEG_INF;
        l_sm[tid] = 0.0f;
    }
    __syncthreads();

    // Per-thread state: acc[ROWS_PER_WARP][D_PER_LANE_MAX].
    // d-индексация: лейн обслуживает d = dp*32 + lane для dp ∈ [0, d_per_lane).
    float acc[ROWS_PER_WARP * D_PER_LANE_MAX];
    #pragma unroll
    for (int i = 0; i < ROWS_PER_WARP * D_PER_LANE_MAX; ++i) acc[i] = 0.0f;

    int n_kv_blocks = (int)((T_cache + BLOCK_KV - 1) / BLOCK_KV);
    unsigned int t_stride_eff = (t_stride > 0) ? t_stride : T_cache;
    size_t kv_base_offset = ((size_t)b * nkv + kv_h) * (size_t)t_stride_eff * hd;
    int row_start = warp_id * ROWS_PER_WARP;

    for (int kv_block = 0; kv_block < n_kv_blocks; ++kv_block) {
        int kv_base = kv_block * BLOCK_KV;
        int rem = (int)T_cache - kv_base;
        int kv_count = rem < BLOCK_KV ? rem : BLOCK_KV;

        // ─── Stage 1: S[r, j] = scale * Q[r] · K[kv_base+j] ───
        // K[j, d] загружается ОДИН раз per j и переиспользуется для всех
        // ROWS_PER_WARP строк этого warp'а.
        #pragma unroll 4
        for (int j = 0; j < BLOCK_KV; ++j) {
            int kv_t = kv_base + j;
            float k_vals[D_PER_LANE_MAX];
            if (j < kv_count) {
                #pragma unroll
                for (int dp = 0; dp < D_PER_LANE_MAX; ++dp) {
                    if ((unsigned int)dp >= d_per_lane) break;
                    int d = dp * 32 + lane;
                    k_vals[dp] = __half2float(__ldg(
                        &k[kv_base_offset + (size_t)kv_t * hd + d]));
                }
            } else {
                #pragma unroll
                for (int dp = 0; dp < D_PER_LANE_MAX; ++dp) k_vals[dp] = 0.0f;
            }

            #pragma unroll
            for (int r_local = 0; r_local < ROWS_PER_WARP; ++r_local) {
                int r_global = row_start + r_local;
                float partial = 0.0f;
                if (r_global < (int)q_count && j < kv_count) {
                    #pragma unroll
                    for (int dp = 0; dp < D_PER_LANE_MAX; ++dp) {
                        if ((unsigned int)dp >= d_per_lane) break;
                        int d = dp * 32 + lane;
                        float qv = __half2float(q_sm[r_global * hd + d]);
                        partial += qv * k_vals[dp];
                    }
                }
                // Warp-reduce sum.
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    partial += __shfl_xor_sync(0xFFFFFFFFu, partial, off);
                }
                if (lane == 0) {
                    float s = partial * scale;
                    int q_pos_global = (int)q_pos_base + (int)q_base + r_global;
                    if (r_global >= (int)q_count) {
                        s = FA_NEG_INF;
                    } else if (j >= kv_count) {
                        s = FA_NEG_INF;
                    } else if (causal && kv_t > q_pos_global) {
                        s = FA_NEG_INF;
                    }
                    s_sm[r_global * BLOCK_KV + j] = s;
                }
            }
        }
        __syncthreads();

        // ─── Stage 2: online softmax 2D per row (each warp on its rows) ───
        #pragma unroll
        for (int r_local = 0; r_local < ROWS_PER_WARP; ++r_local) {
            int r_global = row_start + r_local;
            float s0 = s_sm[r_global * BLOCK_KV + lane];
            float s1 = s_sm[r_global * BLOCK_KV + lane + 32];

            float m_local = (s0 > s1) ? s0 : s1;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float other = __shfl_xor_sync(0xFFFFFFFFu, m_local, off);
                if (other > m_local) m_local = other;
            }
            float m_block = m_local;
            float m_curr = m_sm[r_global];
            float m_new = (m_block > m_curr) ? m_block : m_curr;

            float alpha;
            if (!fa_is_finite(m_curr)) {
                alpha = 0.0f;
            } else if (!fa_is_finite(m_new)) {
                alpha = 1.0f;
            } else {
                alpha = expf(m_curr - m_new);
            }

            float p0, p1;
            if (!fa_is_finite(m_new)) {
                p0 = 0.0f; p1 = 0.0f;
            } else {
                p0 = (s0 == FA_NEG_INF) ? 0.0f : expf(s0 - m_new);
                p1 = (s1 == FA_NEG_INF) ? 0.0f : expf(s1 - m_new);
            }
            float row_sum = p0 + p1;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                row_sum += __shfl_xor_sync(0xFFFFFFFFu, row_sum, off);
            }

            s_sm[r_global * BLOCK_KV + lane]      = p0;
            s_sm[r_global * BLOCK_KV + lane + 32] = p1;

            float l_curr = l_sm[r_global];
            float l_new = l_curr * alpha + row_sum;
            if (lane == 0) {
                m_sm[r_global]     = m_new;
                l_sm[r_global]     = l_new;
                alpha_sm[r_global] = alpha;
            }
        }
        __syncthreads();

        // ─── Stage 3: acc[r,d] = acc[r,d]*alpha[r] + sum_j(P[r,j] * V[kv_base+j, d]) ───
        #pragma unroll
        for (int r_local = 0; r_local < ROWS_PER_WARP; ++r_local) {
            int r_global = row_start + r_local;
            float alpha = alpha_sm[r_global];
            #pragma unroll
            for (int dp = 0; dp < D_PER_LANE_MAX; ++dp) {
                if ((unsigned int)dp >= d_per_lane) break;
                acc[r_local * D_PER_LANE_MAX + dp] *= alpha;
            }
        }

        // V[j, d] загружается ОДИН раз per j и переиспользуется для всех
        // ROWS_PER_WARP строк этого warp'а.
        #pragma unroll 4
        for (int j = 0; j < BLOCK_KV; ++j) {
            if (j >= kv_count) break;
            int kv_t = kv_base + j;
            float v_vals[D_PER_LANE_MAX];
            #pragma unroll
            for (int dp = 0; dp < D_PER_LANE_MAX; ++dp) {
                if ((unsigned int)dp >= d_per_lane) break;
                int d = dp * 32 + lane;
                v_vals[dp] = __half2float(__ldg(
                    &v[kv_base_offset + (size_t)kv_t * hd + d]));
            }
            #pragma unroll
            for (int r_local = 0; r_local < ROWS_PER_WARP; ++r_local) {
                int r_global = row_start + r_local;
                float p = s_sm[r_global * BLOCK_KV + j];
                #pragma unroll
                for (int dp = 0; dp < D_PER_LANE_MAX; ++dp) {
                    if ((unsigned int)dp >= d_per_lane) break;
                    acc[r_local * D_PER_LANE_MAX + dp] += p * v_vals[dp];
                }
            }
        }
        __syncthreads();
    }

    // ─── Stage 4: normalize + write ───
    #pragma unroll
    for (int r_local = 0; r_local < ROWS_PER_WARP; ++r_local) {
        int r_global = row_start + r_local;
        if (r_global >= (int)q_count) break;
        float l = l_sm[r_global];
        float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        #pragma unroll
        for (int dp = 0; dp < D_PER_LANE_MAX; ++dp) {
            if ((unsigned int)dp >= d_per_lane) break;
            int d = dp * 32 + lane;
            float result = acc[r_local * D_PER_LANE_MAX + dp] * inv;
            out[((size_t)((b * nh) + h) * T_chunk + (q_base + r_global)) * hd + d]
                = __float2half(result);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Split-K variant: разбиение KV по T_cache на SPLIT_K кусков (grid.z = SPLIT_K).
// Каждый блок обрабатывает свой kv-диапазон [kv_start, kv_end) и пишет partial
// (m_partial, l_partial, acc_partial[hd]) F32. Финальный merge-kernel объединяет
// через online-softmax-merge.
//
// Цель: убрать underoccupancy на decode T_chunk=1 (grid=B*nh=32 < SM_count=82).
// SPLIT_K=8 → grid=256, waves≈3 → SM throughput с 2.4% (см. ncu отчёт) к ≥20%.
//
// Параметры:
//   grid = (B*nh, T_chunk, SPLIT_K).
//   block = (BLOCK_D=128, 1, 1).
//   shared = такой же как single-row (hd*2 + 64*4 + 12 = ~780 байт для hd=256).
//
// Layout partial-буферов (NOT нормализованные):
//   partial_m[b, h, q_row, split_id]            F32, shape (B*nh*T_chunk*SPLIT_K,)
//   partial_l[b, h, q_row, split_id]            F32, такая же
//   partial_acc[b, h, q_row, split_id, d]       F32, shape (B*nh*T_chunk*SPLIT_K*hd,)
//
// Пустые splits (kv_start >= T_cache, либо все позиции масками отрезаны) пишут
// partial_m = -INF, partial_l = 0, partial_acc = 0 → exp(-INF - m_max) = 0 в merge.
__global__ void flash_attn2_fwd_f16_split(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    float* __restrict__ partial_acc,    // (B*nh*T_chunk*SPLIT_K, hd) F32
    float* __restrict__ partial_m,      // (B*nh*T_chunk*SPLIT_K,)   F32
    float* __restrict__ partial_l,      // (B*nh*T_chunk*SPLIT_K,)   F32
    float scale,
    unsigned int B, unsigned int nh, unsigned int nkv,
    unsigned int T_chunk, unsigned int T_cache, unsigned int hd,
    unsigned int n_rep,
    unsigned int q_pos_base,
    int causal,
    unsigned int split_k,               // count of splits
    unsigned int t_stride,              // 0 → = T_cache (backward compat)
    const unsigned int* __restrict__ T_cache_ptr  // Phase D: NULL → immediate.
) {
    unsigned int bh       = blockIdx.x;
    unsigned int b        = bh / nh;
    unsigned int h        = bh % nh;
    unsigned int q_row    = blockIdx.y;
    unsigned int split_id = blockIdx.z;
    unsigned int tid      = threadIdx.x;

    if (b >= B || q_row >= T_chunk || split_id >= split_k) return;

    if (T_cache_ptr != nullptr) {
        __shared__ unsigned int T_cache_sh;
        if (tid == 0) T_cache_sh = *T_cache_ptr;
        __syncthreads();
        T_cache = T_cache_sh;
    }

    // KV-диапазон этого split'а: равномерное разбиение T_cache на split_k частей.
    unsigned int split_size = (T_cache + split_k - 1) / split_k;
    unsigned int kv_start   = split_id * split_size;
    unsigned int kv_end_unb = kv_start + split_size;
    unsigned int kv_end     = (kv_end_unb < T_cache) ? kv_end_unb : T_cache;

    unsigned int kv_h = h / n_rep;
    unsigned int d_per_thread = hd / BLOCK_D;
    unsigned int n_d_chunks   = hd / 32;

    // Индекс partial-буферов: linear по (B*nh, T_chunk, split_k).
    size_t partial_idx = (((size_t)b * nh + h) * T_chunk + q_row) * split_k + split_id;

    // Если split пустой — пишем NEG_INF/0/0 и выходим.
    if (kv_start >= kv_end) {
        if (tid == 0) {
            partial_m[partial_idx] = FA_NEG_INF;
            partial_l[partial_idx] = 0.0f;
        }
        // Zero acc.
        #pragma unroll D_PER_THREAD_MAX
        for (int dp = 0; dp < D_PER_THREAD_MAX; ++dp) {
            if ((unsigned int)dp >= d_per_thread) break;
            unsigned int d = tid + (unsigned int)dp * BLOCK_D;
            if (d < hd) {
                partial_acc[partial_idx * hd + d] = 0.0f;
            }
        }
        return;
    }

    extern __shared__ unsigned char shm[];
    __half* q_sm = (__half*)shm;
    float*  s_sm = (float*)(q_sm + hd);
    float*  meta = s_sm + BLOCK_KV;

    // Load Q-row.
    #pragma unroll D_PER_THREAD_MAX
    for (int dp = 0; dp < D_PER_THREAD_MAX; ++dp) {
        if ((unsigned int)dp >= d_per_thread) break;
        unsigned int d = tid + (unsigned int)dp * BLOCK_D;
        if (d < hd) {
            q_sm[d] = q[((size_t)((b * nh) + h) * T_chunk + q_row) * hd + d];
        }
    }
    __syncthreads();

    float m_curr = FA_NEG_INF;
    float l_curr = 0.0f;
    float acc[D_PER_THREAD_MAX];
    #pragma unroll
    for (int i = 0; i < D_PER_THREAD_MAX; ++i) acc[i] = 0.0f;

    unsigned int q_pos_global = q_pos_base + q_row;
    unsigned int t_stride_eff = (t_stride > 0) ? t_stride : T_cache;
    size_t kv_base_offset = ((size_t)b * nkv + kv_h) * (size_t)t_stride_eff * hd;

    int warp_id = (int)(tid >> 5);
    int lane    = (int)(tid & 31);

    // KV-блоки только в диапазоне [kv_start, kv_end).
    int kv_block_start = (int)(kv_start / BLOCK_KV);
    int kv_block_end   = (int)((kv_end + BLOCK_KV - 1) / BLOCK_KV);

    for (int kv_block = kv_block_start; kv_block < kv_block_end; ++kv_block) {
        int kv_base = kv_block * BLOCK_KV;
        int rem = (int)kv_end - kv_base;
        int kv_count = rem < BLOCK_KV ? rem : BLOCK_KV;
        int kv_skip  = (kv_base < (int)kv_start) ? ((int)kv_start - kv_base) : 0;

        // ─── Stage 1: S[j] = scale * (Q · K[kv_base+j]).
        #pragma unroll
        for (int j_local = 0; j_local < BLOCK_KV / 4; ++j_local) {
            int j = j_local * 4 + warp_id;
            int kv_t = kv_base + j;
            float partial = 0.0f;
            bool active = (j >= kv_skip) && (j < kv_count);
            if (active) {
                #pragma unroll 4
                for (int d_chunk = 0; d_chunk < 16; ++d_chunk) {
                    if ((unsigned int)d_chunk >= n_d_chunks) break;
                    int d = d_chunk * 32 + lane;
                    float qv = __half2float(q_sm[d]);
                    float kv = __half2float(__ldg(
                        &k[kv_base_offset + (size_t)kv_t * hd + d]));
                    partial += qv * kv;
                }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xFFFFFFFFu, partial, off);
            }
            if (lane == 0) {
                float s = partial * scale;
                if (!active) {
                    s = FA_NEG_INF;
                } else if (causal && kv_t > (int)q_pos_global) {
                    s = FA_NEG_INF;
                }
                s_sm[j] = s;
            }
        }
        __syncthreads();

        // ─── Stage 2: online-softmax.
        if (warp_id == 0) {
            float s0 = s_sm[lane];
            float s1 = s_sm[lane + 32];

            float m_local = (s0 > s1) ? s0 : s1;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float other = __shfl_xor_sync(0xFFFFFFFFu, m_local, off);
                if (other > m_local) m_local = other;
            }
            float m_block = m_local;
            float m_new = (m_block > m_curr) ? m_block : m_curr;

            float alpha;
            if (!fa_is_finite(m_curr)) {
                alpha = 0.0f;
            } else if (!fa_is_finite(m_new)) {
                alpha = 1.0f;
            } else {
                alpha = expf(m_curr - m_new);
            }

            float p0, p1;
            if (!fa_is_finite(m_new)) {
                p0 = 0.0f; p1 = 0.0f;
            } else {
                p0 = (s0 == FA_NEG_INF) ? 0.0f : expf(s0 - m_new);
                p1 = (s1 == FA_NEG_INF) ? 0.0f : expf(s1 - m_new);
            }
            float row_sum = p0 + p1;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                row_sum += __shfl_xor_sync(0xFFFFFFFFu, row_sum, off);
            }

            s_sm[lane]      = p0;
            s_sm[lane + 32] = p1;

            float l_new = l_curr * alpha + row_sum;
            if (lane == 0) {
                meta[0] = m_new;
                meta[1] = l_new;
                meta[2] = alpha;
            }
        }
        __syncthreads();

        float m_new = meta[0];
        float l_new = meta[1];
        float alpha = meta[2];

        // ─── Stage 3: acc[d] = acc[d]*alpha + sum_j(P[j] * V[kv_base+j, d]).
        #pragma unroll D_PER_THREAD_MAX
        for (int dp = 0; dp < D_PER_THREAD_MAX; ++dp) {
            if ((unsigned int)dp >= d_per_thread) break;
            unsigned int d = tid + (unsigned int)dp * BLOCK_D;
            acc[dp] *= alpha;
            #pragma unroll 8
            for (int j = 0; j < BLOCK_KV; ++j) {
                if (j >= kv_skip && j < kv_count) {
                    int kv_t = kv_base + j;
                    float vv = __half2float(__ldg(
                        &v[kv_base_offset + (size_t)kv_t * hd + d]));
                    acc[dp] += s_sm[j] * vv;
                }
            }
        }

        m_curr = m_new;
        l_curr = l_new;
        __syncthreads();
    }

    // ─── Output: write partials (NOT нормализованные).
    if (tid == 0) {
        partial_m[partial_idx] = m_curr;
        partial_l[partial_idx] = l_curr;
    }
    #pragma unroll D_PER_THREAD_MAX
    for (int dp = 0; dp < D_PER_THREAD_MAX; ++dp) {
        if ((unsigned int)dp >= d_per_thread) break;
        unsigned int d = tid + (unsigned int)dp * BLOCK_D;
        if (d < hd) {
            partial_acc[partial_idx * hd + d] = acc[dp];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Merge kernel: объединяет SPLIT_K partial'ов через online softmax merge.
//
//   m_max = max_i(partial_m[i])
//   corr[i] = exp(partial_m[i] - m_max)          (для пустых splits = 0)
//   l_global = sum_i(partial_l[i] * corr[i])
//   out[d] = (sum_i partial_acc[i, d] * corr[i]) / l_global
//
// Grid: (B*nh, T_chunk, 1). Block = (BLOCK_D=128, 1, 1).
// SPLIT_K_MAX = 32 (поднять при необходимости).
#define SPLIT_K_MAX 32

__global__ void flash_attn2_fwd_f16_merge(
    const float* __restrict__ partial_acc,  // (B*nh*T_chunk*SPLIT_K, hd)
    const float* __restrict__ partial_m,    // (B*nh*T_chunk*SPLIT_K,)
    const float* __restrict__ partial_l,    // (B*nh*T_chunk*SPLIT_K,)
    __half* __restrict__ out,                // (B, nh, T_chunk, hd)
    unsigned int B, unsigned int nh, unsigned int T_chunk, unsigned int hd,
    unsigned int split_k
) {
    unsigned int bh    = blockIdx.x;
    unsigned int b     = bh / nh;
    unsigned int h     = bh % nh;
    unsigned int q_row = blockIdx.y;
    unsigned int tid   = threadIdx.x;

    if (b >= B || q_row >= T_chunk) return;

    size_t base_idx = (((size_t)b * nh + h) * T_chunk + q_row) * split_k;

    // Shared: m_max, l_global, corr[SPLIT_K_MAX].
    __shared__ float m_max_sh;
    __shared__ float l_global_sh;
    __shared__ float corr_sh[SPLIT_K_MAX];

    // 1. m_max = max_i(partial_m[i]) — thread 0 делает.
    if (tid == 0) {
        float m_max = FA_NEG_INF;
        for (unsigned int i = 0; i < split_k; ++i) {
            float mi = partial_m[base_idx + i];
            if (mi > m_max) m_max = mi;
        }
        m_max_sh = m_max;

        // 2. corr[i] = exp(partial_m[i] - m_max); l_global = sum partial_l*corr.
        float l_sum = 0.0f;
        bool m_finite = fa_is_finite(m_max);
        for (unsigned int i = 0; i < split_k; ++i) {
            float mi = partial_m[base_idx + i];
            float li = partial_l[base_idx + i];
            float c;
            if (!m_finite) {
                c = 0.0f;
            } else if (!fa_is_finite(mi)) {
                c = 0.0f;
            } else {
                c = expf(mi - m_max);
            }
            corr_sh[i] = c;
            l_sum += li * c;
        }
        l_global_sh = l_sum;
    }
    __syncthreads();

    float l_global = l_global_sh;
    float inv = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;

    // 3. Each thread sums partial_acc[i, d] * corr[i] для своих d-каналов.
    unsigned int d_per_thread = hd / BLOCK_D;
    #pragma unroll D_PER_THREAD_MAX
    for (int dp = 0; dp < D_PER_THREAD_MAX; ++dp) {
        if ((unsigned int)dp >= d_per_thread) break;
        unsigned int d = tid + (unsigned int)dp * BLOCK_D;
        if (d < hd) {
            float acc = 0.0f;
            for (unsigned int i = 0; i < split_k; ++i) {
                size_t row = base_idx + i;
                acc += partial_acc[row * hd + d] * corr_sh[i];
            }
            out[((size_t)((b * nh) + h) * T_chunk + q_row) * hd + d]
                = __float2half(acc * inv);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase A.2: FlashAttention-2 forward с Tensor Cores (WMMA mma.sync.m16n8k16).
//
// Алгоритмически идентичен single-row/tile/split kernels (online softmax + KV-
// tiling + GQA inside), но matmul'ы в Stage 1 (QK^T) и Stage 3 (PV) выполнены
// через `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (sm_80+ Tensor
// Cores). Это эквивалент FA-4 для sm_120 (см. `~/Temp/flash-attention/flash_attn/
// cute/flash_fwd_sm120.py:1` — "SM120 uses the same SM80-era MMA instructions").
//
// Hardcoded для hd=256 (Qwen3.6). T_chunk выровнен до BLOCK_M=16 (decode T=1
// маскируется до 15/16 rows; не оптимально, диспатчер должен направлять
// decode в split-K path).
//
// Tile параметры:
//   BLOCK_M  = 16   (M-dim WMMA; m16n8k16 mma операнд A)
//   BLOCK_KV = 32   (N-dim softmax tile; 4 mma_n=8 sub-tiles per kv-iter)
//   HD       = 256
//   block_dim = 128 = 4 warps
//   grid = (B*nh, ceil(T_chunk/16), 1)
//
// Shared layout (~75 KB, requires opt-in MAX_DYNAMIC_SHARED_SIZE_BYTES ≥ 80KB):
//   q_sm[BM*HD]                    F16 =  8 KB
//   k_sm[2][BN*HD]                 F16 = 32 KB (double-buffer pingpong)
//   v_sm[2][BN*HD]                 F16 = 32 KB (double-buffer pingpong)
//   s_f32[BM*BN]                   F32 =  2 KB (S после Stage 1)
//   p_sm[BM*BN]                    F16 =  1 KB (P после softmax, B-frag для Stage 3)
//   m_sm[BM], l_sm[BM], alpha_sm[BM] F32  = 192 B
//
// Pipeline (cp.async.ca.shared.global, 16B chunks):
//   prologue: load K_0, V_0 → slot 0; commit_group  → 1 group pending
//   for kv = 0..N-1:
//     if kv+1 < N:
//       load K_{kv+1}, V_{kv+1} → slot (kv+1)%2; commit_group  → 2 pending
//       wait_group(1)  // drains group kv
//     else:
//       wait_group(0)  // last iter, drain all
//     __syncthreads()
//     Stage 1: per warp 16 mma_k → S[16][8] в register fragments F32
//     Stage 2: write S → s_f32, sync, warp 0 softmax → p_sm + m_sm/l_sm/alpha_sm
//     sync
//     Stage 3: per warp 8 n_tiles × 2 mma_k → acc[16][64 per warp] += P @ V
//
// Q-fragments: pre-loaded ОДИН раз в registers (16 mma_k slots × 4 half2 = 32
// dword per thread = 32 F32-equivalent registers).
//
// V loading для Stage 3 B-frag: V row-major → mma row.col требует B col-major.
// Per B-frag: 4 strided F16 loads (адреса с шагом HD по kv-измерению), упаковка
// в 2 half2 регистра вручную. Для оптимизации — ldmatrix.x2.trans (TODO).

#define BM 16
#define BN 32
#define HD_WMMA 256
#define N_WARPS 4
#define WMMA_BLOCK_D 128

// ─── PTX helpers ───
// NVRTC не подгружает <cstdint>; используем unsigned int вместо uint32_t.

__device__ __forceinline__ unsigned int fa_smem_ptr(const void* ptr) {
    unsigned int smem_ptr;
    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1;"
        " cvt.u32.u64 %0, smem_ptr; }"
        : "=r"(smem_ptr) : "l"(ptr));
    return smem_ptr;
}

__device__ __forceinline__ void fa_cp_async_16(unsigned int smem_dst, const void* gmem_src) {
    // 16-byte (= 1 uint4 = 8 F16) copy from global → shared, async, cache L1+L2.
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_dst), "l"(gmem_src));
}

__device__ __forceinline__ void fa_cp_async_16_zero(unsigned int smem_dst) {
    // cp.async с src-size=0 → 16 байт нулей в shared (out-of-bounds masking).
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;\n"
                 :: "r"(smem_dst), "l"((const void*)nullptr));
}

__device__ __forceinline__ void fa_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

// `cp.async.wait_group N` требует immediate операнд → используем макрос
// (template был бы запрещён внутри extern "C").
#define FA_CP_ASYNC_WAIT_GROUP(N) \
    asm volatile("cp.async.wait_group " #N ";\n")

__device__ __forceinline__ void fa_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

__device__ __forceinline__ unsigned int fa_pack_h2(__half a, __half b) {
    union { __half2 h; unsigned int u; } x;
    x.h = __halves2half2(a, b);
    return x.u;
}

// Загрузить 2 contiguous F16 из shared как packed unsigned int (half2).
__device__ __forceinline__ unsigned int fa_load_h2_smem(const __half* p) {
    union { __half2 h; unsigned int u; } x;
    x.h = *reinterpret_cast<const __half2*>(p);
    return x.u;
}

// ─────────────────────────────────────────────────────────────────────────────
__global__ void flash_attn2_fwd_f16_wmma(
    const __half* __restrict__ q,    // (B, nh,  T_chunk, hd)
    const __half* __restrict__ k,    // (B, nkv, *T_stride*, hd)
    const __half* __restrict__ v,    // (B, nkv, *T_stride*, hd)
    __half* __restrict__       out,  // (B, nh,  T_chunk, hd)
    float scale,
    unsigned int B, unsigned int nh, unsigned int nkv,
    unsigned int T_chunk, unsigned int T_cache, unsigned int hd_param,
    unsigned int n_rep,
    unsigned int q_pos_base,
    int causal,
    unsigned int t_stride,  // 0 → = T_cache (backward compat)
    const unsigned int* __restrict__ T_cache_ptr  // Phase D: NULL → immediate.
) {
    // hd_param должен быть равен HD_WMMA=256; диспатчер проверяет.
    constexpr int HD = HD_WMMA;

    unsigned int bh     = blockIdx.x;
    unsigned int b      = bh / nh;
    unsigned int h      = bh % nh;
    unsigned int q_tile = blockIdx.y;
    unsigned int tid    = threadIdx.x;
    int warp_id = (int)(tid >> 5);     // 0..3
    int lane    = (int)(tid & 31);     // 0..31

    if (b >= B) return;
    unsigned int q_base = q_tile * BM;
    if (q_base >= T_chunk) return;
    int q_count = (int)((T_chunk - q_base) < BM ? (T_chunk - q_base) : BM);

    if (T_cache_ptr != nullptr) {
        __shared__ unsigned int T_cache_sh;
        if (tid == 0) T_cache_sh = *T_cache_ptr;
        __syncthreads();
        T_cache = T_cache_sh;
    }

    unsigned int kv_h = h / n_rep;
    unsigned int t_stride_eff = (t_stride > 0) ? t_stride : T_cache;
    size_t kv_base_offset = ((size_t)b * nkv + kv_h) * (size_t)t_stride_eff * HD;
    size_t q_base_offset  = ((size_t)b * nh + h) * (size_t)T_chunk * HD;

    // ─── Shared memory layout ───
    extern __shared__ unsigned char smem[];
    __half* q_sm  = (__half*)smem;
    __half* k_sm0 = q_sm + BM * HD;
    __half* v_sm0 = k_sm0 + BN * HD;
    __half* k_sm1 = v_sm0 + BN * HD;
    __half* v_sm1 = k_sm1 + BN * HD;
    float*  s_f32 = (float*)(v_sm1 + BN * HD);     // [BM][BN]
    __half* p_sm  = (__half*)(s_f32 + BM * BN);    // [BM][BN]
    float*  m_sm  = (float*)(p_sm + BM * BN);      // [BM]
    float*  l_sm  = m_sm + BM;                     // [BM]
    float*  alpha_sm = l_sm + BM;                  // [BM]

    __half* k_sm_pp[2] = { k_sm0, k_sm1 };
    __half* v_sm_pp[2] = { v_sm0, v_sm1 };

    // ─── Stage 0a: cooperative load Q → q_sm ───
    // Total half2-chunks = BM * HD/2 = 16 * 128 = 2048. 128 threads × 16 passes
    // = 2048 chunks → каждая thread грузит 1 half2 per pass × 16 passes = 16 half2
    // = 32 F16 elements. (Fix bug: ранее было 4 passes — грузилось только 4 row.)
    {
        for (int pass = 0; pass < 16; ++pass) {
            int linear = pass * WMMA_BLOCK_D + (int)tid;
            int r = linear / (HD / 2);              // 0..15  (treating col by half2)
            int d_h2 = linear % (HD / 2);           // 0..127
            int d = d_h2 * 2;
            __half v0, v1;
            if (r < q_count) {
                size_t off = q_base_offset + (size_t)(q_base + r) * HD + d;
                v0 = q[off];
                v1 = q[off + 1];
            } else {
                v0 = __float2half(0.0f);
                v1 = __float2half(0.0f);
            }
            q_sm[r * HD + d]     = v0;
            q_sm[r * HD + d + 1] = v1;
        }
    }

    // Init m/l per row.
    if (tid < BM) {
        m_sm[tid] = FA_NEG_INF;
        l_sm[tid] = 0.0f;
    }

    // ─── Stage 0b: pre-load K_0, V_0 → slot 0 (async) ───
    // K_tile/V_tile shape (BN, HD) = (32, 256). Total per tile = 32*256 = 8192 F16
    // = 16384 B. cp.async chunk = 16B = 8 F16. Need 16384/16 = 1024 chunks per tile.
    // Distribute over 128 threads × 8 passes = 1024.
    auto issue_kv_load = [&](int kv_block_idx, int slot) {
        int kv_base_local = kv_block_idx * BN;
        __half* k_dst = k_sm_pp[slot];
        __half* v_dst = v_sm_pp[slot];
        for (int pass = 0; pass < 8; ++pass) {
            int chunk = pass * WMMA_BLOCK_D + (int)tid;
            int kv_t_local = chunk / (HD / 8);     // 0..31
            int d_chunk    = chunk % (HD / 8);     // 0..31 (one 8-F16 chunk)
            int d = d_chunk * 8;
            unsigned int k_smem = fa_smem_ptr(k_dst + kv_t_local * HD + d);
            unsigned int v_smem = fa_smem_ptr(v_dst + kv_t_local * HD + d);
            int kv_t = kv_base_local + kv_t_local;
            if ((unsigned)kv_t < T_cache) {
                fa_cp_async_16(k_smem, &k[kv_base_offset + (size_t)kv_t * HD + d]);
                fa_cp_async_16(v_smem, &v[kv_base_offset + (size_t)kv_t * HD + d]);
            } else {
                fa_cp_async_16_zero(k_smem);
                fa_cp_async_16_zero(v_smem);
            }
        }
        fa_cp_async_commit();
    };

    int n_kv_blocks = (int)((T_cache + BN - 1) / BN);
    issue_kv_load(0, 0);
    __syncthreads();

    // ─── Pre-load Q fragments (16 mma_k_step × 4 half2 = 32 dword per thread) ───
    // Q[row=lane/4, col=k_step*16 + 2*(lane%4)..+1]              → a0_h2
    // Q[row=lane/4+8, col=k_step*16 + 2*(lane%4)..+1]            → a1_h2
    // Q[row=lane/4, col=k_step*16 + 2*(lane%4) + 8..+9]          → a2_h2
    // Q[row=lane/4+8, col=k_step*16 + 2*(lane%4) + 8..+9]        → a3_h2
    // (Q_count masked to ≤ q_count rows; out-of-range rows = 0 in shared.)
    unsigned int q_frag[16][4];  // 16 k_step × 4 half2 packs
    {
        int row_lo = lane / 4;          // 0..7
        int row_hi = row_lo + 8;        // 8..15
        int col_lo = (lane % 4) * 2;    // 0,2,4,6
        int col_hi = col_lo + 8;        // 8,10,12,14
        #pragma unroll
        for (int k_step = 0; k_step < 16; ++k_step) {
            int base_k = k_step * 16;
            const __half* p_row_lo = q_sm + row_lo * HD + base_k;
            const __half* p_row_hi = q_sm + row_hi * HD + base_k;
            q_frag[k_step][0] = fa_load_h2_smem(p_row_lo + col_lo);
            q_frag[k_step][1] = fa_load_h2_smem(p_row_hi + col_lo);
            q_frag[k_step][2] = fa_load_h2_smem(p_row_lo + col_hi);
            q_frag[k_step][3] = fa_load_h2_smem(p_row_hi + col_hi);
        }
    }

    // Acc fragment per warp: 8 n_tiles × 4 F32 = 32 F32 acc per thread.
    // n_tile_idx for warp w covers acc cols [w*64 + n_tile_idx*8, +8).
    float acc[8][4];
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[n][r] = 0.0f;
    }

    // ─── Main loop ───
    for (int kv_block = 0; kv_block < n_kv_blocks; ++kv_block) {
        // Prefetch next slot.
        if (kv_block + 1 < n_kv_blocks) {
            issue_kv_load(kv_block + 1, (kv_block + 1) & 1);
            FA_CP_ASYNC_WAIT_GROUP(1);
        } else {
            FA_CP_ASYNC_WAIT_GROUP(0);
        }
        __syncthreads();

        int slot = kv_block & 1;
        __half* k_tile = k_sm_pp[slot];
        __half* v_tile = v_sm_pp[slot];

        int kv_base_local = kv_block * BN;
        int rem = (int)T_cache - kv_base_local;
        int kv_count_local = rem < BN ? rem : BN;

        // ─── Stage 1: S[16][8] = scale * Q @ K^T for warp's column tile ───
        // Per warp: 16 mma_k_step → accumulate S register fragments (4 F32 per thread).
        float s_frag[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        // Per thread: K B-fragment access.
        //   col_n_idx = lane/4  (kv-position within warp's [w*8, w*8+7])
        //   row_k_lo  = 2*(lane%4) ∈ {0,2,4,6}
        //   row_k_hi  = row_k_lo + 8 ∈ {8,10,12,14}
        //   b0_h2 = (K_tile[w*8 + col_n_idx, k_step*16 + row_k_lo],
        //            K_tile[w*8 + col_n_idx, k_step*16 + row_k_lo + 1])  → contiguous d → half2
        //   b1_h2 = same но + 8 в k-направлении.
        int col_n_idx = lane / 4;                    // 0..7
        int row_k_lo  = (lane % 4) * 2;              // 0,2,4,6
        int kv_col    = warp_id * 8 + col_n_idx;     // 0..31

        #pragma unroll
        for (int k_step = 0; k_step < 16; ++k_step) {
            int base_k = k_step * 16;
            const __half* k_row = k_tile + kv_col * HD + base_k;
            unsigned int b0 = fa_load_h2_smem(k_row + row_k_lo);
            unsigned int b1 = fa_load_h2_smem(k_row + row_k_lo + 8);
            float d0, d1, d2, d3;
            fa_mma_m16n8k16(
                d0, d1, d2, d3,
                q_frag[k_step][0], q_frag[k_step][1],
                q_frag[k_step][2], q_frag[k_step][3],
                b0, b1,
                s_frag[0], s_frag[1], s_frag[2], s_frag[3]
            );
            s_frag[0] = d0; s_frag[1] = d1; s_frag[2] = d2; s_frag[3] = d3;
        }

        // Apply scale + masks. S register layout per thread:
        //   c0 = S[row_lo=lane/4,   col=warp_id*8 + 2*(lane%4)]
        //   c1 = S[row_lo,          col=warp_id*8 + 2*(lane%4) + 1]
        //   c2 = S[row_hi=lane/4+8, col=warp_id*8 + 2*(lane%4)]
        //   c3 = S[row_hi,          col=warp_id*8 + 2*(lane%4) + 1]
        {
            int row_lo = lane / 4;
            int row_hi = row_lo + 8;
            int col_a  = warp_id * 8 + (lane % 4) * 2;
            int col_b  = col_a + 1;
            int q_pos_lo = (int)q_pos_base + (int)q_base + row_lo;
            int q_pos_hi = (int)q_pos_base + (int)q_base + row_hi;

            auto apply_mask = [&](float s, int q_row_idx, int q_pos, int kv_col) {
                if (q_row_idx >= q_count) return FA_NEG_INF;
                if (kv_col >= kv_count_local) return FA_NEG_INF;
                if (causal && kv_col + kv_base_local > q_pos) return FA_NEG_INF;
                return s * scale;
            };
            s_frag[0] = apply_mask(s_frag[0], row_lo, q_pos_lo, col_a);
            s_frag[1] = apply_mask(s_frag[1], row_lo, q_pos_lo, col_b);
            s_frag[2] = apply_mask(s_frag[2], row_hi, q_pos_hi, col_a);
            s_frag[3] = apply_mask(s_frag[3], row_hi, q_pos_hi, col_b);
        }

        // Write S register fragments to s_f32[16][32] shared.
        {
            int row_lo = lane / 4;
            int row_hi = row_lo + 8;
            int col_a  = warp_id * 8 + (lane % 4) * 2;
            int col_b  = col_a + 1;
            s_f32[row_lo * BN + col_a] = s_frag[0];
            s_f32[row_lo * BN + col_b] = s_frag[1];
            s_f32[row_hi * BN + col_a] = s_frag[2];
            s_f32[row_hi * BN + col_b] = s_frag[3];
        }
        __syncthreads();

        // ─── Stage 2: warp 0 — row-wise online softmax + cast to F16 ───
        if (warp_id == 0 && lane < BM) {
            int r = lane;
            float row[BN];
            #pragma unroll
            for (int j = 0; j < BN; ++j) row[j] = s_f32[r * BN + j];

            float m_block = FA_NEG_INF;
            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                if (row[j] > m_block) m_block = row[j];
            }

            float m_curr = m_sm[r];
            float m_new = (m_block > m_curr) ? m_block : m_curr;

            float alpha;
            if (!fa_is_finite(m_curr)) {
                alpha = 0.0f;
            } else if (!fa_is_finite(m_new)) {
                alpha = 1.0f;
            } else {
                alpha = expf(m_curr - m_new);
            }

            float row_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                float p;
                if (!fa_is_finite(m_new) || row[j] == FA_NEG_INF) {
                    p = 0.0f;
                } else {
                    p = expf(row[j] - m_new);
                }
                row[j] = p;
                row_sum += p;
            }

            float l_curr = l_sm[r];
            float l_new = l_curr * alpha + row_sum;
            m_sm[r] = m_new;
            l_sm[r] = l_new;
            alpha_sm[r] = alpha;

            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                p_sm[r * BN + j] = __float2half(row[j]);
            }
        }
        __syncthreads();

        // ─── Stage 3a: acc *= alpha[row] ───
        {
            int row_lo = lane / 4;
            int row_hi = row_lo + 8;
            float alpha_lo = alpha_sm[row_lo];
            float alpha_hi = alpha_sm[row_hi];
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                acc[n][0] *= alpha_lo;
                acc[n][1] *= alpha_lo;
                acc[n][2] *= alpha_hi;
                acc[n][3] *= alpha_hi;
            }
        }

        // ─── Stage 3b: acc += P @ V via mma m16n8k16 ───
        // Per warp: 8 n_tiles × 2 mma_k_step = 16 mma.
        // P A-frag per k_step (k=BN=32, 2 k_step of 16):
        //   a0 = (p_sm[row_lo, k_step*16 + col_lo],   p_sm[row_lo, k_step*16 + col_lo + 1])
        //   a1 = (p_sm[row_hi, ...])
        //   a2/a3 = ... + 8 в k.
        // V B-frag per (k_step, n_tile) (n_tile ∈ [warp_id*8, (warp_id+1)*8)):
        //   thread t: col_n = lane/4, row_k_lo = 2*(lane%4)
        //   b0_h2 = (v_tile[k_step*16 + row_k_lo,   n_tile*8 + col_n],
        //            v_tile[k_step*16 + row_k_lo+1, n_tile*8 + col_n])  → strided F16 (по kv).
        //   b1_h2 = same но + 8 в k.
        {
            int row_lo = lane / 4;
            int row_hi = row_lo + 8;
            int col_lo = (lane % 4) * 2;
            int col_hi = col_lo + 8;
            int v_col_n = lane / 4;
            int v_row_k_lo = (lane % 4) * 2;

            #pragma unroll
            for (int k_step = 0; k_step < 2; ++k_step) {
                int base_k = k_step * 16;
                unsigned int a0 = fa_load_h2_smem(p_sm + row_lo * BN + base_k + col_lo);
                unsigned int a1 = fa_load_h2_smem(p_sm + row_hi * BN + base_k + col_lo);
                unsigned int a2 = fa_load_h2_smem(p_sm + row_lo * BN + base_k + col_hi);
                unsigned int a3 = fa_load_h2_smem(p_sm + row_hi * BN + base_k + col_hi);

                #pragma unroll
                for (int n = 0; n < 8; ++n) {
                    int n_col = warp_id * 64 + n * 8 + v_col_n;
                    int k0 = base_k + v_row_k_lo;
                    int k1 = k0 + 1;
                    int k8 = k0 + 8;
                    int k9 = k1 + 8;
                    // Strided F16 loads from v_tile row-major:
                    __half v0 = v_tile[k0 * HD + n_col];
                    __half v1 = v_tile[k1 * HD + n_col];
                    __half v8 = v_tile[k8 * HD + n_col];
                    __half v9 = v_tile[k9 * HD + n_col];
                    unsigned int b0 = fa_pack_h2(v0, v1);
                    unsigned int b1 = fa_pack_h2(v8, v9);

                    float d0, d1, d2, d3;
                    fa_mma_m16n8k16(
                        d0, d1, d2, d3,
                        a0, a1, a2, a3,
                        b0, b1,
                        acc[n][0], acc[n][1], acc[n][2], acc[n][3]
                    );
                    acc[n][0] = d0; acc[n][1] = d1; acc[n][2] = d2; acc[n][3] = d3;
                }
            }
        }
        __syncthreads();   // готовим slot для следующей iter (cp.async write).
    }

    // ─── Epilogue: normalize + write out ───
    // Per thread acc layout:
    //   acc[n][0] = unnormalized out[row_lo, warp_id*64 + n*8 + (lane%4)*2 + 0]
    //   acc[n][1] = ...                           ... + 1
    //   acc[n][2] = ...  out[row_hi, warp_id*64 + n*8 + (lane%4)*2 + 0]
    //   acc[n][3] = ...                           ... + 1
    {
        int row_lo = lane / 4;
        int row_hi = row_lo + 8;
        int col_lo = (lane % 4) * 2;
        int col_hi = col_lo + 1;

        float l_lo = l_sm[row_lo];
        float l_hi = l_sm[row_hi];
        float inv_lo = (l_lo > 0.0f) ? 1.0f / l_lo : 0.0f;
        float inv_hi = (l_hi > 0.0f) ? 1.0f / l_hi : 0.0f;

        bool row_lo_valid = row_lo < q_count;
        bool row_hi_valid = row_hi < q_count;

        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            int d_lo = warp_id * 64 + n * 8 + col_lo;
            int d_hi = warp_id * 64 + n * 8 + col_hi;
            if (row_lo_valid) {
                out[q_base_offset + (size_t)(q_base + row_lo) * HD + d_lo]
                    = __float2half(acc[n][0] * inv_lo);
                out[q_base_offset + (size_t)(q_base + row_lo) * HD + d_hi]
                    = __float2half(acc[n][1] * inv_lo);
            }
            if (row_hi_valid) {
                out[q_base_offset + (size_t)(q_base + row_hi) * HD + d_lo]
                    = __float2half(acc[n][2] * inv_hi);
                out[q_base_offset + (size_t)(q_base + row_hi) * HD + d_hi]
                    = __float2half(acc[n][3] * inv_hi);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase A.2.1: WMMA + Split-K (decode на длинном контексте).
//
// Гибрид WMMA prefill kernel'а и scalar split-K kernel'а:
// - Tile layout, mma.sync, cp.async pipeline, online softmax — те же, что в
//   `flash_attn2_fwd_f16_wmma`.
// - Grid.z = split_id; каждый блок обрабатывает KV[kv_start..kv_end), где
//   split_size = ceil(T_cache/split_k). Цель — устранить underoccupancy
//   на decode T_chunk=1 (B*nh = 32 блока без split, decode 23K ≈ 8% SM
//   occupancy; split_k=16 даёт 512 блоков).
// - Output: НЕ нормализованные partial_acc/partial_m/partial_l, объединяются
//   тем же `flash_attn2_fwd_f16_merge` kernel'ом.
// - hd=256, BLOCK_M=16, BLOCK_KV=32 фиксированы (как в WMMA prefill).
//
// Для decode T_chunk=1 BLOCK_M=16 даёт 15/16 row padding; ничего страшного —
// padded rows получают `m=NEG_INF, l=0, acc=0`, merge корректно их игнорирует
// (corr = 0).
__global__ void flash_attn2_fwd_f16_wmma_split(
    const __half* __restrict__ q,           // (B, nh,  T_chunk, hd)
    const __half* __restrict__ k,           // (B, nkv, *T_stride*, hd)
    const __half* __restrict__ v,           // (B, nkv, *T_stride*, hd)
    float* __restrict__ partial_acc,        // (B*nh*T_chunk*SPLIT_K, hd) F32
    float* __restrict__ partial_m,          // (B*nh*T_chunk*SPLIT_K,)   F32
    float* __restrict__ partial_l,          // (B*nh*T_chunk*SPLIT_K,)   F32
    float scale,
    unsigned int B, unsigned int nh, unsigned int nkv,
    unsigned int T_chunk, unsigned int T_cache, unsigned int hd_param,
    unsigned int n_rep,
    unsigned int q_pos_base,
    int causal,
    unsigned int split_k,
    unsigned int t_stride,  // 0 → = T_cache (backward compat)
    const unsigned int* __restrict__ T_cache_ptr  // Phase D: NULL → immediate.
) {
    constexpr int HD = HD_WMMA;

    unsigned int bh       = blockIdx.x;
    unsigned int b        = bh / nh;
    unsigned int h        = bh % nh;
    unsigned int q_tile   = blockIdx.y;
    unsigned int split_id = blockIdx.z;
    unsigned int tid      = threadIdx.x;
    int warp_id = (int)(tid >> 5);
    int lane    = (int)(tid & 31);

    if (b >= B || split_id >= split_k) return;
    unsigned int q_base = q_tile * BM;
    if (q_base >= T_chunk) return;
    int q_count = (int)((T_chunk - q_base) < BM ? (T_chunk - q_base) : BM);

    if (T_cache_ptr != nullptr) {
        __shared__ unsigned int T_cache_sh;
        if (tid == 0) T_cache_sh = *T_cache_ptr;
        __syncthreads();
        T_cache = T_cache_sh;
    }

    // KV диапазон для этого split.
    unsigned int split_size = (T_cache + split_k - 1) / split_k;
    unsigned int kv_start   = split_id * split_size;
    unsigned int kv_end_unb = kv_start + split_size;
    unsigned int kv_end     = (kv_end_unb < T_cache) ? kv_end_unb : T_cache;

    // Лямбда: записать partial для row r из block'а.
    // partial_idx = (((b*nh + h)*T_chunk + (q_base + r)) * split_k + split_id).
    // Ограничиваем r < q_count, иначе OOB write в соседние (q_base+r) entries.
    auto write_partial_empty = [&]() {
        if (warp_id == 0 && lane < (unsigned)q_count) {
            int r = lane;
            size_t pidx = (((size_t)b * nh + h) * T_chunk + (q_base + r)) * split_k
                        + split_id;
            partial_m[pidx] = FA_NEG_INF;
            partial_l[pidx] = 0.0f;
        }
        for (int r = 0; r < q_count; ++r) {
            size_t pidx = (((size_t)b * nh + h) * T_chunk + (q_base + r)) * split_k
                        + split_id;
            #pragma unroll
            for (int dp = 0; dp < HD / WMMA_BLOCK_D; ++dp) {
                int d = (int)tid + dp * WMMA_BLOCK_D;
                if (d < HD) {
                    partial_acc[pidx * HD + d] = 0.0f;
                }
            }
        }
    };

    // Пустой split → fast-path и выход.
    if (kv_start >= kv_end) {
        write_partial_empty();
        return;
    }

    unsigned int kv_h = h / n_rep;
    unsigned int t_stride_eff = (t_stride > 0) ? t_stride : T_cache;
    size_t kv_base_offset = ((size_t)b * nkv + kv_h) * (size_t)t_stride_eff * HD;

    extern __shared__ unsigned char smem[];
    __half* q_sm  = (__half*)smem;
    __half* k_sm0 = q_sm + BM * HD;
    __half* v_sm0 = k_sm0 + BN * HD;
    __half* k_sm1 = v_sm0 + BN * HD;
    __half* v_sm1 = k_sm1 + BN * HD;
    float*  s_f32 = (float*)(v_sm1 + BN * HD);
    __half* p_sm  = (__half*)(s_f32 + BM * BN);
    float*  m_sm  = (float*)(p_sm + BM * BN);
    float*  l_sm  = m_sm + BM;
    float*  alpha_sm = l_sm + BM;

    __half* k_sm_pp[2] = { k_sm0, k_sm1 };
    __half* v_sm_pp[2] = { v_sm0, v_sm1 };

    // ─── Stage 0a: cooperative load Q → q_sm ───
    {
        size_t q_base_offset = ((size_t)b * nh + h) * (size_t)T_chunk * HD;
        for (int pass = 0; pass < 16; ++pass) {
            int linear = pass * WMMA_BLOCK_D + (int)tid;
            int r = linear / (HD / 2);
            int d_h2 = linear % (HD / 2);
            int d = d_h2 * 2;
            __half v0, v1;
            if (r < q_count) {
                size_t off = q_base_offset + (size_t)(q_base + r) * HD + d;
                v0 = q[off];
                v1 = q[off + 1];
            } else {
                v0 = __float2half(0.0f);
                v1 = __float2half(0.0f);
            }
            q_sm[r * HD + d]     = v0;
            q_sm[r * HD + d + 1] = v1;
        }
    }

    if (tid < BM) {
        m_sm[tid] = FA_NEG_INF;
        l_sm[tid] = 0.0f;
    }

    // ─── Stage 0b: pre-load K_0, V_0 → slot 0 (async) ───
    // Базовый KV-tile теперь относителен kv_start.
    auto issue_kv_load = [&](int kv_block_idx, int slot) {
        int kv_base_local = (int)kv_start + kv_block_idx * BN;
        __half* k_dst = k_sm_pp[slot];
        __half* v_dst = v_sm_pp[slot];
        for (int pass = 0; pass < 8; ++pass) {
            int chunk = pass * WMMA_BLOCK_D + (int)tid;
            int kv_t_local = chunk / (HD / 8);
            int d_chunk    = chunk % (HD / 8);
            int d = d_chunk * 8;
            unsigned int k_smem = fa_smem_ptr(k_dst + kv_t_local * HD + d);
            unsigned int v_smem = fa_smem_ptr(v_dst + kv_t_local * HD + d);
            int kv_t = kv_base_local + kv_t_local;
            // Mask по верхней границе kv_end (а не T_cache): кv_t ≥ kv_end → zero,
            // т.к. за пределами split'а нет данных для этого block'а.
            if ((unsigned)kv_t < kv_end) {
                fa_cp_async_16(k_smem, &k[kv_base_offset + (size_t)kv_t * HD + d]);
                fa_cp_async_16(v_smem, &v[kv_base_offset + (size_t)kv_t * HD + d]);
            } else {
                fa_cp_async_16_zero(k_smem);
                fa_cp_async_16_zero(v_smem);
            }
        }
        fa_cp_async_commit();
    };

    int n_kv_blocks = (int)((kv_end - kv_start + BN - 1) / BN);
    issue_kv_load(0, 0);
    __syncthreads();

    // ─── Pre-load Q fragments ───
    unsigned int q_frag[16][4];
    {
        int row_lo = lane / 4;
        int row_hi = row_lo + 8;
        int col_lo = (lane % 4) * 2;
        int col_hi = col_lo + 8;
        #pragma unroll
        for (int k_step = 0; k_step < 16; ++k_step) {
            int base_k = k_step * 16;
            const __half* p_row_lo = q_sm + row_lo * HD + base_k;
            const __half* p_row_hi = q_sm + row_hi * HD + base_k;
            q_frag[k_step][0] = fa_load_h2_smem(p_row_lo + col_lo);
            q_frag[k_step][1] = fa_load_h2_smem(p_row_hi + col_lo);
            q_frag[k_step][2] = fa_load_h2_smem(p_row_lo + col_hi);
            q_frag[k_step][3] = fa_load_h2_smem(p_row_hi + col_hi);
        }
    }

    float acc[8][4];
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[n][r] = 0.0f;
    }

    // ─── Main loop ───
    for (int kv_block = 0; kv_block < n_kv_blocks; ++kv_block) {
        if (kv_block + 1 < n_kv_blocks) {
            issue_kv_load(kv_block + 1, (kv_block + 1) & 1);
            FA_CP_ASYNC_WAIT_GROUP(1);
        } else {
            FA_CP_ASYNC_WAIT_GROUP(0);
        }
        __syncthreads();

        int slot = kv_block & 1;
        __half* k_tile = k_sm_pp[slot];
        __half* v_tile = v_sm_pp[slot];

        int kv_base_local = (int)kv_start + kv_block * BN;
        int rem = (int)kv_end - kv_base_local;
        int kv_count_local = rem < BN ? rem : BN;

        // ─── Stage 1: S[16][8] = scale * Q @ K^T ───
        float s_frag[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        int col_n_idx = lane / 4;
        int row_k_lo  = (lane % 4) * 2;
        int kv_col    = warp_id * 8 + col_n_idx;

        #pragma unroll
        for (int k_step = 0; k_step < 16; ++k_step) {
            int base_k = k_step * 16;
            const __half* k_row = k_tile + kv_col * HD + base_k;
            unsigned int b0 = fa_load_h2_smem(k_row + row_k_lo);
            unsigned int b1 = fa_load_h2_smem(k_row + row_k_lo + 8);
            float d0, d1, d2, d3;
            fa_mma_m16n8k16(
                d0, d1, d2, d3,
                q_frag[k_step][0], q_frag[k_step][1],
                q_frag[k_step][2], q_frag[k_step][3],
                b0, b1,
                s_frag[0], s_frag[1], s_frag[2], s_frag[3]
            );
            s_frag[0] = d0; s_frag[1] = d1; s_frag[2] = d2; s_frag[3] = d3;
        }

        // Apply scale + masks.
        {
            int row_lo = lane / 4;
            int row_hi = row_lo + 8;
            int col_a  = warp_id * 8 + (lane % 4) * 2;
            int col_b  = col_a + 1;
            int q_pos_lo = (int)q_pos_base + (int)q_base + row_lo;
            int q_pos_hi = (int)q_pos_base + (int)q_base + row_hi;

            auto apply_mask = [&](float s, int q_row_idx, int q_pos, int kv_col_idx) {
                if (q_row_idx >= q_count) return FA_NEG_INF;
                if (kv_col_idx >= kv_count_local) return FA_NEG_INF;
                int kv_t = kv_base_local + kv_col_idx;
                if (causal && kv_t > q_pos) return FA_NEG_INF;
                return s * scale;
            };
            s_frag[0] = apply_mask(s_frag[0], row_lo, q_pos_lo, col_a);
            s_frag[1] = apply_mask(s_frag[1], row_lo, q_pos_lo, col_b);
            s_frag[2] = apply_mask(s_frag[2], row_hi, q_pos_hi, col_a);
            s_frag[3] = apply_mask(s_frag[3], row_hi, q_pos_hi, col_b);
        }

        {
            int row_lo = lane / 4;
            int row_hi = row_lo + 8;
            int col_a  = warp_id * 8 + (lane % 4) * 2;
            int col_b  = col_a + 1;
            s_f32[row_lo * BN + col_a] = s_frag[0];
            s_f32[row_lo * BN + col_b] = s_frag[1];
            s_f32[row_hi * BN + col_a] = s_frag[2];
            s_f32[row_hi * BN + col_b] = s_frag[3];
        }
        __syncthreads();

        // ─── Stage 2: warp 0 row-wise online softmax ───
        if (warp_id == 0 && lane < BM) {
            int r = lane;
            float row[BN];
            #pragma unroll
            for (int j = 0; j < BN; ++j) row[j] = s_f32[r * BN + j];

            float m_block = FA_NEG_INF;
            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                if (row[j] > m_block) m_block = row[j];
            }

            float m_curr = m_sm[r];
            float m_new = (m_block > m_curr) ? m_block : m_curr;

            float alpha;
            if (!fa_is_finite(m_curr)) {
                alpha = 0.0f;
            } else if (!fa_is_finite(m_new)) {
                alpha = 1.0f;
            } else {
                alpha = expf(m_curr - m_new);
            }

            float row_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                float p;
                if (!fa_is_finite(m_new) || row[j] == FA_NEG_INF) {
                    p = 0.0f;
                } else {
                    p = expf(row[j] - m_new);
                }
                row[j] = p;
                row_sum += p;
            }

            float l_curr = l_sm[r];
            float l_new = l_curr * alpha + row_sum;
            m_sm[r] = m_new;
            l_sm[r] = l_new;
            alpha_sm[r] = alpha;

            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                p_sm[r * BN + j] = __float2half(row[j]);
            }
        }
        __syncthreads();

        // ─── Stage 3a: acc *= alpha[row] ───
        {
            int row_lo = lane / 4;
            int row_hi = row_lo + 8;
            float alpha_lo = alpha_sm[row_lo];
            float alpha_hi = alpha_sm[row_hi];
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                acc[n][0] *= alpha_lo;
                acc[n][1] *= alpha_lo;
                acc[n][2] *= alpha_hi;
                acc[n][3] *= alpha_hi;
            }
        }

        // ─── Stage 3b: acc += P @ V via mma ───
        {
            int row_lo = lane / 4;
            int row_hi = row_lo + 8;
            int col_lo = (lane % 4) * 2;
            int col_hi = col_lo + 8;
            int v_col_n = lane / 4;
            int v_row_k_lo = (lane % 4) * 2;

            #pragma unroll
            for (int k_step = 0; k_step < 2; ++k_step) {
                int base_k = k_step * 16;
                unsigned int a0 = fa_load_h2_smem(p_sm + row_lo * BN + base_k + col_lo);
                unsigned int a1 = fa_load_h2_smem(p_sm + row_hi * BN + base_k + col_lo);
                unsigned int a2 = fa_load_h2_smem(p_sm + row_lo * BN + base_k + col_hi);
                unsigned int a3 = fa_load_h2_smem(p_sm + row_hi * BN + base_k + col_hi);

                #pragma unroll
                for (int n = 0; n < 8; ++n) {
                    int n_col = warp_id * 64 + n * 8 + v_col_n;
                    int k0 = base_k + v_row_k_lo;
                    int k1 = k0 + 1;
                    int k8 = k0 + 8;
                    int k9 = k1 + 8;
                    __half v0 = v_tile[k0 * HD + n_col];
                    __half v1 = v_tile[k1 * HD + n_col];
                    __half v8 = v_tile[k8 * HD + n_col];
                    __half v9 = v_tile[k9 * HD + n_col];
                    unsigned int b0 = fa_pack_h2(v0, v1);
                    unsigned int b1 = fa_pack_h2(v8, v9);

                    float d0, d1, d2, d3;
                    fa_mma_m16n8k16(
                        d0, d1, d2, d3,
                        a0, a1, a2, a3,
                        b0, b1,
                        acc[n][0], acc[n][1], acc[n][2], acc[n][3]
                    );
                    acc[n][0] = d0; acc[n][1] = d1; acc[n][2] = d2; acc[n][3] = d3;
                }
            }
        }
        __syncthreads();
    }

    // ─── Epilogue: записать НЕ нормализованные partials ───
    // CRITICAL: ограничиваем запись по `row_lo/row_hi < q_count`. Иначе threads
    // с (lane/4) ≥ q_count корраптят соседние (q_base+row) entries в global
    // memory (особенно критично для decode T_chunk=1, где valid только row 0).
    {
        int row_lo = lane / 4;
        int row_hi = row_lo + 8;
        int col_lo = (lane % 4) * 2;
        int col_hi = col_lo + 1;
        bool row_lo_valid = row_lo < q_count;
        bool row_hi_valid = row_hi < q_count;

        size_t pidx_lo = (((size_t)b * nh + h) * T_chunk + (q_base + row_lo))
                       * split_k + split_id;
        size_t pidx_hi = (((size_t)b * nh + h) * T_chunk + (q_base + row_hi))
                       * split_k + split_id;

        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            int d_lo = warp_id * 64 + n * 8 + col_lo;
            int d_hi = warp_id * 64 + n * 8 + col_hi;
            if (row_lo_valid) {
                partial_acc[pidx_lo * HD + d_lo] = acc[n][0];
                partial_acc[pidx_lo * HD + d_hi] = acc[n][1];
            }
            if (row_hi_valid) {
                partial_acc[pidx_hi * HD + d_lo] = acc[n][2];
                partial_acc[pidx_hi * HD + d_hi] = acc[n][3];
            }
        }
    }

    // m/l: warp 0, threads 0..q_count-1 пишут m/l для real rows.
    if (warp_id == 0 && lane < (unsigned)q_count) {
        int r = lane;
        size_t pidx = (((size_t)b * nh + h) * T_chunk + (q_base + r)) * split_k
                    + split_id;
        partial_m[pidx] = m_sm[r];
        partial_l[pidx] = l_sm[r];
    }
}

} // extern "C"
