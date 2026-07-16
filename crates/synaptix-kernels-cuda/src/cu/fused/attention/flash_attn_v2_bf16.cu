// FlashAttention-2 forward kernel — BF16 native variant.
//
// Алгоритмически идентичен `flash_attn2_fwd_f16` (см. `flash_attn.cu`):
// F32 accumulator + online softmax + KV-tiling + GQA inside kernel.
// Различие: input/output dtype = BF16 (`__nv_bfloat16`), без cast в F16.
//
// Зачем: ACE-Step AR LM (`acestep-qwen-lm`) хранит LM-веса в BF16. Cast в
// F16 для существующего FA-2 kernel'а теряет ~3 ULP precision (BF16 ─ 1+8+7,
// F16 ─ 1+5+10), и через 600+ steps AR generation накапливается до argmax
// drift'а. PyTorch SDPA на Ampere+ использует Dao FlashAttention-2 BF16
// native kernel. Это наш port в стиле существующих kernels.
//
// Параметризовано теми же compile defines (BLOCK_M, BLOCK_KV, BLOCK_D).
// Включаемый файл `flash_attn.cu` ДОЛЖЕН быть включён ранее в NVRTC-source —
// мы переиспользуем его macros и `fa_is_finite` helper. Если файл compile'ит-
// ся standalone — раскомментировать commented header block ниже.

#include <cuda_bf16.h>

// Если файл компилируется отдельно (не объединён с `flash_attn.cu`):
// #include <cuda_fp16.h>
// #define FA_POS_INF (__int_as_float(0x7F800000))
// #define FA_NEG_INF (__int_as_float(0xFF800000))
// #define BLOCK_D 128
// #define BLOCK_KV 64
// #define D_PER_THREAD_MAX 4
// #define D_PER_LANE_MAX 8
// __device__ __forceinline__ bool fa_is_finite(float x) {
//     return (__float_as_int(x) & 0x7F800000) != 0x7F800000;
// }
// #ifndef BLOCK_M
// #define BLOCK_M 32
// #endif
// #define ROWS_PER_WARP (BLOCK_M / 4)

extern "C" {

// ────────────── Single-row BF16 kernel (для decode T_chunk=1) ──────────────
__global__ void flash_attn2_fwd_bf16(
    const __nv_bfloat16* __restrict__ q,    // (B, nh,  T_chunk, hd)
    const __nv_bfloat16* __restrict__ k,    // (B, nkv, *T_stride*, hd)
    const __nv_bfloat16* __restrict__ v,    // (B, nkv, *T_stride*, hd)
    __nv_bfloat16* __restrict__       out,  // (B, nh,  T_chunk, hd)
    float scale,
    unsigned int B, unsigned int nh, unsigned int nkv,
    unsigned int T_chunk, unsigned int T_cache, unsigned int hd,
    unsigned int n_rep,
    unsigned int q_pos_base,
    int causal,
    unsigned int t_stride,
    const unsigned int* __restrict__ T_cache_ptr
) {
    unsigned int bh    = blockIdx.x;
    unsigned int b     = bh / nh;
    unsigned int h     = bh % nh;
    unsigned int q_row = blockIdx.y;
    unsigned int tid   = threadIdx.x;

    if (b >= B || q_row >= T_chunk) return;

    if (T_cache_ptr != nullptr) {
        __shared__ unsigned int T_cache_sh;
        if (tid == 0) T_cache_sh = *T_cache_ptr;
        __syncthreads();
        T_cache = T_cache_sh;
    }

    unsigned int kv_h = h / n_rep;
    unsigned int d_per_thread = hd / BLOCK_D;
    unsigned int n_d_chunks   = hd / 32;

    extern __shared__ unsigned char shm_bf16[];
    __nv_bfloat16* q_sm = (__nv_bfloat16*)shm_bf16;
    float*  s_sm = (float*)(q_sm + hd);
    float*  meta = s_sm + BLOCK_KV;

    // 1. Загрузить Q-row.
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
    int n_kv_blocks = (int)((T_cache + BLOCK_KV - 1) / BLOCK_KV);

    unsigned int t_stride_eff = (t_stride > 0) ? t_stride : T_cache;
    size_t kv_base_offset =
        ((size_t)b * nkv + kv_h) * (size_t)t_stride_eff * hd;

    int warp_id = (int)(tid >> 5);
    int lane    = (int)(tid & 31);

    for (int kv_block = 0; kv_block < n_kv_blocks; ++kv_block) {
        int kv_base = kv_block * BLOCK_KV;
        int rem = (int)T_cache - kv_base;
        int kv_count = rem < BLOCK_KV ? rem : BLOCK_KV;

        // Stage 1: scores.
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
                    float qv = __bfloat162float(q_sm[d]);
                    float kv = __bfloat162float(__ldg(
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
                if (j >= kv_count) {
                    s = FA_NEG_INF;
                } else if (causal && kv_t > (int)q_pos_global) {
                    s = FA_NEG_INF;
                }
                s_sm[j] = s;
            }
        }
        __syncthreads();

        // Stage 2: online softmax (warp 0).
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

        // Stage 3: accumulate P * V.
        #pragma unroll D_PER_THREAD_MAX
        for (int dp = 0; dp < D_PER_THREAD_MAX; ++dp) {
            if ((unsigned int)dp >= d_per_thread) break;
            unsigned int d = tid + (unsigned int)dp * BLOCK_D;
            acc[dp] *= alpha;
            #pragma unroll 8
            for (int j = 0; j < BLOCK_KV; ++j) {
                if (j < kv_count) {
                    int kv_t = kv_base + j;
                    float vv = __bfloat162float(__ldg(
                        &v[kv_base_offset + (size_t)kv_t * hd + d]));
                    acc[dp] += s_sm[j] * vv;
                }
            }
        }

        m_curr = m_new;
        l_curr = l_new;
        __syncthreads();
    }

    // Stage 4: write.
    #pragma unroll D_PER_THREAD_MAX
    for (int dp = 0; dp < D_PER_THREAD_MAX; ++dp) {
        if ((unsigned int)dp >= d_per_thread) break;
        unsigned int d = tid + (unsigned int)dp * BLOCK_D;
        if (d < hd) {
            float result = (l_curr > 0.0f) ? (acc[dp] / l_curr) : 0.0f;
            out[((size_t)((b * nh) + h) * T_chunk + q_row) * hd + d]
                = __float2bfloat16_rn(result);
        }
    }
}

// ────────────── Tiled BF16 kernel (для prefill T_chunk > 1) ──────────────
__global__ void flash_attn2_fwd_bf16_tiled(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__       out,
    float scale,
    unsigned int B, unsigned int nh, unsigned int nkv,
    unsigned int T_chunk, unsigned int T_cache, unsigned int hd,
    unsigned int n_rep,
    unsigned int q_pos_base,
    int causal,
    unsigned int t_stride,
    const unsigned int* __restrict__ T_cache_ptr
) {
    unsigned int bh     = blockIdx.x;
    unsigned int b      = bh / nh;
    unsigned int h      = bh % nh;
    unsigned int q_tile = blockIdx.y;
    unsigned int tid    = threadIdx.x;
    int warp_id = (int)(tid >> 5);
    int lane    = (int)(tid & 31);

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
    unsigned int d_per_lane = hd / 32;

    extern __shared__ unsigned char shm_bf16_tiled[];
    __nv_bfloat16* q_sm    = (__nv_bfloat16*)shm_bf16_tiled;
    float*  s_sm    = (float*)(q_sm + (size_t)BLOCK_M * hd);
    float*  m_sm    = s_sm + (size_t)BLOCK_M * BLOCK_KV;
    float*  l_sm    = m_sm + BLOCK_M;
    float*  alpha_sm = l_sm + BLOCK_M;

    // Stage 0: load Q.
    {
        unsigned int q_total = BLOCK_M * hd;
        for (unsigned int i = tid; i < q_total; i += BLOCK_D) {
            unsigned int r = i / hd;
            unsigned int d = i - r * hd;
            if (r < q_count) {
                q_sm[i] = q[((size_t)((b * nh) + h) * T_chunk + (q_base + r)) * hd + d];
            } else {
                q_sm[i] = __float2bfloat16_rn(0.0f);
            }
        }
    }
    if (tid < BLOCK_M) {
        m_sm[tid] = FA_NEG_INF;
        l_sm[tid] = 0.0f;
    }
    __syncthreads();

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

        // Stage 1: S[r, j].
        #pragma unroll 4
        for (int j = 0; j < BLOCK_KV; ++j) {
            int kv_t = kv_base + j;
            float k_vals[D_PER_LANE_MAX];
            if (j < kv_count) {
                #pragma unroll
                for (int dp = 0; dp < D_PER_LANE_MAX; ++dp) {
                    if ((unsigned int)dp >= d_per_lane) break;
                    int d = dp * 32 + lane;
                    k_vals[dp] = __bfloat162float(__ldg(
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
                        float qv = __bfloat162float(q_sm[r_global * hd + d]);
                        partial += qv * k_vals[dp];
                    }
                }
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

        // Stage 2: online softmax per row.
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
            float m_curr_r = m_sm[r_global];
            float m_new = (m_block > m_curr_r) ? m_block : m_curr_r;

            float alpha;
            if (!fa_is_finite(m_curr_r)) {
                alpha = 0.0f;
            } else if (!fa_is_finite(m_new)) {
                alpha = 1.0f;
            } else {
                alpha = expf(m_curr_r - m_new);
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

            float l_curr_r = l_sm[r_global];
            float l_new = l_curr_r * alpha + row_sum;
            if (lane == 0) {
                m_sm[r_global]     = m_new;
                l_sm[r_global]     = l_new;
                alpha_sm[r_global] = alpha;
            }
        }
        __syncthreads();

        // Stage 3: acc[r, d] = acc * alpha + sum_j P[r,j] * V[j, d].
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

        #pragma unroll 4
        for (int j = 0; j < BLOCK_KV; ++j) {
            if (j >= kv_count) break;
            int kv_t = kv_base + j;
            float v_vals[D_PER_LANE_MAX];
            #pragma unroll
            for (int dp = 0; dp < D_PER_LANE_MAX; ++dp) {
                if ((unsigned int)dp >= d_per_lane) break;
                int d = dp * 32 + lane;
                v_vals[dp] = __bfloat162float(__ldg(
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

    // Stage 4: write.
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
                = __float2bfloat16_rn(result);
        }
    }
}

}  // extern "C"
