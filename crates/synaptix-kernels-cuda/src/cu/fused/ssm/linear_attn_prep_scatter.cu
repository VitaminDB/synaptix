#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Chunk-prefill версия `linear_attn_prep_fused` (T≥1) для GatedDeltaNet.
// Семантика идентична host-loop'у в synaptix-models LinearAttn::forward
// (model.rs:879-907) + gated_delta_decay_beta:
//
//   conv_out: [T, conv_dim] = [T, Q | K | V] row-major (time-major, после
//             causal_conv1d_chunk + SiLU).
//   key_dim  = num_k * hk  (Q-блок занимает первые key_dim каналов,
//                           K-блок — key_dim..2*key_dim,
//                           V-блок — 2*key_dim..conv_dim).
//   n_rep    = num_v / num_k  (GQA repeat для Q/K).
//
//   Для каждого (hi, t), hi∈[0, num_v), t∈[0, T):
//     h_in = hi / n_rep
//     qe[(hi*T + t)*hk + d] = conv_out[t*conv_dim + h_in*hk + d]
//     ke[(hi*T + t)*hk + d] = conv_out[t*conv_dim + key_dim + h_in*hk + d]
//     vv[(hi*T + t)*hv + d] = conv_out[t*conv_dim + 2*key_dim + hi*hv + d]
//
//   Для каждого (t, hi):
//     beta[hi*T + t] = sigmoid(b[t*num_v + hi])
//     g[hi*T + t]    = -exp(a_log[hi]) * softplus(a[t*num_v + hi] + dt_bias[hi])
//     softplus(x) = log(1 + exp(x))
//
// Layout совпадает с тем, что ожидает chunk_gated_delta_rule (BH=h_v, T, HK=hk).
// Все выходы — F32 (как в decode-prep_fused), вход conv_out — параметризован
// (F32/F16/BF16); a/b — F16 (как в decode пути); dt_bias/a_log — F32.
//
// Grid layout:
//   grid = (num_v, T, 4)
//   block = (max(hk, hv, num_v), 1, 1)
//   blockIdx.z (region):
//     0 — beta/g compute (только blockIdx.x == 0, threadIdx.x пробегает num_v)
//     1 — Q scatter:  blockIdx.x = hi, threadIdx.x = d ∈ [0, hk)
//     2 — K scatter:  то же + offset key_dim
//     3 — V scatter:  blockIdx.x = hi, threadIdx.x = d ∈ [0, hv)

__device__ __forceinline__ float ps_ld(const float* p) { return *p; }
__device__ __forceinline__ float ps_ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ps_ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

// T_in: число «реальных» токенов (диапазон чтения для conv_out/a/b).
// T_out: stride записи в qe/ke/vv/g/beta (T_out ≥ T_in). Для prefill с
// padding до chunk_size: T_out = next_multiple(T_in, cs). Threads с t ≥ T_in
// ничего не делают — соответствующие позиции выходов остаются нулями (caller
// аллоцирует через alloc_zeros).
template <typename T>
__device__ __forceinline__ void prep_scatter_impl(
    const __half* __restrict__ b_f16,         // (T_in, num_v)
    const __half* __restrict__ a_f16,         // (T_in, num_v)
    const float*  __restrict__ dt_bias_f32,   // (num_v,)
    const float*  __restrict__ a_log_f32,     // (num_v,)
    float*        __restrict__ beta_f32,      // (num_v, T_out)
    float*        __restrict__ g_f32,         // (num_v, T_out)
    const T*      __restrict__ conv_out,      // (T_in, conv_dim)
    float*        __restrict__ q_f32,         // (num_v, T_out, hk)
    float*        __restrict__ k_f32,         // (num_v, T_out, hk)
    float*        __restrict__ v_f32,         // (num_v, T_out, hv)
    unsigned int T_in,
    unsigned int T_out,
    unsigned int num_v,
    unsigned int n_rep,
    unsigned int hk,
    unsigned int hv,
    unsigned int key_dim
) {
    unsigned int region = blockIdx.z;
    unsigned int t = blockIdx.y;
    if (t >= T_in) return;
    unsigned int conv_dim = 2u * key_dim + num_v * hv;

    if (region == 0u) {
        if (blockIdx.x != 0u) return;
        unsigned int hi = threadIdx.x;
        if (hi >= num_v) return;
        float bv = __half2float(b_f16[t * num_v + hi]);
        beta_f32[hi * T_out + t] = 1.0f / (1.0f + __expf(-bv));
        float av  = __half2float(a_f16[t * num_v + hi]);
        float dt  = av + dt_bias_f32[hi];
        float softplus_dt   = __logf(1.0f + __expf(dt));
        float a_log_neg_exp = -__expf(a_log_f32[hi]);
        g_f32[hi * T_out + t] = softplus_dt * a_log_neg_exp;
        return;
    }

    unsigned int hi = blockIdx.x;
    if (hi >= num_v) return;
    unsigned int d = threadIdx.x;
    unsigned long row_base = (unsigned long)t * conv_dim;

    if (region == 1u) {
        if (d >= hk) return;
        unsigned int h_in = hi / n_rep;
        q_f32[((unsigned long)hi * T_out + t) * hk + d] =
            ps_ld(conv_out + row_base + (unsigned long)h_in * hk + d);
        return;
    }
    if (region == 2u) {
        if (d >= hk) return;
        unsigned int h_in = hi / n_rep;
        k_f32[((unsigned long)hi * T_out + t) * hk + d] =
            ps_ld(conv_out + row_base + (unsigned long)key_dim + (unsigned long)h_in * hk + d);
        return;
    }
    // region == 3 (V scatter)
    if (d >= hv) return;
    v_f32[((unsigned long)hi * T_out + t) * hv + d] =
        ps_ld(conv_out + row_base + (unsigned long)(2u * key_dim) + (unsigned long)hi * hv + d);
}

extern "C" __global__ void linear_attn_prep_scatter_f16(
    const __half* b_f16, const __half* a_f16,
    const float* dt_bias_f32, const float* a_log_f32,
    float* beta_f32, float* g_f32,
    const __half* conv_out_f16,
    float* q_f32, float* k_f32, float* v_f32,
    unsigned int T_in, unsigned int T_out, unsigned int num_v, unsigned int n_rep,
    unsigned int hk, unsigned int hv, unsigned int key_dim) {
    prep_scatter_impl<__half>(
        b_f16, a_f16, dt_bias_f32, a_log_f32, beta_f32, g_f32, conv_out_f16,
        q_f32, k_f32, v_f32, T_in, T_out, num_v, n_rep, hk, hv, key_dim
    );
}

extern "C" __global__ void linear_attn_prep_scatter_bf16(
    const __half* b_f16, const __half* a_f16,
    const float* dt_bias_f32, const float* a_log_f32,
    float* beta_f32, float* g_f32,
    const __nv_bfloat16* conv_out_bf16,
    float* q_f32, float* k_f32, float* v_f32,
    unsigned int T_in, unsigned int T_out, unsigned int num_v, unsigned int n_rep,
    unsigned int hk, unsigned int hv, unsigned int key_dim) {
    prep_scatter_impl<__nv_bfloat16>(
        b_f16, a_f16, dt_bias_f32, a_log_f32, beta_f32, g_f32, conv_out_bf16,
        q_f32, k_f32, v_f32, T_in, T_out, num_v, n_rep, hk, hv, key_dim
    );
}

extern "C" __global__ void linear_attn_prep_scatter_f32(
    const __half* b_f16, const __half* a_f16,
    const float* dt_bias_f32, const float* a_log_f32,
    float* beta_f32, float* g_f32,
    const float* conv_out_f32,
    float* q_f32, float* k_f32, float* v_f32,
    unsigned int T_in, unsigned int T_out, unsigned int num_v, unsigned int n_rep,
    unsigned int hk, unsigned int hv, unsigned int key_dim) {
    prep_scatter_impl<float>(
        b_f16, a_f16, dt_bias_f32, a_log_f32, beta_f32, g_f32, conv_out_f32,
        q_f32, k_f32, v_f32, T_in, T_out, num_v, n_rep, hk, hv, key_dim
    );
}
