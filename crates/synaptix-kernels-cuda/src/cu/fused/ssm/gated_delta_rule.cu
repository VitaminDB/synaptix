// Gated DeltaNet рекуррентный шаг (decode T=1) + fused RmsNormGated вариант.
//
// Портировано из ai-quant/src/kernels/cu_kernels/chunk_fla.cu (валидировано
// bit-exact в проде Qwen3.6). Один block = одна (batch, head) пара,
// block_dim = hk threads, каждый thread держит один key-канал.
//
// Рекуррентность (gated delta rule):
//   q, k нормализуются L2 по hk; q масштабируется на q_scale.
//   g_t = exp(g[b,h]); beta_t = beta[b,h].
//   Для каждого v-канала vi:
//     st       = state[k, vi] * g_t              (decay)
//     kv_mem   = Σ_k st * k_k                     ((g·S)^T k)
//     delta    = (v[vi] - kv_mem) * beta_t
//     state[k, vi] = st + k_k * delta
//     out[vi]  = Σ_k state[k, vi] * q_k
//
// Семантически: S_t = g_t·S_{t-1} + β_t·k_t·(v_t − (g_t·S_{t-1})ᵀk_t)ᵀ;  o_t = S_tᵀ·q_t.

#include <cuda_fp16.h>

extern "C" __global__ void gated_delta_rule_step_f32(
    const float* __restrict__ q,        // (B, H, HK)
    const float* __restrict__ k,        // (B, H, HK)
    const float* __restrict__ v,        // (B, H, HV)
    const float* __restrict__ g,        // (B, H)
    const float* __restrict__ beta,     // (B, H)
    float* __restrict__ state,          // (B, H, HK, HV) in/out
    float* __restrict__ out,            // (B, H, HV)
    float q_scale,
    unsigned int B,
    unsigned int H,
    unsigned int hk,
    unsigned int hv
) {
    unsigned int bi = blockIdx.x;
    unsigned int hi = blockIdx.y;
    if (bi >= B || hi >= H) return;
    unsigned int tid = threadIdx.x;
    if (tid >= hk) return;

    extern __shared__ float shm[];
    float* q_sm      = shm;            // [HK]
    float* k_sm      = shm + hk;       // [HK]
    float* reduce_sm = shm + 2 * hk;   // [HK]
    float* scalars_sm = shm + 3 * hk;  // [4]

    unsigned long long base_qk = (unsigned long long)bi * H * hk + (unsigned long long)hi * hk;
    unsigned long long base_v = (unsigned long long)bi * H * hv + (unsigned long long)hi * hv;
    unsigned long long base_state = (unsigned long long)bi * H * hk * hv + (unsigned long long)hi * hk * hv;

    float qv = q[base_qk + tid];
    float kv = k[base_qk + tid];

    float q_sq = qv * qv;
    float k_sq = kv * kv;
    reduce_sm[tid] = q_sq;
    __syncthreads();
    for (unsigned int s = hk / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_sm[tid] += reduce_sm[tid + s];
        __syncthreads();
    }
    float sum_q_sq = reduce_sm[0];
    __syncthreads();

    reduce_sm[tid] = k_sq;
    __syncthreads();
    for (unsigned int s = hk / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_sm[tid] += reduce_sm[tid + s];
        __syncthreads();
    }
    float sum_k_sq = reduce_sm[0];
    __syncthreads();

    float inv_l2_q = rsqrtf(sum_q_sq + 1e-6f);
    float inv_l2_k = rsqrtf(sum_k_sq + 1e-6f);
    q_sm[tid] = qv * inv_l2_q * q_scale;
    k_sm[tid] = kv * inv_l2_k;

    if (tid == 0) {
        scalars_sm[0] = __expf(g[bi * H + hi]);  // g_t
        scalars_sm[1] = beta[bi * H + hi];        // beta_t
    }
    __syncthreads();
    float g_t = scalars_sm[0];
    float beta_t = scalars_sm[1];

    float q_k = q_sm[tid];
    float k_k = k_sm[tid];

    for (unsigned int vi = 0; vi < hv; ++vi) {
        unsigned long long state_idx = base_state + (unsigned long long)tid * hv + vi;
        float st = state[state_idx] * g_t;

        reduce_sm[tid] = st * k_k;
        __syncthreads();
        for (unsigned int s = hk / 2; s > 0; s >>= 1) {
            if (tid < s) reduce_sm[tid] += reduce_sm[tid + s];
            __syncthreads();
        }
        float kv_mem = reduce_sm[0];
        __syncthreads();

        if (tid == 0) {
            float vt = v[base_v + vi];
            scalars_sm[2] = (vt - kv_mem) * beta_t;  // delta
        }
        __syncthreads();
        float delta = scalars_sm[2];

        float new_st = st + k_k * delta;
        state[state_idx] = new_st;

        reduce_sm[tid] = new_st * q_k;
        __syncthreads();
        for (unsigned int s = hk / 2; s > 0; s >>= 1) {
            if (tid < s) reduce_sm[tid] += reduce_sm[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            out[base_v + vi] = reduce_sm[0];
        }
        __syncthreads();
    }
}

// ───────────── gated_delta_rule_step + RmsNormGated fused ───────────────────
//
// Fused версия + rms_norm_gated. SSM-выход (F32) идёт в shared memory вместо
// global, RMS-фаза в том же block:
//   out_f16 = weight * x / sqrt(mean(x²) + eps) * silu(gate_f16)
// Требование: hk == hv (block layout совпадает). На Qwen3.6 — true (128==128).
//
// Shared: (3*HK + HV + 4) F32.

extern "C" __global__ void gated_delta_rule_step_fused_rms_norm_f32_to_f16(
    const float*  __restrict__ q,            // (B, H, HK)
    const float*  __restrict__ k,            // (B, H, HK)
    const float*  __restrict__ v,            // (B, H, HV)
    const float*  __restrict__ g,            // (B, H)
    const float*  __restrict__ beta,         // (B, H)
    float*        __restrict__ state,        // (B, H, HK, HV) in/out
    const __half* __restrict__ gate_f16,     // (B, H, HV)
    const __half* __restrict__ weight_f16,   // (HV,)
    __half*       __restrict__ out_f16,      // (B, H, HV)
    float q_scale,
    float eps,
    unsigned int B,
    unsigned int H,
    unsigned int hk,
    unsigned int hv
) {
    unsigned int bi = blockIdx.x;
    unsigned int hi = blockIdx.y;
    if (bi >= B || hi >= H) return;
    unsigned int tid = threadIdx.x;
    if (tid >= hk) return;

    extern __shared__ float shm[];
    float* q_sm       = shm;
    float* k_sm       = shm + hk;
    float* reduce_sm  = shm + 2u * hk;
    float* scalars_sm = shm + 3u * hk;          // [4]
    float* core_sm    = shm + 3u * hk + 4u;     // [HV]

    unsigned long long base_qk    = (unsigned long long)bi * H * hk + (unsigned long long)hi * hk;
    unsigned long long base_v     = (unsigned long long)bi * H * hv + (unsigned long long)hi * hv;
    unsigned long long base_state = (unsigned long long)bi * H * hk * hv + (unsigned long long)hi * hk * hv;

    float qv = q[base_qk + tid];
    float kv = k[base_qk + tid];

    float q_sq = qv * qv;
    float k_sq = kv * kv;
    reduce_sm[tid] = q_sq;
    __syncthreads();
    for (unsigned int s = hk / 2u; s > 0u; s >>= 1) {
        if (tid < s) reduce_sm[tid] += reduce_sm[tid + s];
        __syncthreads();
    }
    float sum_q_sq = reduce_sm[0];
    __syncthreads();

    reduce_sm[tid] = k_sq;
    __syncthreads();
    for (unsigned int s = hk / 2u; s > 0u; s >>= 1) {
        if (tid < s) reduce_sm[tid] += reduce_sm[tid + s];
        __syncthreads();
    }
    float sum_k_sq = reduce_sm[0];
    __syncthreads();

    float inv_l2_q = rsqrtf(sum_q_sq + 1e-6f);
    float inv_l2_k = rsqrtf(sum_k_sq + 1e-6f);
    q_sm[tid] = qv * inv_l2_q * q_scale;
    k_sm[tid] = kv * inv_l2_k;

    if (tid == 0u) {
        scalars_sm[0] = __expf(g[bi * H + hi]);
        scalars_sm[1] = beta[bi * H + hi];
    }
    __syncthreads();
    float g_t    = scalars_sm[0];
    float beta_t = scalars_sm[1];

    // Thread-per-vi: каждый thread держит ОДИН value-канал vi=tid и считает его
    // независимо — без cross-thread reduce и без __syncthreads в горячем цикле
    // (старая версия делала ~16 syncthreads на vi → ~2048 на блок). q_sm/k_sm —
    // полные нормализованные вектора (заполнены фазой norm всеми hk потоками,
    // видны после __syncthreads выше). Каждый thread владеет СВОЕЙ колонкой
    // state[:, vi] (state[kk*hv+vi]) — записи не пересекаются между потоками,
    // чтения коалесцированы (соседние vi = соседние адреса). state[kk,vi]
    // читается дважды (kv_mem, затем update+out); между проходами не меняется,
    // поэтому st восстанавливается идентично. Требует hk==hv (block = hv).
    // Порядок сложения по kk идентичен tree-reduce версии (kk возрастает).
    {
        unsigned int vi = tid;
        unsigned long long col = base_state + vi;
        float kv_mem = 0.0f;
        for (unsigned int kk = 0u; kk < hk; ++kk) {
            float st = state[col + (unsigned long long)kk * hv] * g_t;
            kv_mem += st * k_sm[kk];
        }
        float delta = (v[base_v + vi] - kv_mem) * beta_t;
        float out_acc = 0.0f;
        for (unsigned int kk = 0u; kk < hk; ++kk) {
            unsigned long long idx = col + (unsigned long long)kk * hv;
            float new_st = state[idx] * g_t + k_sm[kk] * delta;
            state[idx] = new_st;
            out_acc += new_st * q_sm[kk];
        }
        core_sm[vi] = out_acc;
    }
    __syncthreads();

    // RmsNormGated на core_sm → out_f16.
    if (tid >= hv) return;

    float xv = core_sm[tid];
    float local_sq = xv * xv;

    reduce_sm[tid] = local_sq;
    __syncthreads();
    for (unsigned int s = hv / 2u; s > 0u; s >>= 1) {
        if (tid < s) reduce_sm[tid] += reduce_sm[tid + s];
        __syncthreads();
    }
    float total = reduce_sm[0];
    __syncthreads();

    float mean_sq = total / (float)hv;
    float inv = rsqrtf(mean_sq + eps);

    float w   = __half2float(weight_f16[tid]);
    float gz  = __half2float(gate_f16[base_v + tid]);
    float sig = 1.0f / (1.0f + __expf(-gz));
    float silu = gz * sig;
    float result = w * xv * inv * silu;
    out_f16[base_v + tid] = __float2half(result);
}
