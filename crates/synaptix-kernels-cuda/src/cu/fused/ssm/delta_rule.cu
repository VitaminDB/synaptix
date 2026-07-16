// DeltaNet рекуррентный шаг (без гейта). Частный случай gated delta rule
// с α≡1 (нет decay), без L2-нормализации q/k и без q_scale — «чистый» delta rule.
//
// Рекуррентность:
//   kv_mem   = Σ_k state[k, vi] * k[k]
//   delta    = (v[vi] - kv_mem) * beta
//   state[k, vi] += k[k] * delta
//   out[vi]  = Σ_k state[k, vi] * q[k]
// Семантически: S_t = S_{t-1} + β_t·k_t·(v_t − S_{t-1}ᵀk_t)ᵀ;  o_t = S_tᵀ·q_t.
//
// Один block = (batch, head); block_dim = hk; каждый thread держит key-канал.

extern "C" __global__ void delta_rule_step_f32(
    const float* __restrict__ q,        // (B, H, HK)
    const float* __restrict__ k,        // (B, H, HK)
    const float* __restrict__ v,        // (B, H, HV)
    const float* __restrict__ beta,     // (B, H)
    float* __restrict__ state,          // (B, H, HK, HV) in/out
    float* __restrict__ out,            // (B, H, HV)
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
    float* reduce_sm  = shm;            // [HK]
    float* scalars_sm = shm + hk;       // [2]

    unsigned long long base_qk = (unsigned long long)bi * H * hk + (unsigned long long)hi * hk;
    unsigned long long base_v = (unsigned long long)bi * H * hv + (unsigned long long)hi * hv;
    unsigned long long base_state = (unsigned long long)bi * H * hk * hv + (unsigned long long)hi * hk * hv;

    float q_k = q[base_qk + tid];
    float k_k = k[base_qk + tid];

    if (tid == 0) {
        scalars_sm[0] = beta[bi * H + hi];
    }
    __syncthreads();
    float beta_t = scalars_sm[0];

    for (unsigned int vi = 0; vi < hv; ++vi) {
        unsigned long long state_idx = base_state + (unsigned long long)tid * hv + vi;
        float st = state[state_idx];

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
            scalars_sm[1] = (vt - kv_mem) * beta_t;  // delta
        }
        __syncthreads();
        float delta = scalars_sm[1];

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
