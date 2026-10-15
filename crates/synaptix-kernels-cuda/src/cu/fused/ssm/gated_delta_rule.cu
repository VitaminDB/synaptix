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
    unsigned int hv,
    unsigned int gate_mode                   // 0: silu(gate), 1: sigmoid(gate)
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
    float act = gate_mode ? sig : gz * sig;
    float result = w * xv * inv * act;
    out_f16[base_v + tid] = __float2half(result);
}

// ───────────── decode-шаг GDN одним ядром ───────────────────────────────────
//
// causal_conv1d_update (+SiLU) → β/g → L2-нормы q/k → gated delta rule →
// RmsNormGated — всё в одном блоке. Блок = одна k-голова со всеми её v-головами
// (GQA repeat n_rep = num_v / num_k); нить (sub, ks, d): v-голова
// `kh·n_rep + sub`, канал d, группа ks из KS делит диапазон kk состояния
// (строки kk ≡ ks mod KS) — так в полёте в KS раз больше чтений состояния,
// иначе 16 блоков по 256 нитей не загружают память (состояние 64 КБ на голову).
// Свёртка (ks = 0): нить (0, d) считает Q-канал, нить (1, d) — K-канал (при
// n_rep = 1 обе — нить (0, d)), каждая нить — свой V-канал; состояние свёртки
// сдвигает нить, которая этот канал посчитала (гонок между блоками нет:
// столбцы k-головы читает и пишет только её блок). post_conv округляется в
// f16, как при записи промежуточного буфера трёхъядерного пути; порядок
// редукций q/k-норм и RmsNorm — тот же, суммы по kk — по группам ks.
//
// Требования: hk == hv (степень двойки), KS | hk, n_rep·hv·KS ≤ 1024.
// Shared: (2·hk + (3 + KS)·n_rep·hv + 2·n_rep) F32.

extern "C" __global__ void gdn_decode_step_f16(
    const __half* __restrict__ x,            // [conv_dim] = Q | K | V (выход in_proj)
    __half*       __restrict__ conv_state,   // [K-1, conv_dim] in/out
    const __half* __restrict__ conv_w,       // [conv_dim, K]
    const __half* __restrict__ a_f16,        // [num_v]
    const __half* __restrict__ b_f16,        // [num_v]
    const float*  __restrict__ dt_bias,      // [num_v]
    const float*  __restrict__ a_log,        // [num_v]
    float*        __restrict__ state,        // [num_v, hk, hv] in/out
    const __half* __restrict__ gate_f16,     // [num_v·hv] (z)
    const __half* __restrict__ weight_f16,   // [hv]
    __half*       __restrict__ out_f16,      // [num_v·hv]
    float q_scale,
    float eps,
    unsigned int num_k,
    unsigned int n_rep,
    unsigned int hk,
    unsigned int hv,
    unsigned int conv_k,
    unsigned int ks_n,                       // KS — групп по kk
    unsigned int gate_mode                   // 0: silu(gate), 1: sigmoid(gate)
) {
    unsigned int kh = blockIdx.x;
    if (kh >= num_k) return;
    unsigned int tid = threadIdx.x;
    unsigned int d = tid % hv;
    unsigned int rest = tid / hv;
    unsigned int ks = rest % ks_n;
    unsigned int sub = rest / ks_n;
    if (sub >= n_rep) return;
    unsigned int num_v = num_k * n_rep;
    unsigned int vh = kh * n_rep + sub;
    unsigned int key_dim = num_k * hk;
    unsigned int conv_dim = 2u * key_dim + num_v * hv;
    unsigned int km1 = conv_k - 1u;

    extern __shared__ float shm[];
    float* q_sm      = shm;                          // [hk]
    float* k_sm      = shm + hk;                     // [hk]
    float* v_sm      = shm + 2u * hk;                // [n_rep·hv]
    float* reduce_sm = v_sm + n_rep * hv;            // [n_rep·hv]
    float* core_sm   = reduce_sm + n_rep * hv;       // [n_rep·hv]
    float* part_sm   = core_sm + n_rep * hv;         // [n_rep·KS·hv]
    float* scal_sm   = part_sm + n_rep * ks_n * hv;  // [n_rep·2]

    // ── свёртка одного канала: окно [state[0..K-2, c], x[c]], сдвиг состояния ──
    auto conv_ch = [&](unsigned int c) -> float {
        const __half* wc = conv_w + (size_t)c * conv_k;
        float xc = __half2float(x[c]);
        float acc = 0.0f;
        for (unsigned int j = 0u; j < km1; ++j)
            acc += __half2float(conv_state[(size_t)j * conv_dim + c]) * __half2float(wc[j]);
        acc += xc * __half2float(wc[km1]);
        acc = acc / (1.0f + expf(-acc));
        for (unsigned int j = 0u; j + 1u < km1; ++j)
            conv_state[(size_t)j * conv_dim + c] = conv_state[(size_t)(j + 1u) * conv_dim + c];
        if (km1 > 0u) conv_state[(size_t)(km1 - 1u) * conv_dim + c] = __float2half(xc);
        return __half2float(__float2half(acc));
    };

    if (ks == 0u) {
        if (sub == 0u) q_sm[d] = conv_ch(kh * hk + d);
        if (sub == (n_rep > 1u ? 1u : 0u)) k_sm[d] = conv_ch(key_dim + kh * hk + d);
        v_sm[sub * hv + d] = conv_ch(2u * key_dim + vh * hv + d);
        if (d == 0u) {
            float bv = __half2float(b_f16[vh]);
            scal_sm[sub * 2u + 1u] = 1.0f / (1.0f + __expf(-bv));
            float av = __half2float(a_f16[vh]);
            float dt = av + dt_bias[vh];
            float softplus_dt = __logf(1.0f + __expf(dt));
            float g = softplus_dt * (-__expf(a_log[vh]));
            scal_sm[sub * 2u] = __expf(g);
        }
    }
    __syncthreads();

    // ── L2-нормы q/k (нити sub = 0, ks = 0; порядок редукции как прежде) ──
    bool lead = (sub == 0u && ks == 0u);
    float qv = q_sm[d];
    float kv = k_sm[d];
    if (lead) reduce_sm[d] = qv * qv;
    __syncthreads();
    for (unsigned int s = hk / 2u; s > 0u; s >>= 1) {
        if (lead && d < s) reduce_sm[d] += reduce_sm[d + s];
        __syncthreads();
    }
    float sum_q_sq = reduce_sm[0];
    __syncthreads();
    if (lead) reduce_sm[d] = kv * kv;
    __syncthreads();
    for (unsigned int s = hk / 2u; s > 0u; s >>= 1) {
        if (lead && d < s) reduce_sm[d] += reduce_sm[d + s];
        __syncthreads();
    }
    float sum_k_sq = reduce_sm[0];
    __syncthreads();
    if (lead) {
        q_sm[d] = qv * rsqrtf(sum_q_sq + 1e-6f) * q_scale;
        k_sm[d] = kv * rsqrtf(sum_k_sq + 1e-6f);
    }
    __syncthreads();

    // ── gated delta rule: нить = столбец vi состояния, строки kk ≡ ks (mod KS) ──
    float g_t = scal_sm[sub * 2u];
    float beta_t = scal_sm[sub * 2u + 1u];
    unsigned int vi = d;
    unsigned long long col = (unsigned long long)vh * hk * hv + vi;
    float* part = part_sm + (sub * ks_n) * hv;       // [KS][hv]
    {
        float kv_mem = 0.0f;
        #pragma unroll 8
        for (unsigned int kk = ks; kk < hk; kk += ks_n) {
            float st = state[col + (unsigned long long)kk * hv] * g_t;
            kv_mem += st * k_sm[kk];
        }
        part[ks * hv + vi] = kv_mem;
    }
    __syncthreads();
    float kv_mem = 0.0f;
    for (unsigned int j = 0u; j < ks_n; ++j) kv_mem += part[j * hv + vi];
    float delta = (v_sm[sub * hv + vi] - kv_mem) * beta_t;
    __syncthreads();
    {
        float out_acc = 0.0f;
        #pragma unroll 8
        for (unsigned int kk = ks; kk < hk; kk += ks_n) {
            unsigned long long idx = col + (unsigned long long)kk * hv;
            float new_st = state[idx] * g_t + k_sm[kk] * delta;
            state[idx] = new_st;
            out_acc += new_st * q_sm[kk];
        }
        part[ks * hv + vi] = out_acc;
    }
    __syncthreads();
    if (ks != 0u) return;
    {
        float out_acc = 0.0f;
        for (unsigned int j = 0u; j < ks_n; ++j) out_acc += part[j * hv + vi];
        core_sm[sub * hv + vi] = out_acc;
    }

    // ── RmsNormGated своей v-головы (нити ks = 0: hv на v-голову) ──
    // Барьеры ниже — только для нитей ks = 0; вышедшие нити в них не участвуют
    // (named barrier на hv·n_rep нитей).
    float* red = reduce_sm + sub * hv;
    float xv = core_sm[sub * hv + d];
    red[d] = xv * xv;
    asm volatile("bar.sync 1, %0;" :: "r"(n_rep * hv));
    for (unsigned int s = hv / 2u; s > 0u; s >>= 1) {
        if (d < s) red[d] += red[d + s];
        asm volatile("bar.sync 1, %0;" :: "r"(n_rep * hv));
    }
    float total = red[0];
    float inv = rsqrtf(total / (float)hv + eps);
    float w = __half2float(weight_f16[d]);
    float gz = __half2float(gate_f16[(size_t)vh * hv + d]);
    float sig = 1.0f / (1.0f + __expf(-gz));
    float act = gate_mode ? sig : gz * sig;
    out_f16[(size_t)vh * hv + d] = __float2half(w * xv * inv * act);
}
