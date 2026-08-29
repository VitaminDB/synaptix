// Chunk-FLA helper-ядра для Gated DeltaNet prefill (chunked linear attention).
//
// Портировано из ai-quant/src/kernels/cu_kernels/chunk_fla.cu (валидировано
// bit-exact в проде Qwen3.6). Алгоритм соответствует `torch_chunk_gated_delta_rule`
// из transformers (Qwen3.5/3.6). Все ядра — F32 internal, row-major.
//
// Чанковая форма: последовательность делится на чанки CS (=64). Внутри чанка —
// параллельная intra-attn (closed-form cumulative product), между чанками —
// рекуррентность по state. Эти ядра покрывают chunk-aware element-wise части;
// батчевые GEMM делает оркестратор (см. src/scan/chunk_scan.rs).
//
// Состав:
//   compute_chunk_attn_f32               — intra-chunk attn + decay_mask
//   state_update_decay_f32               — state *= exp(g_last) (по g_last)
//   mul_decay_mask_f32                   — attn_intra *= decay_mask (BH,CS,CS)
//   sub_inplace_f32 / add_inplace_f32    — a ∓= b
//   scale_by_exp_diff_f32                — q_scaled / k_cumdecay / k_decayed
//   sub_chunk_f32                        — value_proc[:,ci] -= v_prime
//   mul_decay_mask_chunk_f32             — attn_intra *= decay_mask[:,ci]
//   scale_k_decayed_chunk_f32            — k_decayed = k[:,ci]*exp(g_last-g_cumsum)
//   state_decay_from_gcumsum_chunk_f32   — state *= exp(g_cumsum[:,ci,CS-1])

extern "C" {

// ─────────────────────────────────────────────────────────────────────────────
// compute_chunk_attn_f32 — intra-chunk attn + decay_mask.
//
//   decay_mask[i,j] = exp(g[i] - g[j])  if j ≤ i else 0
//   attn[i,j]       = -(k_beta[i,:] · key[j,:]) * decay_mask[i,j]  if j < i else 0
//   attn[i,:i]     += Σ_{l<i} attn[i,l] * attn[l,:i]   (closed-form cumprod)
//   attn[i,i]      += 1
//
// Grid: (BH, NC, 1). Block: (CS). Shared: g_sm[CS] + attn_sm[CS*CS].
// ─────────────────────────────────────────────────────────────────────────────
__global__ void compute_chunk_attn_f32(
    const float* __restrict__ k_beta,    // (BH, NC, CS, HK)
    const float* __restrict__ key,       // (BH, NC, CS, HK)
    const float* __restrict__ g_cumsum,  // (BH, NC, CS)
    float* __restrict__ attn_out,        // (BH, NC, CS, CS)
    float* __restrict__ decay_mask_out,  // (BH, NC, CS, CS)
    unsigned int bh,
    unsigned int nc,
    unsigned int cs,
    unsigned int hk
) {
    unsigned int b = blockIdx.x;
    unsigned int c = blockIdx.y;
    unsigned int tid = threadIdx.x;  // 0..cs-1
    if (b >= bh || c >= nc) return;

    extern __shared__ float shm[];
    float* g_sm = shm;          // [CS]
    float* attn_sm = shm + cs;  // [CS * CS]

    unsigned long long base_kv = (unsigned long long)b * nc * cs * hk + (unsigned long long)c * cs * hk;
    unsigned long long base_g  = (unsigned long long)b * nc * cs + (unsigned long long)c * cs;
    unsigned long long base_attn = (unsigned long long)b * nc * cs * cs + (unsigned long long)c * cs * cs;

    if (tid < cs) {
        g_sm[tid] = g_cumsum[base_g + tid];
    }
    __syncthreads();

    // Шаг 1: decay_mask + attn инициализация. Thread tid отвечает за строку i=tid.
    unsigned int i = tid;
    float g_i = g_sm[i];
    for (unsigned int j = 0; j < cs; ++j) {
        float dm = (j <= i) ? __expf(g_i - g_sm[j]) : 0.0f;
        decay_mask_out[base_attn + i * cs + j] = dm;

        float a;
        if (j < i) {
            float acc = 0.0f;
            for (unsigned int d = 0; d < hk; ++d) {
                acc += k_beta[base_kv + i * hk + d] * key[base_kv + j * hk + d];
            }
            a = -acc * dm;
        } else {
            a = 0.0f;  // upper triangle + diag
        }
        attn_sm[i * cs + j] = a;
    }
    __syncthreads();

    // Шаг 2: closed-form cumulative product.
    // RACE-ФИКС: поток j читает attn_sm[row*cs + l] (l<row) — это строка `row`,
    // которую ПАРАЛЛЕЛЬНО пишет поток l (тоже <row) в этой же итерации. fla-эталон
    // использует СТАРЫЕ значения строки (`.clone()`). При cs=64 (2 варпа, не
    // lockstep) гонка → недетерминизм + неверная UT-инверсия. Разделяем фазы:
    // (1) все читают старое и копят new_val в регистр, (2) барьер, (3) все пишут.
    for (unsigned int row = 1; row < cs; ++row) {
        float new_val = 0.0f;
        bool active = (tid < row);
        if (active) {
            unsigned int j = tid;
            float acc = 0.0f;
            for (unsigned int l = 0; l < row; ++l) {
                acc += attn_sm[row * cs + l] * attn_sm[l * cs + j];
            }
            new_val = attn_sm[row * cs + j] + acc;
        }
        __syncthreads();
        if (active) {
            attn_sm[row * cs + tid] = new_val;
        }
        __syncthreads();
    }

    // Шаг 3: attn += I.
    if (tid < cs) {
        attn_sm[tid * cs + tid] += 1.0f;
    }
    __syncthreads();

    for (unsigned int j = 0; j < cs; ++j) {
        attn_out[base_attn + tid * cs + j] = attn_sm[tid * cs + j];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// state_update_decay_f32 — state[b,:,:] *= exp(g_last[b]). Grid (BH, HK, *).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void state_update_decay_f32(
    float* __restrict__ state,           // (BH, HK, HV)
    const float* __restrict__ g_last,    // (BH,)
    unsigned int bh,
    unsigned int hk,
    unsigned int hv
) {
    unsigned int b = blockIdx.x;
    unsigned int k = blockIdx.y;
    unsigned int v = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= bh || k >= hk || v >= hv) return;
    float decay = __expf(g_last[b]);
    unsigned long long idx = (unsigned long long)b * hk * hv + (unsigned long long)k * hv + v;
    state[idx] = state[idx] * decay;
}

// ─────────────────────────────────────────────────────────────────────────────
// Element-wise helpers.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void mul_decay_mask_f32(
    float* __restrict__ attn_intra,           // (BH, CS, CS) in/out
    const float* __restrict__ decay_mask_i,   // (BH, CS, CS) for chunk i
    unsigned int bh,
    unsigned int cs
) {
    unsigned int b = blockIdx.x;
    unsigned int i = blockIdx.y;
    unsigned int j = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= bh || i >= cs || j >= cs) return;
    unsigned long long idx = (unsigned long long)b * cs * cs + (unsigned long long)i * cs + j;
    attn_intra[idx] = attn_intra[idx] * decay_mask_i[idx];
}

__global__ void sub_inplace_f32(float* __restrict__ a, const float* __restrict__ b, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] = a[idx] - b[idx];
}

__global__ void add_inplace_f32(float* __restrict__ a, const float* __restrict__ b, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] = a[idx] + b[idx];
}

// ─────────────────────────────────────────────────────────────────────────────
// scale_by_exp_diff_f32 — common scaling.
//   mode=0 → factor = exp(vec_g[row])                 (q_scaled, k_cumdecay_input)
//   mode=1 → factor = exp(scalar_g[row/cs_in] - vec_g[row])  (k_decayed)
// row кладём в grid_dim.x (без лимита 65535).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void scale_by_exp_diff_f32(
    float* __restrict__ out,             // (total_rows, d)
    const float* __restrict__ in,
    const float* __restrict__ scalar_g,  // (BH * NC,) — может быть == vec_g для mode=0
    const float* __restrict__ vec_g,     // (total_rows,)
    unsigned int total_rows,
    unsigned int d,
    unsigned int cs_in,
    unsigned int mode
) {
    unsigned int row = blockIdx.x;
    unsigned int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= total_rows || col >= d) return;
    float gr = vec_g[row];
    float factor;
    if (mode == 0) {
        factor = __expf(gr);
    } else {
        float gl = scalar_g[row / cs_in];
        factor = __expf(gl - gr);
    }
    unsigned long long idx = (unsigned long long)row * d + col;
    out[idx] = in[idx] * factor;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunk-aware element-wise: работа на (BH, NC, CS, D) без per-BH host-loop.
// ─────────────────────────────────────────────────────────────────────────────

// value_proc[:, ci, :, :] -= v_prime[:].
__global__ void sub_chunk_f32(
    float* __restrict__ value_proc,     // (BH, NC, CS, HV)
    const float* __restrict__ v_prime,  // (BH, CS, HV)
    unsigned int bh,
    unsigned int nc,
    unsigned int cs,
    unsigned int hv,
    unsigned int chunk_idx
) {
    unsigned int b = blockIdx.x;
    unsigned int t = blockIdx.y;
    unsigned int d = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= bh || t >= cs || d >= hv) return;
    unsigned long long off_vp = (unsigned long long)b * nc * cs * hv
                              + (unsigned long long)chunk_idx * cs * hv
                              + (unsigned long long)t * hv + d;
    unsigned long long off_vpt = (unsigned long long)b * cs * hv
                               + (unsigned long long)t * hv + d;
    value_proc[off_vp] = value_proc[off_vp] - v_prime[off_vpt];
}

// attn_intra[bh,:,:] *= decay_mask[bh, ci, :, :].
__global__ void mul_decay_mask_chunk_f32(
    float* __restrict__ attn_intra,           // (BH, CS, CS)
    const float* __restrict__ decay_mask,     // (BH, NC, CS, CS)
    unsigned int bh,
    unsigned int nc,
    unsigned int cs,
    unsigned int chunk_idx
) {
    unsigned int b = blockIdx.x;
    unsigned int i = blockIdx.y;
    unsigned int j = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= bh || i >= cs || j >= cs) return;
    unsigned long long off_a = (unsigned long long)b * cs * cs
                             + (unsigned long long)i * cs + j;
    unsigned long long off_d = (unsigned long long)b * nc * cs * cs
                             + (unsigned long long)chunk_idx * cs * cs
                             + (unsigned long long)i * cs + j;
    attn_intra[off_a] = attn_intra[off_a] * decay_mask[off_d];
}

// k_decayed[bh,:,:] = k[bh, ci, :, :] * exp(g_last[bh] - g_cumsum[bh, ci, :]).
// k_decayed[bh, c, :, :] = k[bh, c, :, :] * exp(g_last[bh,c] - g_cumsum[bh,c,:])
// сразу по всем чанкам: от состояния этот шаг не зависит, а поштучно он стоил
// запуска на чанк при сетке в BH блоков.
__global__ void scale_k_decayed_all_f32(
    float* __restrict__ k_decayed_out,        // (BH, NC, CS, HK)
    const float* __restrict__ k,              // (BH, NC, CS, HK)
    const float* __restrict__ g_cumsum,       // (BH, NC, CS)
    unsigned int bh,
    unsigned int nc,
    unsigned int cs,
    unsigned int hk
) {
    unsigned int row = blockIdx.x;
    unsigned int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= bh * nc * cs || d >= hk) return;
    unsigned int t = row % cs;
    unsigned int rest = row / cs;
    unsigned long long off_g = (unsigned long long)rest * cs;
    float factor = __expf(g_cumsum[off_g + cs - 1] - g_cumsum[off_g + t]);
    unsigned long long off = (unsigned long long)row * hk + d;
    k_decayed_out[off] = k[off] * factor;
}

// Поэлементное a *= b по всему буферу.
__global__ void mul_inplace_f32(
    float* __restrict__ a,
    const float* __restrict__ b,
    unsigned long long n
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = a[i] * b[i];
}

__global__ void scale_k_decayed_chunk_f32(
    float* __restrict__ k_decayed_out,        // (BH, CS, HK)
    const float* __restrict__ k,              // (BH, NC, CS, HK)
    const float* __restrict__ g_cumsum,       // (BH, NC, CS)
    unsigned int bh,
    unsigned int nc,
    unsigned int cs,
    unsigned int hk,
    unsigned int chunk_idx
) {
    unsigned int b = blockIdx.x;
    unsigned int t = blockIdx.y;
    unsigned int d = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= bh || t >= cs || d >= hk) return;
    unsigned long long off_g = (unsigned long long)b * nc * cs
                             + (unsigned long long)chunk_idx * cs;
    float g_last = g_cumsum[off_g + cs - 1];
    float g_t = g_cumsum[off_g + t];
    float factor = __expf(g_last - g_t);
    unsigned long long off_k = (unsigned long long)b * nc * cs * hk
                             + (unsigned long long)chunk_idx * cs * hk
                             + (unsigned long long)t * hk + d;
    unsigned long long off_kd = (unsigned long long)b * cs * hk
                              + (unsigned long long)t * hk + d;
    k_decayed_out[off_kd] = k[off_k] * factor;
}

// state[b,:,:] *= exp(g_cumsum[b, ci, CS-1]). Полностью device-only.
__global__ void state_decay_from_gcumsum_chunk_f32(
    float* __restrict__ state,                // (BH, HK, HV)
    const float* __restrict__ g_cumsum,       // (BH, NC, CS)
    unsigned int bh,
    unsigned int nc,
    unsigned int cs,
    unsigned int hk,
    unsigned int hv,
    unsigned int chunk_idx
) {
    unsigned int b = blockIdx.x;
    unsigned int k = blockIdx.y;
    unsigned int v = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= bh || k >= hk || v >= hv) return;
    unsigned long long off_g = (unsigned long long)b * nc * cs
                             + (unsigned long long)chunk_idx * cs + cs - 1;
    float decay = __expf(g_cumsum[off_g]);
    unsigned long long idx = (unsigned long long)b * hk * hv
                           + (unsigned long long)k * hv + v;
    state[idx] = state[idx] * decay;
}

} // extern "C"
