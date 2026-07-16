// Raw prep-ядра для GatedDeltaNet decode T=1 (linear attention).
//
// Портировано из ai-quant/src/kernels/cu_kernels/linear_attn_raw.cu (валидировано
// bit-exact в проде Qwen3.6). Заменяют 8-12 поэлементных ops под `cuStreamCapture`
// (каждая создавала бы cuEventCreate → INVALIDATED) на 4 fused launch'а.
//
// Все ядра — F32 internal для bit-equivalence с `to_dtype(F32) → ops →
// to_dtype(out_dtype)` паттерном. Layout — row-major. Block sizes подобраны под
// Qwen3.6 27B (num_v_heads = 64, head_k_dim = head_v_dim = 128).
//
// Состав:
//   1) softplus_neg_exp_g  — g[i] = softplus(a[i]+dt_bias[i]) * (-exp(A_log[i]))
//   2) sigmoid_f16_to_f32  — out[i] = sigmoid(in[i])
//   3) repeat_interleave_cast_f16_to_f32 — Q/K repeat-interleave (n_rep) + cast,
//      либо V cast (n_rep = 1).
//   4) rms_norm_gated_f32_in_f16_out — RMSNorm(x) * weight * silu(gate).
//   5) linear_attn_prep_fused_f16 — fused (1)+(2)+(3×3) в один launch.

#include <cuda_fp16.h>

#define WARP 32u

// ──────────────────── 1. softplus_neg_exp_g ─────────────────────────────────
//
// One thread per head. Точное соответствие reference:
//   dt = a_f32 + dt_bias
//   softplus_dt = log(1 + exp(dt))
//   a_log_neg_exp = -exp(A_log)
//   g = softplus_dt * a_log_neg_exp

extern "C" __global__ void softplus_neg_exp_g(
    const __half* __restrict__ a_f16,         // (num_v,) F16
    const float*  __restrict__ dt_bias_f32,   // (num_v,) F32
    const float*  __restrict__ a_log_f32,     // (num_v,) F32
    float*        __restrict__ g_out_f32,     // (num_v,) F32
    unsigned int num_v
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_v) return;
    float a   = __half2float(a_f16[i]);
    float dtb = dt_bias_f32[i];
    float al  = a_log_f32[i];
    float dt  = a + dtb;
    float softplus_dt = __logf(1.0f + __expf(dt));
    float a_log_neg_exp = -__expf(al);
    g_out_f32[i] = softplus_dt * a_log_neg_exp;
}

// ──────────────────── 2. sigmoid_f16_to_f32 ─────────────────────────────────

extern "C" __global__ void sigmoid_f16_to_f32(
    const __half* __restrict__ in_f16,
    float*        __restrict__ out_f32,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(in_f16[i]);
    out_f32[i] = 1.0f / (1.0f + __expf(-v));
}

// ──────────────────── 3. repeat_interleave_cast_f16_to_f32 ──────────────────
//
// Grid: (h_out, ceil(dim/256), 1). Block: (256, 1, 1).
// out[h_out, d] = (float)in[in_offset + (h_out / n_rep) * dim + d].
// При n_rep == 1 → straight cast F16→F32.

extern "C" __global__ void repeat_interleave_cast_f16_to_f32(
    const __half* __restrict__ in_f16,    // base ptr (full buffer)
    unsigned int in_offset,                // offset in elements
    float*        __restrict__ out_f32,   // (h_out, dim), h_out = h_in * n_rep
    unsigned int n_rep,
    unsigned int dim
) {
    unsigned int h_out = blockIdx.x;
    unsigned int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    unsigned int h_in = h_out / n_rep;
    out_f32[h_out * dim + d] = __half2float(in_f16[in_offset + h_in * dim + d]);
}

// ──────────────────── 4. rms_norm_gated_f32_in_f16_out ──────────────────────
//
// Per-row (один block на row). blockDim = next_pow2(dim).
//   var = mean(x^2); inv = rsqrt(var + eps); x_norm = x * inv;
//   out = weight * x_norm * silu(gate)

extern "C" __global__ void rms_norm_gated_f32_in_f16_out(
    const float*  __restrict__ x_f32,         // (n_rows, dim) F32
    const __half* __restrict__ gate_f16,      // (n_rows, dim) F16
    const __half* __restrict__ weight_f16,    // (dim,) F16
    __half*       __restrict__ out_f16,       // (n_rows, dim) F16
    float eps,
    unsigned int dim
) {
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int base = row * dim;

    float xv = (tid < dim) ? x_f32[base + tid] : 0.0f;
    float local_sq = xv * xv;

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sq += __shfl_down_sync(0xffffffffu, local_sq, offset);
    }
    unsigned int lane = tid & 31u;
    unsigned int warp = tid >> 5;
    __shared__ float sm[32];
    if (lane == 0) sm[warp] = local_sq;
    __syncthreads();

    unsigned int n_warps = (blockDim.x + WARP - 1) / WARP;
    float total = 0.0f;
    if (warp == 0) {
        total = (lane < n_warps) ? sm[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffffu, total, offset);
        }
        if (lane == 0) sm[0] = total;
    }
    __syncthreads();
    total = sm[0];

    float mean_sq = total / (float)dim;
    float inv = rsqrtf(mean_sq + eps);

    if (tid >= dim) return;
    float w = __half2float(weight_f16[tid]);
    float g = __half2float(gate_f16[base + tid]);
    float sig = 1.0f / (1.0f + __expf(-g));
    float silu = g * sig;
    float result = w * xv * inv * silu;
    out_f16[base + tid] = __float2half(result);
}

// ──────────────────── 5. linear_attn_prep_fused_f16 ─────────────────────────
//
// Fused kernel — заменяет 5 launch'ей одним. Все output buffers независимы.
// Layout:
//   grid_dim  = (max_h, 1, 4)        max_h = num_v
//   block_dim = (max(hk, hv), 1, 1)
//   blockIdx.z = регион:
//      0 → sigmoid(b)+softplus(a,dt_bias,a_log): только blockIdx.x==0.
//      1 → Q repeat-interleave + cast.
//      2 → K repeat-interleave + cast.
//      3 → V cast (n_rep = 1).

extern "C" __global__ void linear_attn_prep_fused_f16(
    const __half* __restrict__ b_f16,         // (num_v,)
    const __half* __restrict__ a_f16,         // (num_v,)
    const float*  __restrict__ dt_bias_f32,   // (num_v,)
    const float*  __restrict__ a_log_f32,     // (num_v,)
    float*        __restrict__ beta_f32,      // (num_v,)
    float*        __restrict__ g_out_f32,     // (num_v,)
    const __half* __restrict__ post_conv_f16, // (conv_dim,) = [Q|K|V]
    float*        __restrict__ q_out_f32,     // (num_v, hk)
    float*        __restrict__ k_out_f32,     // (num_v, hk)
    float*        __restrict__ v_out_f32,     // (num_v, hv)
    unsigned int num_v,
    unsigned int n_rep,
    unsigned int hk,
    unsigned int hv,
    unsigned int key_dim   // = num_k * hk
) {
    unsigned int region = blockIdx.z;
    unsigned int h_idx  = blockIdx.x;
    unsigned int d      = threadIdx.x;

    if (region == 0u) {
        if (h_idx != 0u) return;
        if (d >= num_v) return;
        float bv = __half2float(b_f16[d]);
        beta_f32[d] = 1.0f / (1.0f + __expf(-bv));
        float av  = __half2float(a_f16[d]);
        float dtb = dt_bias_f32[d];
        float al  = a_log_f32[d];
        float dt  = av + dtb;
        float softplus_dt   = __logf(1.0f + __expf(dt));
        float a_log_neg_exp = -__expf(al);
        g_out_f32[d] = softplus_dt * a_log_neg_exp;
        return;
    }

    if (h_idx >= num_v) return;

    if (region == 1u) {
        if (d >= hk) return;
        unsigned int h_in = h_idx / n_rep;
        q_out_f32[h_idx * hk + d] = __half2float(post_conv_f16[h_in * hk + d]);
        return;
    }
    if (region == 2u) {
        if (d >= hk) return;
        unsigned int h_in = h_idx / n_rep;
        k_out_f32[h_idx * hk + d] = __half2float(post_conv_f16[key_dim + h_in * hk + d]);
        return;
    }
    // region == 3 (V cast, n_rep = 1)
    if (d >= hv) return;
    v_out_f32[h_idx * hv + d] =
        __half2float(post_conv_f16[2u * key_dim + h_idx * hv + d]);
}
