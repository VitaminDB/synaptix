#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Partial Rotary Position Embedding (RoPE) kernel.
//
// Math:
//   half = rotary_dim / 2
//   for d in [0, half):
//     out[d] = x[d] * cos[d] + (-x[d+half]) * sin[d]
//   for d in [half, rotary_dim):
//     out[d] = x[d] * cos[d] +   x[d-half]  * sin[d]
//   for d in [rotary_dim, head_dim):
//     out[d] = x[d]  (pass-through)
//
// Grid: (B * H * T, 1, 1); block: (head_dim, 1, 1). head_dim ≤ 1024.
// `start_pos_ptr` — device-resident u32 (shared broadcast'ом). Совместимо с
// CUDA graph replay.

__device__ __forceinline__ float load_x(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_x(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float load_x(const float* p) { return *p; }

__device__ __forceinline__ void store_x(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_x(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
__device__ __forceinline__ void store_x(float* p, float v) { *p = v; }

template <typename T>
__device__ __forceinline__ void rope_apply_impl(
    const T* __restrict__ x,
    T*       __restrict__ out,
    const T* __restrict__ cos_table,
    const T* __restrict__ sin_table,
    const unsigned int* __restrict__ start_pos_ptr,
    unsigned int B,
    unsigned int H,
    unsigned int T_seq,
    unsigned int head_dim,
    unsigned int rotary_dim
) {
    unsigned int row = blockIdx.x;
    unsigned int d   = threadIdx.x;
    if (d >= head_dim) return;
    unsigned int t = row % T_seq;

    // Per-row start position: start_pos_ptr is [B] (batch-1 passes [1] → b=0
    // reads the same scalar, single-sequence decode unchanged). Enables a
    // batch-2 CFG decode where cond/uncond sit at different absolute positions.
    unsigned int b = row / (H * T_seq);
    unsigned int pos = start_pos_ptr[b] + t;

    size_t base = (size_t)row * head_dim;

    if (d >= rotary_dim) {
        out[base + d] = x[base + d];
        return;
    }

    unsigned int half = rotary_dim >> 1;
    bool low = d < half;
    unsigned int partner = low ? (d + half) : (d - half);

    float c = load_x(cos_table + (size_t)pos * rotary_dim + d);
    float s = load_x(sin_table + (size_t)pos * rotary_dim + d);
    float x_val = load_x(x + base + d);
    float x_partner = load_x(x + base + partner);

    float rotated = low ? -x_partner : x_partner;
    float result = x_val * c + rotated * s;
    store_x(out + base + d, result);
}

extern "C" __global__ void rope_apply_partial_f16(
    const __half* __restrict__ x,
    __half* __restrict__       out,
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    const unsigned int* __restrict__ start_pos_ptr,
    unsigned int B,
    unsigned int H,
    unsigned int T_seq,
    unsigned int head_dim,
    unsigned int rotary_dim
) {
    rope_apply_impl<__half>(x, out, cos_table, sin_table, start_pos_ptr,
                            B, H, T_seq, head_dim, rotary_dim);
}

extern "C" __global__ void rope_apply_partial_bf16(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__       out,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    const unsigned int* __restrict__ start_pos_ptr,
    unsigned int B,
    unsigned int H,
    unsigned int T_seq,
    unsigned int head_dim,
    unsigned int rotary_dim
) {
    rope_apply_impl<__nv_bfloat16>(x, out, cos_table, sin_table, start_pos_ptr,
                                   B, H, T_seq, head_dim, rotary_dim);
}

extern "C" __global__ void rope_apply_partial_f32(
    const float* __restrict__ x,
    float* __restrict__       out,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const unsigned int* __restrict__ start_pos_ptr,
    unsigned int B,
    unsigned int H,
    unsigned int T_seq,
    unsigned int head_dim,
    unsigned int rotary_dim
) {
    rope_apply_impl<float>(x, out, cos_table, sin_table, start_pos_ptr,
                           B, H, T_seq, head_dim, rotary_dim);
}

// ── Split (GPT-NeoX) RoPE, точная копия synaptix-ops apply_rope(Split) ──────
//   half = D/2
//   out[d<half]  = x[d]*cos[s,d]      - x[d+half]*sin[s,d]
//   out[d>=half] = x[d]*cos[s,d-half] + x[d-half]*sin[s,d-half]
// x/out: [.., S, D] type T row-major (rows = numel/D). cos/sin: [S, half] F32
// (как из RopeCache::select_*; ротация в F32, выход в T). Grid (rows); block (D).
template <typename T>
__device__ __forceinline__ void rope_split_impl(
    const T* __restrict__ x,
    T*       __restrict__ out,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    unsigned int S,
    unsigned int D
) {
    unsigned int row = blockIdx.x;
    unsigned int d   = threadIdx.x;
    if (d >= D) return;
    unsigned int s    = row % S;
    unsigned int half = D >> 1;
    size_t base = (size_t)row * D;
    unsigned int idx = (d < half) ? d : (d - half);
    float c  = cos[(size_t)s * half + idx];
    float sn = sin[(size_t)s * half + idx];
    float xv = load_x(x + base + d);
    float result;
    if (d < half) {
        float xp = load_x(x + base + d + half);
        result = xv * c - xp * sn;
    } else {
        float xp = load_x(x + base + d - half);
        result = xv * c + xp * sn;
    }
    store_x(out + base + d, result);
}

extern "C" __global__ void rope_split_f16(
    const __half* __restrict__ x, __half* __restrict__ out,
    const float* __restrict__ cos, const float* __restrict__ sin,
    unsigned int S, unsigned int D
) { rope_split_impl<__half>(x, out, cos, sin, S, D); }

extern "C" __global__ void rope_split_bf16(
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
    const float* __restrict__ cos, const float* __restrict__ sin,
    unsigned int S, unsigned int D
) { rope_split_impl<__nv_bfloat16>(x, out, cos, sin, S, D); }

extern "C" __global__ void rope_split_f32(
    const float* __restrict__ x, float* __restrict__ out,
    const float* __restrict__ cos, const float* __restrict__ sin,
    unsigned int S, unsigned int D
) { rope_split_impl<float>(x, out, cos, sin, S, D); }

// ── Partial split RoPE (MiniMax-H3 MM-RoPE): вращает первые rot_dim из D ──
//   half = rot_dim/2
//   out[d<half]        = x[d]*cos[s,d]      - x[d+half]*sin[s,d]
//   out[half<=d<rot]   = x[d]*cos[s,d-half] + x[d-half]*sin[s,d-half]
//   out[d>=rot_dim]    = x[d]  (pass-through)
// x/out: [.., S, D] type T row-major (rows = numel/D, позиция = row % S —
// broadcast по головам при layout [H,S,D]). cos/sin: [S, rot_dim/2] F32.
template <typename T>
__device__ __forceinline__ void rope_split_partial_impl(
    const T* __restrict__ x,
    T*       __restrict__ out,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    unsigned int S,
    unsigned int D,
    unsigned int rot_dim,
    unsigned int pos_div
) {
    unsigned int row = blockIdx.x;
    unsigned int d   = threadIdx.x;
    if (d >= D) return;
    size_t base = (size_t)row * D;
    if (d >= rot_dim) {
        out[base + d] = x[base + d];
        return;
    }
    unsigned int s    = (row / pos_div) % S;
    unsigned int half = rot_dim >> 1;
    unsigned int idx  = (d < half) ? d : (d - half);
    float c  = cos[(size_t)s * half + idx];
    float sn = sin[(size_t)s * half + idx];
    float xv = load_x(x + base + d);
    float result;
    if (d < half) {
        float xp = load_x(x + base + d + half);
        result = xv * c - xp * sn;
    } else {
        float xp = load_x(x + base + d - half);
        result = xv * c + xp * sn;
    }
    store_x(out + base + d, result);
}

extern "C" __global__ void rope_split_partial_f16(
    const __half* __restrict__ x, __half* __restrict__ out,
    const float* __restrict__ cos, const float* __restrict__ sin,
    unsigned int S, unsigned int D, unsigned int rot_dim, unsigned int pos_div
) { rope_split_partial_impl<__half>(x, out, cos, sin, S, D, rot_dim, pos_div); }

extern "C" __global__ void rope_split_partial_bf16(
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
    const float* __restrict__ cos, const float* __restrict__ sin,
    unsigned int S, unsigned int D, unsigned int rot_dim, unsigned int pos_div
) { rope_split_partial_impl<__nv_bfloat16>(x, out, cos, sin, S, D, rot_dim, pos_div); }

extern "C" __global__ void rope_split_partial_f32(
    const float* __restrict__ x, float* __restrict__ out,
    const float* __restrict__ cos, const float* __restrict__ sin,
    unsigned int S, unsigned int D, unsigned int rot_dim, unsigned int pos_div
) { rope_split_partial_impl<float>(x, out, cos, sin, S, D, rot_dim, pos_div); }

// ── Interleaved (GPT-NeoX adjacent-pair / FLUX use_real_unbind_dim=-1) RoPE ──
//   out[2j]   = x[2j]*cos[s,2j]   - x[2j+1]*sin[s,2j]
//   out[2j+1] = x[2j+1]*cos[s,2j+1] + x[2j]*sin[s,2j+1]   (cos[2j]==cos[2j+1])
// x/out: [B,S,H,D] type T row-major; cos/sin: ПОЛНАЯ таблица [S,D] F32 (как из
// FLUX build_rope, repeat_interleave(2)). Заменяет ~10 decomposed-ops (to_f32 +
// narrow×2 + cat + neg + mul×2 + add + to_bf16). Grid (B*S*H rows); block (D).
template <typename T>
__device__ __forceinline__ void rope_interleaved_impl(
    const T* __restrict__ x,
    T*       __restrict__ out,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    unsigned int H,
    unsigned int S,
    unsigned int D
) {
    unsigned int row = blockIdx.x;
    unsigned int d   = threadIdx.x;
    if (d >= D) return;
    unsigned int s = (row / H) % S;   // layout [B,S,H,D]: позиция = (row/H)%S
    size_t base = (size_t)row * D;
    float c  = cos[(size_t)s * D + d];
    float sn = sin[(size_t)s * D + d];
    float xv = load_x(x + base + d);
    float xp = load_x(x + base + (d ^ 1u));  // партнёр по паре
    float result = (d & 1u) ? (xv * c + xp * sn) : (xv * c - xp * sn);
    store_x(out + base + d, result);
}

extern "C" __global__ void rope_interleaved_f16(
    const __half* __restrict__ x, __half* __restrict__ out,
    const float* __restrict__ cos, const float* __restrict__ sin,
    unsigned int H, unsigned int S, unsigned int D
) { rope_interleaved_impl<__half>(x, out, cos, sin, H, S, D); }

extern "C" __global__ void rope_interleaved_bf16(
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
    const float* __restrict__ cos, const float* __restrict__ sin,
    unsigned int H, unsigned int S, unsigned int D
) { rope_interleaved_impl<__nv_bfloat16>(x, out, cos, sin, H, S, D); }

extern "C" __global__ void rope_interleaved_f32(
    const float* __restrict__ x, float* __restrict__ out,
    const float* __restrict__ cos, const float* __restrict__ sin,
    unsigned int H, unsigned int S, unsigned int D
) { rope_interleaved_impl<float>(x, out, cos, sin, H, S, D); }
