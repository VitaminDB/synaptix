// KV-cache append slice kernel: in-place write нового K/V tile в
// preallocated ring-buffer.
//
// Заменяет Tensor::cat(prev, new, dim=2) который выделяет новый tensor
// (B*kv*(T_old+T_new)*hd) и копирует prev + new — это T_old × B × kv × hd ×
// 2 байта DRAM трафика per layer на decode-step. На 23K decode × 16
// full-attn слоёв × 2 (K и V) = ~1.5 GB трафика per token только на cat.
//
// С preallocated `dst` shape (B, kv, max_seq_len, hd) и slice copy at
// seq_pos: трафик = B × kv × T_new × hd × 2 байт per layer = ~4 KB per
// layer × 16 = 64 KB per token. ×24000 экономия.
//
// Grid: 1D, ceil(n_elements / BLOCK). Block = 128. Per thread — vectorized
// half2 (4 байта) load+store, что даёт coalesced access pattern для hd=256
// (256/2 = 128 half2 на row, по 1 half2 на thread).
//
// **Dev-variants (Phase D — CUDA graphs)**: `seq_pos` передаётся как
// `const uint32_t*` (device pointer) вместо immediate u32. На decode-step
// captured graph хранит этот pointer фиксированно, а значение
// обновляется через memcpy_htod перед каждым replay'ем. Иначе immediate
// `seq_pos` запекается в captured kernel params и на replay'е append
// перезаписал бы старый slot вместо нового. Backward-compat сохранён —
// immediate variants остаются для prefill (где capture не нужен).

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" {

// BF16 path: __nv_bfloat16 bit-identical с unsigned short → копируем через
// uint32 vectorized load/store (2 bf16 на uint).
__global__ void kv_append_slice_bf16(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    unsigned int B, unsigned int kv, unsigned int T_new, unsigned int hd,
    unsigned int max_seq_len, unsigned int seq_pos
) {
    unsigned int n_h2 = (B * kv * T_new * hd) >> 1;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_h2) return;
    unsigned int hd_h2 = hd >> 1;
    unsigned int d_h2 = gid % hd_h2;
    unsigned int t = (gid / hd_h2) % T_new;
    unsigned int k_idx = (gid / (hd_h2 * T_new)) % kv;
    unsigned int b = gid / (hd_h2 * T_new * kv);
    unsigned int d = d_h2 << 1;
    size_t src_off = (((size_t)b * kv + k_idx) * T_new + t) * hd + d;
    size_t dst_off = (((size_t)b * kv + k_idx) * max_seq_len + (seq_pos + t)) * hd + d;
    unsigned int v = *reinterpret_cast<const unsigned int*>(src + src_off);
    *reinterpret_cast<unsigned int*>(dst + dst_off) = v;
}

__global__ void kv_append_slice_bf16_dev(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    unsigned int B, unsigned int kv, unsigned int T_new, unsigned int hd,
    unsigned int max_seq_len,
    const unsigned int* __restrict__ seq_pos_ptr
) {
    // Per-row append position: seq_pos_ptr is [B] (batch-1 passes [1] → b=0 reads
    // the same scalar, so existing single-sequence decode is unchanged). Lets a
    // batch-2 decode (CFG cond+uncond) append at different KV offsets per row.
    unsigned int n_h2 = (B * kv * T_new * hd) >> 1;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_h2) return;
    unsigned int hd_h2 = hd >> 1;
    unsigned int d_h2 = gid % hd_h2;
    unsigned int t = (gid / hd_h2) % T_new;
    unsigned int k_idx = (gid / (hd_h2 * T_new)) % kv;
    unsigned int b = gid / (hd_h2 * T_new * kv);
    unsigned int seq_pos = seq_pos_ptr[b];
    unsigned int d = d_h2 << 1;
    if (seq_pos + t >= max_seq_len) return;
    size_t src_off = (((size_t)b * kv + k_idx) * T_new + t) * hd + d;
    size_t dst_off = (((size_t)b * kv + k_idx) * max_seq_len + (seq_pos + t)) * hd + d;
    unsigned int v = *reinterpret_cast<const unsigned int*>(src + src_off);
    *reinterpret_cast<unsigned int*>(dst + dst_off) = v;
}

// F16 path (наш main case: balance preset compute=F16, kv=F16).
//
// src layout: (B, kv, T_new, hd) contiguous row-major.
// dst layout: (B, kv, max_seq_len, hd) contiguous row-major.
//
// Каждая thread обрабатывает 2 contiguous F16 элемента (одна half2).
// Всего half2-элементов = B * kv * T_new * hd / 2.
__global__ void kv_append_slice_f16(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    unsigned int B,
    unsigned int kv,
    unsigned int T_new,
    unsigned int hd,
    unsigned int max_seq_len,
    unsigned int seq_pos
) {
    unsigned int n_h2 = (B * kv * T_new * hd) >> 1;   // элементов half2
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_h2) return;

    // Декодируем (b, k_idx, t, d_h2) из flat half2-index.
    // d_h2 = 0..(hd/2 - 1) → d = d_h2*2..d_h2*2+1.
    unsigned int hd_h2 = hd >> 1;
    unsigned int d_h2  = gid % hd_h2;
    unsigned int t     = (gid / hd_h2) % T_new;
    unsigned int k_idx = (gid / (hd_h2 * T_new)) % kv;
    unsigned int b     = gid / (hd_h2 * T_new * kv);

    unsigned int d = d_h2 << 1;

    // src offset (4-D row-major).
    size_t src_off = (((size_t)b * kv + k_idx) * T_new + t) * hd + d;
    // dst offset с учётом seq_pos.
    size_t dst_off = (((size_t)b * kv + k_idx) * max_seq_len + (seq_pos + t)) * hd + d;

    // Vectorized half2 load + store.
    __half2 v = *reinterpret_cast<const __half2*>(src + src_off);
    *reinterpret_cast<__half2*>(dst + dst_off) = v;
}

// BF16 path (для policy.balance.kv_dtype == BF16 либо других конфигов).
// __nv_bfloat16 в NVRTC требует cuda_bf16.h — но для простоты используем
// `unsigned short` (16-bit storage, bit-identical с BF16). Bitwise copy.
__global__ void kv_append_slice_u16(
    const unsigned short* __restrict__ src,
    unsigned short* __restrict__ dst,
    unsigned int B,
    unsigned int kv,
    unsigned int T_new,
    unsigned int hd,
    unsigned int max_seq_len,
    unsigned int seq_pos
) {
    unsigned int n_h2 = (B * kv * T_new * hd) >> 1;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_h2) return;

    unsigned int hd_h2 = hd >> 1;
    unsigned int d_h2  = gid % hd_h2;
    unsigned int t     = (gid / hd_h2) % T_new;
    unsigned int k_idx = (gid / (hd_h2 * T_new)) % kv;
    unsigned int b     = gid / (hd_h2 * T_new * kv);

    unsigned int d = d_h2 << 1;

    size_t src_off = (((size_t)b * kv + k_idx) * T_new + t) * hd + d;
    size_t dst_off = (((size_t)b * kv + k_idx) * max_seq_len + (seq_pos + t)) * hd + d;

    // 32-bit (uint) vectorized copy (= 2 × bf16/u16).
    unsigned int v = *reinterpret_cast<const unsigned int*>(src + src_off);
    *reinterpret_cast<unsigned int*>(dst + dst_off) = v;
}

// F32 path (резерв для policy.kv_dtype == F32).
__global__ void kv_append_slice_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int B,
    unsigned int kv,
    unsigned int T_new,
    unsigned int hd,
    unsigned int max_seq_len,
    unsigned int seq_pos
) {
    unsigned int n = B * kv * T_new * hd;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    unsigned int d     = gid % hd;
    unsigned int t     = (gid / hd) % T_new;
    unsigned int k_idx = (gid / (hd * T_new)) % kv;
    unsigned int b     = gid / (hd * T_new * kv);

    size_t src_off = (((size_t)b * kv + k_idx) * T_new + t) * hd + d;
    size_t dst_off = (((size_t)b * kv + k_idx) * max_seq_len + (seq_pos + t)) * hd + d;

    dst[dst_off] = src[src_off];
}

// ─────────────────────────── Device-resident variants ───────────────────────────
//
// `seq_pos` приходит как device pointer (1× uint32). Один warp lane читает
// значение, broadcast'ит через __shfl_sync, остальные logic — идентичны
// immediate variants. Альтернатива (каждый thread читает) дала бы лишние
// L1 transactions; здесь — один load на блок.

__global__ void kv_append_slice_f16_dev(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    unsigned int B,
    unsigned int kv,
    unsigned int T_new,
    unsigned int hd,
    unsigned int max_seq_len,
    const unsigned int* __restrict__ seq_pos_ptr
) {
    __shared__ unsigned int seq_pos;
    if (threadIdx.x == 0) seq_pos = *seq_pos_ptr;
    __syncthreads();

    unsigned int n_h2 = (B * kv * T_new * hd) >> 1;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_h2) return;

    unsigned int hd_h2 = hd >> 1;
    unsigned int d_h2  = gid % hd_h2;
    unsigned int t     = (gid / hd_h2) % T_new;
    unsigned int k_idx = (gid / (hd_h2 * T_new)) % kv;
    unsigned int b     = gid / (hd_h2 * T_new * kv);

    unsigned int d = d_h2 << 1;

    // Phase E.7 bounds-check: silent skip OOB writes когда seq_pos+t >= max_seq_len.
    // Защищает от OOB крашов при corrupted pos_dev (см. NEXT_SESSION_PROMPT Phase E.8).
    if (seq_pos + t >= max_seq_len) return;

    size_t src_off = (((size_t)b * kv + k_idx) * T_new + t) * hd + d;
    size_t dst_off = (((size_t)b * kv + k_idx) * max_seq_len + (seq_pos + t)) * hd + d;

    __half2 v = *reinterpret_cast<const __half2*>(src + src_off);
    *reinterpret_cast<__half2*>(dst + dst_off) = v;
}

__global__ void kv_append_slice_u16_dev(
    const unsigned short* __restrict__ src,
    unsigned short* __restrict__ dst,
    unsigned int B,
    unsigned int kv,
    unsigned int T_new,
    unsigned int hd,
    unsigned int max_seq_len,
    const unsigned int* __restrict__ seq_pos_ptr
) {
    __shared__ unsigned int seq_pos;
    if (threadIdx.x == 0) seq_pos = *seq_pos_ptr;
    __syncthreads();

    unsigned int n_h2 = (B * kv * T_new * hd) >> 1;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_h2) return;

    unsigned int hd_h2 = hd >> 1;
    unsigned int d_h2  = gid % hd_h2;
    unsigned int t     = (gid / hd_h2) % T_new;
    unsigned int k_idx = (gid / (hd_h2 * T_new)) % kv;
    unsigned int b     = gid / (hd_h2 * T_new * kv);

    unsigned int d = d_h2 << 1;

    // Phase E.7 bounds-check (см. f16_dev variant).
    if (seq_pos + t >= max_seq_len) return;

    size_t src_off = (((size_t)b * kv + k_idx) * T_new + t) * hd + d;
    size_t dst_off = (((size_t)b * kv + k_idx) * max_seq_len + (seq_pos + t)) * hd + d;

    unsigned int v = *reinterpret_cast<const unsigned int*>(src + src_off);
    *reinterpret_cast<unsigned int*>(dst + dst_off) = v;
}

__global__ void kv_append_slice_f32_dev(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int B,
    unsigned int kv,
    unsigned int T_new,
    unsigned int hd,
    unsigned int max_seq_len,
    const unsigned int* __restrict__ seq_pos_ptr
) {
    __shared__ unsigned int seq_pos;
    if (threadIdx.x == 0) seq_pos = *seq_pos_ptr;
    __syncthreads();

    unsigned int n = B * kv * T_new * hd;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    unsigned int d     = gid % hd;
    unsigned int t     = (gid / hd) % T_new;
    unsigned int k_idx = (gid / (hd * T_new)) % kv;
    unsigned int b     = gid / (hd * T_new * kv);

    // Phase E.7 bounds-check (см. f16_dev variant).
    if (seq_pos + t >= max_seq_len) return;

    size_t src_off = (((size_t)b * kv + k_idx) * T_new + t) * hd + d;
    size_t dst_off = (((size_t)b * kv + k_idx) * max_seq_len + (seq_pos + t)) * hd + d;

    dst[dst_off] = src[src_off];
}

} // extern "C"
