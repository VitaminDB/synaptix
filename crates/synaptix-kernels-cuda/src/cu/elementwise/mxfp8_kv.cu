#include <cuda_fp16.h>
#include <cuda_bf16.h>

// MXFP8-KV (Blackwell block-scale) quantizing append: BF16/F16 src `(B,nkv,T_new,hd)`
// → MXFP8 E4M3 dst `(B,nkv,max_seq,hd)` slot `seq_pos` + per-32-block E8M0 scale
// `(B,nkv,max_seq,hd/32)`. Scale = exp(amax_блока)/256 (E8M0-байт); деквант в
// attention восстанавливает x = dec_e4m3(byte)·2^(sbyte-127), per-32-block.
// Параллельно FP8 E4M3 (fp8_kv.cu): точнее (scale на 32 элемента, не на всю строку).
//
// Один блок = одна (b,kv,token)-строка. Каждый thread владеет 32-блоками (blk=tid,
// stride bs): локальный amax по своим 32 contiguous-элементам → E8M0-байт → 32 E4M3.
// Шаблон по типу src (bf16/f16): compute=F16 при квант-весах → K/V проекции F16.

__device__ __forceinline__ float mxkv_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float mxkv_to_f(__half x) { return __half2float(x); }

__device__ __forceinline__ unsigned char mxkv_fp8_encode_e4m3(float x) {
    if (isnan(x)) return 0x7F;
    float v = fminf(fmaxf(x, -448.0f), 448.0f);
    unsigned int sign = signbit(v) ? 1 : 0;
    float abs_v = fabsf(v);
    if (abs_v == 0.0f) return (unsigned char)(sign << 7);
    int exp_raw = (int)floorf(log2f(abs_v));
    int exp_biased = exp_raw + 7;
    if (exp_biased < 1) {
        int m = (int)nearbyintf(abs_v * 512.0f);
        m = max(0, min(7, m));
        return (unsigned char)((sign << 7) | (unsigned int)m);
    }
    if (exp_biased > 15) {
        return (unsigned char)((sign << 7) | 0x7E);
    }
    float pow2 = exp2f((float)exp_raw);
    int m = (int)nearbyintf(((abs_v / pow2) - 1.0f) * 8.0f);
    if (m == 8) {
        m = 0;
        exp_biased += 1;
        if (exp_biased > 15) return (unsigned char)((sign << 7) | 0x7E);
    }
    if (exp_biased == 15 && m == 7) m = 6;
    unsigned int mantissa = (unsigned int)(m & 0x07);
    return (unsigned char)((sign << 7) | ((unsigned int)exp_biased << 3) | mantissa);
}

// Квантизация одной (b,kv,token)-строки в slot `pos`: per-32-block amax→E8M0→E4M3.
template <typename T>
__device__ __forceinline__ void mxkv_append_row(
    const T* __restrict__ src, unsigned char* __restrict__ dst,
    unsigned char* __restrict__ scale_dst,
    unsigned int b, unsigned int kv, unsigned int t, unsigned int nkv,
    unsigned int T_new, unsigned int hd, unsigned int max_seq, unsigned int pos,
    int tid, int bs) {
    unsigned int nb = hd / 32u;
    size_t src_base = (((size_t)b * nkv + kv) * T_new + t) * hd;
    size_t dst_base = (((size_t)b * nkv + kv) * max_seq + pos) * hd;
    size_t sc_base  = (((size_t)b * nkv + kv) * max_seq + pos) * nb;
    for (unsigned int blk = tid; blk < nb; blk += bs) {
        size_t boff = (size_t)blk * 32u;
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            amax = fmaxf(amax, fabsf(mxkv_to_f(src[src_base + boff + i])));
        }
        unsigned amax_exp_bits = __float_as_uint(amax) & 0x7F800000u;
        float scale_f = __uint_as_float(amax_exp_bits) / 256.0f;
        unsigned char sbyte = (unsigned char)(__float_as_uint(scale_f) >> 23);
        float sv = fmaxf(__uint_as_float(((unsigned)sbyte) << 23), 1e-12f);
#pragma unroll
        for (int i = 0; i < 32; i++) {
            float x = mxkv_to_f(src[src_base + boff + i]) / sv;
            dst[dst_base + boff + i] = mxkv_fp8_encode_e4m3(x);
        }
        scale_dst[sc_base + blk] = sbyte;
    }
}

template <typename T>
__device__ __forceinline__ void mxkv_append_impl(
    const T* __restrict__ src, unsigned char* __restrict__ dst,
    unsigned char* __restrict__ scale_dst,
    unsigned int B, unsigned int nkv, unsigned int T_new, unsigned int hd,
    unsigned int max_seq, unsigned int seq_pos) {
    unsigned int row = blockIdx.x;
    unsigned int t = row % T_new;
    unsigned int kv = (row / T_new) % nkv;
    unsigned int b = row / ((size_t)T_new * nkv);
    if (b >= B) return;
    if (seq_pos + t >= max_seq) return;
    mxkv_append_row<T>(src, dst, scale_dst, b, kv, t, nkv, T_new, hd, max_seq,
                       seq_pos + t, threadIdx.x, blockDim.x);
}

template <typename T>
__device__ __forceinline__ void mxkv_append_impl_dev(
    const T* __restrict__ src, unsigned char* __restrict__ dst,
    unsigned char* __restrict__ scale_dst,
    unsigned int B, unsigned int nkv, unsigned int T_new, unsigned int hd,
    unsigned int max_seq, const unsigned int* __restrict__ seq_pos_ptr) {
    __shared__ unsigned int seq_pos;
    if (threadIdx.x == 0) seq_pos = *seq_pos_ptr;
    __syncthreads();
    unsigned int row = blockIdx.x;
    unsigned int t = row % T_new;
    unsigned int kv = (row / T_new) % nkv;
    unsigned int b = row / ((size_t)T_new * nkv);
    if (b >= B) return;
    if (seq_pos + t >= max_seq) return;
    mxkv_append_row<T>(src, dst, scale_dst, b, kv, t, nkv, T_new, hd, max_seq,
                       seq_pos + t, threadIdx.x, blockDim.x);
}

extern "C" {

__global__ void kv_quant_append_mxfp8_bf16(
    const __nv_bfloat16* __restrict__ src, unsigned char* __restrict__ dst,
    unsigned char* __restrict__ scale_dst,
    unsigned int B, unsigned int nkv, unsigned int T_new, unsigned int hd,
    unsigned int max_seq, unsigned int seq_pos) {
    mxkv_append_impl<__nv_bfloat16>(src, dst, scale_dst, B, nkv, T_new, hd, max_seq, seq_pos);
}
__global__ void kv_quant_append_mxfp8_f16(
    const __half* __restrict__ src, unsigned char* __restrict__ dst,
    unsigned char* __restrict__ scale_dst,
    unsigned int B, unsigned int nkv, unsigned int T_new, unsigned int hd,
    unsigned int max_seq, unsigned int seq_pos) {
    mxkv_append_impl<__half>(src, dst, scale_dst, B, nkv, T_new, hd, max_seq, seq_pos);
}

// Device-pos варианты (CUDA-graph decode): seq_pos из *seq_pos_ptr.
__global__ void kv_quant_append_mxfp8_bf16_dev(
    const __nv_bfloat16* __restrict__ src, unsigned char* __restrict__ dst,
    unsigned char* __restrict__ scale_dst,
    unsigned int B, unsigned int nkv, unsigned int T_new, unsigned int hd,
    unsigned int max_seq, const unsigned int* __restrict__ seq_pos_ptr) {
    mxkv_append_impl_dev<__nv_bfloat16>(src, dst, scale_dst, B, nkv, T_new, hd, max_seq, seq_pos_ptr);
}
__global__ void kv_quant_append_mxfp8_f16_dev(
    const __half* __restrict__ src, unsigned char* __restrict__ dst,
    unsigned char* __restrict__ scale_dst,
    unsigned int B, unsigned int nkv, unsigned int T_new, unsigned int hd,
    unsigned int max_seq, const unsigned int* __restrict__ seq_pos_ptr) {
    mxkv_append_impl_dev<__half>(src, dst, scale_dst, B, nkv, T_new, hd, max_seq, seq_pos_ptr);
}

}  // extern "C"
