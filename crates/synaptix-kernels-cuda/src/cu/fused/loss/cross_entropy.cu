#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Fused softmax + nll loss за один pass через online stable softmax
// (Milakov-Gimelshein 2018). Один block per sample, T threads (BLOCK=256).
//
// loss[b] = -log_softmax(logits[b])[target[b]]
//        = log_sum_exp(logits[b]) - logits[b, target[b]]
//
// Online soft-max update per thread:
//   m_new = max(m, v); s = s*exp(m-m_new) + exp(v-m_new); m = m_new
// Warp/block reduce обмена (m_acc, s_acc).
// ignore_index: если target == ignore_index → loss=0.

constexpr int BLOCK = 256;
constexpr int WARP = 32;

__device__ __forceinline__ float load_logit(const float* p) { return *p; }
__device__ __forceinline__ float load_logit(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_logit(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ float neg_inf_f32() { return -1.f / 0.f; }

__device__ __forceinline__ void merge_ms(float &m, float &s, float m_o, float s_o) {
    if (s_o == 0.f) return;
    if (s == 0.f) { m = m_o; s = s_o; return; }
    float m_n = fmaxf(m, m_o);
    s = s * __expf(m - m_n) + s_o * __expf(m_o - m_n);
    m = m_n;
}

template <typename T>
__device__ __forceinline__ void cross_entropy_impl(
    const T*   __restrict__ logits,        // (B, V)
    const int* __restrict__ targets,       // (B,)
    float*     __restrict__ losses,        // (B,)
    unsigned int B,
    unsigned int V,
    int ignore_index)
{
    unsigned int b = blockIdx.x;
    if (b >= B) return;
    unsigned int tid = threadIdx.x;

    int target = targets[b];
    if (target == ignore_index) {
        if (tid == 0) losses[b] = 0.f;
        return;
    }

    const T* row = logits + (size_t)b * V;

    float m = neg_inf_f32();
    float s = 0.f;
    for (unsigned int i = tid; i < V; i += BLOCK) {
        float v = load_logit(row + i);
        float m_n = fmaxf(m, v);
        s = s * __expf(m - m_n) + __expf(v - m_n);
        m = m_n;
    }

    for (int off = WARP / 2; off > 0; off >>= 1) {
        float m_o = __shfl_xor_sync(0xFFFFFFFFu, m, off);
        float s_o = __shfl_xor_sync(0xFFFFFFFFu, s, off);
        merge_ms(m, s, m_o, s_o);
    }

    __shared__ float warp_m[BLOCK / WARP];
    __shared__ float warp_s[BLOCK / WARP];
    unsigned int warp_id = tid >> 5;
    unsigned int lane = tid & 31u;
    if (lane == 0) {
        warp_m[warp_id] = m;
        warp_s[warp_id] = s;
    }
    __syncthreads();

    if (warp_id == 0) {
        unsigned int n_warps = BLOCK / WARP;
        float m_b = lane < n_warps ? warp_m[lane] : neg_inf_f32();
        float s_b = lane < n_warps ? warp_s[lane] : 0.f;
        for (int off = WARP / 2; off > 0; off >>= 1) {
            float m_o = __shfl_xor_sync(0xFFFFFFFFu, m_b, off);
            float s_o = __shfl_xor_sync(0xFFFFFFFFu, s_b, off);
            merge_ms(m_b, s_b, m_o, s_o);
        }
        if (lane == 0) {
            float lse = m_b + logf(s_b);
            float tgt_logit = load_logit(row + (unsigned int)target);
            losses[b] = lse - tgt_logit;
        }
    }
}

extern "C" __global__ void cross_entropy_f32(
    const float* __restrict__ logits,
    const int*   __restrict__ targets,
    float*       __restrict__ losses,
    unsigned int B,
    unsigned int V,
    int ignore_index)
{
    cross_entropy_impl<float>(logits, targets, losses, B, V, ignore_index);
}

extern "C" __global__ void cross_entropy_f16(
    const __half* __restrict__ logits,
    const int*    __restrict__ targets,
    float*        __restrict__ losses,
    unsigned int B,
    unsigned int V,
    int ignore_index)
{
    cross_entropy_impl<__half>(logits, targets, losses, B, V, ignore_index);
}

extern "C" __global__ void cross_entropy_bf16(
    const __nv_bfloat16* __restrict__ logits,
    const int*           __restrict__ targets,
    float*               __restrict__ losses,
    unsigned int B,
    unsigned int V,
    int ignore_index)
{
    cross_entropy_impl<__nv_bfloat16>(logits, targets, losses, B, V, ignore_index);
}
