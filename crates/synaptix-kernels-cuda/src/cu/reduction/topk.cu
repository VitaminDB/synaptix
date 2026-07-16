#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Top-K logits per row для sampling. Один block per row, K passes блок-редукции
// с bitmask в dynamic shared memory. Маска размером (V+31)/32 * 4 bytes;
// caller должен передать `shared_mem_bytes` >= это.
//
// Pass k: каждый thread сканирует свой strided участок row, выбирает локальный
// (val, idx) с проверкой что бит idx в mask == 0. Block-wide arg-max reduce
// (val + idx через __shfl). Thread 0 пишет out_vals[k], out_idx[k], ставит бит.

constexpr int BLOCK = 256;

__device__ __forceinline__ float load_f32(const float* p) { return *p; }
__device__ __forceinline__ float load_f32(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_f32(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ float neg_inf_f32() { return -1.f / 0.f; }

template <typename T>
__device__ __forceinline__ void topk_impl(
    const T* __restrict__ logits,   // (B, V)
    float*   __restrict__ out_vals, // (B, K)
    int*     __restrict__ out_idx,  // (B, K)
    int B, int V, int K)
{
    int row = blockIdx.x;
    if (row >= B) return;
    int tid = threadIdx.x;
    const T* row_ptr = logits + (size_t)row * V;

    int mask_words = (V + 31) >> 5;
    extern __shared__ unsigned int s_mem[];
    unsigned int* s_mask = s_mem;
    float* s_block_v = (float*)(s_mem + mask_words);
    int*   s_block_i = (int*)(s_block_v + BLOCK);

    for (int i = tid; i < mask_words; i += BLOCK) {
        s_mask[i] = 0u;
    }
    __syncthreads();

    for (int k = 0; k < K; k++) {
        float best_v = neg_inf_f32();
        int best_i = -1;

        for (int i = tid; i < V; i += BLOCK) {
            unsigned int word = s_mask[i >> 5];
            if (word & (1u << (i & 31))) continue;
            float v = load_f32(row_ptr + i);
            if (v > best_v) {
                best_v = v;
                best_i = i;
            }
        }
        s_block_v[tid] = best_v;
        s_block_i[tid] = best_i;
        __syncthreads();

        if (tid == 0) {
            float win_v = s_block_v[0];
            int   win_i = s_block_i[0];
            for (int t = 1; t < BLOCK; t++) {
                float v = s_block_v[t];
                int   i = s_block_i[t];
                if (v > win_v) { win_v = v; win_i = i; }
            }
            out_vals[(size_t)row * K + k] = win_v;
            out_idx [(size_t)row * K + k] = win_i;
            if (win_i >= 0) {
                s_mask[win_i >> 5] |= (1u << (win_i & 31));
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void topk_f32(
    const float* logits, float* out_vals, int* out_idx, int B, int V, int K
) { topk_impl<float>(logits, out_vals, out_idx, B, V, K); }

extern "C" __global__ void topk_f16(
    const __half* logits, float* out_vals, int* out_idx, int B, int V, int K
) { topk_impl<__half>(logits, out_vals, out_idx, B, V, K); }

extern "C" __global__ void topk_bf16(
    const __nv_bfloat16* logits, float* out_vals, int* out_idx, int B, int V, int K
) { topk_impl<__nv_bfloat16>(logits, out_vals, out_idx, B, V, K); }
