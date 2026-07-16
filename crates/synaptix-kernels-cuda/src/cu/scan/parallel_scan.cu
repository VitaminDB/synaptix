#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Chunked parallel scan baseline (per-row, single-block).
//
// Алгоритм: каждый thread считает sequential prefix по своему чанку, потом
// делает block-wide scan на per-chunk totals (Hillis-Steele в smem),
// потом fixup каждого thread'а offset'ом блочного scan'а. Critical path
// O(log BLOCK + N/BLOCK), work O(N).
//
// Один CUDA block per row. BLOCK=256 threads, MAX_N=8192 (32 elements/thread).
// Поддержка inclusive/exclusive через флаг.

constexpr int BLOCK = 256;
constexpr int MAX_PER_THREAD = 32;

__device__ __forceinline__ float load_f32(const float* p) { return *p; }
__device__ __forceinline__ float load_f32(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_f32(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void store_t(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_t(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_t(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void scan_row_impl(
    const T* __restrict__ x_row,
    T*       __restrict__ y_row,
    int N,
    int inclusive)
{
    int tid = threadIdx.x;
    int per_thread = (N + BLOCK - 1) / BLOCK;

    // Phase 1: per-thread sequential prefix scan, накапливаем total.
    float local[MAX_PER_THREAD];
    float my_total = 0.f;
    int   start = tid * per_thread;
    int   end   = start + per_thread;
    if (end > N) end = N;
    for (int i = start; i < end; i++) {
        float v = load_f32(x_row + i);
        my_total += v;
        local[i - start] = my_total;
    }

    // Phase 2: block-wide exclusive scan по per-thread totals (Hillis-Steele).
    __shared__ float s_totals[BLOCK];
    s_totals[tid] = my_total;
    __syncthreads();

    float offset = 0.f;
    for (int off = 1; off < BLOCK; off <<= 1) {
        float v = (tid >= off) ? s_totals[tid - off] : 0.f;
        __syncthreads();
        s_totals[tid] += v;
        __syncthreads();
    }
    // s_totals[tid] = inclusive prefix sum. Exclusive prefix = s_totals[tid-1] (0 для tid==0).
    offset = (tid > 0) ? s_totals[tid - 1] : 0.f;

    // Phase 3: финальный write с offset'ом.
    if (inclusive) {
        for (int i = start; i < end; i++) {
            store_t(y_row + i, offset + local[i - start]);
        }
    } else {
        for (int i = start; i < end; i++) {
            float incl = offset + local[i - start];
            float v = load_f32(x_row + i);
            store_t(y_row + i, incl - v);
        }
    }
}

extern "C" __global__ void scan_sum_f32(
    const float* x, float* y, int B, int N, int inclusive)
{
    int row = blockIdx.x;
    if (row >= B) return;
    scan_row_impl<float>(x + (size_t)row * N, y + (size_t)row * N, N, inclusive);
}

extern "C" __global__ void scan_sum_f16(
    const __half* x, __half* y, int B, int N, int inclusive)
{
    int row = blockIdx.x;
    if (row >= B) return;
    scan_row_impl<__half>(x + (size_t)row * N, y + (size_t)row * N, N, inclusive);
}

extern "C" __global__ void scan_sum_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* y, int B, int N, int inclusive)
{
    int row = blockIdx.x;
    if (row >= B) return;
    scan_row_impl<__nv_bfloat16>(x + (size_t)row * N, y + (size_t)row * N, N, inclusive);
}
