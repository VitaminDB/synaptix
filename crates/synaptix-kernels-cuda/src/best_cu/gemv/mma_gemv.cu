#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Шаблонные GEMV kernels для F16/BF16/F32: y[row] = sum_k W[row, k] * x[k].
// Один warp (32 lanes) = один output element, K-axis распределён между lanes
// (lane t обрабатывает K=t, t+32, t+64, ...), затем warp-reduce через
// __shfl_xor_sync. F32 accumulator.
//
// Drop-in замена cuBLAS-Lt matmul для M=1 (decode path). Row-major W/x/y,
// без pre-shuffle. Grid (N, 1, 1), block (32, 1, 1).
//
// Vectorized reads (uint4 = 16 bytes = 8 b16 elements или 4 F32):
//   F16: half2 = u32 (2 b16 за 4 bytes), 1 lane = K/32 элементов по 2 за раз
//   BF16: __nv_bfloat162 = u32, аналогично
//   F32: 1 F32 на lane за итерацию (или float2/float4 для лучшего bandwidth)

// ─────────────────────────────── F16 GEMV ───────────────────────────────

extern "C" __global__ void mma_gemv_f16(
    const __half* __restrict__ W,  // (N, K)
    const __half* __restrict__ x,  // (K,)
    __half*       __restrict__ y,  // (N,)
    unsigned int N,
    unsigned int K)
{
    unsigned int row = blockIdx.x;
    if (row >= N) return;
    unsigned int lane = threadIdx.x;
    if (lane >= 32u) return;

    unsigned int K2 = K >> 1;
    const __half2* w_row = reinterpret_cast<const __half2*>(W + row * K);
    const __half2* x_h2  = reinterpret_cast<const __half2*>(x);

    float acc = 0.f;
    for (unsigned int k = lane; k < K2; k += 32u) {
        __half2 wv = w_row[k];
        __half2 xv = x_h2[k];
        float2 wf = __half22float2(wv);
        float2 xf = __half22float2(xv);
        acc = fmaf(wf.x, xf.x, acc);
        acc = fmaf(wf.y, xf.y, acc);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
    }
    if (lane == 0u) {
        y[row] = __float2half(acc);
    }
}

// ─────────────────────────────── BF16 GEMV ──────────────────────────────

extern "C" __global__ void mma_gemv_bf16(
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16*       __restrict__ y,
    unsigned int N,
    unsigned int K)
{
    unsigned int row = blockIdx.x;
    if (row >= N) return;
    unsigned int lane = threadIdx.x;
    if (lane >= 32u) return;

    unsigned int K2 = K >> 1;
    const __nv_bfloat162* w_row = reinterpret_cast<const __nv_bfloat162*>(W + row * K);
    const __nv_bfloat162* x_b2  = reinterpret_cast<const __nv_bfloat162*>(x);

    float acc = 0.f;
    for (unsigned int k = lane; k < K2; k += 32u) {
        __nv_bfloat162 wv = w_row[k];
        __nv_bfloat162 xv = x_b2[k];
        float2 wf = __bfloat1622float2(wv);
        float2 xf = __bfloat1622float2(xv);
        acc = fmaf(wf.x, xf.x, acc);
        acc = fmaf(wf.y, xf.y, acc);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
    }
    if (lane == 0u) {
        y[row] = __float2bfloat16(acc);
    }
}

// ─────────────────────────────── F32 GEMV ───────────────────────────────

extern "C" __global__ void mma_gemv_f32(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float*       __restrict__ y,
    unsigned int N,
    unsigned int K)
{
    unsigned int row = blockIdx.x;
    if (row >= N) return;
    unsigned int lane = threadIdx.x;
    if (lane >= 32u) return;

    const float* w_row = W + row * K;

    float acc = 0.f;
    // Vec4 path: 4 F32 за раз = 16 bytes (uint4).
    unsigned int K4 = K >> 2;
    const float4* w4 = reinterpret_cast<const float4*>(w_row);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (unsigned int k = lane; k < K4; k += 32u) {
        float4 wv = w4[k];
        float4 xv = x4[k];
        acc = fmaf(wv.x, xv.x, acc);
        acc = fmaf(wv.y, xv.y, acc);
        acc = fmaf(wv.z, xv.z, acc);
        acc = fmaf(wv.w, xv.w, acc);
    }
    // Tail (K не кратно 4) — обрабатываем оставшиеся элементы.
    unsigned int tail_start = K4 << 2;
    for (unsigned int k = tail_start + lane; k < K; k += 32u) {
        acc = fmaf(w_row[k], x[k], acc);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
    }
    if (lane == 0u) {
        y[row] = acc;
    }
}
