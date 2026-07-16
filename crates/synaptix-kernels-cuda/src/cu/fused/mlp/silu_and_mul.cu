#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Pointwise fused: out[i] = silu(gate[i]) * up[i],
// silu(x) = x / (1 + exp(-x)). Один проход памяти вместо двух (chain
// `gate.silu().mul(&up)` делал read+write для silu и read+read+write для mul =
// 4 trip'а памяти; здесь 3 trip'а: read gate + read up + write out).

extern "C" {

__device__ __forceinline__ float silu_f(float v) {
    return v / (1.0f + __expf(-v));
}

__global__ void silu_and_mul_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    unsigned int total
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float g = gate[i];
    float u = up[i];
    out[i] = silu_f(g) * u;
}

__global__ void silu_and_mul_f16(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ out,
    unsigned int total
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    out[i] = __float2half(silu_f(g) * u);
}

__global__ void silu_and_mul_bf16(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    unsigned int total
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    out[i] = __float2bfloat16(silu_f(g) * u);
}

}
