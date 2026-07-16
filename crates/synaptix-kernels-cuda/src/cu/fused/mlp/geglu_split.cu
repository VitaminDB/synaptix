#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Fused GEGLU split-activation: вход p[T, 2*I] (выход proj-Linear),
// выход h[T, I],  h[t,i] = p[t, i] * gelu_exact(p[t, I + i]).
// Первая половина = value, вторая = gate (как diffusers GEGLU). Один проход
// вместо narrow×2 + contiguous×2 + gelu + mul. gelu_exact = 0.5·x·(1+erf(x/√2)).

__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ float ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

__device__ __forceinline__ float gelu_exact_f(float v) {
    return 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
}

template <typename T>
__device__ __forceinline__ void geglu_split_impl(
    const T* __restrict__ inp, T* __restrict__ out, long long t, int inner
) {
    long long total = t * (long long)inner;
    long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        long long row = idx / inner;
        int i = (int)(idx - row * inner);
        long long base = row * (long long)inner * 2;
        float val = ld(inp + base + i);
        float gate = ld(inp + base + inner + i);
        st(out + idx, val * gelu_exact_f(gate));
    }
}

extern "C" __global__ void geglu_split_f32(const float* inp, float* out, long long t, int inner) {
    geglu_split_impl<float>(inp, out, t, inner);
}
extern "C" __global__ void geglu_split_f16(const __half* inp, __half* out, long long t, int inner) {
    geglu_split_impl<__half>(inp, out, t, inner);
}
extern "C" __global__ void geglu_split_bf16(const __nv_bfloat16* inp, __nv_bfloat16* out, long long t, int inner) {
    geglu_split_impl<__nv_bfloat16>(inp, out, t, inner);
}
