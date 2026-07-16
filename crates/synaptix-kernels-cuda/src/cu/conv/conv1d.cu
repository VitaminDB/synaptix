#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Conv1d (direct): один thread = один output element.
// input  [B, C_in, L]      row-major
// weight [C_out, C_in, K]  row-major
// bias   [C_out]           optional (nullptr ⟹ skip)
// output [B, C_out, L_out] row-major
//
// L_out = (L + 2*pad - K) / stride + 1
// Padding zeros реализовано через guard (skip out-of-bounds reads).
//
// Grid: (B * C_out, L_out, 1); block: (BLOCK, 1, 1) — каждый thread берёт свой
// output element по l_out. Если BLOCK > L_out, лишние threads ничего не делают.

__device__ __forceinline__ float load_f(const float* p) { return *p; }
__device__ __forceinline__ float load_f(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_f(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void store_f(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_f(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_f(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void conv1d_impl(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int has_bias,
    T* __restrict__ output,
    int B, int C_in, int L,
    int C_out, int K,
    int stride, int padding,
    int L_out
) {
    int bc = blockIdx.x;
    int b = bc / C_out;
    int c_out = bc % C_out;
    int l_out = blockIdx.y * blockDim.x + threadIdx.x;
    if (l_out >= L_out || b >= B) return;

    float acc = 0.0f;
    int l_in_base = l_out * stride - padding;
    for (int c = 0; c < C_in; c++) {
        for (int kk = 0; kk < K; kk++) {
            int l_in = l_in_base + kk;
            if (l_in < 0 || l_in >= L) continue;
            float x = load_f(input + ((size_t)b * C_in + c) * L + l_in);
            float w = load_f(weight + ((size_t)c_out * C_in + c) * K + kk);
            acc += x * w;
        }
    }
    if (has_bias) acc += load_f(bias + c_out);
    store_f(output + ((size_t)b * C_out + c_out) * L_out + l_out, acc);
}

extern "C" __global__ void conv1d_direct_f32(
    const float* input, const float* weight, const float* bias, int has_bias,
    float* output, int B, int C_in, int L, int C_out, int K,
    int stride, int padding, int L_out
) {
    conv1d_impl<float>(input, weight, bias, has_bias, output, B, C_in, L, C_out, K,
                       stride, padding, L_out);
}

extern "C" __global__ void conv1d_direct_f16(
    const __half* input, const __half* weight, const __half* bias, int has_bias,
    __half* output, int B, int C_in, int L, int C_out, int K,
    int stride, int padding, int L_out
) {
    conv1d_impl<__half>(input, weight, bias, has_bias, output, B, C_in, L, C_out, K,
                        stride, padding, L_out);
}

extern "C" __global__ void conv1d_direct_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, int has_bias,
    __nv_bfloat16* output, int B, int C_in, int L, int C_out, int K,
    int stride, int padding, int L_out
) {
    conv1d_impl<__nv_bfloat16>(input, weight, bias, has_bias, output,
                               B, C_in, L, C_out, K, stride, padding, L_out);
}
