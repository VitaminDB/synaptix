#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Conv2d (direct): один thread = один output pixel.
// input  [B, C_in, H, W]               row-major
// weight [C_out, C_in, K_h, K_w]       row-major
// bias   [C_out] optional
// output [B, C_out, H_out, W_out]
//
// H_out = (H + 2*pad_h - K_h) / stride_h + 1
// W_out = (W + 2*pad_w - K_w) / stride_w + 1
//
// Grid: (B * C_out, H_out, W_out_blocks); block: (W_BLOCK, 1, 1).

__device__ __forceinline__ float load_f(const float* p) { return *p; }
__device__ __forceinline__ float load_f(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_f(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void store_f(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_f(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_f(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void conv2d_impl(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int has_bias,
    T* __restrict__ output,
    int B, int C_in, int H, int W,
    int C_out, int Kh, int Kw,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int H_out, int W_out
) {
    int bc = blockIdx.x;
    int b = bc / C_out;
    int c_out = bc % C_out;
    int h_out = blockIdx.y;
    int w_out = blockIdx.z * blockDim.x + threadIdx.x;
    if (w_out >= W_out || h_out >= H_out || b >= B) return;

    float acc = 0.0f;
    int h_in_base = h_out * stride_h - pad_h;
    int w_in_base = w_out * stride_w - pad_w;
    for (int c = 0; c < C_in; c++) {
        for (int kh = 0; kh < Kh; kh++) {
            int h_in = h_in_base + kh;
            if (h_in < 0 || h_in >= H) continue;
            for (int kw = 0; kw < Kw; kw++) {
                int w_in = w_in_base + kw;
                if (w_in < 0 || w_in >= W) continue;
                float x = load_f(input + ((((size_t)b * C_in + c) * H + h_in) * W + w_in));
                float w = load_f(weight + ((((size_t)c_out * C_in + c) * Kh + kh) * Kw + kw));
                acc += x * w;
            }
        }
    }
    if (has_bias) acc += load_f(bias + c_out);
    store_f(output + ((((size_t)b * C_out + c_out) * H_out + h_out) * W_out + w_out), acc);
}

extern "C" __global__ void conv2d_direct_f32(
    const float* input, const float* weight, const float* bias, int has_bias,
    float* output, int B, int C_in, int H, int W, int C_out, int Kh, int Kw,
    int stride_h, int stride_w, int pad_h, int pad_w, int H_out, int W_out
) {
    conv2d_impl<float>(input, weight, bias, has_bias, output, B, C_in, H, W,
                       C_out, Kh, Kw, stride_h, stride_w, pad_h, pad_w, H_out, W_out);
}

extern "C" __global__ void conv2d_direct_f16(
    const __half* input, const __half* weight, const __half* bias, int has_bias,
    __half* output, int B, int C_in, int H, int W, int C_out, int Kh, int Kw,
    int stride_h, int stride_w, int pad_h, int pad_w, int H_out, int W_out
) {
    conv2d_impl<__half>(input, weight, bias, has_bias, output, B, C_in, H, W,
                        C_out, Kh, Kw, stride_h, stride_w, pad_h, pad_w, H_out, W_out);
}

extern "C" __global__ void conv2d_direct_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, int has_bias,
    __nv_bfloat16* output, int B, int C_in, int H, int W, int C_out, int Kh, int Kw,
    int stride_h, int stride_w, int pad_h, int pad_w, int H_out, int W_out
) {
    conv2d_impl<__nv_bfloat16>(input, weight, bias, has_bias, output, B, C_in, H, W,
                               C_out, Kh, Kw, stride_h, stride_w, pad_h, pad_w, H_out, W_out);
}
