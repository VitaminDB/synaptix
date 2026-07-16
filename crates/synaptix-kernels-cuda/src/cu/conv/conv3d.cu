#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Conv3d (direct): один thread = один output voxel.
// input  [B, C_in, D, H, W]                  row-major
// weight [C_out, C_in, Kd, Kh, Kw]           row-major
// bias   [C_out] optional
// output [B, C_out, D_out, H_out, W_out]
//
// D_out = (D + 2*pad_d - Kd) / stride_d + 1
// H_out = (H + 2*pad_h - Kh) / stride_h + 1
// W_out = (W + 2*pad_w - Kw) / stride_w + 1
//
// Grid: (B * C_out * D_out, H_out, W_out_blocks); block: (W_BLOCK, 1, 1).
// d_out свёрнут в gridDim.x (лимит 2^31), а не в gridDim.y — иначе D_out*H_out
// на FullHD (121*1088) превышает gridDim.y-лимит 65535.

__device__ __forceinline__ float load_f(const float* p) { return *p; }
__device__ __forceinline__ float load_f(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_f(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void store_f(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_f(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_f(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void conv3d_impl(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int has_bias,
    T* __restrict__ output,
    int B, int C_in, int D, int H, int W,
    int C_out, int Kd, int Kh, int Kw,
    int sd, int sh, int sw,
    int pd, int ph, int pw,
    int D_out, int H_out, int W_out
) {
    int bcd = blockIdx.x;
    int d_out = bcd % D_out;
    int t = bcd / D_out;
    int c_out = t % C_out;
    int b = t / C_out;
    int h_out = blockIdx.y;
    int w_out = blockIdx.z * blockDim.x + threadIdx.x;
    if (w_out >= W_out || d_out >= D_out || h_out >= H_out || b >= B) return;

    float acc = 0.0f;
    int d_in_base = d_out * sd - pd;
    int h_in_base = h_out * sh - ph;
    int w_in_base = w_out * sw - pw;
    for (int c = 0; c < C_in; c++) {
        for (int kd = 0; kd < Kd; kd++) {
            int d_in = d_in_base + kd;
            if (d_in < 0 || d_in >= D) continue;
            for (int kh = 0; kh < Kh; kh++) {
                int h_in = h_in_base + kh;
                if (h_in < 0 || h_in >= H) continue;
                for (int kw = 0; kw < Kw; kw++) {
                    int w_in = w_in_base + kw;
                    if (w_in < 0 || w_in >= W) continue;
                    size_t i_off = ((((((size_t)b * C_in + c) * D + d_in) * H + h_in)) * W + w_in);
                    size_t w_off = ((((((size_t)c_out * C_in + c) * Kd + kd) * Kh + kh)) * Kw + kw);
                    acc += load_f(input + i_off) * load_f(weight + w_off);
                }
            }
        }
    }
    if (has_bias) acc += load_f(bias + c_out);
    size_t o_off = ((((((size_t)b * C_out + c_out) * D_out + d_out) * H_out + h_out)) * W_out + w_out);
    store_f(output + o_off, acc);
}

extern "C" __global__ void conv3d_direct_f32(
    const float* input, const float* weight, const float* bias, int has_bias,
    float* output, int B, int C_in, int D, int H, int W, int C_out, int Kd, int Kh, int Kw,
    int sd, int sh, int sw, int pd, int ph, int pw, int D_out, int H_out, int W_out
) {
    conv3d_impl<float>(input, weight, bias, has_bias, output, B, C_in, D, H, W,
                       C_out, Kd, Kh, Kw, sd, sh, sw, pd, ph, pw, D_out, H_out, W_out);
}

extern "C" __global__ void conv3d_direct_f16(
    const __half* input, const __half* weight, const __half* bias, int has_bias,
    __half* output, int B, int C_in, int D, int H, int W, int C_out, int Kd, int Kh, int Kw,
    int sd, int sh, int sw, int pd, int ph, int pw, int D_out, int H_out, int W_out
) {
    conv3d_impl<__half>(input, weight, bias, has_bias, output, B, C_in, D, H, W,
                        C_out, Kd, Kh, Kw, sd, sh, sw, pd, ph, pw, D_out, H_out, W_out);
}

extern "C" __global__ void conv3d_direct_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, int has_bias,
    __nv_bfloat16* output, int B, int C_in, int D, int H, int W, int C_out, int Kd, int Kh, int Kw,
    int sd, int sh, int sw, int pd, int ph, int pw, int D_out, int H_out, int W_out
) {
    conv3d_impl<__nv_bfloat16>(input, weight, bias, has_bias, output, B, C_in, D, H, W,
                               C_out, Kd, Kh, Kw, sd, sh, sw, pd, ph, pw, D_out, H_out, W_out);
}
