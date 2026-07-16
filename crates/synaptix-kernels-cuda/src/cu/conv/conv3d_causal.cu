#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Causal Conv3d (LTX-VAE style): causal-pad по времени (слева Kt-1 нулей),
// VALID по пространству. `stride` применяется ко всем 3 осям равномерно.
//
// input  [B, C_in, T, H, W]                  row-major
// weight [C_out, C_in, Kt, Kh, Kw]           row-major
// bias   [C_out] optional
// output [B, C_out, T_out, H_out, W_out]
//
// T_out = (T - 1) / stride + 1     (после causal-padding: T_padded = T + Kt-1, и (T_padded-Kt)/s+1)
// H_out = (H - Kh) / stride + 1    (без padding)
// W_out = (W - Kw) / stride + 1    (без padding)
//
// Grid: (B * C_out, T_out * H_out, W_out_blocks); block: (W_BLOCK, 1, 1).
// F32-accumulator всегда (для F16/BF16 — mixed-precision contract).

__device__ __forceinline__ float load_f(const float* p) { return *p; }
__device__ __forceinline__ float load_f(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_f(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void store_f(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_f(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_f(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename Scalar>
__device__ __forceinline__ void conv3d_causal_impl(
    const Scalar* __restrict__ input,
    const Scalar* __restrict__ weight,
    const Scalar* __restrict__ bias,
    int has_bias,
    Scalar* __restrict__ output,
    int B, int C_in, int T, int H, int W,
    int C_out, int Kt, int Kh, int Kw,
    int stride,
    int T_out, int H_out, int W_out
) {
    int bc = blockIdx.x;
    int b = bc / C_out;
    int c_out = bc % C_out;
    int th = blockIdx.y;
    int t_out = th / H_out;
    int h_out = th % H_out;
    int w_out = blockIdx.z * blockDim.x + threadIdx.x;
    if (w_out >= W_out || t_out >= T_out || h_out >= H_out || b >= B) return;

    // Causal-padding по T: tp = t_out*stride + dt; ti = tp - (Kt-1).
    // Если ti < 0  → положение слева от первого реального сэмпла (нули, skip).
    // Если ti >= T → справа за пределами (не должно случаться при стандартных формулах).
    int h_in_base = h_out * stride;
    int w_in_base = w_out * stride;

    float acc = 0.0f;
    for (int ci = 0; ci < C_in; ci++) {
        for (int kt = 0; kt < Kt; kt++) {
            int tp = t_out * stride + kt;
            if (tp < Kt - 1) continue;
            int ti = tp - (Kt - 1);
            if (ti >= T) continue;
            for (int kh = 0; kh < Kh; kh++) {
                int hi = h_in_base + kh;
                if (hi >= H) continue;
                for (int kw = 0; kw < Kw; kw++) {
                    int wi = w_in_base + kw;
                    if (wi >= W) continue;
                    size_t i_off = ((((size_t)b * C_in + ci) * T + ti) * H + hi) * W + wi;
                    size_t w_off = ((((size_t)c_out * C_in + ci) * Kt + kt) * Kh + kh) * Kw + kw;
                    acc += load_f(input + i_off) * load_f(weight + w_off);
                }
            }
        }
    }
    if (has_bias) acc += load_f(bias + c_out);
    size_t o_off = ((((size_t)b * C_out + c_out) * T_out + t_out) * H_out + h_out) * W_out + w_out;
    store_f(output + o_off, acc);
}

extern "C" __global__ void conv3d_causal_f32(
    const float* input, const float* weight, const float* bias, int has_bias,
    float* output, int B, int C_in, int T, int H, int W,
    int C_out, int Kt, int Kh, int Kw, int stride,
    int T_out, int H_out, int W_out
) {
    conv3d_causal_impl<float>(input, weight, bias, has_bias, output,

                              B, C_in, T, H, W, C_out, Kt, Kh, Kw, stride,
                              T_out, H_out, W_out);
}

extern "C" __global__ void conv3d_causal_f16(
    const __half* input, const __half* weight, const __half* bias, int has_bias,
    __half* output, int B, int C_in, int T, int H, int W,
    int C_out, int Kt, int Kh, int Kw, int stride,
    int T_out, int H_out, int W_out
) {
    conv3d_causal_impl<__half>(input, weight, bias, has_bias, output,
                               B, C_in, T, H, W, C_out, Kt, Kh, Kw, stride,
                               T_out, H_out, W_out);
}

extern "C" __global__ void conv3d_causal_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, int has_bias,
    __nv_bfloat16* output, int B, int C_in, int T, int H, int W,
    int C_out, int Kt, int Kh, int Kw, int stride,
    int T_out, int H_out, int W_out
) {
    conv3d_causal_impl<__nv_bfloat16>(input, weight, bias, has_bias, output,
                                       B, C_in, T, H, W, C_out, Kt, Kh, Kw, stride,
                                       T_out, H_out, W_out);
}
