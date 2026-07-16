#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Fused conv2d-эпилог: вход out2d[M, Cout] (M = B*H*W, row-major NHWC-flattened
// = выход im2col-GEMM), опц. bias[Cout] → выход out[B, Cout, H, W] (NCHW
// contiguous). Заменяет broadcast_add(bias) + permute([0,3,1,2]) +
// contiguous() — два прохода по памяти → один.
//
//   out[b,c,h,w] = out2d[(b*H + h)*W + w, c] + (has_bias ? bias[c] : 0)
//
// Запись coalesced по NCHW (idx идёт contiguous по w → h → c → b).
// Чтение out2d strided по c (stride=Cout), но через L2 / в пределах SM block —
// приемлемо.

__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ float ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void conv_epilogue_impl(
    const T* __restrict__ out2d,
    const T* __restrict__ bias,
    int has_bias,
    const T* __restrict__ residual,
    int has_residual,
    const T* __restrict__ temb_bc,
    int has_temb,
    T* __restrict__ out,
    int B, int C, int H, int W
) {
    long long total = (long long)B * C * H * W;
    long long step = (long long)gridDim.x * blockDim.x;
    long long HW = (long long)H * W;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        int w = (int)(idx % W);
        long long t = idx / W;
        int h = (int)(t % H);
        t /= H;
        int c = (int)(t % C);
        int b = (int)(t / C);
        long long src = ((long long)b * HW + (long long)h * W + w) * (long long)C + c;
        float v = ld(out2d + src);
        if (has_bias) v += ld(bias + c);
        if (has_temb) v += ld(temb_bc + (long long)b * C + c);
        if (has_residual) v += ld(residual + idx);
        st(out + idx, v);
    }
}

extern "C" __global__ void conv_epilogue_f32(
    const float* out2d, const float* bias, int has_bias,
    const float* residual, int has_residual,
    const float* temb_bc, int has_temb,
    float* out, int B, int C, int H, int W
) { conv_epilogue_impl<float>(out2d, bias, has_bias, residual, has_residual, temb_bc, has_temb, out, B, C, H, W); }

extern "C" __global__ void conv_epilogue_f16(
    const __half* out2d, const __half* bias, int has_bias,
    const __half* residual, int has_residual,
    const __half* temb_bc, int has_temb,
    __half* out, int B, int C, int H, int W
) { conv_epilogue_impl<__half>(out2d, bias, has_bias, residual, has_residual, temb_bc, has_temb, out, B, C, H, W); }

extern "C" __global__ void conv_epilogue_bf16(
    const __nv_bfloat16* out2d, const __nv_bfloat16* bias, int has_bias,
    const __nv_bfloat16* residual, int has_residual,
    const __nv_bfloat16* temb_bc, int has_temb,
    __nv_bfloat16* out, int B, int C, int H, int W
) { conv_epilogue_impl<__nv_bfloat16>(out2d, bias, has_bias, residual, has_residual, temb_bc, has_temb, out, B, C, H, W); }
