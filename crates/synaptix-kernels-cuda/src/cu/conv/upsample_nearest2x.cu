#include <cuda_fp16.h>
#include <cuda_bf16.h>

// nearest-neighbour 2× upsample: in [B,C,H,W] → out [B,C,2H,2W]
//   out[b,c,ho,wo] = in[b,c,ho/2,wo/2]   (bit-exact к F.interpolate nearest ×2)
// Один thread = один выходной элемент (grid-stride loop). Чистое копирование —
// dtype-конверсии не нужны.

template <typename T>
__device__ __forceinline__ void upsample_nearest2x_impl(
    const T* __restrict__ input, T* __restrict__ output,
    int B, int C, int H, int W
) {
    int Ho = H * 2;
    int Wo = W * 2;
    long long total = (long long)B * C * Ho * Wo;
    long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        int wo = (int)(idx % Wo);
        long long t = idx / Wo;
        int ho = (int)(t % Ho);
        t /= Ho;
        int c = (int)(t % C);
        int b = (int)(t / C);
        int hi = ho >> 1;
        int wi = wo >> 1;
        output[idx] = input[(((long long)b * C + c) * H + hi) * W + wi];
    }
}

extern "C" __global__ void upsample_nearest2x_f32(
    const float* input, float* output, int B, int C, int H, int W
) { upsample_nearest2x_impl<float>(input, output, B, C, H, W); }

extern "C" __global__ void upsample_nearest2x_f16(
    const __half* input, __half* output, int B, int C, int H, int W
) { upsample_nearest2x_impl<__half>(input, output, B, C, H, W); }

extern "C" __global__ void upsample_nearest2x_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output, int B, int C, int H, int W
) { upsample_nearest2x_impl<__nv_bfloat16>(input, output, B, C, H, W); }
