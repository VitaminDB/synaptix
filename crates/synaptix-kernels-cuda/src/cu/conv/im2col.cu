#include <cuda_fp16.h>
#include <cuda_bf16.h>

// im2col (с row-tiling): input [B, C_in, H, W] (row-major) → col [m_count, K]
//   K = C_in * Kh * Kw,  логическая строка r ∈ [0,m_count) ↔ глобальная
//   m = m_offset + r,  m → (b, ho, wo)  (b = m / (H_out*W_out), ...)
//
// col[r, k] = input[b, c, h_in, w_in]  (или 0 при zero-pad / OOB)
//   k → (c, kh, kw);  h_in = ho*stride_h - pad_h + kh, w_in = wo*stride_w - pad_w + kw
//
// Тайлинг по m нужен, т.к. полный col[B*H_out*W_out, K] на больших spatial×каналах
// (VAE 1024²) не влезает в VRAM. Caller бьёт M на чанки по бюджету памяти.

__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ float ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void im2col_impl(
    const T* __restrict__ input,
    T* __restrict__ col,
    int C_in, int H, int W,
    int Kh, int Kw,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    long long m_offset, long long m_count
) {
    long long Kcols = (long long)C_in * Kh * Kw;
    long long total = m_count * Kcols;
    long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        long long r = idx / Kcols;
        long long k = idx - r * Kcols;
        long long m = m_offset + r;
        int wo = (int)(m % W_out);
        long long mt = m / W_out;
        int ho = (int)(mt % H_out);
        int b = (int)(mt / H_out);
        int kw = (int)(k % Kw);
        long long kt = k / Kw;
        int kh = (int)(kt % Kh);
        int c = (int)(kt / Kh);
        int h_in = ho * stride_h - pad_h + kh;
        int w_in = wo * stride_w - pad_w + kw;
        float v = 0.0f;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            v = ld(input + ((((long long)b * C_in + c) * H + h_in) * W + w_in));
        }
        st(col + idx, v);
    }
}

extern "C" __global__ void im2col_f32(
    const float* input, float* col,
    int C_in, int H, int W, int Kh, int Kw, int H_out, int W_out,
    int stride_h, int stride_w, int pad_h, int pad_w,
    long long m_offset, long long m_count
) {
    im2col_impl<float>(input, col, C_in, H, W, Kh, Kw, H_out, W_out,
                       stride_h, stride_w, pad_h, pad_w, m_offset, m_count);
}

extern "C" __global__ void im2col_f16(
    const __half* input, __half* col,
    int C_in, int H, int W, int Kh, int Kw, int H_out, int W_out,
    int stride_h, int stride_w, int pad_h, int pad_w,
    long long m_offset, long long m_count
) {
    im2col_impl<__half>(input, col, C_in, H, W, Kh, Kw, H_out, W_out,
                        stride_h, stride_w, pad_h, pad_w, m_offset, m_count);
}

extern "C" __global__ void im2col_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* col,
    int C_in, int H, int W, int Kh, int Kw, int H_out, int W_out,
    int stride_h, int stride_w, int pad_h, int pad_w,
    long long m_offset, long long m_count
) {
    im2col_impl<__nv_bfloat16>(input, col, C_in, H, W, Kh, Kw, H_out, W_out,
                               stride_h, stride_w, pad_h, pad_w, m_offset, m_count);
}
