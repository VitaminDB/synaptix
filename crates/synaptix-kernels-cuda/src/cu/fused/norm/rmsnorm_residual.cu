#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Fused: residual = x + residual; y = RMSNorm(residual) * weight
//
// Заменяет 2 kernel calls (1: residual = x + residual; 2: y = rms_norm(residual))
// на 1 launch — экономит memory pass на residual buffer.
//
// Pre-norm transformer block обычно делает:
//   h = h + sa(ln(h))
//   h = h + mlp(ln(h))
// Это значит layer i+1's ln использует i's residual output. Если совмещать
// `h += sa_out` и `ln(h)` в один kernel — экономия одного полного pass'а по
// hidden buffer (на Qwen3.6 ~80 GB/s сэкономлено).
//
// Layout: x, residual, y — (batch, hidden) row-major; weight — (hidden,) F16.
// Grid: (batch, 1, 1), block: (BLOCK, 1, 1). BLOCK = 256 — каждый thread
// проходит strided loop по hidden.

__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ float ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void rmsnorm_residual_impl(
    const T* __restrict__ x,
    T*       __restrict__ residual,
    const T* __restrict__ weight,
    T*       __restrict__ y,
    int batch,
    int hidden,
    float eps,
    int qwen
) {
    int row = blockIdx.x;
    if (row >= batch) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    const T* x_row = x + (long long)row * hidden;
    T*       r_row = residual + (long long)row * hidden;
    T*       y_row = y + (long long)row * hidden;

    __shared__ float warp_sums[32];
    __shared__ float s_rms;

    // ---- Pass 1: residual += x, и одновременно собираем sumsq в F32. ----
    float local_sumsq = 0.0f;
    for (int t = tid; t < hidden; t += bs) {
        float xv = ld(x_row + t);
        float rv = ld(r_row + t);
        float v = xv + rv;
        st(r_row + t, v);
        local_sumsq += v * v;
    }
    unsigned int mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sumsq += __shfl_down_sync(mask, local_sumsq, off, 32);
    }
    int warp = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) warp_sums[warp] = local_sumsq;
    __syncthreads();
    int num_warps = (bs + 31) >> 5;
    if (warp == 0) {
        float v = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(mask, v, off, 32);
        }
        if (lane == 0) {
            float mean = v / (float)hidden;
            s_rms = rsqrtf(mean + eps);
        }
    }
    __syncthreads();
    float rms = s_rms;

    // ---- Pass 2: y = (residual * rms) * weight (qwen: weight = 1 + w) ----
    for (int t = tid; t < hidden; t += bs) {
        float rv = ld(r_row + t);
        float wv = ld(weight + t);
        float scale = qwen ? (1.0f + wv) : wv;
        st(y_row + t, scale * rv * rms);
    }
}

// Out-of-place вариант: hidden_out = x + residual (НЕ мутирует residual);
// y = RMSNorm(hidden_out) * weight. Для Tensor-семантики (residual может быть
// shared-буфером). sumsq берётся от f16-округлённого hidden_out (матчит
// decomposed-путь add→rms_norm: hidden=f16(x+r), затем norm читает f16).
template <typename T>
__device__ __forceinline__ void rmsnorm_residual_split_impl(
    const T* __restrict__ x,
    const T* __restrict__ residual,
    const T* __restrict__ weight,
    T*       __restrict__ hidden_out,
    T*       __restrict__ y,
    int batch,
    int hidden,
    float eps,
    int qwen
) {
    int row = blockIdx.x;
    if (row >= batch) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    const T* x_row = x + (long long)row * hidden;
    const T* r_row = residual + (long long)row * hidden;
    T*       h_row = hidden_out + (long long)row * hidden;
    T*       y_row = y + (long long)row * hidden;

    __shared__ float warp_sums[32];
    __shared__ float s_rms;

    float local_sumsq = 0.0f;
    for (int t = tid; t < hidden; t += bs) {
        float v = ld(x_row + t) + ld(r_row + t);
        st(h_row + t, v);            // f16-округление
        float vr = ld(h_row + t);    // читаем округлённое → совпадает с decomposed sumsq
        local_sumsq += vr * vr;
    }
    unsigned int mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sumsq += __shfl_down_sync(mask, local_sumsq, off, 32);
    }
    int warp = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) warp_sums[warp] = local_sumsq;
    __syncthreads();
    int num_warps = (bs + 31) >> 5;
    if (warp == 0) {
        float v = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(mask, v, off, 32);
        }
        if (lane == 0) {
            float mean = v / (float)hidden;
            s_rms = rsqrtf(mean + eps);
        }
    }
    __syncthreads();
    float rms = s_rms;

    for (int t = tid; t < hidden; t += bs) {
        float rv = ld(h_row + t);
        float wv = ld(weight + t);
        float scale = qwen ? (1.0f + wv) : wv;
        st(y_row + t, scale * rv * rms);
    }
}

extern "C" __global__ void rmsnorm_residual_split_f32(
    const float* x, const float* residual, const float* w, float* hidden_out, float* y,
    int batch, int hidden, float eps, int qwen
) { rmsnorm_residual_split_impl<float>(x, residual, w, hidden_out, y, batch, hidden, eps, qwen); }

extern "C" __global__ void rmsnorm_residual_split_f16(
    const __half* x, const __half* residual, const __half* w, __half* hidden_out, __half* y,
    int batch, int hidden, float eps, int qwen
) { rmsnorm_residual_split_impl<__half>(x, residual, w, hidden_out, y, batch, hidden, eps, qwen); }

extern "C" __global__ void rmsnorm_residual_split_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* residual, const __nv_bfloat16* w,
    __nv_bfloat16* hidden_out, __nv_bfloat16* y,
    int batch, int hidden, float eps, int qwen
) { rmsnorm_residual_split_impl<__nv_bfloat16>(x, residual, w, hidden_out, y, batch, hidden, eps, qwen); }

extern "C" __global__ void rmsnorm_residual_f32(
    const float* x, float* residual, const float* w, float* y,
    int batch, int hidden, float eps, int qwen
) { rmsnorm_residual_impl<float>(x, residual, w, y, batch, hidden, eps, qwen); }

extern "C" __global__ void rmsnorm_residual_f16(
    const __half* x, __half* residual, const __half* w, __half* y,
    int batch, int hidden, float eps, int qwen
) { rmsnorm_residual_impl<__half>(x, residual, w, y, batch, hidden, eps, qwen); }

extern "C" __global__ void rmsnorm_residual_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* residual,
    const __nv_bfloat16* w, __nv_bfloat16* y,
    int batch, int hidden, float eps, int qwen
) { rmsnorm_residual_impl<__nv_bfloat16>(x, residual, w, y, batch, hidden, eps, qwen); }
