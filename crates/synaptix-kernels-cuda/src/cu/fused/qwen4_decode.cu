// Слитые ядра мелких этапов Qwen4Exp (gated-residual потоки, PLE, сборка
// MoE): каждое заменяет цепочку из 5–12 элементарных операций тензорного
// пути. Счёт в F32, вход/выход F16. Все ядра построчные — годятся и на
// префилле (T строк), и на декоде (T = 1).
#include <cuda_fp16.h>

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ __forceinline__ float block_sum(float v, float* red) {
    // warp-редукция, затем по варпам через shared (red — blockDim/32 float).
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? red[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    float r = red[0];
    __syncthreads();
    return r;
}

// out[r, g*G + i] = x[r, g*G + i] * rsqrt(mean_i(x²) + eps) * w[g*G + i]
// Блок на пару (строка, группа); grid = rows * groups.
extern "C" __global__ void group_rms_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    __half* __restrict__ out,
    unsigned int rows,
    unsigned int groups,
    unsigned int group,
    float eps
) {
    __shared__ float red[32];
    unsigned int b = blockIdx.x;
    if (b >= rows * groups) return;
    unsigned int r = b / groups, g = b % groups;
    unsigned long long base = (unsigned long long)r * groups * group + (unsigned long long)g * group;
    float ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < group; i += blockDim.x) {
        float v = __half2float(x[base + i]);
        ss += v * v;
    }
    ss = block_sum(ss, red);
    float inv = rsqrtf(ss / (float)group + eps);
    for (unsigned int i = threadIdx.x; i < group; i += blockDim.x) {
        float v = __half2float(x[base + i]) * inv * __half2float(w[g * group + i]);
        out[base + i] = __float2half(v);
    }
}

// out[r, i] = (1/hc) · Σ_c sigmoid(up[r, c*H + i]) · normed[r, c*H + i]
extern "C" __global__ void hc_mix_f16(
    const __half* __restrict__ up,
    const __half* __restrict__ normed,
    __half* __restrict__ out,
    unsigned int rows,
    unsigned int hc,
    unsigned int H
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)rows * H;
    if (idx >= total) return;
    unsigned int r = (unsigned int)(idx / H), i = (unsigned int)(idx % H);
    unsigned long long base = (unsigned long long)r * hc * H + i;
    float acc = 0.0f;
    for (unsigned int c = 0; c < hc; ++c) {
        unsigned long long p = base + (unsigned long long)c * H;
        acc += sigmoidf_(__half2float(up[p])) * __half2float(normed[p]);
    }
    out[idx] = __float2half(acc / (float)hc);
}

// out[r, c*H + i] = hyper[r, c*H + i] + block[r, i] · w[r, c]
extern "C" __global__ void hc_inject_f16(
    const __half* __restrict__ hyper,
    const __half* __restrict__ block,
    const __half* __restrict__ w,
    __half* __restrict__ out,
    unsigned int rows,
    unsigned int hc,
    unsigned int H
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)rows * hc * H;
    if (idx >= total) return;
    unsigned int r = (unsigned int)(idx / ((unsigned long long)hc * H));
    unsigned int rem = (unsigned int)(idx % ((unsigned long long)hc * H));
    unsigned int c = rem / H, i = rem % H;
    float v = __half2float(hyper[idx])
        + __half2float(block[(unsigned long long)r * H + i]) * __half2float(w[r * hc + c]);
    out[idx] = __float2half(v);
}

// out = act(x · scale): act 0 — silu, 1 — sigmoid, 2 — 2·sigmoid
extern "C" __global__ void scale_act_f16(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    unsigned int n,
    float scale,
    unsigned int act
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]) * scale;
    float s = sigmoidf_(v);
    float r = (act == 0u) ? v * s : (act == 1u) ? s : 2.0f * s;
    out[i] = __float2half(r);
}

// out[t, i] = Σ_j w[t*k + j] · parts[t*k + j, i]
extern "C" __global__ void weighted_rows_sum_f16(
    const __half* __restrict__ parts,
    const float* __restrict__ w,
    __half* __restrict__ out,
    unsigned int t,
    unsigned int k,
    unsigned int H
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)t * H;
    if (idx >= total) return;
    unsigned int row = (unsigned int)(idx / H), i = (unsigned int)(idx % H);
    float acc = 0.0f;
    for (unsigned int j = 0; j < k; ++j) {
        unsigned int p = row * k + j;
        acc += w[p] * __half2float(parts[(unsigned long long)p * H + i]);
    }
    out[idx] = __float2half(acc);
}
