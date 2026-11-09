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

// ───────────── слитый gated-residual (hyper-connection) на декоде ─────────────
//
// Цепочка group_rms → квант → GEMV down → silu → квант → GEMV up → hc_mix и
// GEMV block_inject → 2·sigmoid шла девятью ядрами по 1–6 мкс, 96 раз на
// токен. Здесь — два ядра; веса down/up читаются из перемешанной NVFP4-копии
// с деквантом в регистрах, активация не квантуется (W4A16 — точнее прежнего
// W4A4).
//
//   hc_down:  normed = x·rsqrt(mean_g(x²)+eps)·w_norm (в shared, f32)
//             mix[r]  = silu(inv_hc · Σ_k down[r,k]·normed[k])        r < R
//             inj[c]  = 2·sigmoid(inv_hc · Σ_k inject[c,k]·normed[k]) c < hc
//             norms[g] = rsqrt(mean_g(x²)+eps)
//   hc_up:    mixed[i] = inv_hc · Σ_c sigmoid(Σ_k up[cH+i,k]·mix[k]) · normed[cH+i]

struct HcArgs {
    unsigned long long x, norm_w, down_w, down_s, inj_w, up_w, up_s;
    unsigned long long mix, norms, inj_out, mixed;
    unsigned int rows, hc, H, R;
    float eps, _pad;
};

__device__ __forceinline__ float hc_e4m3(unsigned char byte) {
    const unsigned short hb = (unsigned short)(((byte & 0x80u) << 8) | ((byte & 0x7Fu) << 7));
    return __half2float(__ushort_as_half(hb)) * 256.0f;
}
__device__ __forceinline__ float hc_e2m1(unsigned int nib) {
    const float mag = (float)((0xC8643210u >> ((nib & 7u) * 4u)) & 0xFu) * 0.5f;
    return __int_as_float(__float_as_int(mag) | ((nib & 8u) << 28));
}
__device__ __forceinline__ unsigned hc_scale_off(unsigned outer, unsigned block_col, unsigned sf_inner) {
    unsigned tile_row = outer >> 7, tile_col = block_col >> 2;
    unsigned lo = outer & 127u, li = block_col & 3u;
    return (tile_col * 4u + tile_row * sf_inner) * 128u + (lo & 31u) * 16u + (lo >> 5) * 4u + li;
}
// Σ_j w[row, 32s + j] · v[j] по под-блоку 32 перемешанной NVFP4-копии.
__device__ __forceinline__ float hc_val(float v) { return v; }
__device__ __forceinline__ float hc_val(__half v) { return __half2float(v); }
template <typename V>
__device__ __forceinline__ float hc_dot32(const unsigned char* shuf, const unsigned char* scales, unsigned row,
                                          unsigned K, unsigned s, const V* v) {
    const unsigned sf_inner = ((K + 63u) / 64u) * 4u;
    const size_t off = (size_t)(row >> 4) * (K >> 6) * 512u + (size_t)(s >> 1) * 512u + (row & 15u) * 32u + (s & 1u) * 16u;
    const uint4 raw = *reinterpret_cast<const uint4*>(shuf + off);
    const unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
    float acc = 0.0f;
    // Два масштаба под-блока лежат соседними байтами (чётное смещение).
    const unsigned short sc2 = *reinterpret_cast<const unsigned short*>(scales + hc_scale_off(row, s * 2u, sf_inner));
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const float sc = hc_e4m3((unsigned char)(sc2 >> (8 * h)));
        float part = 0.0f;
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const unsigned w = words[h * 2 + q];
            #pragma unroll
            for (int j = 0; j < 8; ++j) part += hc_e2m1((w >> (4 * j)) & 0xFu) * hc_val(v[h * 16 + q * 8 + j]);
        }
        acc += part * sc;
    }
    return acc;
}

// 8 half одной 16-байтной загрузкой: число последовательных обращений к
// глобальной памяти на нить и есть время ядра (варп ждёт каждую загрузку).
__device__ __forceinline__ void hc_ld8(const __half* p, float* v) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const __half2* h = reinterpret_cast<const __half2*>(&raw);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __half22float2(h[i]);
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
}

#ifndef HC_RPB
#define HC_RPB 4u   // строк down на блок (варпов на строку — 8 / HC_RPB)
#endif
// grid = (R/HC_RPB + (inj ? 1 : 0), rows), block = 256, shared = (hc·H + 64) f32.
// Блок: две строки down (по 4 варпа на строку) либо block_inject (hc строк).
extern "C" __global__ void hc_down_f16(HcArgs a) {
    extern __shared__ float sm[];
    const unsigned K = a.hc * a.H;
    float* red = sm;                // [8 варпов][hc ≤ 8]
    float* xs = sm + 64;            // [K] нормированный вход
    const unsigned t = blockIdx.y;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const __half* x = (const __half*)a.x + (size_t)t * K;
    const __half* nw = (const __half*)a.norm_w;
    const unsigned vecs = K >> 3, vec_per_g = a.H >> 3;

    float ss[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) ss[g] = 0.0f;
    for (unsigned v = tid; v < vecs; v += blockDim.x) {
        float f[8];
        hc_ld8(x + v * 8u, f);
        float s = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            xs[v * 8u + j] = f[j];
            s += f[j] * f[j];
        }
        ss[v / vec_per_g] += s;
    }
    for (unsigned g = 0; g < a.hc; ++g) {
        float s = ss[g];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) red[warp * 8u + g] = s;
    }
    __syncthreads();
    float inv[8];
    for (unsigned g = 0; g < a.hc; ++g) {
        float s = 0.0f;
        for (unsigned w = 0; w < (blockDim.x >> 5); ++w) s += red[w * 8u + g];
        inv[g] = rsqrtf(s / (float)a.H + a.eps);
    }
    for (unsigned v = tid; v < vecs; v += blockDim.x) {
        float f[8];
        hc_ld8(nw + v * 8u, f);
        const float ig = inv[v / vec_per_g];
        #pragma unroll
        for (int j = 0; j < 8; ++j) xs[v * 8u + j] *= ig * f[j];
    }
    if (blockIdx.x == 0 && tid < a.hc) ((float*)a.norms)[t * a.hc + tid] = inv[tid];
    __syncthreads();

    const float inv_hc = 1.0f / (float)a.hc;
    const unsigned down_blocks = a.R / HC_RPB;
    float* red2 = red;              // [8 варпов] частичные суммы строк
    if (blockIdx.x < down_blocks) {
        const unsigned wpr = 8u / HC_RPB;                   // варпов на строку
        const unsigned row = blockIdx.x * HC_RPB + warp / wpr;
        const unsigned sub = (warp % wpr) * 32u + lane;
        float acc = 0.0f;
        for (unsigned s = sub; s < (K >> 5); s += wpr * 32u) {
            acc += hc_dot32((const unsigned char*)a.down_w, (const unsigned char*)a.down_s, row, K, s, xs + s * 32u);
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) red2[warp] = acc;
        __syncthreads();
        if (lane == 0 && (warp % wpr) == 0u) {
            float sum = 0.0f;
            for (unsigned j = 0; j < wpr; ++j) sum += red2[warp + j];
            float v = sum * inv_hc;
            ((float*)a.mix)[t * a.R + row] = v / (1.0f + __expf(-v));
        }
    } else {
        // block_inject: плотные f16-строки [hc, K]; на строку — 8/hc варпов.
        const unsigned wpr = 8u / a.hc;                     // варпов на строку (hc ≤ 8)
        const unsigned row = warp / wpr;
        const unsigned sub = (warp % wpr) * 32u + lane;
        float acc = 0.0f;
        if (row < a.hc) {
            const __half* w = (const __half*)a.inj_w + (size_t)row * K;
            for (unsigned v = sub; v < vecs; v += wpr * 32u) {
                float f[8];
                hc_ld8(w + v * 8u, f);
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc = fmaf(f[j], xs[v * 8u + j], acc);
            }
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) red2[warp] = acc;
        __syncthreads();
        if (lane == 0 && row < a.hc && (warp % wpr) == 0u) {
            float sum = 0.0f;
            for (unsigned j = 0; j < wpr; ++j) sum += red2[warp + j];
            float v = sum * inv_hc;
            ((__half*)a.inj_out)[t * a.hc + row] = __float2half(2.0f / (1.0f + __expf(-v)));
        }
    }
}

// grid = (H/64, rows), block = hc·64 (нить — (c, i)), shared = R + hc·64 f32.
extern "C" __global__ void hc_up_f16(HcArgs a) {
    extern __shared__ float sm[];
    float* mix = sm;                // [R]
    float* part = sm + a.R;         // [hc][64]
    const unsigned t = blockIdx.y;
    const unsigned tid = threadIdx.x;
    const unsigned c = tid >> 6, il = tid & 63u;
    const unsigned i = blockIdx.x * 64u + il;
    const unsigned K = a.hc * a.H;
    const float* mix_g = (const float*)a.mix + (size_t)t * a.R;
    for (unsigned k = tid; k < a.R; k += blockDim.x) mix[k] = mix_g[k];
    __syncthreads();
    const unsigned row = c * a.H + i;
    float up = 0.0f;
    for (unsigned s = 0; s < (a.R >> 5); ++s) {
        up += hc_dot32((const unsigned char*)a.up_w, (const unsigned char*)a.up_s, row, a.R, s, mix + s * 32u);
    }
    const float nrm = ((const float*)a.norms)[t * a.hc + c];
    const float xn = __half2float(((const __half*)a.x)[(size_t)t * K + row]) * nrm * __half2float(((const __half*)a.norm_w)[row]);
    part[c * 64u + il] = xn / (1.0f + __expf(-up));
    __syncthreads();
    if (c == 0) {
        float acc = 0.0f;
        for (unsigned cc = 0; cc < a.hc; ++cc) acc += part[cc * 64u + il];
        ((__half*)a.mixed)[(size_t)t * a.H + i] = __float2half(acc / (float)a.hc);
    }
}

// ───────────── слитый shared-эксперт MoE на декоде ─────────────
//
//   se_up:   h[r]  = silu(Σ gate[r,k]·x[k]) · Σ up[r,k]·x[k]      r < I
//            sg    = sigmoid(Σ router[k]·x[k])
//   se_down: out[i] = sg · Σ down[i,k]·h[k]                        i < H
// Веса gate/up/down — перемешанные NVFP4-копии, деквант в регистрах,
// активации не квантуются.

struct SeArgs {
    unsigned long long x, gate_w, gate_s, up_w, up_s, down_w, down_s, router;
    unsigned long long h, sg, out;
    unsigned int rows, H, I, _pad;
};

// grid = (I/4 + 1, rows), block = 256 (4 строки: варп — (строка, gate|up)),
// shared = (H + 16) f32.
extern "C" __global__ void se_up_f16(SeArgs a) {
    extern __shared__ float sm[];
    float* red = sm;            // [8]
    float* xs = sm + 16;        // [H]
    const unsigned t = blockIdx.y, tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const __half* x = (const __half*)a.x + (size_t)t * a.H;
    for (unsigned v = tid; v < (a.H >> 3); v += blockDim.x) {
        float f[8];
        hc_ld8(x + v * 8u, f);
        #pragma unroll
        for (int j = 0; j < 8; ++j) xs[v * 8u + j] = f[j];
    }
    __syncthreads();
    const unsigned up_blocks = a.I / 4u;
    if (blockIdx.x < up_blocks) {
        const unsigned row = blockIdx.x * 4u + (warp >> 1);
        const bool is_up = (warp & 1u) != 0u;
        const unsigned char* w = (const unsigned char*)(is_up ? a.up_w : a.gate_w);
        const unsigned char* s = (const unsigned char*)(is_up ? a.up_s : a.gate_s);
        float acc = 0.0f;
        for (unsigned sb = lane; sb < (a.H >> 5); sb += 32u) acc += hc_dot32(w, s, row, a.H, sb, xs + sb * 32u);
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) red[warp] = acc;
        __syncthreads();
        if (lane == 0 && !is_up) {
            const float g = red[warp], u = red[warp + 1];
            ((float*)a.h)[(size_t)t * a.I + row] = g / (1.0f + __expf(-g)) * u;
        }
    } else {
        const float4* r4 = reinterpret_cast<const float4*>((const float*)a.router);
        float acc = 0.0f;
        for (unsigned i = tid; i < (a.H >> 2); i += blockDim.x) {
            const float4 wv = r4[i];
            acc = fmaf(wv.x, xs[4 * i], acc);
            acc = fmaf(wv.y, xs[4 * i + 1], acc);
            acc = fmaf(wv.z, xs[4 * i + 2], acc);
            acc = fmaf(wv.w, xs[4 * i + 3], acc);
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) red[warp] = acc;
        __syncthreads();
        if (tid == 0) {
            float s = 0.0f;
            for (unsigned w = 0; w < (blockDim.x >> 5); ++w) s += red[w];
            ((float*)a.sg)[t] = 1.0f / (1.0f + __expf(-s));
        }
    }
}

// grid = (H/128, rows), block = 256 (нить — (половина K, i)), shared = (I + 256) f32.
extern "C" __global__ void se_down_f16(SeArgs a) {
    extern __shared__ float sm[];
    float* hs = sm;             // [I]
    float* part = sm + a.I;     // [2][128]
    const unsigned t = blockIdx.y, tid = threadIdx.x;
    const unsigned half = tid >> 7, il = tid & 127u;
    const unsigned i = blockIdx.x * 128u + il;
    const float* hg = (const float*)a.h + (size_t)t * a.I;
    for (unsigned k = tid; k < a.I; k += blockDim.x) hs[k] = hg[k];
    __syncthreads();
    const unsigned subs = a.I >> 5, mid = subs >> 1;
    const unsigned s0 = half ? mid : 0u, s1 = half ? subs : mid;
    float acc = 0.0f;
    for (unsigned sb = s0; sb < s1; ++sb) {
        acc += hc_dot32((const unsigned char*)a.down_w, (const unsigned char*)a.down_s, i, a.I, sb, hs + sb * 32u);
    }
    part[half * 128u + il] = acc;
    __syncthreads();
    if (half == 0) {
        const float sg = ((const float*)a.sg)[t];
        ((__half*)a.out)[(size_t)t * a.H + i] = __float2half((part[il] + part[128u + il]) * sg);
    }
}
