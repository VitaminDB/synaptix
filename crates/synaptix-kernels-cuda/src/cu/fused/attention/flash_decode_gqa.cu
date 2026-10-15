#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Flash-decode для одного запроса с GQA: блок = (KV-голова, сегмент ключей),
// внутри блока 4 варпа делят ГОЛОВЫ группы и/или ключи сегмента. K/V читаются
// один раз на группу запросных голов (прежнее split-ядро тянуло их отдельно
// на каждую из NREP голов, а оконное splitq-ядро держало по одному блоку на
// голову и 255 регистров — 16 блоков на 82 SM).
//
//   q    [nh, HD] T (bf16/f16, после нормы и RoPE)
//   k/v  [nkv, cap, HD] T — предвыделенный кэш, либо MXFP8 (e4m3 + E8M0
//        масштабы [nkv, cap, HD/32]) — деквант при чтении; активная длина tkv_ptr[0]
//   окно window > 0: ключи j ≥ tkv − window (иначе все от 0)
//   partials: m/l [nh, S], acc [nh, S, HD] f32, S = splits · WKS
//
// Грид (nkv, splits) статичен — под захват графа; пустые сегменты пишут
// m = −inf, l = 0, merge их обнуляет.

#define FDG_NEG_INF (__int_as_float(0xFF800000))

typedef __nv_bfloat16 fdg_bf16;

__device__ __forceinline__ float fdg_e8m0(unsigned char b) {
    return fmaxf(__uint_as_float(((unsigned)b) << 23), 1e-12f);
}

// EPL элементов подряд (8 → uint4, 16 → 2×uint4) → float.
template <int EPL>
__device__ __forceinline__ void fdg_load_row(const __nv_bfloat16* __restrict__ p, float* out) {
    const uint4* src = reinterpret_cast<const uint4*>(p);
    #pragma unroll
    for (int c = 0; c < EPL / 8; ++c) {
        uint4 u = src[c];
        __nv_bfloat162 h[4];
        *reinterpret_cast<uint4*>(h) = u;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f = __bfloat1622float2(h[i]);
            out[c * 8 + 2 * i] = f.x;
            out[c * 8 + 2 * i + 1] = f.y;
        }
    }
}

template <int EPL>
__device__ __forceinline__ void fdg_load_row(const __half* __restrict__ p, float* out) {
    const uint4* src = reinterpret_cast<const uint4*>(p);
    #pragma unroll
    for (int c = 0; c < EPL / 8; ++c) {
        uint4 u = src[c];
        __half2 h[4];
        *reinterpret_cast<uint4*>(h) = u;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f = __half22float2(h[i]);
            out[c * 8 + 2 * i] = f.x;
            out[c * 8 + 2 * i + 1] = f.y;
        }
    }
}

// MXFP8-кэш: EPL байт e4m3 (8 → uint2, 16 → uint4), масштаб E8M0 общий на 32
// элемента — у лейна все EPL элементов внутри одного 32-блока (EPL | 32).
template <int EPL>
__device__ __forceinline__ void fdg_load_row_mx(const unsigned char* __restrict__ p, float sc, float* out) {
    #pragma unroll
    for (int c = 0; c < EPL / 8; ++c) {
        uint2 u = reinterpret_cast<const uint2*>(p)[c];
        unsigned int w[2] = {u.x, u.y};
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                unsigned short pair = (unsigned short)((w[i] >> (16 * j)) & 0xFFFFu);
                __half2 h = __half2(__nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3));
                float2 f = __half22float2(h);
                out[c * 8 + i * 4 + j * 2] = f.x * sc;
                out[c * 8 + i * 4 + j * 2 + 1] = f.y * sc;
            }
        }
    }
}

// Шаг split: один ключ j — K/V группы прочитаны, обновление online-softmax по головам варпа.
template <int HD, int NREP, typename T, bool MX>
__device__ __forceinline__ void fdg_split_impl(
    const T* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    const unsigned char* __restrict__ k_scale,
    const unsigned char* __restrict__ v_scale,
    const unsigned int* __restrict__ tkv_ptr,
    float scale, int window, int cap,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    int splits)
{
    constexpr int EPL = HD / 32;
    constexpr int HPW = (NREP + 3) / 4;
    constexpr int WPH = (NREP + HPW - 1) / HPW;
    constexpr int WKS = 4 / WPH;
    static_assert(WPH * WKS == 4, "4 варпа: головы × разбиения ключей");
    static_assert(EPL % 8 == 0, "голова кратна 256");
    static_assert(!MX || (32 % EPL == 0), "MXFP8: элементы лейна внутри одного 32-блока");

    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hgrp = warp % WPH;
    const int ks = warp / WPH;
    const int tkv = (int)tkv_ptr[0];
    const int lo = window > 0 ? max(0, tkv - window) : 0;
    const int n = tkv - lo;
    const int S = splits * WKS;
    const int sidx = split * WKS + ks;
    const int per = (n + S - 1) / S;
    const int j0 = lo + sidx * per;
    const int j1 = min(j0 + per, tkv);

    float qr[HPW][EPL];
    #pragma unroll
    for (int hh = 0; hh < HPW; ++hh) {
        int r = hgrp * HPW + hh;
        if (r < NREP) {
            fdg_load_row<EPL>(q + ((size_t)(kvh * NREP + r)) * HD + lane * EPL, qr[hh]);
        } else {
            #pragma unroll
            for (int i = 0; i < EPL; ++i) qr[hh][i] = 0.f;
        }
    }
    float m[HPW], l[HPW], acc[HPW][EPL];
    #pragma unroll
    for (int hh = 0; hh < HPW; ++hh) {
        m[hh] = FDG_NEG_INF;
        l[hh] = 0.f;
        #pragma unroll
        for (int i = 0; i < EPL; ++i) acc[hh][i] = 0.f;
    }
    const T* kb = reinterpret_cast<const T*>(k) + (size_t)kvh * cap * HD + lane * EPL;
    const T* vb = reinterpret_cast<const T*>(v) + (size_t)kvh * cap * HD + lane * EPL;
    const unsigned char* kbm = reinterpret_cast<const unsigned char*>(k) + (size_t)kvh * cap * HD + lane * EPL;
    const unsigned char* vbm = reinterpret_cast<const unsigned char*>(v) + (size_t)kvh * cap * HD + lane * EPL;
    const unsigned char* ksb = k_scale + (size_t)kvh * cap * (HD / 32) + (lane * EPL) / 32;
    const unsigned char* vsb = v_scale + (size_t)kvh * cap * (HD / 32) + (lane * EPL) / 32;

    for (int j = j0; j < j1; ++j) {
        float kv[EPL], vv[EPL];
        if (MX) {
            fdg_load_row_mx<EPL>(kbm + (size_t)j * HD, fdg_e8m0(ksb[(size_t)j * (HD / 32)]), kv);
            fdg_load_row_mx<EPL>(vbm + (size_t)j * HD, fdg_e8m0(vsb[(size_t)j * (HD / 32)]), vv);
        } else {
            fdg_load_row<EPL>(kb + (size_t)j * HD, kv);
            fdg_load_row<EPL>(vb + (size_t)j * HD, vv);
        }
        #pragma unroll
        for (int hh = 0; hh < HPW; ++hh) {
            float s = 0.f;
            #pragma unroll
            for (int i = 0; i < EPL; ++i) s = fmaf(qr[hh][i], kv[i], s);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
            s *= scale;
            float m_new = fmaxf(m[hh], s);
            float corr = expf(m[hh] - m_new);
            float p = expf(s - m_new);
            l[hh] = l[hh] * corr + p;
            #pragma unroll
            for (int i = 0; i < EPL; ++i) acc[hh][i] = fmaf(p, vv[i], acc[hh][i] * corr);
            m[hh] = m_new;
        }
    }
    #pragma unroll
    for (int hh = 0; hh < HPW; ++hh) {
        int r = hgrp * HPW + hh;
        if (r >= NREP) continue;
        int head = kvh * NREP + r;
        size_t pi = (size_t)head * S + sidx;
        if (lane == 0) {
            part_m[pi] = m[hh];
            part_l[pi] = l[hh];
        }
        float* dst = part_acc + pi * HD + lane * EPL;
        #pragma unroll
        for (int i = 0; i < EPL; ++i) dst[i] = acc[hh][i];
    }
}

__device__ __forceinline__ float fdg_rnd(const __nv_bfloat16*, float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float fdg_rnd(const __half*, float v) { return __half2float(__float2half(v)); }
__device__ __forceinline__ void fdg_st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
__device__ __forceinline__ void fdg_st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ float fdg_ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float fdg_ld(const __half* p) { return __half2float(*p); }

// Слияние partials: блок = голова, нити по HD (blockDim = HD, кратно 32).
// Гейт выхода (attn_output_gate, Qwen3.5): y ·= sigmoid(gate[head·gate_stride + d]).
// Эпилог: T-выход и/или MXFP8-квант строки внимания (вход o_proj) —
// natural-раскладка [nh·HD] e4m3 + [nh·HD/32] E8M0, арифметика как у
// mxfp8_quant_natural (amax по 32 = один варп, shfl).
template <int HD, typename T>
__device__ __forceinline__ void fdg_merge_impl(
    const float* __restrict__ part_m, const float* __restrict__ part_l, const float* __restrict__ part_acc,
    const T* __restrict__ gate, int gate_stride,
    T* __restrict__ out, unsigned char* __restrict__ mx_packed, unsigned char* __restrict__ mx_scales,
    int S)
{
    __shared__ float s_w[64];
    __shared__ float s_l;
    const int head = blockIdx.x;
    if (threadIdx.x == 0) {
        float mx = FDG_NEG_INF;
        for (int s = 0; s < S; ++s) mx = fmaxf(mx, part_m[(size_t)head * S + s]);
        float lsum = 0.f;
        for (int s = 0; s < S; ++s) {
            float ms = part_m[(size_t)head * S + s];
            float w = (ms == FDG_NEG_INF) ? 0.f : expf(ms - mx);
            s_w[s] = w;
            lsum += w * part_l[(size_t)head * S + s];
        }
        s_l = lsum;
    }
    __syncthreads();
    float inv = 1.0f / s_l;
    const int d = threadIdx.x;
    float a = 0.f;
    for (int s = 0; s < S; ++s) {
        float w = s_w[s];
        if (w != 0.f) a = fmaf(w, part_acc[((size_t)head * S + s) * HD + d], a);
    }
    float y = a * inv;
    if (gate) {
        float g = fdg_ld(gate + (size_t)head * gate_stride + d);
        y *= 1.0f / (1.0f + expf(-g));
    }
    float yr = fdg_rnd(out, y);
    if (out) fdg_st(out + (size_t)head * HD + d, yr);
    if (mx_packed) {
        float v = yr;
        float amax = fabsf(v);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
        unsigned char sb = (unsigned char)((__float_as_uint(
            __uint_as_float(__float_as_uint(amax) & 0x7F800000u) / 256.0f)) >> 23);
        float sv = fmaxf(__uint_as_float(((unsigned)sb) << 23), 1e-12f);
        size_t idx = (size_t)head * HD + d;
        mx_packed[idx] = __nv_fp8_e4m3(fminf(fmaxf(v / sv, -448.0f), 448.0f)).__x;
        if ((d & 31) == 0) mx_scales[idx >> 5] = sb;
    }
}

extern "C" {

#define FDG_SPLIT(HD, NREP, T, TN, MX, MXN)                                                 \
__global__ void __launch_bounds__(128) fdg_split_hd##HD##_r##NREP##_##TN##MXN(               \
    const T* q, const void* k, const void* v, const unsigned char* ks, const unsigned char* vs, \
    const unsigned int* tkv_ptr, float scale, int window, int cap,                          \
    float* part_m, float* part_l, float* part_acc, int splits) {                            \
    fdg_split_impl<HD, NREP, T, MX>(q, k, v, ks, vs, tkv_ptr, scale, window, cap, part_m, part_l, part_acc, splits); \
}

FDG_SPLIT(256, 1, fdg_bf16, bf16, false, )
FDG_SPLIT(256, 2, fdg_bf16, bf16, false, )
FDG_SPLIT(256, 4, fdg_bf16, bf16, false, )
FDG_SPLIT(256, 8, fdg_bf16, bf16, false, )
FDG_SPLIT(512, 2, fdg_bf16, bf16, false, )
FDG_SPLIT(512, 4, fdg_bf16, bf16, false, )
FDG_SPLIT(512, 8, fdg_bf16, bf16, false, )
FDG_SPLIT(256, 1, __half, f16, false, )
FDG_SPLIT(256, 2, __half, f16, false, )
FDG_SPLIT(256, 4, __half, f16, false, )
FDG_SPLIT(256, 8, __half, f16, false, )
FDG_SPLIT(512, 2, __half, f16, false, )
FDG_SPLIT(512, 4, __half, f16, false, )
FDG_SPLIT(512, 8, __half, f16, false, )
FDG_SPLIT(256, 1, __half, f16, true, _mx)
FDG_SPLIT(256, 2, __half, f16, true, _mx)
FDG_SPLIT(256, 4, __half, f16, true, _mx)
FDG_SPLIT(256, 8, __half, f16, true, _mx)
FDG_SPLIT(256, 1, fdg_bf16, bf16, true, _mx)
FDG_SPLIT(256, 2, fdg_bf16, bf16, true, _mx)
FDG_SPLIT(256, 4, fdg_bf16, bf16, true, _mx)
FDG_SPLIT(256, 8, fdg_bf16, bf16, true, _mx)

#define FDG_MERGE(HD, T, TN)                                                                 \
__global__ void fdg_merge_hd##HD##_##TN(const float* pm, const float* pl, const float* pa,   \
                                        const T* gate, int gate_stride, T* out,              \
                                        unsigned char* mxp, unsigned char* mxs, int S) {     \
    fdg_merge_impl<HD, T>(pm, pl, pa, gate, gate_stride, out, mxp, mxs, S);                  \
}

FDG_MERGE(256, fdg_bf16, bf16)
FDG_MERGE(512, fdg_bf16, bf16)
FDG_MERGE(256, __half, f16)
FDG_MERGE(512, __half, f16)

}  // extern "C"
