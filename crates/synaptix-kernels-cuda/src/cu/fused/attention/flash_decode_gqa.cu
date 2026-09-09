#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Flash-decode для одного запроса с GQA: блок = (KV-голова, сегмент ключей),
// внутри блока 4 варпа делят ГОЛОВЫ группы и/или ключи сегмента. K/V читаются
// один раз на группу запросных голов (прежнее split-ядро тянуло их отдельно
// на каждую из NREP голов, а оконное splitq-ядро держало по одному блоку на
// голову и 255 регистров — 16 блоков на 82 SM).
//
//   q    [nh, HD] bf16 (после нормы и RoPE)
//   k/v  [nkv, cap, HD] bf16 — предвыделенный кэш; активная длина tkv_ptr[0]
//   окно window > 0: ключи j ≥ tkv − window (иначе все от 0)
//   partials: m/l [nh, S], acc [nh, S, HD] f32, S = splits · WKS
//
// Грид (nkv, splits) статичен — под захват графа; пустые сегменты пишут
// m = −inf, l = 0, merge их обнуляет.

#define FDG_NEG_INF (__int_as_float(0xFF800000))

__device__ __forceinline__ float fdg_bf(const __nv_bfloat16& v) { return __bfloat162float(v); }

template <int EPL>
__device__ __forceinline__ void fdg_load_row(const __nv_bfloat16* __restrict__ p, float* out) {
    // EPL элементов подряд: 8 → uint4, 16 → 2×uint4.
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

template <int HD, int NREP>
__device__ __forceinline__ void fdg_split_impl(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
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
    const __nv_bfloat16* kb = k + (size_t)kvh * cap * HD + lane * EPL;
    const __nv_bfloat16* vb = v + (size_t)kvh * cap * HD + lane * EPL;

    for (int j = j0; j < j1; ++j) {
        float kv[EPL], vv[EPL];
        fdg_load_row<EPL>(kb + (size_t)j * HD, kv);
        fdg_load_row<EPL>(vb + (size_t)j * HD, vv);
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

// Слияние partials: блок = голова, нити по HD (blockDim = HD, кратно 32).
// Эпилог: bf16-выход и/или MXFP8-квант строки внимания (вход o_proj) —
// natural-раскладка [nh·HD] e4m3 + [nh·HD/32] E8M0, арифметика как у
// mxfp8_quant_natural (amax по 32 = один варп, shfl).
template <int HD>
__device__ __forceinline__ void fdg_merge_impl(
    const float* __restrict__ part_m, const float* __restrict__ part_l, const float* __restrict__ part_acc,
    __nv_bfloat16* __restrict__ out, unsigned char* __restrict__ mx_packed, unsigned char* __restrict__ mx_scales,
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
    __nv_bfloat16 yb = __float2bfloat16(y);
    if (out) out[(size_t)head * HD + d] = yb;
    if (mx_packed) {
        float v = __bfloat162float(yb);
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

#define FDG_SPLIT(HD, NREP)                                                                 \
__global__ void __launch_bounds__(128) fdg_split_hd##HD##_r##NREP(                          \
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,                 \
    const unsigned int* tkv_ptr, float scale, int window, int cap,                          \
    float* part_m, float* part_l, float* part_acc, int splits) {                            \
    fdg_split_impl<HD, NREP>(q, k, v, tkv_ptr, scale, window, cap, part_m, part_l, part_acc, splits); \
}

FDG_SPLIT(256, 1)
FDG_SPLIT(256, 2)
FDG_SPLIT(256, 4)
FDG_SPLIT(256, 8)
FDG_SPLIT(512, 2)
FDG_SPLIT(512, 4)
FDG_SPLIT(512, 8)

__global__ void fdg_merge_hd256(const float* pm, const float* pl, const float* pa, __nv_bfloat16* out,
                                unsigned char* mxp, unsigned char* mxs, int S) {
    fdg_merge_impl<256>(pm, pl, pa, out, mxp, mxs, S);
}
__global__ void fdg_merge_hd512(const float* pm, const float* pl, const float* pa, __nv_bfloat16* out,
                                unsigned char* mxp, unsigned char* mxs, int S) {
    fdg_merge_impl<512>(pm, pl, pa, out, mxp, mxs, S);
}

}  // extern "C"
