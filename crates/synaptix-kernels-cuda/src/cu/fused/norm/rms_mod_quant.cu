#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Fused «adaLN-модуляция + квант» (эпилог нормы): за ОДИН launch
//   y = round_T(round_T(round_T(x*rms) * round_T(scale+1)) + shift)   [B*T, D]
//   (packed, scales) = quant(f16(y)),  FMT 0 = NVFP4, 1 = MXFP8
// БИТ-В-БИТ с цепочкой rms_no_gain → add_scalar(1) → broadcast_mul →
// broadcast_add → to(F16) → quantize (nvfp4_fast | mxfp8_quant_natural):
//   • редукция sumsq = копия row_sumsq из rms_norm.cu (тот же страйд/шаффл);
//   • каждое округление цепочки воспроизведено (все элементвайзы f32-математика
//     с раундом в dtype на сторе; bf16→f16 каст точен — мантисса 8→10 бит);
//   • квант-хелперы = копия nvfp4_quant.cu / mxfp8_quant.cu (тот же
//     amax-порядок/encode/div; e4m3 mxfp8 — аппаратный __nv_fp8_e4m3).
// Заменяет ~10 DRAM-проходов (rms r+w, add_scalar r+w, mul 2r+w, add 2r+w,
// cast r+w, quant r+w) на 3r+3w и 6 запусков на 1.
// NVFP4: хвост строк m..m_cov (128-тайл scale-раскладки) — нулевые scale-байты
// (контракт quantize-обёртки). MXFP8: natural-раскладка [m, K/32] — без хвоста,
// грид ровно m.

struct RmsModQuantParams {
    int   batch;     // реальные строки m
    int   batch_cov; // ceil128(m) — грид по хвосту зануляет скейлы
    int   hidden;    // D (кратно 16)
    float eps;
    int   sf_inner_dim; // K/64*4 — страйд scale-раскладки
};

__device__ __forceinline__ float ld_f32(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ld_f32(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void st_t(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void st_t(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
__device__ __forceinline__ float rnd_t(const __half*, float v) {
    return __half2float(__float2half(v));
}
__device__ __forceinline__ float rnd_t(const __nv_bfloat16*, float v) {
    return __bfloat162float(__float2bfloat16(v));
}
__device__ __forceinline__ __half to_h(const __half* p) { return *p; }
__device__ __forceinline__ __half to_h(const __nv_bfloat16* p) {
    return __float2half(__bfloat162float(*p));
}

// ── копия row_sumsq из reduction/rms_norm.cu (бит-в-бит редукция) ──
template <typename T>
__device__ __forceinline__ float rmq_row_sumsq(
    const T* __restrict__ x_row, int hidden, int tid, int block_size, float* warp_sums) {
    float local = 0.0f;
    for (int t = tid; t < hidden; t += block_size) {
        float v = ld_f32(x_row + t);
        local += v * v;
    }
    unsigned int mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) local += __shfl_down_sync(mask, local, off, 32);
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) warp_sums[warp_id] = local;
    __syncthreads();
    int num_warps = block_size >> 5;
    if (warp_id == 0) {
        float v = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off, 32);
        if (lane == 0) warp_sums[0] = v;
    }
    __syncthreads();
    return warp_sums[0];
}

// ── копии квант-хелперов из elementwise/nvfp4_quant.cu (бит-в-бит) ──
__device__ __forceinline__ float rmq_decode_e4m3(unsigned char byte) {
    bool sign = (byte & 0x80) != 0;
    int exp_bits = (byte >> 3) & 0x0F;
    int mantissa = byte & 0x07;
    if (exp_bits == 15 && mantissa == 7) return __int_as_float(0x7FC00000);
    float val;
    if (exp_bits == 0) {
        val = mantissa * 0.001953125f;
    } else {
        int exp_raw = exp_bits - 7;
        float frac = 1.0f + mantissa * 0.125f;
        val = frac * exp2f((float)exp_raw);
    }
    return sign ? -val : val;
}

__device__ __forceinline__ unsigned char rmq_encode_e4m3(float x) {
    if (isnan(x)) return 0x7F;
    float v = fminf(fmaxf(x, -448.0f), 448.0f);
    unsigned int sign = signbit(v) ? 1 : 0;
    float abs_v = fabsf(v);
    if (abs_v == 0.0f) return (unsigned char)(sign << 7);
    int exp_raw = (int)floorf(log2f(abs_v));
    int exp_biased = exp_raw + 7;
    unsigned int mantissa;
    if (exp_biased < 1) {
        int m = (int)nearbyintf(abs_v * 512.0f);
        m = max(0, min(7, m));
        return (unsigned char)((sign << 7) | (unsigned int)m);
    }
    if (exp_biased > 15) return (unsigned char)((sign << 7) | 0x7E);
    float pow2 = exp2f((float)exp_raw);
    int m = (int)nearbyintf(((abs_v / pow2) - 1.0f) * 8.0f);
    if (m == 8) {
        m = 0;
        exp_biased += 1;
        if (exp_biased > 15) return (unsigned char)((sign << 7) | 0x7E);
    }
    if (exp_biased == 15 && m == 7) m = 6;
    mantissa = (unsigned int)(m & 0x07);
    return (unsigned char)((sign << 7) | ((unsigned int)exp_biased << 3) | mantissa);
}

__device__ __forceinline__ unsigned char rmq_encode_e2m1_rtne(float x) {
    unsigned int sign = signbit(x) ? 0x08 : 0x00;
    float abs_x = fabsf(x);
    unsigned int idx;
    if (abs_x >= 5.0f) idx = 7;
    else if (abs_x >= 3.5f) idx = 6;
    else if (abs_x >= 2.5f) idx = 5;
    else if (abs_x >= 1.75f) idx = 4;
    else if (abs_x >= 1.25f) idx = 3;
    else if (abs_x >= 0.75f) idx = 2;
    else if (abs_x >= 0.25f) idx = 1;
    else idx = 0;
    return (unsigned char)(sign | idx);
}

__device__ __forceinline__ unsigned int rmq_tile_scale_offset(
    unsigned int outer, unsigned int block_col, unsigned int sf_inner_dim) {
    unsigned int tile_row = outer >> 7;
    unsigned int tile_col = block_col >> 2;
    unsigned int local_outer = outer & 127u;
    unsigned int local_inner = block_col & 3u;
    unsigned int tile_base = (tile_col * 4u + tile_row * sf_inner_dim) * 128u;
    unsigned int off_in_tile =
        (local_outer & 31u) * 16u + (local_outer >> 5) * 4u + local_inner;
    return tile_base + off_in_tile;
}

// ── общий квант-эпилог из f16-копии строки (s_y16): FMT 0 = NVFP4 (группы 16,
// tiled-scales, как quantize_f16_to_nvfp4_fast), 1 = MXFP8 (блоки 32, natural
// [row, K/32], арифметика = копия mxfp8_quant_natural: E8M0 бит-трюком из amax,
// e4m3 аппаратным __nv_fp8_e4m3, clamp ±448). ──
template <int FMT>
__device__ __forceinline__ void rmq_quant_epilogue(
    const __half* __restrict__ s_y16,
    unsigned char* __restrict__ packed,
    unsigned char* __restrict__ scales_out,
    int row, int tid, int block_size, RmsModQuantParams p)
{
    if constexpr (FMT == 0) {
        int groups = p.hidden >> 4;
        unsigned char* p_row = packed + (long long)row * (p.hidden >> 1);
        for (int g = tid; g < groups; g += block_size) {
            const __half2* hp = reinterpret_cast<const __half2*>(s_y16 + g * 16);
            __half2 h[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) h[i] = hp[i];
            float amax = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float2 f = __half22float2(h[i]);
                amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
            float scale_raw = (amax > 0.0f) ? (amax / 6.0f) : 1e-9f;
            unsigned char sb = rmq_encode_e4m3(scale_raw);
            float scale_q = rmq_decode_e4m3(sb);
            if (scale_q == 0.0f) scale_q = 1e-9f;
            scales_out[rmq_tile_scale_offset((unsigned)row, (unsigned)g, p.sf_inner_dim)] = sb;
            unsigned long long out8;
            unsigned char* ob = reinterpret_cast<unsigned char*>(&out8);
#ifdef SYN_E2M1_THRESHOLD
            // Пороговый e2m1 (синхронно с nvfp4_quant.cu, оба пути вместе):
            // |x| против точных кратных scale — без округления деления.
            float t1 = 0.25f * scale_q, t2 = 0.75f * scale_q, t3 = 1.25f * scale_q;
            float t4 = 1.75f * scale_q, t5 = 2.5f * scale_q, t6 = 3.5f * scale_q;
            float t7 = 5.0f * scale_q;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float2 f = __half22float2(h[i]);
                float ax = fabsf(f.x), ay = fabsf(f.y);
                unsigned int lo = (unsigned int)(ax >= t1) + (ax >= t2) + (ax >= t3)
                                + (ax >= t4) + (ax >= t5) + (ax >= t6) + (ax >= t7);
                unsigned int hi = (unsigned int)(ay >= t1) + (ay >= t2) + (ay >= t3)
                                + (ay >= t4) + (ay >= t5) + (ay >= t6) + (ay >= t7);
                lo |= signbit(f.x) ? 8u : 0u;
                hi |= signbit(f.y) ? 8u : 0u;
                ob[i] = (unsigned char)((lo & 0x0F) | ((hi & 0x0F) << 4));
            }
#else
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float2 f = __half22float2(h[i]);
                unsigned char lo = rmq_encode_e2m1_rtne(f.x / scale_q);
                unsigned char hi = rmq_encode_e2m1_rtne(f.y / scale_q);
                ob[i] = (unsigned char)((lo & 0x0F) | ((hi & 0x0F) << 4));
            }
#endif
            *reinterpret_cast<unsigned long long*>(p_row + g * 8) = out8;
        }
    } else {
        int groups = p.hidden >> 5;
        unsigned char* p_row = packed + (long long)row * p.hidden;
        unsigned char* s_row = scales_out + (long long)row * groups;
        for (int g = tid; g < groups; g += block_size) {
            const __half2* hp = reinterpret_cast<const __half2*>(s_y16 + g * 32);
            __half2 h[16];
            #pragma unroll
            for (int i = 0; i < 16; ++i) h[i] = hp[i];
            float v[32];
            float amax = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                float2 f = __half22float2(h[i]);
                v[2 * i] = f.x;
                v[2 * i + 1] = f.y;
                amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
            unsigned char sb = (unsigned char)((__float_as_uint(
                __uint_as_float(__float_as_uint(amax) & 0x7F800000u) / 256.0f)) >> 23);
            float sv = fmaxf(__uint_as_float(((unsigned)sb) << 23), 1e-12f);
            unsigned char ob[32];
            #pragma unroll
            for (int i = 0; i < 32; ++i)
                ob[i] = __nv_fp8_e4m3(fminf(fmaxf(v[i] / sv, -448.0f), 448.0f)).__x;
            uint4* dst = reinterpret_cast<uint4*>(p_row + g * 32);
            dst[0] = *reinterpret_cast<uint4*>(&ob[0]);
            dst[1] = *reinterpret_cast<uint4*>(&ob[16]);
            s_row[g] = sb;
        }
    }
}

// Хвост грида m..m_cov: NVFP4 зануляет scale-байты 128-тайла; MXFP8 без хвоста.
template <int FMT>
__device__ __forceinline__ bool rmq_tail_row(
    unsigned char* __restrict__ scales_out, int row, int tid, int block_size,
    RmsModQuantParams p)
{
    if (row < p.batch)
        return false;
    if constexpr (FMT == 0) {
        if (row < p.batch_cov) {
            int groups = p.hidden >> 4;
            for (int g = tid; g < groups; g += block_size)
                scales_out[rmq_tile_scale_offset((unsigned)row, (unsigned)g, p.sf_inner_dim)] = 0;
        }
    }
    return true;
}

template <typename T, int FMT>
__device__ __forceinline__ void rms_mod_quant_impl(
    const T* __restrict__ x,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    T*       __restrict__ y,
    unsigned char* __restrict__ packed,
    unsigned char* __restrict__ scales_e4m3,
    RmsModQuantParams p)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    if (rmq_tail_row<FMT>(scales_e4m3, row, tid, block_size, p))
        return;
    __shared__ float warp_sums[32];
    extern __shared__ __half s_y16[]; // D половинок (после warp_sums — отдельный extern)

    const T* x_row = x + (long long)row * p.hidden;
    const T* s_row = scale + (long long)row * p.hidden;
    const T* b_row = shift + (long long)row * p.hidden;
    T*       y_row = y + (long long)row * p.hidden;

    float sumsq = rmq_row_sumsq(x_row, p.hidden, tid, block_size, warp_sums);
    __shared__ float s_rms;
    if (tid == 0) {
        float mean = sumsq / (float)p.hidden;
        s_rms = rsqrtf(mean + p.eps);
    }
    __syncthreads();
    float rms = s_rms;

    for (int t = tid; t < p.hidden; t += block_size) {
        float xv = ld_f32(x_row + t);
        float n0 = rnd_t(x_row, xv * rms);                 // rms_no_gain: (1.0*xv)*rms
        float m1 = rnd_t(x_row, ld_f32(s_row + t) + 1.0f); // add_scalar(1.0) = affine(1,1)
        float y1 = rnd_t(x_row, n0 * m1);                  // broadcast_mul
        float y2 = rnd_t(x_row, y1 + ld_f32(b_row + t));   // broadcast_add
        st_t(y_row + t, y2);
        s_y16[t] = __float2half(y2); // f16 от УЖЕ округлённого y (bf16→f16 точен)
    }
    __syncthreads();

    rmq_quant_epilogue<FMT>(s_y16, packed, scales_e4m3, row, tid, block_size, p);
}

// ── LN-вариант (FLUX adaLN): y = LN(x)·(1+scale)+shift, scale/shift —
// per-batch векторы [B, D] (mod_div = строк на батч; LTX-стиль = 1).
// Редукция/арифметика = копия reduction/layernorm.cu (sum+sumsq за проход,
// var = sumsq/n − mean², gamma=1, beta нет): бит-в-бит с layer_norm(x, ones).
template <typename T, int FMT>
__device__ __forceinline__ void ln_mod_quant_impl(
    const T* __restrict__ x,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    T*       __restrict__ y,
    unsigned char* __restrict__ packed,
    unsigned char* __restrict__ scales_e4m3,
    RmsModQuantParams p,
    int mod_div)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    if (rmq_tail_row<FMT>(scales_e4m3, row, tid, block_size, p))
        return;
    extern __shared__ __half s_y16[];

    const T* x_row = x + (long long)row * p.hidden;
    const T* s_row = scale + (long long)(row / mod_div) * p.hidden;
    const T* b_row = shift + (long long)(row / mod_div) * p.hidden;
    T*       y_row = y + (long long)row * p.hidden;

    float local_sum = 0.f, local_sumsq = 0.f;
    for (int t = tid; t < p.hidden; t += block_size) {
        float v = ld_f32(x_row + t);
        local_sum += v;
        local_sumsq += v * v;
    }
    unsigned int mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sum   += __shfl_down_sync(mask, local_sum, off, 32);
        local_sumsq += __shfl_down_sync(mask, local_sumsq, off, 32);
    }
    __shared__ float warp_sum[32];
    __shared__ float warp_sumsq[32];
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) {
        warp_sum[warp_id] = local_sum;
        warp_sumsq[warp_id] = local_sumsq;
    }
    __syncthreads();
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    if (warp_id == 0) {
        int num_warps = block_size >> 5;
        float vs = (lane < num_warps) ? warp_sum[lane] : 0.f;
        float vq = (lane < num_warps) ? warp_sumsq[lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            vs += __shfl_down_sync(mask, vs, off, 32);
            vq += __shfl_down_sync(mask, vq, off, 32);
        }
        if (lane == 0) {
            float n = (float)p.hidden;
            float mean = vs / n;
            float var = vq / n - mean * mean;
            if (var < 0.f) var = 0.f;
            s_mean = mean;
            s_inv_std = rsqrtf(var + p.eps);
        }
    }
    __syncthreads();
    float mean = s_mean;
    float inv_std = s_inv_std;

    for (int t = tid; t < p.hidden; t += block_size) {
        float xv = ld_f32(x_row + t);
        float n0 = rnd_t(x_row, (xv - mean) * inv_std * 1.0f); // LN(gamma=1) → round
        float m1 = rnd_t(x_row, ld_f32(s_row + t) + 1.0f);     // add_scalar(1.0)
        float y1 = rnd_t(x_row, n0 * m1);                      // broadcast_mul
        float y2 = rnd_t(x_row, y1 + ld_f32(b_row + t));       // broadcast_add
        st_t(y_row + t, y2);
        s_y16[t] = __float2half(y2);
    }
    __syncthreads();

    rmq_quant_epilogue<FMT>(s_y16, packed, scales_e4m3, row, tid, block_size, p);
}

// ── RMS+вес+квант (LLM prefill, без модуляции): y = round_T(scale·xv·rms),
// scale = qwen? (1+w): w — порядок/редукция = копия rms_norm_impl. ──
template <typename T, int FMT>
__device__ __forceinline__ void rms_w_quant_impl(
    const T* __restrict__ x,
    const T* __restrict__ w,
    T*       __restrict__ y,
    unsigned char* __restrict__ packed,
    unsigned char* __restrict__ scales_e4m3,
    RmsModQuantParams p,
    int qwen)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    if (rmq_tail_row<FMT>(scales_e4m3, row, tid, block_size, p))
        return;
    __shared__ float warp_sums[32];
    extern __shared__ __half s_y16[];

    const T* x_row = x + (long long)row * p.hidden;
    T*       y_row = y + (long long)row * p.hidden;

    float sumsq = rmq_row_sumsq(x_row, p.hidden, tid, block_size, warp_sums);
    __shared__ float s_rms;
    if (tid == 0) {
        float mean = sumsq / (float)p.hidden;
        s_rms = rsqrtf(mean + p.eps);
    }
    __syncthreads();
    float rms = s_rms;

    for (int t = tid; t < p.hidden; t += block_size) {
        float xv = ld_f32(x_row + t);
        float wv = ld_f32(w + t);
        float scale = qwen ? (1.0f + wv) : wv;
        float yv = rnd_t(x_row, scale * xv * rms); // как rms_norm_impl: ((scale*xv)*rms)
        st_t(y_row + t, yv);
        s_y16[t] = __float2half(yv);
    }
    __syncthreads();

    rmq_quant_epilogue<FMT>(s_y16, packed, scales_e4m3, row, tid, block_size, p);
}

extern "C" __global__ void rms_w_quant_nvfp4_f16(
    const __half* x, const __half* w, const __half* shift_unused, __half* y,
    unsigned char* packed, unsigned char* scales, RmsModQuantParams p, int qwen)
{ (void)shift_unused; rms_w_quant_impl<__half, 0>(x, w, y, packed, scales, p, qwen); }

extern "C" __global__ void rms_w_quant_nvfp4_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* shift_unused,
    __nv_bfloat16* y, unsigned char* packed, unsigned char* scales, RmsModQuantParams p, int qwen)
{ (void)shift_unused; rms_w_quant_impl<__nv_bfloat16, 0>(x, w, y, packed, scales, p, qwen); }

extern "C" __global__ void ln_mod_quant_nvfp4_f16(
    const __half* x, const __half* scale, const __half* shift, __half* y,
    unsigned char* packed, unsigned char* scales, RmsModQuantParams p, int mod_div)
{ ln_mod_quant_impl<__half, 0>(x, scale, shift, y, packed, scales, p, mod_div); }

extern "C" __global__ void ln_mod_quant_nvfp4_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* y, unsigned char* packed, unsigned char* scales, RmsModQuantParams p,
    int mod_div)
{ ln_mod_quant_impl<__nv_bfloat16, 0>(x, scale, shift, y, packed, scales, p, mod_div); }

extern "C" __global__ void rms_mod_quant_nvfp4_f16(
    const __half* x, const __half* scale, const __half* shift, __half* y,
    unsigned char* packed, unsigned char* scales, RmsModQuantParams p)
{ rms_mod_quant_impl<__half, 0>(x, scale, shift, y, packed, scales, p); }

extern "C" __global__ void rms_mod_quant_nvfp4_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* y, unsigned char* packed, unsigned char* scales, RmsModQuantParams p)
{ rms_mod_quant_impl<__nv_bfloat16, 0>(x, scale, shift, y, packed, scales, p); }

// ── MXFP8-варианты (эпилог = mxfp8_quant_natural бит-в-бит): packed [m, K] u8,
// scales natural [m, K/32] u8; грид ровно m (без 128-хвоста). ──
extern "C" __global__ void rms_w_quant_mxfp8_f16(
    const __half* x, const __half* w, const __half* shift_unused, __half* y,
    unsigned char* packed, unsigned char* scales, RmsModQuantParams p, int qwen)
{ (void)shift_unused; rms_w_quant_impl<__half, 1>(x, w, y, packed, scales, p, qwen); }

extern "C" __global__ void rms_w_quant_mxfp8_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* shift_unused,
    __nv_bfloat16* y, unsigned char* packed, unsigned char* scales, RmsModQuantParams p, int qwen)
{ (void)shift_unused; rms_w_quant_impl<__nv_bfloat16, 1>(x, w, y, packed, scales, p, qwen); }

extern "C" __global__ void ln_mod_quant_mxfp8_f16(
    const __half* x, const __half* scale, const __half* shift, __half* y,
    unsigned char* packed, unsigned char* scales, RmsModQuantParams p, int mod_div)
{ ln_mod_quant_impl<__half, 1>(x, scale, shift, y, packed, scales, p, mod_div); }

extern "C" __global__ void ln_mod_quant_mxfp8_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* y, unsigned char* packed, unsigned char* scales, RmsModQuantParams p,
    int mod_div)
{ ln_mod_quant_impl<__nv_bfloat16, 1>(x, scale, shift, y, packed, scales, p, mod_div); }

extern "C" __global__ void rms_mod_quant_mxfp8_f16(
    const __half* x, const __half* scale, const __half* shift, __half* y,
    unsigned char* packed, unsigned char* scales, RmsModQuantParams p)
{ rms_mod_quant_impl<__half, 1>(x, scale, shift, y, packed, scales, p); }

extern "C" __global__ void rms_mod_quant_mxfp8_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* y, unsigned char* packed, unsigned char* scales, RmsModQuantParams p)
{ rms_mod_quant_impl<__nv_bfloat16, 1>(x, scale, shift, y, packed, scales, p); }
