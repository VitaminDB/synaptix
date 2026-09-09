#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Слитые ядра шага декода (один токен, batch = 1) для общего декодера LLM.
//
// Зачем: у Gemma-4 26B шаг графа состоял из ~2200 ядер по 1–3 мкс — каждая
// норма, каждый add, каждый cast и каждое квантование отдельным запуском.
// Ядра ниже собирают цепочки элементвайзов одного слоя в один запуск и сразу
// отдают активацию в том виде, какой нужен следующему GEMV (NVFP4/MXFP8-пара).
//
// Нумерика ПОВТОРЯЕТ host-путь: каждое место, где host округлял в bf16
// (выход нормы, сумма с residual, произведение gelu·up), округляется так же,
// а квант-эпилоги — копии nvfp4_quant.cu / mxfp8_quant.cu. Отличие только в
// порядке суммирования редукций.
//
// Все ядра — одиночные строки (декод), поэтому квант NVFP4 пишет лишь строку
// 0 tile-раскладки масштабов: GEMV читает у активации только свою строку,
// хвост 128-тайла ему не нужен (контракт GEMM с паддингом сюда не относится).

typedef __nv_bfloat16 bf16_t;

__device__ __forceinline__ float ldf(const bf16_t* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float ldf(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float ldf(const float* p) { return *p; }
__device__ __forceinline__ float rnd_bf16(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float rnd_t(const bf16_t*, float v) { return rnd_bf16(v); }
__device__ __forceinline__ float rnd_t(const __half*, float v) { return __half2float(__float2half(v)); }
__device__ __forceinline__ void stf(bf16_t* p, float v) { *p = __float2bfloat16(v); }
__device__ __forceinline__ void stf(__half* p, float v) { *p = __float2half(v); }

// Сумма по блоку (все нити получают результат). `red` — smem [32].
__device__ __forceinline__ float block_sum(float v, float* red) {
    unsigned int mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off, 32);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    float r = 0.f;
    if (warp == 0) {
        r = (lane < nw) ? red[lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) r += __shfl_down_sync(mask, r, off, 32);
        if (lane == 0) red[0] = r;
    }
    __syncthreads();
    r = red[0];
    __syncthreads();
    return r;
}

// rms по строке из smem (значения уже f32).
__device__ __forceinline__ float row_rms(const float* s, int h, float eps, float* red) {
    float local = 0.f;
    for (int t = threadIdx.x; t < h; t += blockDim.x) local += s[t] * s[t];
    float sumsq = block_sum(local, red);
    return rsqrtf(sumsq / (float)h + eps);
}

// ── квант-хелперы: копии nvfp4_quant.cu / mxfp8_quant.cu (бит-в-бит) ──
__device__ __forceinline__ float dq_decode_e4m3(unsigned char byte) {
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

__device__ __forceinline__ unsigned char dq_encode_e4m3(float x) {
    if (isnan(x)) return 0x7F;
    float v = fminf(fmaxf(x, -448.0f), 448.0f);
    unsigned int sign = signbit(v) ? 1 : 0;
    float abs_v = fabsf(v);
    if (abs_v == 0.0f) return (unsigned char)(sign << 7);
    int exp_raw = (int)floorf(log2f(abs_v));
    int exp_biased = exp_raw + 7;
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
    return (unsigned char)((sign << 7) | ((unsigned int)exp_biased << 3) | (unsigned int)(m & 0x07));
}

__device__ __forceinline__ unsigned char dq_encode_e2m1_rtne(float x) {
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

__device__ __forceinline__ unsigned int dq_tile_scale_offset(
    unsigned int outer, unsigned int block_col, unsigned int sf_inner_dim) {
    unsigned int tile_row = outer >> 7;
    unsigned int tile_col = block_col >> 2;
    unsigned int local_outer = outer & 127u;
    unsigned int local_inner = block_col & 3u;
    unsigned int tile_base = (tile_col * 4u + tile_row * sf_inner_dim) * 128u;
    unsigned int off_in_tile = (local_outer & 31u) * 16u + (local_outer >> 5) * 4u + local_inner;
    return tile_base + off_in_tile;
}

// Квант 16 значений (уже округлённых в bf16/f16) → 8 байт packed + scale.
__device__ __forceinline__ void dq_nvfp4_group(
    const float* v, unsigned char* packed8, unsigned char* scale_byte) {
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
    float scale_raw = (amax > 0.0f) ? (amax / 6.0f) : 1e-9f;
    unsigned char sb = dq_encode_e4m3(scale_raw);
    float scale_q = dq_decode_e4m3(sb);
    if (scale_q == 0.0f) scale_q = 1e-9f;
    *scale_byte = sb;
    unsigned long long out8;
    unsigned char* ob = reinterpret_cast<unsigned char*>(&out8);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned char lo = dq_encode_e2m1_rtne(v[2 * i] / scale_q);
        unsigned char hi = dq_encode_e2m1_rtne(v[2 * i + 1] / scale_q);
        ob[i] = (unsigned char)((lo & 0x0F) | ((hi & 0x0F) << 4));
    }
    *reinterpret_cast<unsigned long long*>(packed8) = out8;
}

// Строка `s[h]` (значения bf16-точности) → NVFP4-пара строки `row`.
__device__ __forceinline__ void dq_nvfp4_row(
    const float* s, int h, unsigned char* packed, unsigned char* scales, int row, int sf_inner_dim) {
    int groups = h >> 4;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        float v[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = s[g * 16 + i];
        unsigned char sb;
        dq_nvfp4_group(v, packed + ((size_t)row * h + (size_t)g * 16) / 2, &sb);
        scales[dq_tile_scale_offset((unsigned)row, (unsigned)g, (unsigned)sf_inner_dim)] = sb;
    }
}

// Строка `s[h]` → MXFP8 natural (packed [h] e4m3, scales [h/32] E8M0).
__device__ __forceinline__ void dq_mxfp8_row(
    const float* s, int h, unsigned char* packed, unsigned char* scales) {
    int groups = h >> 5;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(s[g * 32 + i]));
        unsigned char sb = (unsigned char)((__float_as_uint(
            __uint_as_float(__float_as_uint(amax) & 0x7F800000u) / 256.0f)) >> 23);
        float sv = fmaxf(__uint_as_float(((unsigned)sb) << 23), 1e-12f);
        unsigned char ob[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i)
            ob[i] = __nv_fp8_e4m3(fminf(fmaxf(s[g * 32 + i] / sv, -448.0f), 448.0f)).__x;
        uint4* dst = reinterpret_cast<uint4*>(packed + (size_t)g * 32);
        dst[0] = *reinterpret_cast<uint4*>(&ob[0]);
        dst[1] = *reinterpret_cast<uint4*>(&ob[16]);
        scales[g] = sb;
    }
}

// y = bf16(w · x · rms) построчно в smem (порядок как в rms_norm_impl: (w*x)*rms).
__device__ __forceinline__ void norm_into(
    const float* s_x, float rms, const bf16_t* w, float* s_y, int h) {
    for (int t = threadIdx.x; t < h; t += blockDim.x) {
        float wv = w ? ldf(w + t) : 1.0f;
        s_y[t] = rnd_bf16(wv * s_x[t] * rms);
    }
}

extern "C" {

// ── Хвост после внимания ─────────────────────────────────────────────────
//
//   y      = post_w ? bf16(norm(attn_out)·post_w) : attn_out
//   hidden = bf16(y + hidden_in)                     → hidden_out
//   a      = bf16(norm(hidden)·w_a)  → NVFP4-пара    (вход плотного MLP)
//   b      = bf16(norm(hidden)·w_b)  → NVFP4-пара    (вход экспертов)
//   c      = bf16(norm(hidden)·w_c)  → bf16          (вход роутера)
//
// Любой из выходов a/b/c можно выключить нулевым указателем веса. Один блок;
// dynamic smem = 2·H·4 байт.
__global__ void dec_attn_tail_bf16(
    const bf16_t* __restrict__ attn_out,
    const bf16_t* __restrict__ post_w,
    const bf16_t* __restrict__ hidden_in,
    bf16_t* __restrict__ hidden_out,
    const bf16_t* __restrict__ w_a, unsigned char* __restrict__ a_packed, unsigned char* __restrict__ a_scales,
    const bf16_t* __restrict__ w_b, unsigned char* __restrict__ b_packed, unsigned char* __restrict__ b_scales,
    const bf16_t* __restrict__ w_c, bf16_t* __restrict__ c_out,
    int h, float eps_post, float eps, int sf_inner_dim)
{
    extern __shared__ float smem[];
    float* s_h = smem;
    float* s_y = smem + h;
    __shared__ float red[32];

    // y = post-norm(attn_out) либо сам attn_out.
    for (int t = threadIdx.x; t < h; t += blockDim.x) s_y[t] = ldf(attn_out + t);
    if (post_w) {
        __syncthreads();
        float rms = row_rms(s_y, h, eps_post, red);
        for (int t = threadIdx.x; t < h; t += blockDim.x)
            s_y[t] = rnd_bf16(ldf(post_w + t) * s_y[t] * rms);
    }
    // hidden = bf16(y + residual).
    for (int t = threadIdx.x; t < h; t += blockDim.x) {
        float v = rnd_bf16(s_y[t] + ldf(hidden_in + t));
        s_h[t] = v;
        stf(hidden_out + t, v);
    }
    __syncthreads();
    float rms = row_rms(s_h, h, eps, red);

    if (w_a) {
        norm_into(s_h, rms, w_a, s_y, h);
        __syncthreads();
        dq_nvfp4_row(s_y, h, a_packed, a_scales, 0, sf_inner_dim);
        __syncthreads();
    }
    if (w_b) {
        norm_into(s_h, rms, w_b, s_y, h);
        __syncthreads();
        dq_nvfp4_row(s_y, h, b_packed, b_scales, 0, sf_inner_dim);
        __syncthreads();
    }
    if (w_c) {
        for (int t = threadIdx.x; t < h; t += blockDim.x)
            stf(c_out + t, ldf(w_c + t) * s_h[t] * rms);
    }
}

// ── Хвост FFN-части блока ────────────────────────────────────────────────
//
//   m = moe_acc ? bf16( bf16(norm(dense)·w_pd) + bf16(norm(bf16(moe_acc))·w_pm) ) : dense
//   m = w_post_mlp ? bf16(norm(m)·w_post_mlp) : m
//   hidden = bf16(hidden_in + m); hidden = bf16(hidden · layer_scalar)   → hidden_out
//   next   = bf16(norm(hidden)·w_next) → bf16 и/или MXFP8-пара (вход внимания
//            следующего слоя)
//
// `moe_acc` — f32-сумма взвешенных выходов экспертов (эпилог индексного GEMV).
__global__ void dec_ffn_tail_bf16(
    const bf16_t* __restrict__ dense_out,
    const float* __restrict__ moe_acc,
    const bf16_t* __restrict__ hidden_in,
    const bf16_t* __restrict__ w_post_dense,
    const bf16_t* __restrict__ w_post_moe,
    const bf16_t* __restrict__ w_post_mlp,
    float layer_scalar,
    bf16_t* __restrict__ hidden_out,
    const bf16_t* __restrict__ w_next,
    bf16_t* __restrict__ next_bf16,
    unsigned char* __restrict__ next_mx_packed,
    unsigned char* __restrict__ next_mx_scales,
    int h, float eps_post, float eps)
{
    extern __shared__ float smem[];
    float* s_m = smem;
    float* s_t = smem + h;
    __shared__ float red[32];

    for (int t = threadIdx.x; t < h; t += blockDim.x) s_m[t] = ldf(dense_out + t);
    if (moe_acc) {
        __syncthreads();
        float rms_d = row_rms(s_m, h, eps_post, red);
        for (int t = threadIdx.x; t < h; t += blockDim.x) s_t[t] = rnd_bf16(moe_acc[t]);
        __syncthreads();
        float rms_s = row_rms(s_t, h, eps_post, red);
        for (int t = threadIdx.x; t < h; t += blockDim.x) {
            float d = rnd_bf16(ldf(w_post_dense + t) * s_m[t] * rms_d);
            float s = rnd_bf16(ldf(w_post_moe + t) * s_t[t] * rms_s);
            s_m[t] = rnd_bf16(d + s);
        }
    }
    if (w_post_mlp) {
        __syncthreads();
        float rms_m = row_rms(s_m, h, eps_post, red);
        for (int t = threadIdx.x; t < h; t += blockDim.x)
            s_m[t] = rnd_bf16(ldf(w_post_mlp + t) * s_m[t] * rms_m);
    }
    for (int t = threadIdx.x; t < h; t += blockDim.x) {
        float v = rnd_bf16(ldf(hidden_in + t) + s_m[t]);
        if (layer_scalar != 1.0f) v = rnd_bf16(v * layer_scalar);
        s_m[t] = v;
        stf(hidden_out + t, v);
    }
    if (!w_next) return;
    __syncthreads();
    float rms = row_rms(s_m, h, eps, red);
    for (int t = threadIdx.x; t < h; t += blockDim.x) {
        float y = rnd_bf16(ldf(w_next + t) * s_m[t] * rms);
        s_t[t] = y;
        if (next_bf16) stf(next_bf16 + t, y);
    }
    if (next_mx_packed) {
        __syncthreads();
        dq_mxfp8_row(s_t, h, next_mx_packed, next_mx_scales);
    }
}

// ── Роутер MoE: логиты, top-k, софтмакс по k, per-expert scale ──────────
//
// Один запуск вместо цепочки cast → gemv_f32 → topk → 5 ядер софтмакса →
// gather → mul. Логиты считает несколько блоков (варп на эксперта, порядок
// fmaf — как в mma_gemv_f32, бит-в-бит), последний финиширующий блок (счётчик
// `atomicInc` с автосбросом) выбирает top-k и попутно обнуляет f32-аккумулятор
// выхода экспертов `acc_zero[h]`, чтобы индексному GEMV было куда складывать.
//
//   x      bf16 [h]; w f32 [e, h]; pes f32 [e] | null
//   logits f32 [e] (скретч); counter u32 [1] (изначально 0)
//   out_idx u32 [k]; out_w f32 [k]
// Грид: ceil(e / 8) блоков по 256 нитей. Dynamic smem = h·4 + e·4 байт.
__global__ void dec_router_topk_bf16(
    const bf16_t* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ pes,
    float* __restrict__ logits,
    unsigned int* __restrict__ counter,
    unsigned int* __restrict__ out_idx,
    float* __restrict__ out_w,
    float* __restrict__ acc_zero,
    int e, int h, int k)
{
    extern __shared__ float smem[];
    float* s_x = smem;          // [h]
    float* s_l = smem + h;      // [e] — только у финишного блока
    __shared__ float red_v[256];
    __shared__ unsigned int red_i[256];
    __shared__ bool s_last;
    __shared__ float s_top[64];

    for (int t = threadIdx.x; t < h; t += blockDim.x) s_x[t] = ldf(x + t);
    __syncthreads();

    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row < e) {
        const float* w_row = w + (size_t)row * h;
        const float4* w4 = reinterpret_cast<const float4*>(w_row);
        const float4* x4 = reinterpret_cast<const float4*>(s_x);
        float acc = 0.f;
        int k4 = h >> 2;
        for (int i = lane; i < k4; i += 32) {
            float4 wv = w4[i];
            float4 xv = x4[i];
            acc = fmaf(wv.x, xv.x, acc);
            acc = fmaf(wv.y, xv.y, acc);
            acc = fmaf(wv.z, xv.z, acc);
            acc = fmaf(wv.w, xv.w, acc);
        }
        for (int i = (k4 << 2) + lane; i < h; i += 32) acc = fmaf(w_row[i], s_x[i], acc);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0) logits[row] = acc;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int old = atomicInc(counter, gridDim.x - 1);
        s_last = (old == gridDim.x - 1);
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();

    for (int i = threadIdx.x; i < e; i += blockDim.x) s_l[i] = __ldcg(logits + i);
    __syncthreads();

    // top-k: k раз максимум по блоку, при равенстве — меньший индекс (как topk_rows).
    for (int slot = 0; slot < k; ++slot) {
        float best = __int_as_float(0xFF800000);
        unsigned int best_i = 0;
        for (int i = threadIdx.x; i < e; i += blockDim.x) {
            float v = s_l[i];
            if (v > best) { best = v; best_i = (unsigned)i; }
        }
        red_v[threadIdx.x] = best;
        red_i[threadIdx.x] = best_i;
        __syncthreads();
        for (unsigned int off = blockDim.x >> 1; off > 0; off >>= 1) {
            if (threadIdx.x < off) {
                float o = red_v[threadIdx.x + off];
                unsigned int oi = red_i[threadIdx.x + off];
                if (o > red_v[threadIdx.x] || (o == red_v[threadIdx.x] && oi < red_i[threadIdx.x])) {
                    red_v[threadIdx.x] = o;
                    red_i[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            out_idx[slot] = red_i[0];
            s_top[slot] = red_v[0];
            s_l[red_i[0]] = __int_as_float(0xFF800000);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = s_top[0];
        for (int s = 1; s < k; ++s) m = fmaxf(m, s_top[s]);
        float sum = 0.f;
        for (int s = 0; s < k; ++s) { s_top[s] = expf(s_top[s] - m); sum += s_top[s]; }
        for (int s = 0; s < k; ++s) {
            float wv = s_top[s] / sum;
            if (pes) wv *= pes[out_idx[s]];
            out_w[s] = wv;
        }
    }
    if (acc_zero) {
        for (int t = threadIdx.x; t < h; t += blockDim.x) acc_zero[t] = 0.f;
    }
}

// ── gelu_tanh(gate) · up → NVFP4-пара ────────────────────────────────────
//
// Строка `r`: gate = gate_ptr + r·stride, up = up_ptr + r·stride (у сцепленного
// [gate|up] up_ptr = gate_ptr + inter, stride = 2·inter; у раздельных буферов
// stride = inter). Округления как у host-пути: gelu → T, произведение → T.
// Одна нить = группа из 16; scales — tile-раскладка строки `r` (без хвоста).
__device__ __forceinline__ float gelu_tanh_f(float x) {
    float c = sqrtf(2.0f / 3.14159265358979323846f);
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

template <typename T>
__device__ __forceinline__ void geglu_quant_impl(
    const T* __restrict__ gate, const T* __restrict__ up, long long stride,
    unsigned char* __restrict__ packed, unsigned char* __restrict__ scales,
    int rows, int inter, int sf_inner_dim)
{
    int groups = inter >> 4;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)rows * groups) return;
    int row = (int)(g / groups);
    int col = (int)(g % groups);
    const T* gp = gate + (long long)row * stride + (long long)col * 16;
    const T* upp = up + (long long)row * stride + (long long)col * 16;
    float v[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        float gv = ldf(gp + i);
        float uv = ldf(upp + i);
        float a = rnd_t(gp, gelu_tanh_f(gv));
        v[i] = rnd_t(gp, a * uv);
    }
    unsigned char sb;
    dq_nvfp4_group(v, packed + ((size_t)row * inter + (size_t)col * 16) / 2, &sb);
    scales[dq_tile_scale_offset((unsigned)row, (unsigned)col, (unsigned)sf_inner_dim)] = sb;
}

__global__ void dec_geglu_quant_nvfp4_f16(
    const __half* gate, const __half* up, long long stride,
    unsigned char* packed, unsigned char* scales, int rows, int inter, int sf_inner_dim)
{ geglu_quant_impl<__half>(gate, up, stride, packed, scales, rows, inter, sf_inner_dim); }

__global__ void dec_geglu_quant_nvfp4_bf16(
    const bf16_t* gate, const bf16_t* up, long long stride,
    unsigned char* packed, unsigned char* scales, int rows, int inter, int sf_inner_dim)
{ geglu_quant_impl<bf16_t>(gate, up, stride, packed, scales, rows, inter, sf_inner_dim); }

// ── Подготовка внимания: нормы голов + RoPE + запись в KV ───────────────
//
// Блок = одна голова (grid = nh + nkv + nkv), blockDim = hd.
//   q: bf16(norm_q(q))·RoPE → q_out
//   k: bf16(norm_k(k))·RoPE → k_cache[(kvh·max_seq + kv_pos)·hd + d]
//   v: v_norm ? bf16(norm(v)) : v → v_cache[…]   (v_in == null → V = сырой K,
//      снятый ДО нормы и RoPE — `attention_k_eq_v`)
// RoPE — как rope_apply_impl: пары (d, d±half) по rotary_dim, таблицы
// [cap, rotary_dim] bf16, позиция pos_ptr[0].
__global__ void dec_attn_prep_bf16(
    const bf16_t* __restrict__ q_in,
    const bf16_t* __restrict__ k_in,
    const bf16_t* __restrict__ v_in,
    const bf16_t* __restrict__ q_norm_w,
    const bf16_t* __restrict__ k_norm_w,
    int v_norm,
    const bf16_t* __restrict__ cos_t,
    const bf16_t* __restrict__ sin_t,
    const unsigned int* __restrict__ pos_ptr,
    int rotary_dim,
    const unsigned int* __restrict__ kv_pos_ptr,
    bf16_t* __restrict__ q_out,
    bf16_t* __restrict__ k_cache,
    bf16_t* __restrict__ v_cache,
    int max_seq, int nh, int nkv, int hd, float eps)
{
    extern __shared__ float smem[];
    float* s_y = smem;  // [hd]
    __shared__ float red[32];
    int d = threadIdx.x;
    int blk = blockIdx.x;
    int kind = blk < nh ? 0 : (blk < nh + nkv ? 1 : 2);  // 0 = q, 1 = k, 2 = v
    int head = kind == 0 ? blk : (kind == 1 ? blk - nh : blk - nh - nkv);
    const bf16_t* src;
    const bf16_t* wn;
    bool do_norm;
    if (kind == 0) { src = q_in + (size_t)head * hd; wn = q_norm_w; do_norm = q_norm_w != nullptr; }
    else if (kind == 1) { src = k_in + (size_t)head * hd; wn = k_norm_w; do_norm = k_norm_w != nullptr; }
    else { src = (v_in ? v_in : k_in) + (size_t)head * hd; wn = nullptr; do_norm = v_norm != 0; }

    float x = d < hd ? ldf(src + d) : 0.f;
    float y = x;
    if (do_norm) {
        float sumsq = block_sum(x * x, red);
        float rms = rsqrtf(sumsq / (float)hd + eps);
        float wv = wn ? ldf(wn + d) : 1.0f;
        y = rnd_bf16(wv * x * rms);
    }
    if (d >= hd) return;
    if (kind == 2) {
        unsigned int pos = kv_pos_ptr[0];
        stf(v_cache + ((size_t)head * max_seq + pos) * hd + d, y);
        return;
    }
    s_y[d] = y;
    __syncthreads();
    float out = y;
    if (d < rotary_dim) {
        unsigned int pos = pos_ptr[0];
        int half = rotary_dim >> 1;
        bool low = d < half;
        int partner = low ? d + half : d - half;
        float c = ldf(cos_t + (size_t)pos * rotary_dim + d);
        float s = ldf(sin_t + (size_t)pos * rotary_dim + d);
        float xp = s_y[partner];
        float rot = low ? -xp : xp;
        out = y * c + rot * s;
    }
    if (kind == 0) {
        stf(q_out + (size_t)head * hd + d, out);
    } else {
        unsigned int pos = kv_pos_ptr[0];
        stf(k_cache + ((size_t)head * max_seq + pos) * hd + d, out);
    }
}

}  // extern "C"
