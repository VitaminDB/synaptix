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

// Две суммы по блоку за одну синхронизацию (результат у всех нитей). `red` — smem [64].
__device__ __forceinline__ void block_sum2(float& a, float& b, float* red) {
    unsigned int mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_down_sync(mask, a, off, 32);
        b += __shfl_down_sync(mask, b, off, 32);
    }
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __syncthreads();
    if (lane == 0) { red[warp] = a; red[32 + warp] = b; }
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    if (warp == 0) {
        float ra = (lane < nw) ? red[lane] : 0.f;
        float rb = (lane < nw) ? red[32 + lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            ra += __shfl_down_sync(mask, ra, off, 32);
            rb += __shfl_down_sync(mask, rb, off, 32);
        }
        if (lane == 0) { red[0] = ra; red[32] = rb; }
    }
    __syncthreads();
    a = red[0];
    b = red[32];
    __syncthreads();
}

// Элементов строки на нить: строка H ≤ DEC_MAXE·blockDim живёт в регистрах,
// все векторы грузятся разом в начале ядра — латентности перекрываются, а не
// выстраиваются в цепочку по фазам. 4 на нить: у H=2816 это 704 нити — квант
// эпилога (четвёрка нитей на группу) укладывается в один проход.
#define DEC_MAXE 4

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

// Тот же результат, что у nvfp4_quant.cu::encode_e4m3, но без log2f/exp2f:
// floor(log2|v|) для нормального float — его поле экспоненты, 2^e — сборка
// битами (в ветку exp_biased < 1 попадают и субнормали: у них поле 0 →
// e = −127 < −6, как и у log2f).
__device__ __forceinline__ unsigned char dq_encode_e4m3(float x) {
    if (isnan(x)) return 0x7F;
    float v = fminf(fmaxf(x, -448.0f), 448.0f);
    unsigned int sign = signbit(v) ? 1 : 0;
    float abs_v = fabsf(v);
    if (abs_v == 0.0f) return (unsigned char)(sign << 7);
    int exp_raw = (int)((__float_as_uint(abs_v) >> 23) & 0xFFu) - 127;
    int exp_biased = exp_raw + 7;
    if (exp_biased < 1) {
        int m = (int)nearbyintf(abs_v * 512.0f);
        m = max(0, min(7, m));
        return (unsigned char)((sign << 7) | (unsigned int)m);
    }
    if (exp_biased > 15) return (unsigned char)((sign << 7) | 0x7E);
    float pow2 = __uint_as_float((unsigned int)(exp_raw + 127) << 23);
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

// Квант 4 значений группы из 16 (четвёрка нитей на группу): amax по группе
// собирается shfl_xor внутри четвёрки, каждая нить кодирует свои 4 значения в
// 2 байта. Одна нить на группу с 16 encode подряд держала активными лишь
// 176 нитей из 1024 и стоила ~4 мкс на строку.
__device__ __forceinline__ void dq_nvfp4_quad(
    const float* v4, bool valid, unsigned char* packed2, unsigned char* scale_byte, int sub) {
    float amax = 0.f;
    if (valid) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) amax = fmaxf(amax, fabsf(v4[i]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));
    if (!valid) return;
    float scale_raw = (amax > 0.0f) ? (amax / 6.0f) : 1e-9f;
    unsigned char sb = dq_encode_e4m3(scale_raw);
    float scale_q = dq_decode_e4m3(sb);
    if (scale_q == 0.0f) scale_q = 1e-9f;
    unsigned char b0 = dq_encode_e2m1_rtne(v4[0] / scale_q);
    unsigned char b1 = dq_encode_e2m1_rtne(v4[1] / scale_q);
    unsigned char b2 = dq_encode_e2m1_rtne(v4[2] / scale_q);
    unsigned char b3 = dq_encode_e2m1_rtne(v4[3] / scale_q);
    unsigned short o = (unsigned short)((b0 & 0x0F) | ((b1 & 0x0F) << 4) | ((b2 & 0x0F) << 8) | ((b3 & 0x0F) << 12));
    *reinterpret_cast<unsigned short*>(packed2) = o;
    if (sub == 0) *scale_byte = sb;
}

// Строка `s[h]` (значения bf16-точности) → NVFP4-пара строки `row`. Все нити
// блока участвуют в shfl (число итераций одинаково для всех).
__device__ __forceinline__ void dq_nvfp4_row(
    const float* s, int h, unsigned char* packed, unsigned char* scales, int row, int sf_inner_dim) {
    int groups = h >> 4;
    int quad = threadIdx.x >> 2;
    int sub = threadIdx.x & 3;
    int quads = blockDim.x >> 2;
    for (int base = 0; base < groups; base += quads) {
        int g = base + quad;
        bool valid = g < groups;
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        if (valid) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = s[g * 16 + sub * 4 + i];
        }
        unsigned char* p2 = packed + ((size_t)row * h + (size_t)g * 16) / 2 + sub * 2;
        unsigned char* sc = scales + (valid ? dq_tile_scale_offset((unsigned)row, (unsigned)g, (unsigned)sf_inner_dim) : 0u);
        dq_nvfp4_quad(v, valid, p2, sc, sub);
    }
}

// Строка `s[h]` → MXFP8 natural (packed [h] e4m3, scales [h/32] E8M0):
// восьмёрка нитей на группу из 32, по 4 значения на нить.
__device__ __forceinline__ void dq_mxfp8_row(
    const float* s, int h, unsigned char* packed, unsigned char* scales) {
    int groups = h >> 5;
    int oct = threadIdx.x >> 3;
    int sub = threadIdx.x & 7;
    int octs = blockDim.x >> 3;
    for (int base = 0; base < groups; base += octs) {
        int g = base + oct;
        bool valid = g < groups;
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        float amax = 0.f;
        if (valid) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[i] = s[g * 32 + sub * 4 + i];
                amax = fmaxf(amax, fabsf(v[i]));
            }
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 4));
        if (!valid) continue;
        unsigned char sb = (unsigned char)((__float_as_uint(
            __uint_as_float(__float_as_uint(amax) & 0x7F800000u) / 256.0f)) >> 23);
        float sv = fmaxf(__uint_as_float(((unsigned)sb) << 23), 1e-12f);
        unsigned char ob[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            ob[i] = __nv_fp8_e4m3(fminf(fmaxf(v[i] / sv, -448.0f), 448.0f)).__x;
        *reinterpret_cast<unsigned int*>(packed + (size_t)g * 32 + sub * 4) =
            *reinterpret_cast<unsigned int*>(&ob[0]);
        if (sub == 0) scales[g] = sb;
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
extern "C" __global__ void dec_attn_tail_bf16(
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
    float* s_y = smem;  // [h]
    __shared__ float red[64];
    const int tid = threadIdx.x, bs = blockDim.x;

    float xv[DEC_MAXE], rv[DEC_MAXE], pw[DEC_MAXE], wa[DEC_MAXE], wb[DEC_MAXE], wc[DEC_MAXE];
    #pragma unroll
    for (int i = 0; i < DEC_MAXE; ++i) {
        int t = tid + i * bs;
        bool ok = t < h;
        xv[i] = ok ? ldf(attn_out + t) : 0.f;
        rv[i] = ok ? ldf(hidden_in + t) : 0.f;
        pw[i] = (ok && post_w) ? ldf(post_w + t) : 1.f;
        wa[i] = (ok && w_a) ? ldf(w_a + t) : 0.f;
        wb[i] = (ok && w_b) ? ldf(w_b + t) : 0.f;
        wc[i] = (ok && w_c) ? ldf(w_c + t) : 0.f;
    }
    // y = post-norm(attn_out) либо сам attn_out.
    if (post_w) {
        float sq = 0.f;
        #pragma unroll
        for (int i = 0; i < DEC_MAXE; ++i) sq += xv[i] * xv[i];
        sq = block_sum(sq, red);
        float rms = rsqrtf(sq / (float)h + eps_post);
        #pragma unroll
        for (int i = 0; i < DEC_MAXE; ++i) xv[i] = rnd_bf16(pw[i] * xv[i] * rms);
    }
    // hidden = bf16(y + residual).
    float sq = 0.f;
    #pragma unroll
    for (int i = 0; i < DEC_MAXE; ++i) {
        int t = tid + i * bs;
        float v = rnd_bf16(xv[i] + rv[i]);
        xv[i] = v;
        if (t < h) stf(hidden_out + t, v);
        sq += v * v;
    }
    sq = block_sum(sq, red);
    float rms = rsqrtf(sq / (float)h + eps);

    if (w_a) {
        #pragma unroll
        for (int i = 0; i < DEC_MAXE; ++i) {
            int t = tid + i * bs;
            if (t < h) s_y[t] = rnd_bf16(wa[i] * xv[i] * rms);
        }
        __syncthreads();
        dq_nvfp4_row(s_y, h, a_packed, a_scales, 0, sf_inner_dim);
        __syncthreads();
    }
    if (w_b) {
        #pragma unroll
        for (int i = 0; i < DEC_MAXE; ++i) {
            int t = tid + i * bs;
            if (t < h) s_y[t] = rnd_bf16(wb[i] * xv[i] * rms);
        }
        __syncthreads();
        dq_nvfp4_row(s_y, h, b_packed, b_scales, 0, sf_inner_dim);
        __syncthreads();
    }
    if (w_c) {
        #pragma unroll
        for (int i = 0; i < DEC_MAXE; ++i) {
            int t = tid + i * bs;
            if (t < h) stf(c_out + t, wc[i] * xv[i] * rms);
        }
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
extern "C" __global__ void dec_ffn_tail_bf16(
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
    float* s_t = smem;  // [h]
    __shared__ float red[64];
    const int tid = threadIdx.x, bs = blockDim.x;

    float mv[DEC_MAXE], av[DEC_MAXE], hv[DEC_MAXE], wd[DEC_MAXE], wm[DEC_MAXE], wp[DEC_MAXE], wn[DEC_MAXE];
    #pragma unroll
    for (int i = 0; i < DEC_MAXE; ++i) {
        int t = tid + i * bs;
        bool ok = t < h;
        mv[i] = ok ? ldf(dense_out + t) : 0.f;
        av[i] = (ok && moe_acc) ? rnd_bf16(moe_acc[t]) : 0.f;
        hv[i] = ok ? ldf(hidden_in + t) : 0.f;
        wd[i] = (ok && w_post_dense) ? ldf(w_post_dense + t) : 0.f;
        wm[i] = (ok && w_post_moe) ? ldf(w_post_moe + t) : 0.f;
        wp[i] = (ok && w_post_mlp) ? ldf(w_post_mlp + t) : 0.f;
        wn[i] = (ok && w_next) ? ldf(w_next + t) : 0.f;
    }
    if (moe_acc) {
        float sd = 0.f, ss = 0.f;
        #pragma unroll
        for (int i = 0; i < DEC_MAXE; ++i) { sd += mv[i] * mv[i]; ss += av[i] * av[i]; }
        block_sum2(sd, ss, red);
        float rms_d = rsqrtf(sd / (float)h + eps_post);
        float rms_s = rsqrtf(ss / (float)h + eps_post);
        #pragma unroll
        for (int i = 0; i < DEC_MAXE; ++i) {
            float d = rnd_bf16(wd[i] * mv[i] * rms_d);
            float sv = rnd_bf16(wm[i] * av[i] * rms_s);
            mv[i] = rnd_bf16(d + sv);
        }
    }
    if (w_post_mlp) {
        float sq = 0.f;
        #pragma unroll
        for (int i = 0; i < DEC_MAXE; ++i) sq += mv[i] * mv[i];
        sq = block_sum(sq, red);
        float rms_m = rsqrtf(sq / (float)h + eps_post);
        #pragma unroll
        for (int i = 0; i < DEC_MAXE; ++i) mv[i] = rnd_bf16(wp[i] * mv[i] * rms_m);
    }
    float sq = 0.f;
    #pragma unroll
    for (int i = 0; i < DEC_MAXE; ++i) {
        int t = tid + i * bs;
        float v = rnd_bf16(hv[i] + mv[i]);
        if (layer_scalar != 1.0f) v = rnd_bf16(v * layer_scalar);
        mv[i] = v;
        if (t < h) stf(hidden_out + t, v);
        sq += v * v;
    }
    if (!w_next) return;
    sq = block_sum(sq, red);
    float rms = rsqrtf(sq / (float)h + eps);
    #pragma unroll
    for (int i = 0; i < DEC_MAXE; ++i) {
        int t = tid + i * bs;
        if (t < h) {
            float y = rnd_bf16(wn[i] * mv[i] * rms);
            if (next_bf16) stf(next_bf16 + t, y);
            s_t[t] = y;
        }
    }
    if (next_mx_packed) {
        __syncthreads();
        dq_mxfp8_row(s_t, h, next_mx_packed, next_mx_scales);
    }
}

// ── top-k роутера MoE одним варпом ───────────────────────────────────────
//
// Логиты уже посчитаны (хвостовые блоки группового GEMV); здесь top-k,
// софтмакс по k, `per_expert_scale` и обнуление f32-аккумулятора выхода
// экспертов. Логиты в регистрах: lane держит PER значений (шаблон, чтобы
// циклы раскрылись без предикатов), k раз максимум через shfl, при
// равенстве — меньший индекс (как topk_rows). Софтмакс — по лейнам.
template <int PER>
__device__ __forceinline__ void router_topk_warp_t(
    const float* __restrict__ logits, const float* __restrict__ pes,
    unsigned int* __restrict__ out_idx, float* __restrict__ out_w,
    int e, int k, int lane)
{
    float lv[PER];
    #pragma unroll
    for (int i = 0; i < PER; ++i) {
        int j = lane + 32 * i;
        lv[i] = (j < e) ? __ldg(logits + j) : __int_as_float(0xFF800000);
    }
    float my_v = __int_as_float(0xFF800000);
    int my_i = 0;
    for (int slot = 0; slot < k; ++slot) {
        float best = __int_as_float(0xFF800000);
        int best_i = 0x7FFFFFFF;
        #pragma unroll
        for (int i = 0; i < PER; ++i) {
            int j = lane + 32 * i;
            if (lv[i] > best) { best = lv[i]; best_i = j; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xFFFFFFFFu, best, off);
            int oi = __shfl_xor_sync(0xFFFFFFFFu, best_i, off);
            if (ov > best || (ov == best && oi < best_i)) { best = ov; best_i = oi; }
        }
        if (lane == slot) { my_v = best; my_i = best_i; }
        if ((best_i & 31) == lane) {
            #pragma unroll
            for (int i = 0; i < PER; ++i) if (i == (best_i >> 5)) lv[i] = __int_as_float(0xFF800000);
        }
    }
    // Софтмакс по k выбранным: lane s < k держит свой логит.
    float m = my_v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, off));
    float ev = (lane < k) ? expf(my_v - m) : 0.f;
    float sum = ev;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
    if (lane < k) {
        float wv = ev / sum;
        if (pes) wv *= pes[my_i];
        out_idx[lane] = (unsigned)my_i;
        out_w[lane] = wv;
    }
}

__device__ __forceinline__ void router_topk_warp(
    const float* __restrict__ logits, const float* __restrict__ pes,
    unsigned int* __restrict__ out_idx, float* __restrict__ out_w,
    int e, int k, int lane)
{
    if (e <= 128) router_topk_warp_t<4>(logits, pes, out_idx, out_w, e, k, lane);
    else if (e <= 256) router_topk_warp_t<8>(logits, pes, out_idx, out_w, e, k, lane);
    else if (e <= 512) router_topk_warp_t<16>(logits, pes, out_idx, out_w, e, k, lane);
    else router_topk_warp_t<32>(logits, pes, out_idx, out_w, e, k, lane);
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

// Четвёрка нитей на группу из 16 (см. dq_nvfp4_quad).
template <typename T>
__device__ __forceinline__ void geglu_quant_impl(
    const T* __restrict__ gate, const T* __restrict__ up, long long stride,
    unsigned char* __restrict__ packed, unsigned char* __restrict__ scales,
    int rows, int inter, int sf_inner_dim)
{
    int groups = inter >> 4;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long g = tid >> 2;
    int sub = (int)(tid & 3);
    bool valid = g < (long long)rows * groups;
    int row = valid ? (int)(g / groups) : 0;
    int col = valid ? (int)(g % groups) : 0;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (valid) {
        const T* gp = gate + (long long)row * stride + (long long)col * 16 + sub * 4;
        const T* upp = up + (long long)row * stride + (long long)col * 16 + sub * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float gv = ldf(gp + i);
            float uv = ldf(upp + i);
            float a = rnd_t(gp, gelu_tanh_f(gv));
            v[i] = rnd_t(gp, a * uv);
        }
    }
    unsigned char* p2 = packed + ((size_t)row * inter + (size_t)col * 16) / 2 + sub * 2;
    unsigned char* sc = scales + (valid ? dq_tile_scale_offset((unsigned)row, (unsigned)col, (unsigned)sf_inner_dim) : 0u);
    dq_nvfp4_quad(v, valid, p2, sc, sub);
}

// Последний блок грида (при r_e > 0) — top-k роутера: варп 0 выбирает
// экспертов, остальные варпы обнуляют аккумулятор. Так top-k едет в одном
// запуске с geglu плотного MLP, а не отдельным латентным ядром.
template <typename T>
__device__ __forceinline__ void geglu_topk_kernel(
    const T* gate, const T* up, long long stride,
    unsigned char* packed, unsigned char* scales, int rows, int inter, int sf_inner_dim,
    const float* logits, const float* pes, unsigned int* out_idx, float* out_w, float* acc_zero,
    int r_e, int r_k, int r_h)
{
    if (r_e > 0 && blockIdx.x == gridDim.x - 1) {
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        if (warp == 0) {
            router_topk_warp(logits, pes, out_idx, out_w, r_e, r_k, lane);
        } else if (acc_zero) {
            for (int t = threadIdx.x - 32; t < r_h; t += blockDim.x - 32) acc_zero[t] = 0.f;
        }
        return;
    }
    geglu_quant_impl<T>(gate, up, stride, packed, scales, rows, inter, sf_inner_dim);
}

extern "C" __global__ void dec_geglu_quant_nvfp4_f16(
    const __half* gate, const __half* up, long long stride,
    unsigned char* packed, unsigned char* scales, int rows, int inter, int sf_inner_dim,
    const float* logits, const float* pes, unsigned int* out_idx, float* out_w, float* acc_zero,
    int r_e, int r_k, int r_h)
{
    geglu_topk_kernel<__half>(gate, up, stride, packed, scales, rows, inter, sf_inner_dim,
                              logits, pes, out_idx, out_w, acc_zero, r_e, r_k, r_h);
}

extern "C" __global__ void dec_geglu_quant_nvfp4_bf16(
    const bf16_t* gate, const bf16_t* up, long long stride,
    unsigned char* packed, unsigned char* scales, int rows, int inter, int sf_inner_dim,
    const float* logits, const float* pes, unsigned int* out_idx, float* out_w, float* acc_zero,
    int r_e, int r_k, int r_h)
{
    geglu_topk_kernel<bf16_t>(gate, up, stride, packed, scales, rows, inter, sf_inner_dim,
                              logits, pes, out_idx, out_w, acc_zero, r_e, r_k, r_h);
}

// ── Подготовка внимания: нормы голов + RoPE + запись в KV ───────────────
//
// Блок = одна голова (grid = nh + nkv + nkv), blockDim = hd.
//   q: bf16(norm_q(q))·RoPE → q_out
//   k: bf16(norm_k(k))·RoPE → k_cache[(kvh·max_seq + kv_pos)·hd + d]
//   v: v_norm ? bf16(norm(v)) : v → v_cache[…]   (v_in == null → V = сырой K,
//      снятый ДО нормы и RoPE — `attention_k_eq_v`)
// RoPE — как rope_apply_impl: пары (d, d±half) по rotary_dim, таблицы
// [cap, rotary_dim] bf16, позиция pos_ptr[0].
extern "C" __global__ void dec_attn_prep_bf16(
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

