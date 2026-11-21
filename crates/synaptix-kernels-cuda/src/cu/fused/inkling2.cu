// Ядра Inkling-2 (thinkingmachines): этапы, которых нет среди общих операций —
// смесь блоков по глубине (attention-residuals), короткие каузальные свёртки
// с состоянием, подготовка q/k/v/r с L2-клипом и записью в KV, внимание с
// относительным bias по расстоянию и синком на (токен, голову), softcap-SwiGLU
// экспертов с общим множителем NVFP4, маршрутизация и сборка MoE.
// Счёт в F32; KV и входы экспертов — BF16. Аргументы — одна структура Ink2Args
// (см. synaptix_core::backend::Ink2Launch).
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

struct Ink2Args {
    unsigned long long a[12];
    float f[4];
    unsigned int u[16];
};

#define PF(i) ((float*)args.a[i])
#define CPF(i) ((const float*)args.a[i])
#define CPB(i) ((const __nv_bfloat16*)args.a[i])
#define PB(i) ((__nv_bfloat16*)args.a[i])

// Элемент i входа, лежащего в F32 либо BF16 (выход GEMV как есть, без приведения и копии).
__device__ __forceinline__ float ink2_ld(unsigned long long p, unsigned long long i, unsigned bf) {
    return bf ? __bfloat162float(((const __nv_bfloat16*)p)[i]) : ((const float*)p)[i];
}

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Сумма по блоку; red — не меньше blockDim/32 float.
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    float r = 0.0f;
    for (int i = 0; i < nw; ++i) r += red[i];
    return r;
}

// ---- Квантованный KV полных слотов (u11 — формат) ----
// 0 — BF16 [nkv,cap,128]; 1 — MXFP8: E4M3 [nkv,cap,128] байт + масштаб E8M0 на 32 канала;
// 2 — INT8 + масштаб E8M0 на 32 канала. Масштабы — a11 U8 [2][nkv][cap][4] (k, затем v).
// Масштаб блока — наименьшая степень двойки sc, при которой amax/sc ≤ 448 (E4M3) / 127 (INT8):
// без насыщения крайних значений. Деквант e4m3·2^k и int8·2^k точно представим в BF16, так что
// ядра внимания после декванта считают ровно как на BF16-кэше.
#define INK2_KV_BF16 0u
#define INK2_KV_MXFP8 1u
#define INK2_KV_INT8 2u

__device__ __forceinline__ float ink2_e8m0(unsigned s) { return __uint_as_float(s << 23); }  // s ≥ 1

__device__ __forceinline__ unsigned char ink2_f2e4m3(float y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.0f), "f"(y));
    return (unsigned char)(r & 0xFFu);
#else
    unsigned sign = signbit(y) ? 0x80u : 0u;
    float a = fminf(fabsf(y), 448.0f);
    if (a < 0.015625f) return (unsigned char)(sign | (unsigned)min(8, (int)nearbyintf(a * 512.0f)));  // субнормали (8 → 0x08 = 2^-6)
    int e;
    float m = frexpf(a, &e);  // a = m·2^e, m ∈ [0.5, 1)
    int q = (int)nearbyintf(m * 16.0f);  // 8..16
    if (q == 16) { q = 8; ++e; }
    return (unsigned char)(sign | ((unsigned)(e + 6) << 3) | (unsigned)(q - 8));
#endif
}

// Квантование значения x варпом (32 лейна = блок из 32 каналов): байт и E8M0 блока.
__device__ __forceinline__ unsigned char ink2_kv_enc(float x, unsigned fmt, unsigned& sbyte) {
    float amax = fabsf(x);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (!(amax > 0.0f)) { sbyte = 127u; return 0; }
    int e;
    float m = frexpf(amax / (fmt == INK2_KV_MXFP8 ? 448.0f : 127.0f), &e);
    if (m == 0.5f) --e;  // r = 2^(e-1) ровно
    e = max(-126, min(127, e));
    sbyte = (unsigned)(e + 127);
    float y = ldexpf(x, -e);
    if (fmt == INK2_KV_MXFP8) return ink2_f2e4m3(y);
    int q = max(-127, min(127, __float2int_rn(y)));
    return (unsigned char)(signed char)q;
}

// 4 байта (младший — первый канал) → 4 значения × масштаб.
__device__ __forceinline__ void ink2_kv_dec4(unsigned w, unsigned fmt, float sc, float* y) {
    if (fmt == INK2_KV_INT8) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) y[j] = (float)(signed char)(w >> (8 * j)) * sc;
        return;
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        unsigned h2;
        unsigned short pair = (unsigned short)(w >> (16 * j));
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(pair));
        float2 f = __half22float2(*reinterpret_cast<__half2*>(&h2));
        y[2 * j] = f.x * sc;
        y[2 * j + 1] = f.y * sc;
    }
#else
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        unsigned b = (w >> (8 * j)) & 0xFFu, ex = (b >> 3) & 15u, mt = b & 7u;
        float v = ex ? ldexpf(1.0f + mt / 8.0f, (int)ex - 7) : ldexpf((float)mt, -9);
        y[j] = ((b & 0x80u) ? -v : v) * sc;
    }
#endif
}

// RMS-норма строки: out = x·rsqrt(mean(x²)+eps)·w (w = 0 — без веса).
// a0 x f32 [T,H], a1 w f32 [H]|0, a2 out; u0 T, u1 H, u2 out_bf16; f0 eps.
// Несколько входов одним запуском (blockIdx.z = j): вход j — с j·u3 элемента a0, вес — a1 | a4 | a5,
// выход — a2 | a6 | a7 (отдельные тензоры).
extern "C" __global__ void ink2_rms(Ink2Args args) {
    __shared__ float red[32];
    unsigned int t = blockIdx.x, H = args.u[1], z = blockIdx.z;
    const float* x = CPF(0) + (unsigned long long)z * args.u[3] + (unsigned long long)t * H;
    const float* w = z == 0 ? CPF(1) : (const float*)args.a[3 + z];
    const unsigned long long outp = z == 0 ? args.a[2] : args.a[5 + z];
    float ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < H; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss, red);
    float inv = rsqrtf(ss / (float)H + args.f[0]);
    for (unsigned int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = x[i] * inv * (w ? w[i] : 1.0f);
        if (args.u[2]) ((__nv_bfloat16*)outp)[(unsigned long long)t * H + i] = __float2bfloat16(v);
        else ((float*)outp)[(unsigned long long)t * H + i] = v;
    }
}

// Короткая каузальная свёртка с остатком: y_t = (Σ_j w[c,j]·x_{t-K+1+j} + x_t)·scale,
// w[c, K-1] — текущий токен. Динамические отводы (полные слои): + Σ_lag dyn[t, h·K+lag]·x_{t-lag},
// h = c / (C/heads). Состояние — последние K-1 входов (старший первый), обновляется.
// a0 x [T,C] (шаг строки u5, 0 — C), a1 out f32 [T,C], a2 state f32 [K-1,C], a3 w f32 [C,K],
// a4 dyn [T,heads·K]|0 (шаг строки u6, 0 — heads·K), a5 остаток f32 [T,C]|0 (out = остаток + свёртка);
// x и dyn — F32 либо BF16 (u4); u0 T, u1 C, u2 K (≤8), u3 heads; f0 scale.
// Вторая свёртка той же формы в том же запуске (blockIdx.z = 1, k и v): a6 x, a7 out, a8 state,
// a9 w, a10 dyn (без остатка).
// gridDim.y = 1 — поток проходит все T токенов и обновляет состояние (декод); gridDim.y = T — блок
// на токен (префилл, без цикла по T), состояние затем обновляет ink2_conv_state. Сумма для
// токена в обоих режимах одна и та же.
struct Ink2Conv {
    unsigned long long x, dyn;
    float* out;
    float* st;
    const float* w;
    const float* res;
};

__device__ __forceinline__ Ink2Conv ink2_conv_args(const Ink2Args& args) {
    const bool z = blockIdx.z != 0;
    Ink2Conv c;
    c.x = z ? args.a[6] : args.a[0];
    c.out = (float*)(z ? args.a[7] : args.a[1]);
    c.st = (float*)(z ? args.a[8] : args.a[2]);
    c.w = (const float*)(z ? args.a[9] : args.a[3]);
    c.dyn = z ? args.a[10] : args.a[4];
    c.res = z ? nullptr : CPF(5);
    return c;
}

extern "C" __global__ void ink2_conv(Ink2Args args) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int T = args.u[0], C = args.u[1], K = args.u[2], heads = args.u[3];
    if (c >= C) return;
    const Ink2Conv cv = ink2_conv_args(args);
    const unsigned bf = args.u[4];
    const unsigned long long xs = args.u[5] ? args.u[5] : C, ds = args.u[6] ? args.u[6] : heads * K;
    const float* w = cv.w + (unsigned long long)c * K;
    unsigned int h = heads ? c / (C / heads) : 0;
    const bool par = gridDim.y > 1;
    const unsigned t0 = par ? blockIdx.y : 0u, t1 = par ? blockIdx.y + 1u : T;
    float hist[8];
    for (unsigned int j = 0; j + 1 < K; ++j) {
        int p = (int)t0 - (int)(K - 1) + (int)j;  // позиция входа отвода j для токена t0
        hist[j] = p >= 0 ? ink2_ld(cv.x, (unsigned long long)p * xs + c, bf) : cv.st[(unsigned long long)(p + (int)K - 1) * C + c];
    }
    for (unsigned int t = t0; t < t1; ++t) {
        float xt = ink2_ld(cv.x, (unsigned long long)t * xs + c, bf);
        hist[K - 1] = xt;
        float acc = 0.0f;
        for (unsigned int j = 0; j < K; ++j) acc += w[j] * hist[j];
        if (cv.dyn) {
            const unsigned long long d = (unsigned long long)t * ds + h * K;
            for (unsigned int lag = 0; lag < K; ++lag) acc += ink2_ld(cv.dyn, d + lag, bf) * hist[K - 1 - lag];
        }
        // Явные _rn: без слияния в FMA — ровно как свёртка и отдельное сложение с потоком.
        float y = __fmul_rn(acc + xt, args.f[0]);
        if (cv.res) y = __fadd_rn(cv.res[(unsigned long long)t * C + c], y);
        cv.out[(unsigned long long)t * C + c] = y;
        for (unsigned int j = 0; j + 1 < K; ++j) hist[j] = hist[j + 1];
    }
    if (!par)
        for (unsigned int j = 0; j + 1 < K; ++j) cv.st[(unsigned long long)j * C + c] = hist[j];
}

// Состояние после префилла (режим блока на токен): последние K-1 входов. Аргументы — как у ink2_conv.
extern "C" __global__ void ink2_conv_state(Ink2Args args) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int T = args.u[0], C = args.u[1], K = args.u[2];
    if (c >= C) return;
    const Ink2Conv cv = ink2_conv_args(args);
    const unsigned long long xs = args.u[5] ? args.u[5] : C;
    // Ячейка j читает вход T-K+1+j либо прежнюю ячейку T+j (> j — ещё не перезаписана).
    for (unsigned int j = 0; j + 1 < K; ++j) {
        int p = (int)T - (int)(K - 1) + (int)j;
        cv.st[(unsigned long long)j * C + c] = p >= 0 ? ink2_ld(cv.x, (unsigned long long)p * xs + c, args.u[4]) : cv.st[(unsigned long long)(p + (int)K - 1) * C + c];
    }
}

// Подготовка внимания: блок на (токен, вектор); вектора 0..nh-1 — q (и r той же
// головы), nh..nh+nkv-1 — k, дальше v. q: ·scale → L2-клип → ·tau; r: ·rscale →
// клип → ·tau; k: клип, в кэш; v: RMS без веса, в кэш. tau = 1 + α·ln(max(1,(pos+1)/floor))
// у полных слоёв, иначе 1.
// a0 q, a1 k, a2 v, a3 r (f32 [T, ·]), a4 qscale f32 [nh·d], a5 rscale f32 [nh·dr],
// a6 q_out f32, a7 r_out f32, a8 kcache bf16 [nkv,cap,d], a9 vcache bf16;
// u0 T, u1 nh, u2 nkv, u3 d, u4 dr, u5 cap, u6 past (или a10 — u32 на карте), u7 full, u8 floor,
// u9 кольцо (позиция p — ячейка p mod кольцо; 0 — линейно), u11 формат KV (0 — BF16; иначе
// байты в a8/a9 и масштабы в a11, см. «Квантованный KV»); q и r — F32 либо BF16 (u12), шаг строки
// у обоих u13 (0 — nh·d и nh·dr); f0 qk_clip, f1 rel_clip, f2 eps, f3 alpha.
extern "C" __global__ void ink2_qkv(Ink2Args args) {
    __shared__ float red[32];
    unsigned int t = blockIdx.x, vi = blockIdx.y;
    unsigned int nh = args.u[1], nkv = args.u[2], d = args.u[3], dr = args.u[4], cap = args.u[5];
    unsigned int past = args.a[10] ? *(const unsigned int*)args.a[10] : args.u[6];
    unsigned int pos = past + t;
    float tau = 1.0f;
    if (args.u[7]) tau = 1.0f + args.f[3] * logf(fmaxf(1.0f, (float)(pos + 1) / (float)args.u[8]));
    unsigned int i = threadIdx.x;
    if (vi < nh) {
        unsigned long long o = ((unsigned long long)t * nh + vi) * d;
        const unsigned qbf = args.u[12];
        const unsigned long long qi = (unsigned long long)t * (args.u[13] ? args.u[13] : nh * d) + vi * d;
        const unsigned long long ri = (unsigned long long)t * (args.u[13] ? args.u[13] : nh * dr) + vi * dr;
        float v = i < d ? ink2_ld(args.a[0], qi + i, qbf) * CPF(4)[vi * d + i] : 0.0f;
        float n = sqrtf(block_sum(v * v, red));
        float k = fminf(1.0f, args.f[0] / (n + 1e-6f));
        if (i < d) PF(6)[o + i] = v * k * tau;
        unsigned long long orr = ((unsigned long long)t * nh + vi) * dr;
        float r = i < dr ? ink2_ld(args.a[3], ri + i, qbf) * CPF(5)[vi * dr + i] : 0.0f;
        float nr = sqrtf(block_sum(r * r, red));
        float kr = fminf(1.0f, args.f[1] / (nr + 1e-6f));
        if (i < dr) PF(7)[orr + i] = r * kr * tau;
    } else if (vi < nh + nkv) {
        unsigned int h = vi - nh;
        float v = i < d ? CPF(1)[((unsigned long long)t * nkv + h) * d + i] : 0.0f;
        float n = sqrtf(block_sum(v * v, red));
        float k = fminf(1.0f, args.f[0] / (n + 1e-6f));
        unsigned long long row = (unsigned long long)h * cap + (args.u[9] ? pos % args.u[9] : pos);
        if (args.u[11]) {
            unsigned sb;
            unsigned char b = ink2_kv_enc(v * k, args.u[11], sb);
            ((unsigned char*)args.a[8])[row * d + i] = b;
            if ((i & 31u) == 0u) ((unsigned char*)args.a[11])[row * 4 + i / 32] = (unsigned char)sb;
        } else if (i < d) PB(8)[row * d + i] = __float2bfloat16(v * k);
    } else {
        unsigned int h = vi - nh - nkv;
        float v = i < d ? CPF(2)[((unsigned long long)t * nkv + h) * d + i] : 0.0f;
        float ss = block_sum(v * v, red);
        float inv = rsqrtf(ss / (float)d + args.f[2]);
        unsigned long long row = (unsigned long long)h * cap + (args.u[9] ? pos % args.u[9] : pos);
        if (args.u[11]) {
            unsigned sb;
            unsigned char b = ink2_kv_enc(v * inv, args.u[11], sb);
            ((unsigned char*)args.a[9])[row * d + i] = b;
            if ((i & 31u) == 0u) ((unsigned char*)args.a[11])[((unsigned long long)nkv * cap + row) * 4 + i / 32] = (unsigned char)sb;
        } else if (i < d) PB(9)[row * d + i] = __float2bfloat16(v * inv);
    }
}

// Внимание: блок на (токен, голову, часть ключей), 4 варпа делят ключи части; лейн держит
// 4 из 128 измерений. Логит = q·k·scale + bias(dist), bias = Σ_c r[c]·proj[c, dist] при
// dist < extent, иначе 0; лишний логит синка с нулевым value; окно window (0 — нет).
// splits ≤ 1: результат сразу ·sigmoid(gate) в out. Иначе каждая часть пишет (m, l, acc[128])
// в a8, сборку делает ink2_attn_merge. Длина части считается от позиции на карте — число
// частей фиксировано (от ёмкости), поэтому шаг годится для CUDA-графа.
// a0 q f32 [T,nh,d], a1 r f32 [T,nh,dr], a2 proj f32 [dr,extent], a3 sink f32 [T,nh],
// a4 kcache, a5 vcache bf16 [nkv,cap,d], a6 gate f32 [T,nh·d], a7 out f32 [T,nh·d], a8 части f32 [T,nh,S,130];
// u0 T, u1 nh, u2 nkv, u3 d (=128), u4 dr (≤32), u5 cap, u6 past (или a10 — u32 на карте), u7 window, u8 extent,
// u9 splits, u10 кольцо KV (0 — линейно), u11 формат KV (a11 — масштабы); sink и gate — F32 либо
// BF16 (u12), шаг строки у обоих u13 (0 — nh и nh·d); f0 scale.
extern "C" __global__ void ink2_attn(Ink2Args args) {
    __shared__ float sm_m[4], sm_l[4];
    __shared__ float sm_acc[4][128];
    unsigned int t = blockIdx.x, h = blockIdx.y, sp = blockIdx.z;
    unsigned int nh = args.u[1], nkv = args.u[2], d = 128, dr = args.u[4], cap = args.u[5];
    unsigned int window = args.u[7], extent = args.u[8], S = args.u[9] ? args.u[9] : 1;
    unsigned int past = args.a[10] ? *(const unsigned int*)args.a[10] : args.u[6];
    unsigned int pos = past + t;
    unsigned int kvh = h / (nh / nkv);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    unsigned long long qo = ((unsigned long long)t * nh + h) * d;
    float q[4];
    for (int j = 0; j < 4; ++j) q[j] = CPF(0)[qo + lane * 4 + j] * args.f[0];
    float rr = lane < (int)dr ? CPF(1)[((unsigned long long)t * nh + h) * dr + lane] : 0.0f;
    const float* proj = CPF(2);
    const __nv_bfloat16* kc = CPB(4) + (unsigned long long)kvh * cap * d;
    const __nv_bfloat16* vc = CPB(5) + (unsigned long long)kvh * cap * d;
    const unsigned fmt = args.u[11];
    const unsigned char* k8 = (const unsigned char*)args.a[4] + (unsigned long long)kvh * cap * d;
    const unsigned char* v8 = (const unsigned char*)args.a[5] + (unsigned long long)kvh * cap * d;
    const unsigned char* ks8 = fmt ? (const unsigned char*)args.a[11] + (unsigned long long)kvh * cap * 4 : nullptr;
    const unsigned char* vs8 = fmt ? ks8 + (unsigned long long)nkv * cap * 4 : nullptr;
    unsigned int lo0 = (window && pos + 1 > window) ? pos + 1 - window : 0;
    unsigned int len = pos + 1 - lo0, chunk = (len + S - 1) / S;
    unsigned int lo = lo0 + sp * chunk, hi = min(lo + chunk, pos + 1);
    float m = -INFINITY, l = 0.0f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    // Сырые K/V (и срез proj) следующего ключа варпа грузятся до счёта текущего: задержка памяти
    // перекрывается, порядок онлайн-softmax тот же.
    uint2 kw = make_uint2(0u, 0u), vw = kw;
    unsigned ksb = 0, vsb = 0;
    float pj = 0.0f;
    auto fetch = [&](unsigned j) {
        unsigned long long jr = args.u[10] ? j % args.u[10] : j;
        if (fmt) {
            kw.x = *(const unsigned*)(k8 + jr * d + lane * 4);
            vw.x = *(const unsigned*)(v8 + jr * d + lane * 4);
            ksb = ks8[jr * 4 + (lane >> 3)];
            vsb = vs8[jr * 4 + (lane >> 3)];
        } else {
            kw = *(const uint2*)(kc + jr * d + lane * 4);
            vw = *(const uint2*)(vc + jr * d + lane * 4);
        }
        unsigned dist = pos - j;
        pj = (dist < extent && lane < (int)dr) ? proj[(unsigned long long)lane * extent + dist] : 0.0f;
    };
    if (lo + wid < hi) fetch(lo + wid);
    for (unsigned int j = lo + wid; j < hi; j += 4) {
        const uint2 kc0 = kw, vc0 = vw;
        const unsigned ks0 = ksb, vs0 = vsb;
        const float pj0 = pj;
        if (j + 4 < hi) fetch(j + 4);
        float kf[4], vf[4];
        if (fmt) {
            ink2_kv_dec4(kc0.x, fmt, ink2_e8m0(ks0), kf);
        } else {
            float2 k01 = __bfloat1622float2(*(const __nv_bfloat162*)&kc0.x);
            float2 k23 = __bfloat1622float2(*(const __nv_bfloat162*)&kc0.y);
            kf[0] = k01.x; kf[1] = k01.y; kf[2] = k23.x; kf[3] = k23.y;
        }
        float part = q[0] * kf[0] + q[1] * kf[1] + q[2] * kf[2] + q[3] * kf[3];
        unsigned int dist = pos - j;
        if (dist < extent && lane < (int)dr) part += rr * pj0;
        float s = warp_sum(part);
        float mn = fmaxf(m, s);
        float corr = expf(m - mn), p = expf(s - mn);
        if (fmt) {
            ink2_kv_dec4(vc0.x, fmt, ink2_e8m0(vs0), vf);
        } else {
            float2 v01 = __bfloat1622float2(*(const __nv_bfloat162*)&vc0.x);
            float2 v23 = __bfloat1622float2(*(const __nv_bfloat162*)&vc0.y);
            vf[0] = v01.x; vf[1] = v01.y; vf[2] = v23.x; vf[3] = v23.y;
        }
        acc[0] = acc[0] * corr + p * vf[0];
        acc[1] = acc[1] * corr + p * vf[1];
        acc[2] = acc[2] * corr + p * vf[2];
        acc[3] = acc[3] * corr + p * vf[3];
        l = l * corr + p;
        m = mn;
    }
    if (lane == 0) { sm_m[wid] = m; sm_l[wid] = l; }
    for (int j = 0; j < 4; ++j) sm_acc[wid][lane * 4 + j] = acc[j];
    __syncthreads();
    unsigned int i = threadIdx.x;  // 128 потоков — по измерению
    if (S > 1) {
        float M = -INFINITY;
        for (int w = 0; w < 4; ++w) if (sm_l[w] > 0.0f) M = fmaxf(M, sm_m[w]);
        float L = 0.0f, o = 0.0f;
        for (int w = 0; w < 4; ++w) if (sm_l[w] > 0.0f) { float c = expf(sm_m[w] - M); L += sm_l[w] * c; o += sm_acc[w][i] * c; }
        float* pp = PF(8) + (((unsigned long long)t * nh + h) * S + sp) * 130;
        pp[2 + i] = o;
        if (i == 0) { pp[0] = M; pp[1] = L; }
        return;
    }
    float sink = ink2_ld(args.a[3], (unsigned long long)t * (args.u[13] ? args.u[13] : nh) + h, args.u[12]);
    float M = sink;
    for (int w = 0; w < 4; ++w) if (sm_l[w] > 0.0f) M = fmaxf(M, sm_m[w]);
    float L = expf(sink - M);
    for (int w = 0; w < 4; ++w) if (sm_l[w] > 0.0f) L += sm_l[w] * expf(sm_m[w] - M);
    float o = 0.0f;
    for (int w = 0; w < 4; ++w) if (sm_l[w] > 0.0f) o += sm_acc[w][i] * expf(sm_m[w] - M);
    o /= L;
    float g = ink2_ld(args.a[6], (unsigned long long)t * (args.u[13] ? args.u[13] : nh * d) + h * d + i, args.u[12]);
    PF(7)[(unsigned long long)t * nh * d + h * d + i] = o * sigm(g);
}

// Префилл-внимание на тензорных ядрах (flash, mma.sync m16n8k16 bf16 → F32): та же семантика,
// что у ink2_attn (bias по расстоянию, окно, кольцо KV, синк, гейт). Блок — 64 запроса одной
// головы (4 варпа × 16 строк), ключи плитками по 32, часть z из S по ключам (blockIdx.z).
// Точность как у F32-пути: q и p расщеплены на bf16 hi + lo (два mma), k/v — bf16 кэша как
// есть; online softmax и накопление — F32 (основание 2: q и r заранее ×log2e).
// Bias плитки — маленький GEMM [16 строк × 16 каналов] × [16 × 48 расстояний] на варп
// (r и срез proj расщеплены на hi + lo, три mma), таблица в shared варпа, сборка в логиты по
// dist = pos_q − pos_k. Плитки дальше extent bias не считают (большинство на длинном контексте).
// a0 q f32 [T,nh,128], a1 r f32 [T,nh,16], a2 proj f32 [16,extent], a3 sink f32 [T,nh],
// a4 kcache, a5 vcache bf16 [nkv,cap,128], a6 gate f32 [T,nh·128], a7 out f32 [T,nh·128],
// a8 части f32 [T,nh,S,130] (при S > 1; сборка — ink2_attn_merge с u9 = S), a10 позиция u32 | u6;
// u0 T, u1 nh, u2 nkv, u5 cap, u7 window (0 — нет), u8 extent, u9 S (≤ 1 — сразу в out),
// u10 кольцо KV (0 — линейно), u11 формат KV (a11 — масштабы; плитка деквантуется в BF16 shared
// точно); sink и gate — F32 либо BF16 (u12), шаг строки u13 (0 — nh и nh·128); f0 scale. grid = (ceil(T/64), nh, S), block = 128.
#define FA_BM 64
#define FA_BN 32
#define FA_LD 136  // bf16 на строку K/V в shared (128 + 8: без конфликтов банков у ldmatrix)
#define FA_PD 96   // строк среза proj на блок (расстояния плитки для 64 строк)
#define FA_PLD 24  // bf16 на строку среза (16 каналов + паддинг)
#define FA_BLD 52  // float на строку таблицы bias варпа (48 расстояний + паддинг)
#define FA_L2E 1.4426950408889634f
#define FA_LN2 0.6931471805599453f

__device__ __forceinline__ void ink2_mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Пара (x0 — младший адрес, x1) → bf16x2 hi и остаток lo (x ≈ hi + lo с точностью ~2^-17).
__device__ __forceinline__ void ink2_split2(float x0, float x1, unsigned& hi, unsigned& lo) {
    __nv_bfloat162 h = __floats2bfloat162_rn(x0, x1);
    float2 hf = __bfloat1622float2(h);
    __nv_bfloat162 l = __floats2bfloat162_rn(x0 - hf.x, x1 - hf.y);
    hi = *(unsigned*)&h;
    lo = *(unsigned*)&l;
}

__device__ __forceinline__ unsigned ink2_smem(const void* p) {
    unsigned r;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p));
    return r;
}

__device__ __forceinline__ void ink2_cp16(unsigned dst, const void* src, bool ok) {
    // ok = false — заполнить нулями (src-size 0).
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(ok ? 16 : 0));
}

__device__ __forceinline__ void ink2_ldm4(unsigned* r, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void ink2_ldm4t(unsigned* r, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

extern "C" __global__ void __launch_bounds__(128, 2) ink2_attn_fa(Ink2Args args) {
    __shared__ __align__(16) __nv_bfloat16 ks[FA_BN * FA_LD];
    __shared__ __align__(16) __nv_bfloat16 vs[FA_BN * FA_LD];
    __shared__ __align__(16) __nv_bfloat16 prh[FA_PD * FA_PLD];
    __shared__ __align__(16) __nv_bfloat16 prl[FA_PD * FA_PLD];
    __shared__ __align__(16) float btab[4][16 * FA_BLD];
    const unsigned T = args.u[0], nh = args.u[1], nkv = args.u[2], cap = args.u[5];
    const unsigned window = args.u[7], extent = args.u[8], S = args.u[9] > 1 ? args.u[9] : 1u, ring = args.u[10];
    const unsigned h = blockIdx.y, kvh = h / (nh / nkv), z = blockIdx.z;
    const unsigned past = args.a[10] ? *(const unsigned int*)args.a[10] : args.u[6];
    const unsigned t0 = blockIdx.x * FA_BM;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tg = lane & 3;
    const unsigned wrow = t0 + w * 16;           // первая строка варпа
    const unsigned ta = wrow + g, tb = ta + 8;   // строки потока
    const bool va = ta < T, vb = tb < T;
    const unsigned pa = past + ta, pb = past + tb;

    // Ключи блока: [kbeg, kend), плитки [tile0, tile1) делятся на S частей.
    const unsigned tlast = min(t0 + FA_BM, T) - 1u;
    const unsigned kend = past + tlast + 1u;
    const unsigned kbeg = (window && past + t0 + 1u > window) ? past + t0 + 1u - window : 0u;
    const unsigned tile0 = kbeg / FA_BN, tile1 = (kend + FA_BN - 1u) / FA_BN;
    const unsigned per = (tile1 - tile0 + S - 1u) / S;
    const unsigned my0 = tile0 + z * per, my1 = min(my0 + per, tile1);

    // q (×scale·log2e) и r (×log2e) — A-фрагменты hi/lo в регистрах.
    unsigned qh[8][4], ql[8][4], rh[4], rl[4];
    {
        const float qs = args.f[0] * FA_L2E;
        const float* q = CPF(0);
        #pragma unroll
        for (int kq = 0; kq < 8; ++kq) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                unsigned row = (r & 1) ? tb : ta;
                unsigned col = kq * 16 + 2 * tg + ((r & 2) ? 8 : 0);
                float2 x = make_float2(0.0f, 0.0f);
                if (row < T) x = *(const float2*)(q + ((unsigned long long)row * nh + h) * 128 + col);
                ink2_split2(x.x * qs, x.y * qs, qh[kq][r], ql[kq][r]);
            }
        }
        const float* rr = CPF(1);
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            unsigned row = (r & 1) ? tb : ta;
            unsigned col = 2 * tg + ((r & 2) ? 8 : 0);
            float2 x = make_float2(0.0f, 0.0f);
            if (row < T) x = *(const float2*)(rr + ((unsigned long long)row * nh + h) * 16 + col);
            ink2_split2(x.x * FA_L2E, x.y * FA_L2E, rh[r], rl[r]);
        }
    }
    float o[16][4];
    #pragma unroll
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
    const __nv_bfloat16* kc = CPB(4) + (unsigned long long)kvh * cap * 128;
    const __nv_bfloat16* vc = CPB(5) + (unsigned long long)kvh * cap * 128;
    const unsigned fmt = args.u[11];
    const unsigned char* k8 = (const unsigned char*)args.a[4] + (unsigned long long)kvh * cap * 128;
    const unsigned char* v8 = (const unsigned char*)args.a[5] + (unsigned long long)kvh * cap * 128;
    const unsigned char* ks8 = fmt ? (const unsigned char*)args.a[11] + (unsigned long long)kvh * cap * 4 : nullptr;
    const unsigned char* vs8 = fmt ? ks8 + (unsigned long long)nkv * cap * 4 : nullptr;
    const float* proj = CPF(2);
    float* bt = btab[w];
    // Последняя строка варпа (для пропуска плиток) — в пределах T.
    const unsigned wlast = wrow < T ? min(wrow + 15u, T - 1u) : 0u;

    for (unsigned kt = my0; kt < my1; ++kt) {
        const unsigned j0 = kt * FA_BN;
        __syncthreads();  // прошлая плитка дочитана
        if (fmt) {
            // Квант-KV: 16 байт строки → 16 BF16 в shared (деквант точен).
            for (int i = tid; i < FA_BN * 8; i += 128) {
                int r = i >> 3, c = (i & 7) * 16;
                unsigned j = j0 + r;
                uint4 kw = make_uint4(0u, 0u, 0u, 0u), vw = kw;
                float kscl = 0.0f, vscl = 0.0f;
                if (j < kend) {
                    unsigned long long jr = ring ? j % ring : j;
                    kw = *(const uint4*)(k8 + jr * 128 + c);
                    vw = *(const uint4*)(v8 + jr * 128 + c);
                    kscl = ink2_e8m0(ks8[jr * 4 + (c >> 5)]);
                    vscl = ink2_e8m0(vs8[jr * 4 + (c >> 5)]);
                }
                const unsigned kws[4] = {kw.x, kw.y, kw.z, kw.w}, vws[4] = {vw.x, vw.y, vw.z, vw.w};
                unsigned kb[8], vb[8];
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    float y[4];
                    ink2_kv_dec4(kws[u], fmt, kscl, y);
                    __nv_bfloat162 a0 = __floats2bfloat162_rn(y[0], y[1]), a1 = __floats2bfloat162_rn(y[2], y[3]);
                    kb[2 * u] = *(unsigned*)&a0;
                    kb[2 * u + 1] = *(unsigned*)&a1;
                    ink2_kv_dec4(vws[u], fmt, vscl, y);
                    a0 = __floats2bfloat162_rn(y[0], y[1]);
                    a1 = __floats2bfloat162_rn(y[2], y[3]);
                    vb[2 * u] = *(unsigned*)&a0;
                    vb[2 * u + 1] = *(unsigned*)&a1;
                }
                uint4* kd = (uint4*)(ks + r * FA_LD + c);
                uint4* vd = (uint4*)(vs + r * FA_LD + c);
                kd[0] = make_uint4(kb[0], kb[1], kb[2], kb[3]);
                kd[1] = make_uint4(kb[4], kb[5], kb[6], kb[7]);
                vd[0] = make_uint4(vb[0], vb[1], vb[2], vb[3]);
                vd[1] = make_uint4(vb[4], vb[5], vb[6], vb[7]);
            }
        } else {
            for (int i = tid; i < FA_BN * 16; i += 128) {
                int r = i >> 4, c = (i & 15) * 8;
                unsigned j = j0 + r;
                bool ok = j < kend;
                unsigned long long jr = ok ? (ring ? j % ring : j) : 0ull;
                ink2_cp16(ink2_smem(ks + r * FA_LD + c), kc + jr * 128 + c, ok);
                ink2_cp16(ink2_smem(vs + r * FA_LD + c), vc + jr * 128 + c, ok);
            }
        }
        asm volatile("cp.async.commit_group;\n");
        // Ближняя плитка (у какой-то строки dist < extent): срез proj по расстояниям
        // base0 .. base0+95, base0 = dist(строка t0, ключ j0+31).
        const int base0 = (int)(past + t0) - (int)(j0 + FA_BN - 1);
        const bool near_blk = base0 < (int)extent;
        if (near_blk) {
            for (int i = tid; i < FA_PD * 16; i += 128) {
                int ch = i / FA_PD, di = i - ch * FA_PD, dist = base0 + di;
                float v = (dist >= 0 && dist < (int)extent) ? proj[(unsigned long long)ch * extent + dist] : 0.0f;
                __nv_bfloat16 hv = __float2bfloat16_rn(v);
                prh[di * FA_PLD + ch] = hv;
                prl[di * FA_PLD + ch] = __float2bfloat16_rn(v - __bfloat162float(hv));
            }
        }
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        // Варп без видимых ключей в плитке — пропуск (синхронизации общие).
        if (wrow >= T || j0 > past + wlast) continue;
        if (window && (int)(past + wrow) - (int)(j0 + FA_BN - 1) >= (int)window) continue;

        // S = Q·Kᵀ (q = hi + lo).
        float s[4][4];
        #pragma unroll
        for (int n = 0; n < 4; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
        #pragma unroll
        for (int kp = 0; kp < 4; ++kp) {
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                unsigned kb[4];
                ink2_ldm4(kb, ink2_smem(ks + (n * 8 + (lane & 7)) * FA_LD + kp * 32 + (lane >> 3) * 8));
                ink2_mma(s[n], ql[2 * kp], kb[0], kb[1]);
                ink2_mma(s[n], qh[2 * kp], kb[0], kb[1]);
                ink2_mma(s[n], ql[2 * kp + 1], kb[2], kb[3]);
                ink2_mma(s[n], qh[2 * kp + 1], kb[2], kb[3]);
            }
        }

        // Bias: таблица варпа bt[r][idx] = r_row · proj[:, base_w + idx], idx = r + 31 − c.
        const int base_w = base0 + w * 16;
        if (base_w < (int)extent) {
            #pragma unroll
            for (int nt = 0; nt < 6; ++nt) {
                float c4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const int pr = (w * 16 + nt * 8 + g) * FA_PLD + 2 * tg;
                unsigned bh0 = *(const unsigned*)(prh + pr), bh1 = *(const unsigned*)(prh + pr + 8);
                unsigned bl0 = *(const unsigned*)(prl + pr), bl1 = *(const unsigned*)(prl + pr + 8);
                ink2_mma(c4, rl, bh0, bh1);
                ink2_mma(c4, rh, bl0, bl1);
                ink2_mma(c4, rh, bh0, bh1);
                *(float2*)(bt + g * FA_BLD + nt * 8 + 2 * tg) = make_float2(c4[0], c4[1]);
                *(float2*)(bt + (g + 8) * FA_BLD + nt * 8 + 2 * tg) = make_float2(c4[2], c4[3]);
            }
            __syncwarp();
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                #pragma unroll
                for (int e = 0; e < 2; ++e) {
                    int c = n * 8 + 2 * tg + e;
                    s[n][e] += bt[g * FA_BLD + g + (FA_BN - 1) - c];
                    s[n][2 + e] += bt[(g + 8) * FA_BLD + g + 8 + (FA_BN - 1) - c];
                }
            }
            __syncwarp();
        }

        // Маска: каузальная, окно, строки за T. Плитка целиком видна всем строкам варпа — без неё.
        const bool full = (wrow + 15u < T) && (j0 + FA_BN - 1u <= past + wrow) &&
                          (!window || (int)(past + wrow + 15u) - (int)j0 < (int)window);
        if (!full) {
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                #pragma unroll
                for (int e = 0; e < 2; ++e) {
                    unsigned j = j0 + n * 8 + 2 * tg + e;
                    if (!va || j > pa || (window && pa - j >= window)) s[n][e] = -INFINITY;
                    if (!vb || j > pb || (window && pb - j >= window)) s[n][2 + e] = -INFINITY;
                }
            }
        }

        // Online softmax (основание 2), строка — 4 лейна.
        float mxa = -INFINITY, mxb = -INFINITY;
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            mxa = fmaxf(mxa, fmaxf(s[n][0], s[n][1]));
            mxb = fmaxf(mxb, fmaxf(s[n][2], s[n][3]));
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float mna = fmaxf(ma, mxa), mnb = fmaxf(mb, mxb);
        const float ca = mna == -INFINITY ? 1.0f : exp2f(ma - mna);
        const float cb = mnb == -INFINITY ? 1.0f : exp2f(mb - mnb);
        float sa = 0.0f, sb = 0.0f;
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                float p0 = mna == -INFINITY ? 0.0f : exp2f(s[n][e] - mna);
                float p1 = mnb == -INFINITY ? 0.0f : exp2f(s[n][2 + e] - mnb);
                s[n][e] = p0;
                s[n][2 + e] = p1;
                sa += p0;
                sb += p1;
            }
        }
        sa += __shfl_xor_sync(0xffffffffu, sa, 1);
        sa += __shfl_xor_sync(0xffffffffu, sa, 2);
        sb += __shfl_xor_sync(0xffffffffu, sb, 1);
        sb += __shfl_xor_sync(0xffffffffu, sb, 2);
        la = la * ca + sa;
        lb = lb * cb + sb;
        ma = mna;
        mb = mnb;
        // O = O·c + P·V (p = hi + lo), V через ldmatrix.trans. P·V плитки — в свежем аккумуляторе,
        // в O — обычным FMA F32: аккумулятор mma складывает с усечением, и за тысячи плиток
        // (200K ключей) ошибка копилась бы линейно (~6·10⁻⁴ отн.); так — как у F32-пути.
        unsigned ph[2][4], pl[2][4];
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            ink2_split2(s[2 * kk][0], s[2 * kk][1], ph[kk][0], pl[kk][0]);
            ink2_split2(s[2 * kk][2], s[2 * kk][3], ph[kk][1], pl[kk][1]);
            ink2_split2(s[2 * kk + 1][0], s[2 * kk + 1][1], ph[kk][2], pl[kk][2]);
            ink2_split2(s[2 * kk + 1][2], s[2 * kk + 1][3], ph[kk][3], pl[kk][3]);
        }
        #pragma unroll
        for (int np = 0; np < 8; ++np) {
            float pv[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned vb4[4];
                ink2_ldm4t(vb4, ink2_smem(vs + (kk * 16 + (lane & 15)) * FA_LD + np * 16 + (lane >> 4) * 8));
                ink2_mma(pv[0], pl[kk], vb4[0], vb4[1]);
                ink2_mma(pv[0], ph[kk], vb4[0], vb4[1]);
                ink2_mma(pv[1], pl[kk], vb4[2], vb4[3]);
                ink2_mma(pv[1], ph[kk], vb4[2], vb4[3]);
            }
            #pragma unroll
            for (int x = 0; x < 2; ++x) {
                o[2 * np + x][0] = fmaf(o[2 * np + x][0], ca, pv[x][0]);
                o[2 * np + x][1] = fmaf(o[2 * np + x][1], ca, pv[x][1]);
                o[2 * np + x][2] = fmaf(o[2 * np + x][2], cb, pv[x][2]);
                o[2 * np + x][3] = fmaf(o[2 * np + x][3], cb, pv[x][3]);
            }
        }
    }

    // Выход: S = 1 — синк, нормировка, гейт → out; иначе часть (m — в натуральных единицах).
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        const unsigned t = half ? tb : ta;
        if (t >= T) continue;
        const float m = half ? mb : ma, l = half ? lb : la;
        const int oi = half ? 2 : 0;
        if (S > 1) {
            float* d = PF(8) + (((unsigned long long)t * nh + h) * S + z) * 130;
            #pragma unroll
            for (int nd = 0; nd < 16; ++nd)
                *(float2*)(d + 2 + nd * 8 + 2 * tg) = make_float2(o[nd][oi], o[nd][oi + 1]);
            if (tg == 0) { d[0] = l > 0.0f ? m * FA_LN2 : -INFINITY; d[1] = l; }
        } else {
            const float sk = ink2_ld(args.a[3], (unsigned long long)t * (args.u[13] ? args.u[13] : nh) + h, args.u[12]) * FA_L2E;
            const float M = l > 0.0f ? fmaxf(m, sk) : sk;
            const float c = l > 0.0f ? exp2f(m - M) : 0.0f;
            const float inv = c / (l * c + exp2f(sk - M));
            const unsigned long long gt = (unsigned long long)t * (args.u[13] ? args.u[13] : nh * 128) + h * 128;
            float* out = PF(7) + (unsigned long long)t * nh * 128 + h * 128;
            #pragma unroll
            for (int nd = 0; nd < 16; ++nd) {
                int col = nd * 8 + 2 * tg;
                float2 gg = make_float2(ink2_ld(args.a[6], gt + col, args.u[12]), ink2_ld(args.a[6], gt + col + 1, args.u[12]));
                *(float2*)(out + col) = make_float2(o[nd][oi] * inv * sigm(gg.x), o[nd][oi + 1] * inv * sigm(gg.y));
            }
        }
    }
}

// Сборка частей ink2_attn: блок на (токен, голову), 128 потоков; синк и гейт здесь.
extern "C" __global__ void ink2_attn_merge(Ink2Args args) {
    unsigned int t = blockIdx.x, h = blockIdx.y, nh = args.u[1], S = args.u[9], d = 128, i = threadIdx.x;
    const float* pp = CPF(8) + ((unsigned long long)t * nh + h) * S * 130;
    float sink = ink2_ld(args.a[3], (unsigned long long)t * (args.u[13] ? args.u[13] : nh) + h, args.u[12]);
    float M = sink;
    for (unsigned int s = 0; s < S; ++s) if (pp[s * 130 + 1] > 0.0f) M = fmaxf(M, pp[s * 130]);
    float L = expf(sink - M), o = 0.0f;
    for (unsigned int s = 0; s < S; ++s) {
        float ls = pp[s * 130 + 1];
        if (ls > 0.0f) { float c = expf(pp[s * 130] - M); L += ls * c; o += pp[s * 130 + 2 + i] * c; }
    }
    o /= L;
    float g = ink2_ld(args.a[6], (unsigned long long)t * (args.u[13] ? args.u[13] : nh * d) + h * d + i, args.u[12]);
    PF(7)[(unsigned long long)t * nh * d + h * d + i] = o * sigm(g);
}

// Активация эксперта: g = α·gu[:I], u = α·gu[I:]; out = tanh(g/cg)·cg·σ(g)·tanh(u/cu)·cu (BF16).
// α — a1[r] либо, при a3 ≠ 0, a1[idx[r]] (таблица множителей по экспертам, idx = a3 u32).
// a0 gu bf16 [R,2I], a1 alpha f32, a2 out bf16 [R,I], a3 idx u32 [R]|0; u0 R, u1 I; f0 cg, f1 cu.
extern "C" __global__ void ink2_act(Ink2Args args) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y, I = args.u[1];
    if (i >= I) return;
    const unsigned int* idx = (const unsigned int*)args.a[3];
    float a = CPF(1)[idx ? idx[r] : r];
    const __nv_bfloat16* gu = CPB(0) + (unsigned long long)r * 2 * I;
    float g = __bfloat162float(gu[i]) * a, u = __bfloat162float(gu[I + i]) * a;
    float cg = args.f[0], cu = args.f[1];
    float v = tanhf(g / cg) * cg * sigm(g) * tanhf(u / cu) * cu;
    PB(2)[(unsigned long long)r * I + i] = __float2bfloat16(v);
}

// Маршрут: s = σ(logit); выбор top-k по s + bias; вес = s/(Σ s + 1e-20)·scale.
// a0 logits f32 [T,E], a1 bias f32 [E], a2 idx u32 [T,k], a3 w f32 [T,k], a4 множитель веса по эксперту f32 [E]|0;
// a5 f32 [u3]|0 — обнулить (накопитель MoE декода: без отдельного memset в графе);
// u0 T, u1 E (≤1024), u2 k; f0 scale.
extern "C" __global__ void ink2_route(Ink2Args args) {
    // Один варп на токен: у лейна E/32 ключей в регистрах, k раундов argmax по варпу.
    unsigned int t = blockIdx.x, E = args.u[1], K = args.u[2];
    int lane = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;
    if (args.a[5] && t == 0)
        for (unsigned int i = lane; i < args.u[3]; i += 32) PF(5)[i] = 0.0f;
    const float* lg = CPF(0) + (unsigned long long)t * E;
    if (E <= 128 && K <= 32) {
        // E ≤ 128: ключи и σ — в регистрах (развёрнуто, без local memory), σ выбранного берётся
        // шаффлом у лейна-владельца, веса пишут лейны k < K. Значения и порядок суммы те же.
        float key[4], sv[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int e = j * 32 + lane;
            sv[j] = e < E ? sigm(lg[e]) : 0.0f;
            key[j] = e < E ? sv[j] + CPF(1)[e] : -INFINITY;
        }
        int my_i = 0;
        float my_s = 0.0f, sum = 0.0f;
        for (unsigned int k = 0; k < K; ++k) {
            float best = -INFINITY;
            int idx = 0x7fffffff;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (key[j] > best) { best = key[j]; idx = j * 32 + lane; }
            }
            for (int o = 16; o > 0; o >>= 1) {
                float ob = __shfl_xor_sync(0xffffffffu, best, o);
                int oi = __shfl_xor_sync(0xffffffffu, idx, o);
                if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
            }
            const int jw = idx >> 5;
            float own = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) if (j == jw) { own = sv[j]; if ((idx & 31) == lane) key[j] = -INFINITY; }
            const float s = __shfl_sync(0xffffffffu, own, idx & 31);
            if ((unsigned)lane == k) { my_i = idx; my_s = s; }
            sum += s;
        }
        if ((unsigned)lane < K) {
            ((unsigned int*)args.a[2])[t * K + lane] = (unsigned int)my_i;
            float m = args.a[4] ? CPF(4)[my_i] : 1.0f;
            PF(3)[t * K + lane] = my_s / (sum + 1e-20f) * args.f[0] * m;
        }
        return;
    }
    float key[32];
    unsigned int per = (E + 31) / 32;
    for (unsigned int j = 0; j < per; ++j) {
        unsigned int e = j * 32 + lane;
        key[j] = e < E ? sigm(lg[e]) + CPF(1)[e] : -INFINITY;
    }
    int sel_i[16];
    float sel_s[16], sum = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
        float best = -INFINITY;
        int idx = 0x7fffffff;
        for (unsigned int j = 0; j < per; ++j) {
            int e = (int)(j * 32 + lane);
            if (key[j] > best) { best = key[j]; idx = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            float ob = __shfl_xor_sync(0xffffffffu, best, o);
            int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
        }
        if ((unsigned int)(idx & 31) == (unsigned int)lane) key[idx / 32] = -INFINITY;
        sel_i[k] = idx;
        sel_s[k] = sigm(lg[idx]);
        sum += sel_s[k];
    }
    if (lane == 0) {
        for (unsigned int k = 0; k < K; ++k) {
            ((unsigned int*)args.a[2])[t * K + k] = (unsigned int)sel_i[k];
            float m = args.a[4] ? CPF(4)[sel_i[k]] : 1.0f;
            PF(3)[t * K + k] = sel_s[k] / (sum + 1e-20f) * args.f[0] * m;
        }
    }
}

// Сборка MoE: out[t] = Σ_k w[t·K+k]·y[rowmap[t·K+k]] (F32).
// a0 y bf16 [P,H], a1 w f32 [T·K], a2 rowmap u32 [T·K], a3 out f32 [T,H]; u0 T, u1 H, u2 K.
extern "C" __global__ void ink2_combine(Ink2Args args) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y, H = args.u[1], K = args.u[2];
    if (i >= H) return;
    const unsigned int* rm = (const unsigned int*)args.a[2];
    float acc = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
        unsigned int p = t * K + k;
        acc += CPF(1)[p] * __bfloat162float(CPB(0)[(unsigned long long)rm[p] * H + i]);
    }
    PF(3)[(unsigned long long)t * H + i] = acc;
}

// Логиты: softcap и срез до несдвинутого словаря. a0 in (bf16|f32) [T,V], a1 out f32 [T,Vu];
// u0 T, u1 V, u2 Vu, u3 in_bf16; f0 cap (0 — без).
extern "C" __global__ void ink2_logits(Ink2Args args) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y, V = args.u[1], Vu = args.u[2];
    if (i >= Vu) return;
    unsigned long long o = (unsigned long long)t * V + i;
    float v = args.u[3] ? __bfloat162float(CPB(0)[o]) : CPF(0)[o];
    float c = args.f[0];
    if (c > 0.0f) v = tanhf(v / c) * c;
    PF(1)[(unsigned long long)t * Vu + i] = v;
}

// Контроллер, шаг 1: d[t, r] = Σ_j down[r, j]·z_t[j], z_t = [x_t·inv_x; m_t·inv_m] (RMS без веса, eps 1e-6).
// Блок на (токен, строку ранга).
// a0 x f32 [T,H], a1 m f32 [T,H], a2 down f32 [R, 2H], a3 d f32 [T,R]; u0 T, u1 H, u2 R.
// Несколько контроллеров одним запуском (blockIdx.z = j): шаги между ними (в элементах) — x u3,
// m u4, down u5, d u6.
extern "C" __global__ void ink2_ctrl_down(Ink2Args args) {
    __shared__ float red[32];
    unsigned int t = blockIdx.x, r = blockIdx.y, H = args.u[1], R = args.u[2];
    const unsigned long long z = blockIdx.z;
    const float* x = CPF(0) + z * args.u[3] + (unsigned long long)t * H;
    const float* m = CPF(1) + z * args.u[4] + (unsigned long long)t * H;
    const float* w = CPF(2) + z * args.u[5] + (unsigned long long)r * 2 * H;
    float sx = 0.0f, sm = 0.0f, ax = 0.0f, am = 0.0f;
    for (unsigned int i = threadIdx.x; i < H; i += blockDim.x) {
        float xv = x[i], mv = m[i];
        sx += xv * xv;
        sm += mv * mv;
        ax += w[i] * xv;
        am += w[H + i] * mv;
    }
    sx = block_sum(sx, red);
    sm = block_sum(sm, red);
    ax = block_sum(ax, red);
    am = block_sum(am, red);
    if (threadIdx.x == 0) PF(3)[z * args.u[6] + (unsigned long long)t * R + r] = ax * rsqrtf(sx / (float)H + 1e-6f) + am * rsqrtf(sm / (float)H + 1e-6f);
}

// Контроллер, шаг 2: out = (1 − tanh(F·d))·x + tanh(G·d)·m. Поток на канал; gain/forget
// транспонированы — `[R, H]`, чтение подряд по каналам.
// a0 x, a1 m f32 [T,H], a2 d f32 [T,R], a3 gain f32 [R,H], a4 forget f32 [R,H], a5 out f32 [T,H]; u0 T, u1 H, u2 R (≤128).
// Несколько контроллеров (blockIdx.z = j): шаги x u3, m u4, gain/forget u5, d u6, выход — T·H.
extern "C" __global__ void ink2_ctrl_out(Ink2Args args) {
    __shared__ float d[128];
    unsigned int t = blockIdx.y, H = args.u[1], R = args.u[2];
    const unsigned long long z = blockIdx.z;
    for (unsigned int r = threadIdx.x; r < R; r += blockDim.x) d[r] = CPF(2)[z * args.u[6] + (unsigned long long)t * R + r];
    __syncthreads();
    unsigned int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    const float* gw = CPF(3) + z * args.u[5];
    const float* fw = CPF(4) + z * args.u[5];
    float ag = 0.0f, af = 0.0f;
    for (unsigned int r = 0; r < R; ++r) {
        ag += gw[(unsigned long long)r * H + h] * d[r];
        af += fw[(unsigned long long)r * H + h] * d[r];
    }
    unsigned long long o = (unsigned long long)t * H + h;
    PF(5)[z * args.u[0] * H + o] = (1.0f - tanhf(af)) * CPF(0)[z * args.u[3] + o] + tanhf(ag) * CPF(1)[z * args.u[4] + o];
}

// Логиты роутера: l[t, e] = w[e]·x_t, варп на эксперта. a0 x [T,H] F32 либо BF16 (u3), a1 w f32 [E,H],
// a2 l f32 [T,E]; u0 T, u1 H, u2 E.
extern "C" __global__ void ink2_router(Ink2Args args) {
    unsigned int t = blockIdx.x, H = args.u[1], E = args.u[2];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    unsigned int e = blockIdx.y * 8 + wid;
    if (e >= E) return;
    const float* w = CPF(1) + (unsigned long long)e * H;
    const unsigned long long xo = (unsigned long long)t * H;
    float acc = 0.0f;
    for (unsigned int i = lane; i < H; i += 32) acc += w[i] * ink2_ld(args.a[0], xo + i, args.u[3]);
    acc = warp_sum(acc);
    if (lane == 0) PF(2)[(unsigned long long)t * E + e] = acc;
}

// Внимание по глубине: состояния s = a0..a(ns-1) (f32 [T,H] каждое), нормируются RMS без веса
// (eps 1e-6); для каждого из nq запросов (a8 [nq,H]) — softmax(q·ŝ/√H) по состояниям и
// out[q] = Σ p·ŝ. Два прохода, много блоков на токен. Проход 1: блок (t, b) считает по своему
// куску H (256 каналов) суммы x² и x·q для всех состояний → a10 [T, B, 40].
// a9 out f32 [nq,T,H], a10 частичные суммы; u0 ns (≤8), u1 T, u2 H, u3 nq (≤4), u4 B.
extern "C" __global__ void ink2_mix_part(Ink2Args args) {
    __shared__ float part[8][40];
    unsigned int t = blockIdx.x, b = blockIdx.y, ns = args.u[0], H = args.u[2], nq = args.u[3], B = args.u[4];
    unsigned int nv = ns * (1 + nq), i = b * blockDim.x + threadIdx.x;
    const float* q = CPF(8);
    float acc[40];
    for (unsigned int v = 0; v < nv; ++v) acc[v] = 0.0f;
    if (i < H) {
        float qv[4];
        for (unsigned int j = 0; j < nq; ++j) qv[j] = q[j * H + i];
        for (unsigned int s = 0; s < ns; ++s) {
            float x = CPF(s)[(unsigned long long)t * H + i];
            acc[s * (1 + nq)] += x * x;
            for (unsigned int j = 0; j < nq; ++j) acc[s * (1 + nq) + 1 + j] += x * qv[j];
        }
    }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    for (unsigned int v = 0; v < nv; ++v) {
        float x = warp_sum(acc[v]);
        if (lane == 0) part[wid][v] = x;
    }
    __syncthreads();
    if (threadIdx.x < nv) {
        float x = 0.0f;
        for (int w = 0; w < nw; ++w) x += part[w][threadIdx.x];
        PF(10)[((unsigned long long)t * B + b) * 40 + threadIdx.x] = x;
    }
}

// Проход 2: блок (t, b) складывает частичные суммы, считает softmax и пишет свой кусок выхода.
extern "C" __global__ void ink2_mix_out(Ink2Args args) {
    __shared__ float tot[40];
    __shared__ float prob[4][8];
    __shared__ float inv[8];
    unsigned int t = blockIdx.x, b = blockIdx.y, ns = args.u[0], T = args.u[1], H = args.u[2], nq = args.u[3], B = args.u[4];
    unsigned int nv = ns * (1 + nq);
    if (threadIdx.x < nv) {
        float x = 0.0f;
        for (unsigned int c = 0; c < B; ++c) x += CPF(10)[((unsigned long long)t * B + c) * 40 + threadIdx.x];
        tot[threadIdx.x] = x;
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        unsigned int j = threadIdx.x;
        float m = -INFINITY;
        for (unsigned int s = 0; s < ns; ++s) {
            float iv = rsqrtf(tot[s * (1 + nq)] / (float)H + 1e-6f);
            prob[j][s] = tot[s * (1 + nq) + 1 + j] * iv * rsqrtf((float)H);
            m = fmaxf(m, prob[j][s]);
        }
        float l = 0.0f;
        for (unsigned int s = 0; s < ns; ++s) { float e = expf(prob[j][s] - m); prob[j][s] = e; l += e; }
        for (unsigned int s = 0; s < ns; ++s) prob[j][s] /= l;
    }
    if (threadIdx.x < ns) inv[threadIdx.x] = rsqrtf(tot[threadIdx.x * (1 + nq)] / (float)H + 1e-6f);
    __syncthreads();
    unsigned int i = b * blockDim.x + threadIdx.x;
    if (i >= H) return;
    float xs[8];
    for (unsigned int s = 0; s < ns; ++s) xs[s] = CPF(s)[(unsigned long long)t * H + i] * inv[s];
    for (unsigned int j = 0; j < nq; ++j) {
        float a = 0.0f;
        for (unsigned int s = 0; s < ns; ++s) a += prob[j][s] * xs[s];
        PF(9)[((unsigned long long)j * T + t) * H + i] = a;
    }
}
