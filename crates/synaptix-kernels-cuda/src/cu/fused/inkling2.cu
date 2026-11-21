// Ядра Inkling-2 (thinkingmachines): этапы, которых нет среди общих операций —
// смесь блоков по глубине (attention-residuals), короткие каузальные свёртки
// с состоянием, подготовка q/k/v/r с L2-клипом и записью в KV, внимание с
// относительным bias по расстоянию и синком на (токен, голову), softcap-SwiGLU
// экспертов с общим множителем NVFP4, маршрутизация и сборка MoE.
// Счёт в F32; KV и входы экспертов — BF16. Аргументы — одна структура Ink2Args
// (см. synaptix_core::backend::Ink2Launch).
#include <cuda_bf16.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

struct Ink2Args {
    unsigned long long a[12];
    float f[4];
    unsigned int u[12];
};

#define PF(i) ((float*)args.a[i])
#define CPF(i) ((const float*)args.a[i])
#define CPB(i) ((const __nv_bfloat16*)args.a[i])
#define PB(i) ((__nv_bfloat16*)args.a[i])

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

// RMS-норма строки: out = x·rsqrt(mean(x²)+eps)·w (w = 0 — без веса).
// a0 x f32 [T,H], a1 w f32 [H]|0, a2 out; u0 T, u1 H, u2 out_bf16; f0 eps.
extern "C" __global__ void ink2_rms(Ink2Args args) {
    __shared__ float red[32];
    unsigned int t = blockIdx.x, H = args.u[1];
    const float* x = CPF(0) + (unsigned long long)t * H;
    const float* w = CPF(1);
    float ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < H; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss, red);
    float inv = rsqrtf(ss / (float)H + args.f[0]);
    for (unsigned int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = x[i] * inv * (w ? w[i] : 1.0f);
        if (args.u[2]) PB(2)[(unsigned long long)t * H + i] = __float2bfloat16(v);
        else PF(2)[(unsigned long long)t * H + i] = v;
    }
}

// Внимание по глубине: состояния s = a0..a(ns-1) (f32 [T,H] каждое), нормируются
// RMS без веса (eps 1e-6); для каждого из nq запросов (a8 [nq,H]) —
// softmax(q·ŝ/√H) по состояниям и out[q] = Σ p·ŝ.
// a9 out f32 [nq,T,H]; u0 ns (≤8), u1 T, u2 H, u3 nq (≤4).
extern "C" __global__ void ink2_mix(Ink2Args args) {
    __shared__ float red[32];
    __shared__ float inv[8];
    __shared__ float prob[4][8];
    unsigned int t = blockIdx.x, ns = args.u[0], T = args.u[1], H = args.u[2], nq = args.u[3];
    const float* q = CPF(8);
    for (unsigned int s = 0; s < ns; ++s) {
        const float* x = CPF(s) + (unsigned long long)t * H;
        float ss = 0.0f;
        float dq[4] = {0.f, 0.f, 0.f, 0.f};
        for (unsigned int i = threadIdx.x; i < H; i += blockDim.x) {
            float v = x[i];
            ss += v * v;
            for (unsigned int j = 0; j < nq; ++j) dq[j] += v * q[j * H + i];
        }
        ss = block_sum(ss, red);
        float iv = rsqrtf(ss / (float)H + 1e-6f);
        for (unsigned int j = 0; j < nq; ++j) {
            float d = block_sum(dq[j], red);
            if (threadIdx.x == 0) prob[j][s] = d * iv * rsqrtf((float)H);
        }
        if (threadIdx.x == 0) inv[s] = iv;
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        unsigned int j = threadIdx.x;
        float m = -INFINITY;
        for (unsigned int s = 0; s < ns; ++s) m = fmaxf(m, prob[j][s]);
        float l = 0.0f;
        for (unsigned int s = 0; s < ns; ++s) { float e = expf(prob[j][s] - m); prob[j][s] = e; l += e; }
        for (unsigned int s = 0; s < ns; ++s) prob[j][s] /= l;
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < H; i += blockDim.x) {
        float xs[8];
        for (unsigned int s = 0; s < ns; ++s) xs[s] = CPF(s)[(unsigned long long)t * H + i] * inv[s];
        for (unsigned int j = 0; j < nq; ++j) {
            float acc = 0.0f;
            for (unsigned int s = 0; s < ns; ++s) acc += prob[j][s] * xs[s];
            PF(9)[((unsigned long long)j * T + t) * H + i] = acc;
        }
    }
}

// Короткая каузальная свёртка с остатком: y_t = (Σ_j w[c,j]·x_{t-K+1+j} + x_t)·scale,
// w[c, K-1] — текущий токен. Динамические отводы (полные слои): + Σ_lag dyn[t, h·K+lag]·x_{t-lag},
// h = c / (C/heads). Состояние — последние K-1 входов (старший первый), обновляется.
// a0 x f32 [T,C], a1 out f32 [T,C], a2 state f32 [K-1,C], a3 w f32 [C,K], a4 dyn f32 [T,heads·K]|0;
// u0 T, u1 C, u2 K (≤8), u3 heads; f0 scale.
extern "C" __global__ void ink2_conv(Ink2Args args) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int T = args.u[0], C = args.u[1], K = args.u[2], heads = args.u[3];
    if (c >= C) return;
    const float* x = CPF(0);
    float* st = PF(2);
    const float* w = CPF(3) + (unsigned long long)c * K;
    const float* dyn = CPF(4);
    unsigned int h = heads ? c / (C / heads) : 0;
    float hist[8];
    for (unsigned int j = 0; j + 1 < K; ++j) hist[j] = st[(unsigned long long)j * C + c];
    for (unsigned int t = 0; t < T; ++t) {
        float xt = x[(unsigned long long)t * C + c];
        hist[K - 1] = xt;
        float acc = 0.0f;
        for (unsigned int j = 0; j < K; ++j) acc += w[j] * hist[j];
        if (dyn) {
            const float* d = dyn + (unsigned long long)t * heads * K + h * K;
            for (unsigned int lag = 0; lag < K; ++lag) acc += d[lag] * hist[K - 1 - lag];
        }
        PF(1)[(unsigned long long)t * C + c] = (acc + xt) * args.f[0];
        for (unsigned int j = 0; j + 1 < K; ++j) hist[j] = hist[j + 1];
    }
    for (unsigned int j = 0; j + 1 < K; ++j) st[(unsigned long long)j * C + c] = hist[j];
}

// Подготовка внимания: блок на (токен, вектор); вектора 0..nh-1 — q (и r той же
// головы), nh..nh+nkv-1 — k, дальше v. q: ·scale → L2-клип → ·tau; r: ·rscale →
// клип → ·tau; k: клип, в кэш; v: RMS без веса, в кэш. tau = 1 + α·ln(max(1,(pos+1)/floor))
// у полных слоёв, иначе 1.
// a0 q, a1 k, a2 v, a3 r (f32 [T, ·]), a4 qscale f32 [nh·d], a5 rscale f32 [nh·dr],
// a6 q_out f32, a7 r_out f32, a8 kcache bf16 [nkv,cap,d], a9 vcache bf16;
// u0 T, u1 nh, u2 nkv, u3 d, u4 dr, u5 cap, u6 past, u7 full, u8 floor;
// f0 qk_clip, f1 rel_clip, f2 eps, f3 alpha.
extern "C" __global__ void ink2_qkv(Ink2Args args) {
    __shared__ float red[32];
    unsigned int t = blockIdx.x, vi = blockIdx.y;
    unsigned int nh = args.u[1], nkv = args.u[2], d = args.u[3], dr = args.u[4], cap = args.u[5];
    unsigned int pos = args.u[6] + t;
    float tau = 1.0f;
    if (args.u[7]) tau = 1.0f + args.f[3] * logf(fmaxf(1.0f, (float)(pos + 1) / (float)args.u[8]));
    unsigned int i = threadIdx.x;
    if (vi < nh) {
        unsigned long long o = ((unsigned long long)t * nh + vi) * d;
        float v = i < d ? CPF(0)[o + i] * CPF(4)[vi * d + i] : 0.0f;
        float n = sqrtf(block_sum(v * v, red));
        float k = fminf(1.0f, args.f[0] / (n + 1e-6f));
        if (i < d) PF(6)[o + i] = v * k * tau;
        unsigned long long orr = ((unsigned long long)t * nh + vi) * dr;
        float r = i < dr ? CPF(3)[orr + i] * CPF(5)[vi * dr + i] : 0.0f;
        float nr = sqrtf(block_sum(r * r, red));
        float kr = fminf(1.0f, args.f[1] / (nr + 1e-6f));
        if (i < dr) PF(7)[orr + i] = r * kr * tau;
    } else if (vi < nh + nkv) {
        unsigned int h = vi - nh;
        float v = i < d ? CPF(1)[((unsigned long long)t * nkv + h) * d + i] : 0.0f;
        float n = sqrtf(block_sum(v * v, red));
        float k = fminf(1.0f, args.f[0] / (n + 1e-6f));
        if (i < d) PB(8)[((unsigned long long)h * cap + pos) * d + i] = __float2bfloat16(v * k);
    } else {
        unsigned int h = vi - nh - nkv;
        float v = i < d ? CPF(2)[((unsigned long long)t * nkv + h) * d + i] : 0.0f;
        float ss = block_sum(v * v, red);
        float inv = rsqrtf(ss / (float)d + args.f[2]);
        if (i < d) PB(9)[((unsigned long long)h * cap + pos) * d + i] = __float2bfloat16(v * inv);
    }
}

// Внимание: блок на (токен, голову), 4 варпа делят ключи; лейн держит 4 из 128
// измерений. Логит = q·k·scale + bias(dist), bias = Σ_c r[c]·proj[c, dist] при
// dist < extent, иначе 0; лишний логит синка с нулевым value; окно window (0 — нет).
// Выход ·sigmoid(gate).
// a0 q f32 [T,nh,d], a1 r f32 [T,nh,dr], a2 proj f32 [dr,extent], a3 sink f32 [T,nh],
// a4 kcache, a5 vcache bf16 [nkv,cap,d], a6 gate f32 [T,nh·d], a7 out f32 [T,nh·d];
// u0 T, u1 nh, u2 nkv, u3 d (=128), u4 dr (≤32), u5 cap, u6 past, u7 window, u8 extent; f0 scale.
extern "C" __global__ void ink2_attn(Ink2Args args) {
    __shared__ float sm_m[4], sm_l[4];
    __shared__ float sm_acc[4][128];
    unsigned int t = blockIdx.x, h = blockIdx.y;
    unsigned int nh = args.u[1], nkv = args.u[2], d = 128, dr = args.u[4], cap = args.u[5];
    unsigned int window = args.u[7], extent = args.u[8];
    unsigned int pos = args.u[6] + t;
    unsigned int kvh = h / (nh / nkv);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    unsigned long long qo = ((unsigned long long)t * nh + h) * d;
    float q[4];
    for (int j = 0; j < 4; ++j) q[j] = CPF(0)[qo + lane * 4 + j] * args.f[0];
    float rr = lane < (int)dr ? CPF(1)[((unsigned long long)t * nh + h) * dr + lane] : 0.0f;
    const float* proj = CPF(2);
    const __nv_bfloat16* kc = CPB(4) + (unsigned long long)kvh * cap * d;
    const __nv_bfloat16* vc = CPB(5) + (unsigned long long)kvh * cap * d;
    unsigned int lo = (window && pos + 1 > window) ? pos + 1 - window : 0;
    float m = -INFINITY, l = 0.0f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (unsigned int j = lo + wid; j <= pos; j += 4) {
        const __nv_bfloat16* kr = kc + (unsigned long long)j * d + lane * 4;
        float2 k01 = __bfloat1622float2(*(const __nv_bfloat162*)kr);
        float2 k23 = __bfloat1622float2(*(const __nv_bfloat162*)(kr + 2));
        float part = q[0] * k01.x + q[1] * k01.y + q[2] * k23.x + q[3] * k23.y;
        unsigned int dist = pos - j;
        if (dist < extent && lane < (int)dr) part += rr * proj[(unsigned long long)lane * extent + dist];
        float s = warp_sum(part);
        float mn = fmaxf(m, s);
        float corr = expf(m - mn), p = expf(s - mn);
        const __nv_bfloat16* vr = vc + (unsigned long long)j * d + lane * 4;
        float2 v01 = __bfloat1622float2(*(const __nv_bfloat162*)vr);
        float2 v23 = __bfloat1622float2(*(const __nv_bfloat162*)(vr + 2));
        acc[0] = acc[0] * corr + p * v01.x;
        acc[1] = acc[1] * corr + p * v01.y;
        acc[2] = acc[2] * corr + p * v23.x;
        acc[3] = acc[3] * corr + p * v23.y;
        l = l * corr + p;
        m = mn;
    }
    if (lane == 0) { sm_m[wid] = m; sm_l[wid] = l; }
    for (int j = 0; j < 4; ++j) sm_acc[wid][lane * 4 + j] = acc[j];
    __syncthreads();
    float sink = CPF(3)[(unsigned long long)t * nh + h];
    float M = sink;
    for (int w = 0; w < 4; ++w) M = fmaxf(M, sm_m[w]);
    float L = expf(sink - M);
    for (int w = 0; w < 4; ++w) if (sm_l[w] > 0.0f) L += sm_l[w] * expf(sm_m[w] - M);
    unsigned int i = threadIdx.x;  // 128 потоков — по измерению
    float o = 0.0f;
    for (int w = 0; w < 4; ++w) if (sm_l[w] > 0.0f) o += sm_acc[w][i] * expf(sm_m[w] - M);
    o /= L;
    float g = CPF(6)[(unsigned long long)t * nh * d + h * d + i];
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
// u0 T, u1 E (≤1024), u2 k; f0 scale.
extern "C" __global__ void ink2_route(Ink2Args args) {
    __shared__ float key[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ float sel_s[16];
    __shared__ int sel_i[16];
    unsigned int t = blockIdx.x, E = args.u[1], K = args.u[2];
    const float* lg = CPF(0) + (unsigned long long)t * E;
    for (unsigned int e = threadIdx.x; e < E; e += blockDim.x) key[e] = sigm(lg[e]) + CPF(1)[e];
    __syncthreads();
    for (unsigned int k = 0; k < K; ++k) {
        float best = -INFINITY;
        int idx = -1;
        for (unsigned int e = threadIdx.x; e < E; e += blockDim.x) {
            if (key[e] > best) { best = key[e]; idx = (int)e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            float ob = __shfl_xor_sync(0xffffffffu, best, o);
            int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ob > best || (ob == best && oi >= 0 && (idx < 0 || oi < idx))) { best = ob; idx = oi; }
        }
        int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        if (lane == 0) { bv[wid] = best; bi[wid] = idx; }
        __syncthreads();
        if (threadIdx.x == 0) {
            int nw = (blockDim.x + 31) >> 5;
            float b = bv[0];
            int ii = bi[0];
            for (int w = 1; w < nw; ++w) if (bv[w] > b || (bv[w] == b && bi[w] >= 0 && bi[w] < ii)) { b = bv[w]; ii = bi[w]; }
            sel_i[k] = ii;
            sel_s[k] = sigm(lg[ii]);
            key[ii] = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < K; ++k) sum += sel_s[k];
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
