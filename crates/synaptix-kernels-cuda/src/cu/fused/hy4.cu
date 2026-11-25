// Ядра Hy4-flash (tencent, hy_v4_flash): этапы, которых нет среди общих
// операций — iHC (сведение 4 потоков и возврат в них), RoPE со сдвигом в
// строке, MLA-внимание по латентному кэшу с синками (окно / выбранные группы),
// сжатый индексатор (пулинг групп, оценки), гейт engram.
// Счёт в F32. Аргументы — одна структура Hy4Args (synaptix_core::backend::Ink2Launch).
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

struct Hy4Args {
    unsigned long long a[12];
    float f[4];
    unsigned int u[16];
};

#define PF(i) ((float*)args.a[i])
#define CPF(i) ((const float*)args.a[i])

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Сумма по блоку; red — не меньше 32 float.
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

__device__ __forceinline__ float block_max(float v, float* red) {
    v = warp_max(v);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    float r = -INFINITY;
    for (int i = 0; i < nw; ++i) r = fmaxf(r, red[i]);
    return r;
}

// Элемент входа в F32 / BF16 / F16 (u-флаг: 0 / 1 / 2).
__device__ __forceinline__ float ld_any(unsigned long long p, unsigned long long i, unsigned kind) {
    if (kind == 1) return __bfloat162float(((const __nv_bfloat16*)p)[i]);
    if (kind == 2) return __half2float(((const __half*)p)[i]);
    return ((const float*)p)[i];
}

__device__ __forceinline__ void st_any(unsigned long long p, unsigned long long i, unsigned kind, float v) {
    if (kind == 1) ((__nv_bfloat16*)p)[i] = __float2bfloat16(v);
    else if (kind == 2) ((__half*)p)[i] = __float2half(v);
    else ((float*)p)[i] = v;
}

// ---- iHC: сведение потоков ----
// a0 X [T, M, H] F32; a1 mixes [T, NM] F32 (flat·fnᵀ, без нормы); a2 base [NM];
// a3 scale [2] (голова: [1]); a4 → h [T, H] F32; a5 → post [T, M] F32 (не голова).
// u0 T, u1 H, u2 M, u3 голова (только pre: NM = M, один масштаб).
// f0 eps нормы, f1 hc_eps, f2 множитель post.
// pre_i = σ(mix_i·r·s0 + b_i) + eps; post_i = mag·σ(mix_{M+i}·r·s1 + b_{M+i}) + eps;
// r = rsqrt(mean(X²) + eps) по всей строке M·H; h = Σ pre_i·X_i.
extern "C" __global__ void hy4_hc_pre(Hy4Args args) {
    __shared__ float red[32];
    __shared__ float pre[8];
    const unsigned t = blockIdx.x, H = args.u[1], M = args.u[2], head = args.u[3];
    const float* x = CPF(0) + (size_t)t * M * H;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < M * H; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / (float)(M * H) + args.f[0]);
    const unsigned nm = head ? M : 2 * M;
    if (threadIdx.x < nm) {
        const unsigned i = threadIdx.x;
        const float mix = CPF(1)[(size_t)t * nm + i] * r;
        const float* sc = CPF(3);
        const float b = CPF(2)[i];
        if (i < M) {
            pre[i] = sigm(mix * sc[0] + b) + args.f[1];
        } else {
            PF(5)[(size_t)t * M + (i - M)] = args.f[2] * sigm(mix * sc[1] + b) + args.f[1];
        }
    }
    __syncthreads();
    float* h = PF(4) + (size_t)t * H;
    for (unsigned d = threadIdx.x; d < H; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned i = 0; i < M; ++i) acc += pre[i] * x[(size_t)i * H + d];
        h[d] = acc;
    }
}

// ---- iHC: возврат в потоки ----
// a0 X [T, M, H] F32 (на месте); a1 post [T, M]; a2 y [T, H] (u3 — тип: 0 F32, 1 BF16, 2 F16).
// u0 T, u1 H, u2 M.
extern "C" __global__ void hy4_hc_post(Hy4Args args) {
    const unsigned t = blockIdx.x, H = args.u[1], M = args.u[2];
    float* x = PF(0) + (size_t)t * M * H;
    const float* post = CPF(1) + (size_t)t * M;
    for (unsigned d = threadIdx.x; d < H; d += blockDim.x) {
        const float y = ld_any(args.a[2], (size_t)t * H + d, args.u[3]);
        for (unsigned i = 0; i < M; ++i) x[(size_t)i * H + d] += post[i] * y;
    }
}

// ---- RMS-норма строк с весом ----
// a0 x [T, H] (u3 — тип входа), a1 w [H] F32 (0 — без веса), a2 → out [T, H] (u4 — тип выхода).
// u0 T, u1 H, u2 шаг строки входа (элементов; 0 — H). f0 eps.
extern "C" __global__ void hy4_rms(Hy4Args args) {
    __shared__ float red[32];
    const unsigned t = blockIdx.x, H = args.u[1];
    const size_t stride = args.u[2] ? args.u[2] : H;
    float ss = 0.0f;
    for (unsigned d = threadIdx.x; d < H; d += blockDim.x) {
        const float v = ld_any(args.a[0], t * stride + d, args.u[3]);
        ss += v * v;
    }
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / (float)H + args.f[0]);
    const float* w = (const float*)args.a[1];
    for (unsigned d = threadIdx.x; d < H; d += blockDim.x) {
        const float v = ld_any(args.a[0], t * stride + d, args.u[3]) * r * (w ? w[d] : 1.0f);
        st_any(args.a[2], (size_t)t * H + d, args.u[4], v);
    }
}

// ---- RoPE со сдвигом (чередующиеся пары, как apply_rotary_pos_emb Hy4) ----
// a0 x F32 (на месте): строка t, голова h — x[t·u1 + h·u2 + u3 + …];
// a1 позиции [T] I32. u0 T, u1 шаг строки, u2 шаг головы, u3 сдвиг, u4 голов, u5 ширина RoPE.
// f0 log2(theta). Пара j (каналы 2j, 2j+1) крутится на pos·theta^(−2j/width).
extern "C" __global__ void hy4_rope(Hy4Args args) {
    const unsigned t = blockIdx.x, h = blockIdx.y, W = args.u[5];
    const int pos = ((const int*)args.a[1])[t];
    float* x = PF(0) + (size_t)t * args.u[1] + (size_t)h * args.u[2] + args.u[3];
    for (unsigned j = threadIdx.x; j < W / 2; j += blockDim.x) {
        // Частота как в torch: 1/(theta^(2j/W)) в F32, угол pos·inv_freq в F32.
        const float inv = 1.0f / exp2f(args.f[0] * (float)(2 * j) / (float)W);
        const float ang = (float)pos * inv;
        float s, c;
        sincosf(ang, &s, &c);
        const float a = x[2 * j], b = x[2 * j + 1];
        x[2 * j] = a * c - b * s;
        x[2 * j + 1] = b * c + a * s;
    }
}

// ---- MLA-внимание по латентному кэшу с синками ----
// a0 qa [T, NH, L+R] F32 (латентная часть q·W_uk и q_rot); a1 кэш [CAP, L+R] (u7 — тип: 0 F32, 1 BF16);
// a2 sinks [NH] F32; a3 → o [T, NH, L] F32; a4 позиции запросов [T] I32;
// a5 группы [T, NG] U32 (режим 1; u32::MAX — пусто) — плюс своя группа запроса.
// u0 T, u1 NH, u2 L, u3 R, u4 CAP (слот = pos mod CAP), u5 режим (0 окно, 1 группы),
// u6 окно (режим 0: ключи pos−окно+1 … pos), u8 NG, u9 токенов в группе.
// f0 масштаб.
#define HY4_MAXK 2048
extern "C" __global__ void hy4_mla_attn(Hy4Args args) {
    __shared__ float q[640];
    __shared__ float sc[HY4_MAXK];
    __shared__ int kp[HY4_MAXK];
    __shared__ float red[32];
    __shared__ int nk_s;
    const unsigned t = blockIdx.x, h = blockIdx.y, NH = args.u[1], L = args.u[2], R = args.u[3];
    const unsigned D = L + R, CAP = args.u[4], kind = args.u[7];
    const int pos = ((const int*)args.a[4])[t];
    const float* qp = CPF(0) + ((size_t)t * NH + h) * D;
    for (unsigned d = threadIdx.x; d < D; d += blockDim.x) q[d] = qp[d];
    // Список позиций ключей.
    if (args.u[5] == 0) {
        const int w = (int)args.u[6];
        const int lo = max(0, pos - w + 1);
        const int n = min(pos - lo + 1, HY4_MAXK);
        const int lo2 = pos - n + 1;
        for (int j = threadIdx.x; j < n; j += blockDim.x) kp[j] = lo2 + j;
        if (threadIdx.x == 0) nk_s = n;
    } else {
        const unsigned NG = args.u[8], G = args.u[9];
        const unsigned* gl = (const unsigned*)args.a[5] + (size_t)t * NG;
        if (threadIdx.x == 0) {
            int n = 0;
            const int own = pos / (int)G;
            for (unsigned i = 0; i <= NG && n + (int)G <= HY4_MAXK; ++i) {
                const unsigned g = i < NG ? gl[i] : (unsigned)own;
                if (g == 0xffffffffu) continue;
                if (i < NG && (int)g == own) continue;
                for (unsigned o = 0; o < G; ++o) {
                    const int p = (int)(g * G + o);
                    if (p <= pos) kp[n++] = p;
                }
            }
            nk_s = n;
        }
    }
    __syncthreads();
    const int nk = nk_s;
    const float scale = args.f[0];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int j = wid; j < nk; j += nw) {
        const size_t row = (size_t)((unsigned)kp[j] % CAP) * D;
        float acc = 0.0f;
        for (unsigned d = lane; d < D; d += 32) acc += q[d] * ld_any(args.a[1], row + d, kind);
        acc = warp_sum(acc);
        if (lane == 0) sc[j] = acc * scale;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = block_max(m, red);
    const float sink = CPF(2)[h];
    m = fmaxf(m, sink);
    float s = 0.0f;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) {
        const float e = expf(sc[j] - m);
        sc[j] = e;
        s += e;
    }
    s = block_sum(s, red);
    const float inv = 1.0f / (s + expf(sink - m));
    float* o = PF(3) + ((size_t)t * NH + h) * L;
    for (unsigned d = threadIdx.x; d < L; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < nk; ++j) acc += sc[j] * ld_any(args.a[1], (size_t)((unsigned)kp[j] % CAP) * D + d, kind);
        o[d] = acc * inv;
    }
}

// ---- Запись латентов в кэш ----
// a0 src [T, D] F32; a1 кэш [CAP, D] (u3 — тип: 0 F32, 1 BF16); a2 позиции [T] I32.
// u0 T, u1 D, u2 CAP.
extern "C" __global__ void hy4_cache_put(Hy4Args args) {
    const unsigned t = blockIdx.x, D = args.u[1];
    const int pos = ((const int*)args.a[2])[t];
    const size_t row = (size_t)((unsigned)pos % args.u[2]) * D;
    for (unsigned d = threadIdx.x; d < D; d += blockDim.x) st_any(args.a[1], row + d, args.u[3], CPF(0)[(size_t)t * D + d]);
}

// ---- Индексатор: пулинг групп ----
// a0 ключи токенов [CAPT, C] F32 (нормированные; слот = pos mod CAPT); a1 Wv [C, C] F32;
// a2 Wg [C, C] F32; a3 ape [G, C] F32; a4 вес key_norm [C] F32; a5 → ключи групп [CAPG, C] F32.
// u0 первая группа, u1 C, u2 G (токенов в группе), u3 CAPT, u4 CAPG, u5 ширина RoPE (хвост строки).
// f0 eps, f1 log2(theta). Позиция группы — позиция её первого токена (g·G).
extern "C" __global__ void hy4_idx_pool(Hy4Args args) {
    __shared__ float k[4][256];
    __shared__ float pooled[256];
    __shared__ float red[32];
    const unsigned g = args.u[0] + blockIdx.x, C = args.u[1], G = args.u[2];
    for (unsigned i = 0; i < G; ++i)
        for (unsigned c = threadIdx.x; c < C; c += blockDim.x)
            k[i][c] = CPF(0)[(size_t)((g * G + i) % args.u[3]) * C + c];
    __syncthreads();
    for (unsigned c = threadIdx.x; c < C; c += blockDim.x) {
        float v[4], gt[4];
        float mx = -INFINITY;
        for (unsigned i = 0; i < G; ++i) {
            float av = 0.0f, ag = 0.0f;
            for (unsigned j = 0; j < C; ++j) {
                av += CPF(1)[(size_t)c * C + j] * k[i][j];
                ag += CPF(2)[(size_t)c * C + j] * k[i][j];
            }
            v[i] = av;
            gt[i] = ag + CPF(3)[(size_t)i * C + c];
            mx = fmaxf(mx, gt[i]);
        }
        float s = 0.0f, acc = 0.0f;
        for (unsigned i = 0; i < G; ++i) {
            const float e = expf(gt[i] - mx);
            s += e;
            acc += v[i] * e;
        }
        pooled[c] = acc / s;
    }
    __syncthreads();
    float ss = 0.0f;
    for (unsigned c = threadIdx.x; c < C; c += blockDim.x) ss += pooled[c] * pooled[c];
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / (float)C + args.f[0]);
    for (unsigned c = threadIdx.x; c < C; c += blockDim.x) pooled[c] = pooled[c] * r * CPF(4)[c];
    __syncthreads();
    const unsigned W = args.u[5], off = C - W;
    float* out = PF(5) + (size_t)(g % args.u[4]) * C;
    const float pos = (float)(g * G);
    for (unsigned c = threadIdx.x; c < C; c += blockDim.x) {
        float v = pooled[c];
        if (c >= off) {
            const unsigned j = (c - off) / 2;
            const float inv = 1.0f / exp2f(args.f[1] * (float)(2 * j) / (float)W);
            float s, co;
            sincosf(pos * inv, &s, &co);
            const float a = pooled[off + 2 * j], b = pooled[off + 2 * j + 1];
            v = ((c - off) & 1) ? (b * co + a * s) : (a * co - b * s);
        }
        out[c] = v;
    }
}

// ---- Индексатор: оценки групп ----
// a0 q [T, NHI, C] F32 (RoPE уже на хвосте); a1 веса голов [T, NHI] F32 (с масштабом);
// a2 ключи групп [CAPG, C]; a3 → оценки [T, NGMAX] F32; a4 позиции [T] I32.
// u0 T, u1 NHI, u2 C, u3 NGMAX (ширина строки оценок), u4 G, u5 CAPG.
// Группа g < своей группы запроса: Σ_h w_h·relu(q_h·K_g); остальные не пишутся
// (их отсекает valid у topk_wide).
extern "C" __global__ void hy4_idx_score(Hy4Args args) {
    __shared__ float qs[16 * 128];
    __shared__ float ws[16];
    const unsigned t = blockIdx.x, NHI = args.u[1], C = args.u[2];
    const int pos = ((const int*)args.a[4])[t];
    const unsigned own = (unsigned)pos / args.u[4];
    for (unsigned i = threadIdx.x; i < NHI * C; i += blockDim.x) qs[i] = CPF(0)[(size_t)t * NHI * C + i];
    if (threadIdx.x < NHI) ws[threadIdx.x] = CPF(1)[(size_t)t * NHI + threadIdx.x];
    __syncthreads();
    const unsigned g = blockIdx.y * blockDim.x + threadIdx.x;
    if (g >= own || g >= args.u[3]) return;
    const float* kg = CPF(2) + (size_t)(g % args.u[5]) * C;
    float total = 0.0f;
    for (unsigned h = 0; h < NHI; ++h) {
        float d = 0.0f;
        for (unsigned c = 0; c < C; ++c) d += qs[h * C + c] * kg[c];
        total += ws[h] * fmaxf(d, 0.0f);
    }
    PF(3)[(size_t)t * args.u[3] + g] = total;
}

// ---- Гейт engram ----
// a0 X [T, M, H] F32 (на месте); a1 proj [T, (M+1)·H] F32 (ключи потоков, затем value);
// a2 key_norm [M, H]; a3 query_norm [M, H]; a4 pre_gamma [H]; a5 hc_scale [M, H];
// a6 live [T] U8. u0 T, u1 H, u2 M. f0 eps.
// key = rms(k_c)·kn_c, query = rms(X_c)·qn_c, dot = Σ key·query / √H,
// gate = σ(sign·√max(|dot|, 1e-6)), signal = pre_gamma·value·rsqrt(mean(value²)+eps),
// X_c += (gate + hc_scale_c)·signal.
extern "C" __global__ void hy4_engram_gate(Hy4Args args) {
    __shared__ float red[32];
    const unsigned t = blockIdx.x / args.u[2], c = blockIdx.x % args.u[2], H = args.u[1], M = args.u[2];
    if (!((const unsigned char*)args.a[6])[t]) return;
    float* x = PF(0) + ((size_t)t * M + c) * H;
    const float* key = CPF(1) + (size_t)t * (M + 1) * H + (size_t)c * H;
    const float* val = CPF(1) + (size_t)t * (M + 1) * H + (size_t)M * H;
    const float* kn = CPF(2) + (size_t)c * H;
    const float* qn = CPF(3) + (size_t)c * H;
    const float eps = args.f[0];
    float kk = 0.0f, xx = 0.0f, vv = 0.0f;
    for (unsigned d = threadIdx.x; d < H; d += blockDim.x) {
        kk += key[d] * key[d];
        xx += x[d] * x[d];
        vv += val[d] * val[d];
    }
    kk = block_sum(kk, red);
    xx = block_sum(xx, red);
    vv = block_sum(vv, red);
    const float rk = rsqrtf(kk / (float)H + eps), rx = rsqrtf(xx / (float)H + eps), rv = rsqrtf(vv / (float)H + eps);
    float dot = 0.0f;
    for (unsigned d = threadIdx.x; d < H; d += blockDim.x) dot += (key[d] * rk * kn[d]) * (x[d] * rx * qn[d]);
    dot = block_sum(dot, red) * rsqrtf((float)H);
    const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float sign = dot > 0.0f ? 1.0f : (dot < 0.0f ? -1.0f : 0.0f);
    const float gate = sigm(sign * mag);
    __syncthreads();
    for (unsigned d = threadIdx.x; d < H; d += blockDim.x) {
        const float sig = CPF(4)[d] * val[d] * rv;
        x[d] += (gate + CPF(5)[(size_t)c * H + d]) * sig;
    }
}

// ---- Гейт внимания: o[t, h, :] ·= σ(gate[t, h, :]) и приведение ----
// a0 o [T, NH·V] F32; a1 gate [T, NH·V] (u2 — тип); a2 → out [T, NH·V] (u3 — тип).
// u0 T, u1 NH·V.
extern "C" __global__ void hy4_gate_mul(Hy4Args args) {
    const unsigned t = blockIdx.x, N = args.u[1];
    for (unsigned i = threadIdx.x; i < N; i += blockDim.x) {
        const size_t k = (size_t)t * N + i;
        st_any(args.a[2], k, args.u[3], CPF(0)[k] * sigm(ld_any(args.a[1], k, args.u[2])));
    }
}

// Поглощённые проекции MLA на декоде: по головам out[t,h,:] = x[t,h,:] · W[h]
// (W BF16 [NH, Din, Dout]; x, out — F32 со своими шагами по t и h). Общий
// батч-matmul на одну строку медленный; здесь вес читается один раз на все
// T ≤ 8 строк. Блок 32×8: варп — 64 соседних столбца (пары bf16), 8 варпов
// делят Din и сводятся в shared. Сетка (NH, Dout/64).
//   a0 x, a1 W, a2 out; u0 T, u1 NH, u2 Din, u3 Dout, u4 x_t, u5 x_h, u6 o_t, u7 o_h
#define HY4_HG_T 8
extern "C" __global__ void hy4_head_gemv(Hy4Args args) {
    __shared__ float xs[HY4_HG_T][512];
    __shared__ float red[8][HY4_HG_T][64];
    const unsigned h = blockIdx.x, T = args.u[0], Din = args.u[2], Dout = args.u[3];
    const unsigned lx = threadIdx.x & 31, ly = threadIdx.x >> 5;
    const float* x = CPF(0);
    for (unsigned i = threadIdx.x; i < T * Din; i += blockDim.x) {
        const unsigned t = i / Din, d = i % Din;
        xs[t][d] = x[(size_t)t * args.u[4] + (size_t)h * args.u[5] + d];
    }
    __syncthreads();
    const unsigned c = blockIdx.y * 64 + 2 * lx;
    float acc[HY4_HG_T][2];
#pragma unroll
    for (int t = 0; t < HY4_HG_T; ++t) acc[t][0] = acc[t][1] = 0.0f;
    if (c < Dout) {
        const __nv_bfloat162* w = (const __nv_bfloat162*)((const __nv_bfloat16*)args.a[1] + (size_t)h * Din * Dout + c);
        for (unsigned d = ly; d < Din; d += 8) {
            const float2 wf = __bfloat1622float2(w[(size_t)d * (Dout / 2)]);
#pragma unroll
            for (int t = 0; t < HY4_HG_T; ++t) {
                if (t < (int)T) {
                    acc[t][0] += xs[t][d] * wf.x;
                    acc[t][1] += xs[t][d] * wf.y;
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < HY4_HG_T; ++t) {
        red[ly][t][2 * lx] = acc[t][0];
        red[ly][t][2 * lx + 1] = acc[t][1];
    }
    __syncthreads();
    float* out = PF(2);
    for (unsigned i = threadIdx.x; i < T * 64; i += blockDim.x) {
        const unsigned t = i / 64, j = i % 64, col = blockIdx.y * 64 + j;
        if (col >= Dout) continue;
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) s += red[k][t][j];
        out[(size_t)t * args.u[6] + (size_t)h * args.u[7] + col] = s;
    }
}
