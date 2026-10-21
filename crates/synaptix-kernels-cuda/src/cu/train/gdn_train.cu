// Gated DeltaNet для обучения: каузальная depthwise-свёртка с номерами
// примеров и рекуррентное правило дельты с затуханием (прямой проход с
// контрольными точками состояния и обратный проход назад по времени).
//
// Рекуррентность на v-голову (k-голова — hv / (Hv / Hk)), состояние
// S [DK, dv]:
//   f_t = начало примера ? 0 : exp(g_t),   A = f_t · S_{t−1},
//   u_t = β_t · (v_t − Aᵀ·k_t),   S_t = A + k_t·u_tᵀ,   o_t = S_tᵀ·q_t.
// Столбцы S (по dv) независимы: блок ведёт 32 столбца одной головы, варп w —
// строки w·32 … w·32+31, нить — один столбец. q, k уже нормированы (q — с
// множителем 1/√DK).

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define DK 128
#define CS 32      // столбцов на блок
#define TC 32      // шагов в куске подкачки
#define CKPT 64    // шаг контрольных точек состояния

__device__ __forceinline__ float ldx(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float ldx(const __nv_bfloat16* p, long long i) { return __bfloat162float(p[i]); }

__device__ __forceinline__ bool seg_start(const int* seg, int t) {
    return t == 0 || (seg != nullptr && seg[t] != seg[t - 1]);
}

// ── Свёртка ─────────────────────────────────────────────────────────────────
// y[t, c] = Σ_j w[c, j] · x[t − (K−1) + j, c], слагаемые из другого примера —
// нули. x [T, C], w [C, K] F32, y [T, C] F32.
template <typename T>
__device__ void conv_fwd(const T* __restrict__ x, const float* __restrict__ w, const int* __restrict__ seg,
                         float* __restrict__ y, int Tn, int C, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)Tn * C) return;
    int t = (int)(i / C), c = (int)(i % C);
    int st = seg ? seg[t] : 0;
    float acc = 0.f;
    for (int j = 0; j < K; j++) {
        int s = t - (K - 1) + j;
        if (s < 0 || (seg && seg[s] != st)) continue;
        acc += w[c * K + j] * ldx(x, (long long)s * C + c);
    }
    y[i] = acc;
}

// dx[s, c] = Σ_j w[c, j] · dy[s + (K−1) − j, c] (тот же пример).
extern "C" __global__ void gdn_conv_bwd(const float* __restrict__ dy, const float* __restrict__ w, const int* __restrict__ seg,
                             float* __restrict__ dx, int Tn, int C, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)Tn * C) return;
    int s = (int)(i / C), c = (int)(i % C);
    int ss = seg ? seg[s] : 0;
    float acc = 0.f;
    for (int j = 0; j < K; j++) {
        int t = s + (K - 1) - j;
        if (t >= Tn || (seg && seg[t] != ss)) continue;
        acc += w[c * K + j] * dy[(long long)t * C + c];
    }
    dx[i] = acc;
}

// ── Рекуррентность: прямой проход ───────────────────────────────────────────
// q, k [T, Hk, DK]; v [T, Hv, dv]; g, beta [T, Hv] F32. o [T, Hv, dv] F32,
// ckpt [Hv, ⌈T/CKPT⌉, DK, dv] — состояние перед шагом t = n·CKPT.
// Сетка (dv/CS, Hv), блок 128 нитей.
extern "C" __global__ void __launch_bounds__(128) gdn_fwd(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta, const int* __restrict__ seg,
    float* __restrict__ o, float* __restrict__ ckpt, int Tn, int Hk, int Hv, int dv) {
    __shared__ float ks[TC][DK], qs[TC][DK], vs[TC][CS], fs[TC], bs[TC];
    __shared__ float red[2][4][CS];
    const int hv = blockIdx.y, hk = hv / (Hv / Hk);
    const int c0 = blockIdx.x * CS;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nck = (Tn + CKPT - 1) / CKPT;
    float s[32];
#pragma unroll
    for (int i = 0; i < 32; i++) s[i] = 0.f;

    for (int t0 = 0; t0 < Tn; t0 += TC) {
        __syncthreads();
        for (int i = threadIdx.x; i < TC * DK; i += blockDim.x) {
            int tt = i / DK, r = i % DK, t = t0 + tt;
            ks[tt][r] = t < Tn ? k[((long long)t * Hk + hk) * DK + r] : 0.f;
            qs[tt][r] = t < Tn ? q[((long long)t * Hk + hk) * DK + r] : 0.f;
        }
        for (int i = threadIdx.x; i < TC * CS; i += blockDim.x) {
            int tt = i / CS, cc = i % CS, t = t0 + tt;
            vs[tt][cc] = (t < Tn && c0 + cc < dv) ? v[((long long)t * Hv + hv) * dv + c0 + cc] : 0.f;
        }
        for (int tt = threadIdx.x; tt < TC; tt += blockDim.x) {
            int t = t0 + tt;
            fs[tt] = t < Tn ? (seg_start(seg, t) ? 0.f : __expf(g[(long long)t * Hv + hv])) : 1.f;
            bs[tt] = t < Tn ? beta[(long long)t * Hv + hv] : 0.f;
        }
        __syncthreads();
        for (int tt = 0; tt < TC; tt++) {
            int t = t0 + tt;
            if (t >= Tn) break;
            if (t % CKPT == 0) {
                float* cp = ckpt + (((long long)hv * nck + t / CKPT) * DK + w * 32) * dv + c0 + lane;
#pragma unroll
                for (int i = 0; i < 32; i++) cp[(long long)i * dv] = s[i];
            }
            float f = fs[tt];
            float pk = 0.f;
#pragma unroll
            for (int i = 0; i < 32; i++) {
                s[i] *= f;
                pk += s[i] * ks[tt][w * 32 + i];
            }
            red[0][w][lane] = pk;
            __syncthreads();
            float kv = red[0][0][lane] + red[0][1][lane] + red[0][2][lane] + red[0][3][lane];
            float u = bs[tt] * (vs[tt][lane] - kv);
            float po = 0.f;
#pragma unroll
            for (int i = 0; i < 32; i++) {
                s[i] += ks[tt][w * 32 + i] * u;
                po += s[i] * qs[tt][w * 32 + i];
            }
            red[1][w][lane] = po;
            __syncthreads();
            if (w == 0 && c0 + lane < dv)
                o[((long long)t * Hv + hv) * dv + c0 + lane] = red[1][0][lane] + red[1][1][lane] + red[1][2][lane] + red[1][3][lane];
        }
    }
}

// Сумма 32 значений по 32 нитям варпа: нить i получает Σ_lanes x[i]
// (31 обмен вместо 32·5).
__device__ __forceinline__ float warp_transpose_sum(float (&x)[32], int lane) {
#pragma unroll
    for (int step = 16; step >= 1; step >>= 1) {
        bool upper = (lane & step) != 0;
#pragma unroll
        for (int i = 0; i < step; i++) {
            // Пара (i, i+step): нижняя половина нитей оставляет i, верхняя — i+step.
            float send = upper ? x[i] : x[i + step];
            float keep = upper ? x[i + step] : x[i];
            x[i] = keep + __shfl_xor_sync(0xffffffffu, send, step);
        }
    }
    return x[0];
}

// ── Рекуррентность: обратный проход ─────────────────────────────────────────
// do_ [T, Hv, dv] F32. Выходы — частичные суммы по блокам столбцов
// (P = dv/CS): dq_p, dk_p [P, T, Hv, DK], dg_p, db_p [P, T, Hv]; dv_out
// [T, Hv, dv] — целиком. scratch [P·Hv, CKPT, DK, CS] — состояния куска.
extern "C" __global__ void __launch_bounds__(128) gdn_bwd(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta, const int* __restrict__ seg,
    const float* __restrict__ ckpt, const float* __restrict__ dout,
    float* __restrict__ dq_p, float* __restrict__ dk_p, float* __restrict__ dg_p, float* __restrict__ db_p,
    float* __restrict__ dv_out, float* __restrict__ scratch, int Tn, int Hk, int Hv, int dv) {
    __shared__ float red[4][CS];
    __shared__ float red2[4][CS];
    __shared__ float kk[DK], qq[DK];
    __shared__ float sc[2];
    const int hv = blockIdx.y, hk = hv / (Hv / Hk);
    const int P = gridDim.x, p = blockIdx.x;
    const int c0 = p * CS;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c = c0 + lane;
    const bool col_ok = c < dv;
    const int nck = (Tn + CKPT - 1) / CKPT;
    float* scr = scratch + ((long long)(p * Hv + hv) * CKPT) * DK * CS;

    float ds[32];
#pragma unroll
    for (int i = 0; i < 32; i++) ds[i] = 0.f;

    for (int ch = nck - 1; ch >= 0; ch--) {
        const int t0 = ch * CKPT;
        const int t1 = min(Tn, t0 + CKPT);
        // 1. Состояния перед каждым шагом куска — вперёд от контрольной точки.
        float s[32];
        {
            const float* cp = ckpt + (((long long)hv * nck + ch) * DK + w * 32) * dv + c;
#pragma unroll
            for (int i = 0; i < 32; i++) s[i] = col_ok ? cp[(long long)i * dv] : 0.f;
        }
        for (int t = t0; t < t1; t++) {
            float* sp = scr + ((long long)(t - t0) * DK + w * 32) * CS + lane;
#pragma unroll
            for (int i = 0; i < 32; i++) sp[i * CS] = s[i];
            __syncthreads();
            if (threadIdx.x < DK) kk[threadIdx.x] = k[((long long)t * Hk + hk) * DK + threadIdx.x];
            if (threadIdx.x == 0) {
                sc[0] = seg_start(seg, t) ? 0.f : __expf(g[(long long)t * Hv + hv]);
                sc[1] = beta[(long long)t * Hv + hv];
            }
            __syncthreads();
            float f = sc[0];
            float pk = 0.f;
#pragma unroll
            for (int i = 0; i < 32; i++) {
                s[i] *= f;
                pk += s[i] * kk[w * 32 + i];
            }
            red[w][lane] = pk;
            __syncthreads();
            float kv = red[0][lane] + red[1][lane] + red[2][lane] + red[3][lane];
            float vv = col_ok ? v[((long long)t * Hv + hv) * dv + c] : 0.f;
            float u = sc[1] * (vv - kv);
#pragma unroll
            for (int i = 0; i < 32; i++) s[i] += kk[w * 32 + i] * u;
        }
        // 2. Назад по шагам куска.
        for (int t = t1 - 1; t >= t0; t--) {
            __syncthreads();
            if (threadIdx.x < DK) {
                kk[threadIdx.x] = k[((long long)t * Hk + hk) * DK + threadIdx.x];
                qq[threadIdx.x] = q[((long long)t * Hk + hk) * DK + threadIdx.x];
            }
            if (threadIdx.x == 0) {
                sc[0] = seg_start(seg, t) ? 0.f : __expf(g[(long long)t * Hv + hv]);
                sc[1] = beta[(long long)t * Hv + hv];
            }
            __syncthreads();
            const float f = sc[0], b = sc[1];
            float a[32];
            {
                const float* sp = scr + ((long long)(t - t0) * DK + w * 32) * CS + lane;
#pragma unroll
                for (int i = 0; i < 32; i++) a[i] = f * sp[i * CS];
            }
            float pk = 0.f;
#pragma unroll
            for (int i = 0; i < 32; i++) pk += a[i] * kk[w * 32 + i];
            red[w][lane] = pk;
            __syncthreads();
            const float kv = red[0][lane] + red[1][lane] + red[2][lane] + red[3][lane];
            const float vv = col_ok ? v[((long long)t * Hv + hv) * dv + c] : 0.f;
            const float u = b * (vv - kv);
            const float dov = col_ok ? dout[((long long)t * Hv + hv) * dv + c] : 0.f;

            // dq[r] = Σ_c S_t[r, c] · do[c],  S_t = A + k·uᵀ
            float tmp[32];
#pragma unroll
            for (int i = 0; i < 32; i++) tmp[i] = (a[i] + kk[w * 32 + i] * u) * dov;
            float dqr = warp_transpose_sum(tmp, lane);
            dq_p[(((long long)p * Tn + t) * Hv + hv) * DK + w * 32 + lane] = dqr;

            // dS += q·doᵀ;  du[c] = Σ_r dS[r, c] · k[r]
            float pdu = 0.f;
#pragma unroll
            for (int i = 0; i < 32; i++) {
                ds[i] += qq[w * 32 + i] * dov;
                pdu += ds[i] * kk[w * 32 + i];
            }
            red2[w][lane] = pdu;
            __syncthreads();
            const float du = red2[0][lane] + red2[1][lane] + red2[2][lane] + red2[3][lane];
            const float dkv = -b * du;
            if (w == 0) {
                if (col_ok) dv_out[((long long)t * Hv + hv) * dv + c] = b * du;
                float dbp = col_ok ? du * (vv - kv) : 0.f;
                for (int o2 = 16; o2 > 0; o2 >>= 1) dbp += __shfl_xor_sync(0xffffffffu, dbp, o2);
                if (lane == 0) db_p[((long long)p * Tn + t) * Hv + hv] = dbp;
            }
            // dk[r] = Σ_c (dS[r, c]·u[c] + A[r, c]·dkv[c]);  dA = dS + k·dkvᵀ;
            // dg = Σ A∘dA;  dS_{t−1} = f·dA.
            float pg = 0.f;
#pragma unroll
            for (int i = 0; i < 32; i++) {
                tmp[i] = ds[i] * u + a[i] * dkv;
                float da = ds[i] + kk[w * 32 + i] * dkv;
                pg += a[i] * da;
                ds[i] = f * da;
            }
            float dkr = warp_transpose_sum(tmp, lane);
            dk_p[(((long long)p * Tn + t) * Hv + hv) * DK + w * 32 + lane] = dkr;
            for (int o2 = 16; o2 > 0; o2 >>= 1) pg += __shfl_xor_sync(0xffffffffu, pg, o2);
            __syncthreads();
            if (lane == 0) red[w][0] = pg;
            __syncthreads();
            if (threadIdx.x == 0) dg_p[((long long)p * Tn + t) * Hv + hv] = red[0][0] + red[1][0] + red[2][0] + red[3][0];
        }
    }
}

extern "C" {
__global__ void gdn_conv_fwd_f32(const float* x, const float* w, const int* seg, float* y, int T, int C, int K) { conv_fwd(x, w, seg, y, T, C, K); }
__global__ void gdn_conv_fwd_bf16(const __nv_bfloat16* x, const float* w, const int* seg, float* y, int T, int C, int K) { conv_fwd(x, w, seg, y, T, C, K); }
}
