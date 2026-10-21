// Flash-attention для обучения (FA2): прямой проход с log-sum-exp строк и
// обратный проход по нему, BF16 на тензорных ядрах (WMMA 16×16×16, F32
// накопление). Каузальная маска с номерами примеров (упаковка): строка i
// видит ключ j, если j ≤ i и seg[j] == seg[i]. Номера примеров не убывают по
// позиции, поэтому блоки ключей чужих примеров пропускаются целиком.
//
// Раскладка: q [H, T, D], k/v [Hkv, T, D] BF16 contiguous, голова h читает
// kv-голову h / (H / Hkv) (GQA без повторения K/V).

#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

#define NEG_BIG (-3.0e38f)

__device__ __forceinline__ bool visible(const int* seg, int i, int j) {
    return j <= i && (seg == nullptr || seg[i] == seg[j]);
}

// Первый ключ, видимый хоть одной строке из [q0, ...): у упаковки — начало
// примера строки q0.
__device__ __forceinline__ int first_key(const int* seg, int q0) {
    if (seg == nullptr) return 0;
    int s = seg[q0];
    int j = q0;
    while (j > 0 && seg[j - 1] == s) j--;
    return j;
}

// Копия тайла rows×D BF16 из глобальной памяти в shared (нули за T).
template <int D>
__device__ __forceinline__ void load_tile(__nv_bfloat16* dst, const __nv_bfloat16* src, int row0, int rows, int T) {
    const int vec = 8;  // 16 байт
    int total = rows * D / vec;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int r = (i * vec) / D, c = (i * vec) % D;
        int4 v = make_int4(0, 0, 0, 0);
        if (row0 + r < T) v = *reinterpret_cast<const int4*>(src + (long long)(row0 + r) * D + c);
        *reinterpret_cast<int4*>(dst + r * D + c) = v;
    }
}

// ── Прямой проход ────────────────────────────────────────────────────────────
// Блок — WARPS·16 строк одной головы; варп — 16 строк. Тайл ключей — BC.
// Shared: K, V [BC, D] BF16; на варп O [16, D] F32 и S [16, BC] F32 (поверх S
// пишется P в BF16). Q варпа — во фрагментах (D/16 штук).
template <int D, int WARPS, int BC>
__device__ void flash_fwd(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                          const __nv_bfloat16* __restrict__ v, const int* __restrict__ seg,
                          __nv_bfloat16* __restrict__ o, float* __restrict__ lse,
                          int T, int H, int Hkv, float scale) {
    extern __shared__ __align__(128) unsigned char smem[];
    __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Vs = Ks + BC * D;
    float* Ow = reinterpret_cast<float*>(Vs + BC * D);  // [WARPS][16][D]
    float* Sw = Ow + WARPS * 16 * D;                    // [WARPS][16][BC]

    const int h = blockIdx.y;
    const int hk = h / (H / Hkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int q0 = blockIdx.x * WARPS * 16;
    const int qw = q0 + warp * 16;
    const __nv_bfloat16* qh = q + (long long)h * T * D;
    const __nv_bfloat16* kh = k + (long long)hk * T * D;
    const __nv_bfloat16* vh = v + (long long)hk * T * D;
    float* O = Ow + warp * 16 * D;
    float* S = Sw + warp * 16 * BC;
    __nv_bfloat16* P = reinterpret_cast<__nv_bfloat16*>(S);

    // Q блока — через буфер K (с нулями за T), затем во фрагменты варпа.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf[D / 16];
    {
        // WARPS·16 ≤ BC не обязательно: грузим по 16 строк на варп в его O.
        __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(O);
        for (int i = lane; i < 16 * D / 8; i += 32) {
            int r = (i * 8) / D, c = (i * 8) % D;
            int4 val = make_int4(0, 0, 0, 0);
            if (qw + r < T) val = *reinterpret_cast<const int4*>(qh + (long long)(qw + r) * D + c);
            *reinterpret_cast<int4*>(qs + r * D + c) = val;
        }
        __syncwarp();
#pragma unroll
        for (int kk = 0; kk < D / 16; kk++) wmma::load_matrix_sync(qf[kk], qs + kk * 16, D);
        __syncwarp();
    }
    for (int i = lane; i < 16 * D; i += 32) O[i] = 0.f;

    // Строка варпа: lane/2, половина ключей тайла — lane%2.
    const int row = lane >> 1, half = lane & 1;
    const int qi = qw + row;
    float m = NEG_BIG, l = 0.f;

    const int q_last = min(q0 + WARPS * 16, T) - 1;
    const int k_begin = (first_key(seg, q0) / BC) * BC;
    for (int k0 = k_begin; k0 <= q_last; k0 += BC) {
        __syncthreads();
        load_tile<D>(Ks, kh, k0, BC, T);
        load_tile<D>(Vs, vh, k0, BC, T);
        __syncthreads();
        if (qw > q_last) continue;  // варп за концом последовательности

        // S = Q·Kᵀ
#pragma unroll
        for (int c = 0; c < BC / 16; c++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
#pragma unroll
            for (int kk = 0; kk < D / 16; kk++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
                wmma::load_matrix_sync(kf, Ks + c * 16 * D + kk * 16, D);
                wmma::mma_sync(acc, qf[kk], kf, acc);
            }
            wmma::store_matrix_sync(S + c * 16, acc, BC, wmma::mem_row_major);
        }
        __syncwarp();

        // Онлайн-softmax: строки держат по две нити.
        float sv[BC / 2];
        float mx = NEG_BIG;
#pragma unroll
        for (int c = 0; c < BC / 2; c++) {
            int j = k0 + half * (BC / 2) + c;
            float s = S[row * BC + half * (BC / 2) + c] * scale;
            bool ok = qi < T && j < T && visible(seg, qi, j);
            sv[c] = ok ? s : __int_as_float(0xff800000);
            mx = fmaxf(mx, ok ? s : NEG_BIG);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        float m_new = fmaxf(m, mx);
        float alpha = __expf(m - m_new);
        float sum = 0.f;
        __syncwarp();
#pragma unroll
        for (int c = 0; c < BC / 2; c++) {
            float p = __expf(sv[c] - m_new);
            sum += p;
            P[row * BC + half * (BC / 2) + c] = __float2bfloat16(p);
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        l = l * alpha + sum;
        m = m_new;
        // O *= alpha (строка — пара нитей, по половине D).
        for (int c = 0; c < D / 2; c++) O[row * D + half * (D / 2) + c] *= alpha;
        __syncwarp();

        // O += P·V
#pragma unroll
        for (int d = 0; d < D / 16; d++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::load_matrix_sync(acc, O + d * 16, D, wmma::mem_row_major);
#pragma unroll
            for (int kk = 0; kk < BC / 16; kk++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
                wmma::load_matrix_sync(pf, P + kk * 16, BC);
                wmma::load_matrix_sync(vf, Vs + kk * 16 * D + d * 16, D);
                wmma::mma_sync(acc, pf, vf, acc);
            }
            wmma::store_matrix_sync(O + d * 16, acc, D, wmma::mem_row_major);
        }
        __syncwarp();
    }

    if (qi < T) {
        float inv = 1.f / l;
        __nv_bfloat16* orow = o + ((long long)h * T + qi) * D;
        for (int c = 0; c < D / 2; c++) {
            int d = half * (D / 2) + c;
            orow[d] = __float2bfloat16(O[row * D + d] * inv);
        }
        if (half == 0) lse[(long long)h * T + qi] = m + __logf(l);
    }
}

// ── Обратный проход ──────────────────────────────────────────────────────────
// Блок — 32 ключа одной kv-головы (2 варпа по 16 ключей), цикл по головам
// группы и тайлам запросов по 32 строки. dK, dV копятся прямо в выходных
// буферах (блок владеет своими строками; буферы обнулены), dQ — атомарно.
//   Pᵀ = exp(scale·K·Qᵀ − lse),  dV += Pᵀ·dO,  dPᵀ = V·dOᵀ,
//   dSᵀ = Pᵀ ∘ (dPᵀ − Δ),  dK += scale·dSᵀ·Q,  dQ += scale·dS·K.
#define BWD_BK 32
#define BWD_BQ 32
template <int D>
__device__ void flash_bwd(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                          const __nv_bfloat16* __restrict__ v, const int* __restrict__ seg,
                          const __nv_bfloat16* __restrict__ dout, const float* __restrict__ lse,
                          const float* __restrict__ delta, float* __restrict__ dq, float* __restrict__ dk,
                          float* __restrict__ dv, int T, int Tp, int H, int Hkv, float scale) {
    extern __shared__ __align__(128) unsigned char smem[];
    __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(smem);  // [BK, D]
    __nv_bfloat16* Vs = Ks + BWD_BK * D;
    __nv_bfloat16* Qs = Vs + BWD_BK * D;                          // [BQ, D]
    __nv_bfloat16* dOs = Qs + BWD_BQ * D;
    float* Ls = reinterpret_cast<float*>(dOs + BWD_BQ * D);        // [BQ]
    float* Ds = Ls + BWD_BQ;                                       // [BQ]
    float* Sw = Ds + BWD_BQ;                                       // [2][16][BQ] F32
    __nv_bfloat16* PTw = reinterpret_cast<__nv_bfloat16*>(Sw + 2 * 16 * BWD_BQ);  // [2][16][BQ]
    __nv_bfloat16* dSTw = PTw + 2 * 16 * BWD_BQ;                                    // [2][16][BQ]
    float* Stw = reinterpret_cast<float*>(dSTw + 2 * 16 * BWD_BQ);                  // [2][16][16]

    const int hk = blockIdx.y;
    const int group = H / Hkv;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int k0 = blockIdx.x * BWD_BK;
    const int kw = warp * 16;  // ключи варпа внутри блока
    float* S = Sw + warp * 16 * BWD_BQ;
    __nv_bfloat16* PT = PTw + warp * 16 * BWD_BQ;
    __nv_bfloat16* dST = dSTw + warp * 16 * BWD_BQ;
    float* St = Stw + warp * 256;
    // dK/dV — [Hkv, Tp, D], Tp кратно 32: wmma пишет строки тайла целиком.
    float* dkh = dk + ((long long)hk * Tp + k0 + kw) * D;
    float* dvh = dv + ((long long)hk * Tp + k0 + kw) * D;
    const bool warp_live = k0 + kw < T;

    load_tile<D>(Ks, k + (long long)hk * T * D, k0, BWD_BK, T);
    load_tile<D>(Vs, v + (long long)hk * T * D, k0, BWD_BK, T);

    // Последняя строка, видящая хоть один ключ блока: конец примера ключа k_last.
    const int k_last = min(k0 + BWD_BK, T) - 1;
    int q_end = T;
    if (seg != nullptr) {
        int s = seg[k_last];
        q_end = k_last + 1;
        while (q_end < T && seg[q_end] == s) q_end++;
    }
    const int q_begin = (k0 / BWD_BQ) * BWD_BQ;

    for (int hh = 0; hh < group; hh++) {
        const int h = hk * group + hh;
        const __nv_bfloat16* qh = q + (long long)h * T * D;
        const __nv_bfloat16* doh = dout + (long long)h * T * D;
        for (int qb = q_begin; qb < q_end; qb += BWD_BQ) {
            __syncthreads();
            load_tile<D>(Qs, qh, qb, BWD_BQ, T);
            load_tile<D>(dOs, doh, qb, BWD_BQ, T);
            for (int i = threadIdx.x; i < BWD_BQ; i += blockDim.x) {
                int qi = qb + i;
                Ls[i] = qi < T ? lse[(long long)h * T + qi] : 0.f;
                Ds[i] = qi < T ? delta[(long long)h * T + qi] : 0.f;
            }
            __syncthreads();
            if (!warp_live) continue;

            // Sᵀ = K_w·Qᵀ [16, BQ]
#pragma unroll
            for (int c = 0; c < BWD_BQ / 16; c++) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::fill_fragment(acc, 0.f);
#pragma unroll
                for (int kk = 0; kk < D / 16; kk++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
                    wmma::load_matrix_sync(af, Ks + kw * D + kk * 16, D);
                    wmma::load_matrix_sync(bf, Qs + c * 16 * D + kk * 16, D);
                    wmma::mma_sync(acc, af, bf, acc);
                }
                wmma::store_matrix_sync(S + c * 16, acc, BWD_BQ, wmma::mem_row_major);
            }
            __syncwarp();
            // Pᵀ: нить — ключ lane/2, половина запросов lane%2.
            const int kr = lane >> 1, half = lane & 1;
            const int kj = k0 + kw + kr;
#pragma unroll
            for (int c = 0; c < BWD_BQ / 2; c++) {
                int col = half * (BWD_BQ / 2) + c;
                int qi = qb + col;
                bool ok = qi < T && kj < T && visible(seg, qi, kj);
                float p = ok ? __expf(S[kr * BWD_BQ + col] * scale - Ls[col]) : 0.f;
                PT[kr * BWD_BQ + col] = __float2bfloat16(p);
                S[kr * BWD_BQ + col] = p;  // F32-копия для dS
            }
            __syncwarp();

            // dV_w += Pᵀ·dO
#pragma unroll
            for (int d = 0; d < D / 16; d++) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::load_matrix_sync(acc, dvh + d * 16, D, wmma::mem_row_major);
#pragma unroll
                for (int kk = 0; kk < BWD_BQ / 16; kk++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
                    wmma::load_matrix_sync(af, PT + kk * 16, BWD_BQ);
                    wmma::load_matrix_sync(bf, dOs + kk * 16 * D + d * 16, D);
                    wmma::mma_sync(acc, af, bf, acc);
                }
                wmma::store_matrix_sync(dvh + d * 16, acc, D, wmma::mem_row_major);
            }

            // dPᵀ = V_w·dOᵀ [16, BQ] → dSᵀ = Pᵀ ∘ (dPᵀ − Δ)
#pragma unroll
            for (int c = 0; c < BWD_BQ / 16; c++) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::fill_fragment(acc, 0.f);
#pragma unroll
                for (int kk = 0; kk < D / 16; kk++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
                    wmma::load_matrix_sync(af, Vs + kw * D + kk * 16, D);
                    wmma::load_matrix_sync(bf, dOs + c * 16 * D + kk * 16, D);
                    wmma::mma_sync(acc, af, bf, acc);
                }
                wmma::store_matrix_sync(St, acc, 16, wmma::mem_row_major);
                __syncwarp();
                for (int i = lane; i < 256; i += 32) {
                    int r = i / 16, cc = i % 16;
                    int col = c * 16 + cc;
                    float p = S[r * BWD_BQ + col];
                    dST[r * BWD_BQ + col] = __float2bfloat16(p * (St[i] - Ds[col]) * scale);
                }
                __syncwarp();
            }

            // dK_w += dSᵀ·Q
#pragma unroll
            for (int d = 0; d < D / 16; d++) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::load_matrix_sync(acc, dkh + d * 16, D, wmma::mem_row_major);
#pragma unroll
                for (int kk = 0; kk < BWD_BQ / 16; kk++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
                    wmma::load_matrix_sync(af, dST + kk * 16, BWD_BQ);
                    wmma::load_matrix_sync(bf, Qs + kk * 16 * D + d * 16, D);
                    wmma::mma_sync(acc, af, bf, acc);
                }
                wmma::store_matrix_sync(dkh + d * 16, acc, D, wmma::mem_row_major);
            }

            // dQ += dS·K_w: dS = (dSᵀ)ᵀ — тот же буфер как col_major.
#pragma unroll
            for (int c = 0; c < BWD_BQ / 16; c++) {
#pragma unroll
                for (int d = 0; d < D / 16; d++) {
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                    wmma::fill_fragment(acc, 0.f);
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> af;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
                    wmma::load_matrix_sync(af, dST + c * 16, BWD_BQ);
                    wmma::load_matrix_sync(bf, Ks + kw * D + d * 16, D);
                    wmma::mma_sync(acc, af, bf, acc);
                    wmma::store_matrix_sync(St, acc, 16, wmma::mem_row_major);
                    __syncwarp();
                    for (int i = lane; i < 256; i += 32) {
                        int r = i / 16, cc = i % 16;
                        int qi = qb + c * 16 + r;
                        if (qi < T) atomicAdd(dq + ((long long)h * T + qi) * D + d * 16 + cc, St[i]);
                    }
                    __syncwarp();
                }
            }
        }
    }
}

#define FWD_INST(D, W, BC)                                                                                    \
    extern "C" __global__ void __launch_bounds__(W * 32) flash_train_fwd_d##D(                               \
        const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const int* seg,              \
        __nv_bfloat16* o, float* lse, int T, int H, int Hkv, float scale) {                                  \
        flash_fwd<D, W, BC>(q, k, v, seg, o, lse, T, H, Hkv, scale);                                         \
    }
#define BWD_INST(D)                                                                                           \
    extern "C" __global__ void __launch_bounds__(64) flash_train_bwd_d##D(                                   \
        const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, const int* seg,              \
        const __nv_bfloat16* dout, const float* lse, const float* delta, float* dq, float* dk, float* dv,    \
        int T, int Tp, int H, int Hkv, float scale) {                                                        \
        flash_bwd<D>(q, k, v, seg, dout, lse, delta, dq, dk, dv, T, Tp, H, Hkv, scale);                          \
    }

FWD_INST(64, 4, 64)
FWD_INST(128, 4, 64)
FWD_INST(256, 2, 32)
BWD_INST(64)
BWD_INST(128)
BWD_INST(256)
