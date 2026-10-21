// Ядра обучения (LoRA / QLoRA поверх замороженной базы).
//
// Все обратные проходы считают в f32: градиенты по остаточному потоку
// копятся в f32, а bf16 остаются только входы, сохранённые прямым проходом.
// Типы входов задаются шаблоном, наружу торчат extern "C" обёртки с
// суффиксом типа (_f32 / _bf16 / _f16).

#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float ld(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float ld(const __nv_bfloat16* p, long long i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ float ld(const __half* p, long long i) { return __half2float(p[i]); }

__device__ __forceinline__ void st(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void st(__nv_bfloat16* p, long long i, float v) { p[i] = __float2bfloat16(v); }
__device__ __forceinline__ void st(__half* p, long long i, float v) { p[i] = __float2half(v); }

// Сумма по блоку (blockDim.x кратен 32, до 1024 нитей). `sh` — 32 float.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    float r = sh[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float block_max(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : -3.402823466e38f;
    if (wid == 0) {
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    float r = sh[0];
    __syncthreads();
    return r;
}

// ── RMSNorm backward ─────────────────────────────────────────────────────────
// y = x · r · w, r = 1/sqrt(mean(x²) + eps). Вес заморожен — нужен только dx:
//   g = dy · w,  dx = r · g − x · r³ · Σ(g·x) / H.
// Блок на строку.
template <typename T>
__device__ void rms_norm_bwd(const T* __restrict__ x, const T* __restrict__ w,
                             const float* __restrict__ dy, float* __restrict__ dx,
                             int H, float eps) {
    __shared__ float sh[32];
    long long row = blockIdx.x;
    const T* xr = x + row * H;
    const float* dyr = dy + row * H;
    float* dxr = dx + row * H;
    float sxx = 0.f, sgx = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float xv = ld(xr, i);
        float g = dyr[i] * ld(w, i);
        sxx += xv * xv;
        sgx += g * xv;
    }
    sxx = block_sum(sxx, sh);
    sgx = block_sum(sgx, sh);
    float r = rsqrtf(sxx / (float)H + eps);
    float c = r * r * r * sgx / (float)H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float xv = ld(xr, i);
        float g = dyr[i] * ld(w, i);
        dxr[i] = r * g - xv * c;
    }
}

// ── silu(gate)·up backward ───────────────────────────────────────────────────
template <typename T>
__device__ void silu_mul_bwd(const T* __restrict__ gate, const T* __restrict__ up,
                             const float* __restrict__ dy, float* __restrict__ dgate,
                             float* __restrict__ dup, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = ld(gate, i), u = ld(up, i), d = dy[i];
    float s = 1.0f / (1.0f + __expf(-g));
    float silu = g * s;
    dgate[i] = d * u * s * (1.0f + g * (1.0f - s));
    dup[i] = d * silu;
}

// ── Cross-entropy: прямой и обратный проход разом ───────────────────────────
// Строка логитов [V]; label < 0 — позиция без лосса (нулевой градиент).
// loss[row] = lse − x[label];  dlogits = (softmax − onehot) · scale.
template <typename T, typename G>
__device__ void ce_fwd_bwd(const T* __restrict__ logits, const int* __restrict__ labels,
                           float* __restrict__ loss, G* __restrict__ dlogits,
                           int V, float scale) {
    __shared__ float sh[32];
    long long row = blockIdx.x;
    const T* xr = logits + row * V;
    G* gr = dlogits + row * V;
    int label = labels[row];
    if (label < 0 || label >= V) {
        for (int i = threadIdx.x; i < V; i += blockDim.x) st(gr, i, 0.0f);
        if (threadIdx.x == 0) loss[row] = 0.0f;
        return;
    }
    float m = -3.402823466e38f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) m = fmaxf(m, ld(xr, i));
    m = block_max(m, sh);
    float s = 0.f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) s += __expf(ld(xr, i) - m);
    s = block_sum(s, sh);
    float inv = 1.0f / s;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float p = __expf(ld(xr, i) - m) * inv;
        if (i == label) p -= 1.0f;
        st(gr, i, p * scale);
    }
    if (threadIdx.x == 0) loss[row] = (m + __logf(s)) - ld(xr, label);
}

// ── Каузальный softmax по строкам скоров внимания ───────────────────────────
// scores [B, T, T] f32 (B = головы). Строка i видит ключи j ≤ i с тем же
// номером примера seg (упаковка нескольких примеров в одну
// последовательность). Замаскированные вероятности — ровно 0.
extern "C" __global__ void causal_softmax_f32(const float* __restrict__ scores, const int* __restrict__ seg,
                                   float* __restrict__ probs, int T, float scale, int has_seg) {
    __shared__ float sh[32];
    long long row = blockIdx.x;          // b * T + i
    int i = (int)(row % T);
    const float* sr = scores + row * T;
    float* pr = probs + row * T;
    int si = has_seg ? seg[i] : 0;
    float m = -3.402823466e38f;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        if (has_seg && seg[j] != si) continue;
        m = fmaxf(m, sr[j] * scale);
    }
    m = block_max(m, sh);
    float s = 0.f;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        if (has_seg && seg[j] != si) continue;
        s += __expf(sr[j] * scale - m);
    }
    s = block_sum(s, sh);
    float inv = 1.0f / s;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float p = 0.0f;
        if (j <= i && (!has_seg || seg[j] == si)) p = __expf(sr[j] * scale - m) * inv;
        pr[j] = p;
    }
}

// ds = scale · p · (dp − Σ dp·p) по строкам длины L.
extern "C" __global__ void softmax_bwd_f32(const float* __restrict__ p, const float* __restrict__ dp,
                                float* __restrict__ ds, int L, float scale) {
    __shared__ float sh[32];
    long long row = blockIdx.x;
    const float* pr = p + row * L;
    const float* dr = dp + row * L;
    float* outr = ds + row * L;
    float acc = 0.f;
    for (int j = threadIdx.x; j < L; j += blockDim.x) acc += pr[j] * dr[j];
    acc = block_sum(acc, sh);
    for (int j = threadIdx.x; j < L; j += blockDim.x) outr[j] = scale * pr[j] * (dr[j] - acc);
}

// ── NVFP4: деквант из перемешанной раскладки ────────────────────────────────
// Перемешанная копия (`nvfp4_w_repack`): тайлы 16 строк × 64 значения по 512
// байт подряд, внутри — строка за строкой по 32 байта. Шкалы — в обычной
// тайл-мажорной раскладке cuBLASLt (как у линейного packed).
__device__ __forceinline__ float dec_e4m3(unsigned char byte) {
    bool sign = (byte & 0x80) != 0;
    int e = (byte >> 3) & 0x0F;
    int mnt = byte & 0x07;
    if (e == 15 && mnt == 7) return __int_as_float(0x7FC00000);
    float v = (e == 0) ? mnt * 0.001953125f : (1.0f + mnt * 0.125f) * exp2f((float)(e - 7));
    return sign ? -v : v;
}

__device__ __forceinline__ float dec_e2m1(unsigned char nib) {
    const float tbl[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float v = tbl[nib & 7u];
    return (nib & 8u) ? -v : v;
}

__device__ __forceinline__ unsigned int sf_off(unsigned int outer, unsigned int bc, unsigned int sf_inner) {
    unsigned int tile_base = ((bc >> 2) * 4u + (outer >> 7) * sf_inner) * 128u;
    unsigned int lo = outer & 127u;
    return tile_base + (lo & 31u) * 16u + (lo >> 5) * 4u + (bc & 3u);
}

// Одна нить — один байт (два значения). Выход bf16 [N, K].
extern "C" __global__ void nvfp4_dequant_shuf_bf16(const unsigned char* __restrict__ shuf,
                                        const unsigned char* __restrict__ scales,
                                        __nv_bfloat16* __restrict__ out,
                                        unsigned int N, unsigned int K) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)N * (K >> 1);
    if (idx >= total) return;
    unsigned int row = (unsigned int)(idx / (K >> 1));
    unsigned int kb = (unsigned int)(idx % (K >> 1));
    unsigned int src = (row >> 4) * (K >> 6) * 512u + (kb >> 5) * 512u + (row & 15u) * 32u + (kb & 31u);
    unsigned char byte = shuf[src];
    unsigned int col = kb * 2u;
    unsigned int sf_inner = ((K + 63u) / 64u) * 4u;
    float sc = dec_e4m3(scales[sf_off(row, col >> 4, sf_inner)]);
    __nv_bfloat162 v;
    v.x = __float2bfloat16(dec_e2m1(byte & 0x0F) * sc);
    v.y = __float2bfloat16(dec_e2m1(byte >> 4) * sc);
    *reinterpret_cast<__nv_bfloat162*>(out + (unsigned long long)row * K + col) = v;
}

// ── AdamW (f32) ──────────────────────────────────────────────────────────────
// Новые p, m, v пишутся в отдельные буферы (тензоры неизменяемы).
extern "C" __global__ void adamw_f32(const float* __restrict__ p, const float* __restrict__ g,
                          const float* __restrict__ m, const float* __restrict__ v,
                          float* __restrict__ p_out, float* __restrict__ m_out, float* __restrict__ v_out,
                          long long n, float lr, float b1, float b2, float eps, float wd,
                          float bc1, float bc2) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = b1 * m[i] + (1.0f - b1) * gi;
    float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
    float mh = mi / bc1, vh = vi / bc2;
    float pi = p[i];
    p_out[i] = pi - lr * (mh / (sqrtf(vh) + eps) + wd * pi);
    m_out[i] = mi;
    v_out[i] = vi;
}

extern "C" {

__global__ void rms_norm_bwd_f32(const float* x, const float* w, const float* dy, float* dx, int H, float eps) { rms_norm_bwd(x, w, dy, dx, H, eps); }
__global__ void rms_norm_bwd_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w, const float* dy, float* dx, int H, float eps) { rms_norm_bwd(x, w, dy, dx, H, eps); }
__global__ void rms_norm_bwd_f16(const __half* x, const __half* w, const float* dy, float* dx, int H, float eps) { rms_norm_bwd(x, w, dy, dx, H, eps); }

__global__ void silu_mul_bwd_f32(const float* g, const float* u, const float* dy, float* dg, float* du, long long n) { silu_mul_bwd(g, u, dy, dg, du, n); }
__global__ void silu_mul_bwd_bf16(const __nv_bfloat16* g, const __nv_bfloat16* u, const float* dy, float* dg, float* du, long long n) { silu_mul_bwd(g, u, dy, dg, du, n); }
__global__ void silu_mul_bwd_f16(const __half* g, const __half* u, const float* dy, float* dg, float* du, long long n) { silu_mul_bwd(g, u, dy, dg, du, n); }

__global__ void ce_fwd_bwd_f32_f32(const float* x, const int* l, float* loss, float* g, int V, float s) { ce_fwd_bwd(x, l, loss, g, V, s); }
__global__ void ce_fwd_bwd_f32_bf16(const float* x, const int* l, float* loss, __nv_bfloat16* g, int V, float s) { ce_fwd_bwd(x, l, loss, g, V, s); }
__global__ void ce_fwd_bwd_bf16_f32(const __nv_bfloat16* x, const int* l, float* loss, float* g, int V, float s) { ce_fwd_bwd(x, l, loss, g, V, s); }
__global__ void ce_fwd_bwd_bf16_bf16(const __nv_bfloat16* x, const int* l, float* loss, __nv_bfloat16* g, int V, float s) { ce_fwd_bwd(x, l, loss, g, V, s); }
__global__ void ce_fwd_bwd_f16_f32(const __half* x, const int* l, float* loss, float* g, int V, float s) { ce_fwd_bwd(x, l, loss, g, V, s); }
__global__ void ce_fwd_bwd_f16_bf16(const __half* x, const int* l, float* loss, __nv_bfloat16* g, int V, float s) { ce_fwd_bwd(x, l, loss, g, V, s); }
}
