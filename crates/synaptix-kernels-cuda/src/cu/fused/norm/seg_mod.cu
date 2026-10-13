#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Слитые ядра DiT с сегментной модуляцией (MiniMax-H3) и qk-нормой.
//
// gate_residual_seg: out = x + round_T(y*g), g — строка таблицы gate[R, D] по
//   ridx[row]. Арифметика = FMA_ROWB_KERNEL (elementwise.cu) бит-в-бит; один
//   запуск на весь [m, D] вместо нарезки по сегментам с копиями туда-обратно.
//
// qk_norm_rope: голова [D] (D % 32 == 0, D ≤ 256) типа T →
//   n = (w·x)·rms (F32, как rms_norm_f32 c блоком 256: тот же шаффл-порядок
//   редукции по варпам) → RoPE split-partial (как rope_split_partial_f32) →
//   round_T. Вместо cast→rms_norm_f32→rope_f32→cast (4 прохода F32/T).
//   Один варп на строку-голову; позиция = pos_off + row / heads.

__device__ __forceinline__ float qk_ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float qk_ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void qk_st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void qk_st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T, int VEC_N>
__device__ __forceinline__ void gate_residual_seg_impl(
    const T* __restrict__ x, const T* __restrict__ y, const T* __restrict__ g,
    const int* __restrict__ ridx, T* __restrict__ out, long long numel, int d)
{
    long long i = ((long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x) * VEC_N;
    if (i >= numel) return;
    long long row = i / d;
    int j = (int)(i - row * d);
    const T* g_row = g + (long long)ridx[row] * d;
    if (i + VEC_N <= numel && j + VEC_N <= d) {
        uint4 xv = *reinterpret_cast<const uint4*>(x + i);
        uint4 yv = *reinterpret_cast<const uint4*>(y + i);
        uint4 gv = *reinterpret_cast<const uint4*>(g_row + j);
        T xe[VEC_N], ye[VEC_N], ge[VEC_N], oe[VEC_N];
        *reinterpret_cast<uint4*>(xe) = xv;
        *reinterpret_cast<uint4*>(ye) = yv;
        *reinterpret_cast<uint4*>(ge) = gv;
        #pragma unroll
        for (int k = 0; k < VEC_N; ++k) {
            float yg = qk_ld(&ye[k]) * qk_ld(&ge[k]);
            T ygt; qk_st(&ygt, yg);
            float o = qk_ld(&xe[k]) + qk_ld(&ygt);
            qk_st(&oe[k], o);
        }
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<uint4*>(oe);
    } else {
        for (int k = 0; k < VEC_N && i + k < numel; ++k) {
            long long ii = i + k;
            long long r = ii / d;
            int jj = (int)(ii - r * d);
            float yg = qk_ld(y + ii) * qk_ld(g + (long long)ridx[r] * d + jj);
            T ygt; qk_st(&ygt, yg);
            qk_st(out + ii, qk_ld(x + ii) + qk_ld(&ygt));
        }
    }
}

extern "C" __global__ void gate_residual_seg_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* y, const __nv_bfloat16* g, const int* ridx,
    __nv_bfloat16* out, long long numel, int d)
{ gate_residual_seg_impl<__nv_bfloat16, 8>(x, y, g, ridx, out, numel, d); }

extern "C" __global__ void gate_residual_seg_f16(
    const __half* x, const __half* y, const __half* g, const int* ridx,
    __half* out, long long numel, int d)
{ gate_residual_seg_impl<__half, 8>(x, y, g, ridx, out, numel, d); }

// Один варп = одна строка-голова [D]. Блок = 8 варпов.
template <typename T>
__device__ __forceinline__ void qk_norm_rope_impl(
    const T* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    T* __restrict__ out, long long rows, int heads, int d, int rot_dim,
    int pos_off, float eps)
{
    __shared__ float s_n[8][256];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    long long row = (long long)blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const T* xr = x + row * d;
    int nw = d >> 5; // варпов исходного rms_norm_f32 с данными (блок 256)

    // Редукция = row_sumsq(rms_norm.cu) при блоке 256: варп k держит
    // элементы 32k..32k+31 (по одному на нить), шаффл-дерево внутри варпа,
    // затем шаффл-дерево по 8 частичным суммам (пустые варпы — нули).
    float xv[8];
    float part[8];
    unsigned int mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float v = (k < nw) ? qk_ld(xr + k * 32 + lane) : 0.0f;
        xv[k] = v;
        float local = v * v;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) local += __shfl_down_sync(mask, local, off, 32);
        part[k] = local; // значение lane 0 — сумма варпа k
    }
    float v = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float pk = __shfl_sync(mask, part[k], 0, 32);
        if (lane == k) v = pk;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off, 32);
    float sumsq = __shfl_sync(mask, v, 0, 32);
    float mean = sumsq / (float)d;
    float rms = rsqrtf(mean + eps);

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k < nw) {
            int t = k * 32 + lane;
            float wv = w[t];
            s_n[warp][t] = wv * xv[k] * rms; // как rms_norm_impl: (scale*xv)*rms
        }
    }
    __syncwarp();

    int s = pos_off + (int)(row / heads);
    int half = rot_dim >> 1;
    T* orow = out + row * d;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k < nw) {
            int t = k * 32 + lane;
            float result;
            if (t >= rot_dim) {
                result = s_n[warp][t];
            } else {
                int idx = (t < half) ? t : (t - half);
                float c = cos_t[(size_t)s * half + idx];
                float sn = sin_t[(size_t)s * half + idx];
                float xn = s_n[warp][t];
                if (t < half) {
                    float xp = s_n[warp][t + half];
                    result = xn * c - xp * sn;
                } else {
                    float xp = s_n[warp][t - half];
                    result = xn * c + xp * sn;
                }
            }
            qk_st(orow + t, result);
        }
    }
}

extern "C" __global__ void qk_norm_rope_bf16(
    const __nv_bfloat16* x, const float* w, const float* cos_t, const float* sin_t,
    __nv_bfloat16* out, long long rows, int heads, int d, int rot_dim, int pos_off, float eps)
{ qk_norm_rope_impl<__nv_bfloat16>(x, w, cos_t, sin_t, out, rows, heads, d, rot_dim, pos_off, eps); }

extern "C" __global__ void qk_norm_rope_f16(
    const __half* x, const float* w, const float* cos_t, const float* sin_t,
    __half* out, long long rows, int heads, int d, int rot_dim, int pos_off, float eps)
{ qk_norm_rope_impl<__half>(x, w, cos_t, sin_t, out, rows, heads, d, rot_dim, pos_off, eps); }
