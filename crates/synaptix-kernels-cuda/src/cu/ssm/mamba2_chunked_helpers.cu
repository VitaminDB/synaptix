#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Mamba2 chunked-SSD helpers. Все kernels — light-weight permute/cast/cumsum/
// elementwise + один transpose. Workspace для multi-kernel pipeline по плану
// plan/mamba2_chunked_stage2_handover.md.
//
// Соглашения:
//   B  — batch, H — heads, L = T·Q — длина, T — число chunks, Q — chunk size,
//   P  — head_dim, N — d_state, BH = B·H.
//   Внутри chunked pipeline всё реорганизовано в **(T, BH, *) row-major** —
//   chunk-axis самый внешний, чтобы per-chunk slice одной операцией
//   `buf[c*per_chunk..(c+1)*per_chunk]` давал contiguous view для всех BH.
//
// NVRTC: математика — встроенные __device__ intrinsics (expf/__expf).
// `-1e30f` sentinel НЕ используется (не нужны).

// ─── Загрузка/выгрузка с приведением типа ───────────────────────────────────
__device__ __forceinline__ float load_f(const float* p) { return *p; }
__device__ __forceinline__ float load_f(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_f(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void store_f(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_f(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_f(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// ─── 1. alpha_cum_f32 ───────────────────────────────────────────────────────
// Per (t, bh): cumsum_j(A[h] * dt[b, t*Q+j, h]).
// Input:  dt [B, L, H] dtype-вход, A [H] dtype-вход.
// Output: alpha_cum [T, BH, Q] f32.  ← chunk-outermost.
// Один block = (bh, t), threads = Q (≤ 1024, степень 2 — не требуется, любое Q).
template <typename T>
__device__ __forceinline__ void alpha_cum_impl(
    const T* __restrict__ dt,
    const T* __restrict__ A,
    float*    __restrict__ alpha_cum,
    int B, int H, int T_, int Q)
{
    int bh = blockIdx.x;
    int t  = blockIdx.y;
    int j  = threadIdx.x;
    if (bh >= B * H || t >= T_ || j >= Q) return;

    int b = bh / H;
    int h = bh % H;
    int l = t * Q + j;

    float a_h  = load_f(A + h);
    float dt_j = load_f(dt + ((size_t)b * (size_t)(T_ * Q) + l) * (size_t)H + h);
    float step = a_h * dt_j;

    // Plain sequential scan в shared (Q ≤ 1024, типично 16/32/64 — overhead мал).
    extern __shared__ float sh_alpha_cum[];
    sh_alpha_cum[j] = step;
    __syncthreads();
    if (j == 0) {
        float acc = 0.0f;
        for (int k = 0; k < Q; ++k) {
            acc += sh_alpha_cum[k];
            sh_alpha_cum[k] = acc;
        }
    }
    __syncthreads();
    // Layout (T, BH, Q): index = (t * BH + bh) * Q + j.
    alpha_cum[((size_t)t * (B * H) + bh) * Q + j] = sh_alpha_cum[j];
}

extern "C" __global__ void mamba2_alpha_cum_f32_in(
    const float* dt, const float* A, float* alpha_cum, int B, int H, int T_, int Q)
{ alpha_cum_impl<float>(dt, A, alpha_cum, B, H, T_, Q); }

extern "C" __global__ void mamba2_alpha_cum_f16_in(
    const __half* dt, const __half* A, float* alpha_cum, int B, int H, int T_, int Q)
{ alpha_cum_impl<__half>(dt, A, alpha_cum, B, H, T_, Q); }

extern "C" __global__ void mamba2_alpha_cum_bf16_in(
    const __nv_bfloat16* dt, const __nv_bfloat16* A, float* alpha_cum,
    int B, int H, int T_, int Q)
{ alpha_cum_impl<__nv_bfloat16>(dt, A, alpha_cum, B, H, T_, Q); }

// ─── 2. permute_blhx_to_tbhqx (f32/bf16/f16 → bf16) ────────────────────────
// Input:  src [B, L, H, X] dtype-вход (L = T*Q).
// Output: dst [T, BH, Q, X] bf16.  ← chunk-outermost.
template <typename Tin>
__device__ __forceinline__ void permute_blhx_impl(
    const Tin* __restrict__ src, __nv_bfloat16* __restrict__ dst,
    int B, int L, int H, int X, int Q)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blh   = blockIdx.z * gridDim.y + blockIdx.y;  // split: blh может быть > 65535.
    if (x_idx >= X) return;
    if (blh >= B * L * H) return;

    int h = blh % H;
    int l = (blh / H) % L;
    int b = blh / (L * H);
    int t = l / Q;
    int q = l % Q;
    int bh = b * H + h;
    int BH = B * H;

    float v = load_f(src + ((size_t)b * L * H + (size_t)l * H + h) * X + x_idx);
    // (T, BH, Q, X): index = ((t * BH + bh) * Q + q) * X + x.
    dst[(((size_t)t * BH + bh) * Q + q) * X + x_idx] = __float2bfloat16(v);
}

extern "C" __global__ void mamba2_permute_blhx_f32_to_bf16(
    const float* src, __nv_bfloat16* dst, int B, int L, int H, int X, int Q)
{ permute_blhx_impl<float>(src, dst, B, L, H, X, Q); }

extern "C" __global__ void mamba2_permute_blhx_f16_to_bf16(
    const __half* src, __nv_bfloat16* dst, int B, int L, int H, int X, int Q)
{ permute_blhx_impl<__half>(src, dst, B, L, H, X, Q); }

extern "C" __global__ void mamba2_permute_blhx_bf16_to_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst, int B, int L, int H, int X, int Q)
{ permute_blhx_impl<__nv_bfloat16>(src, dst, B, L, H, X, Q); }

// ─── 3. compute_dt_x_to_bf16 ────────────────────────────────────────────────
// dt_x[t, bh, q, p] = dt[b, l, h] * x[b, l, h, p].
// Input:  dt [B, L, H], x [B, L, H, P] dtype-вход.
// Output: dt_x [T, BH, Q, P] bf16.  ← chunk-outermost.
template <typename Tin>
__device__ __forceinline__ void compute_dt_x_impl(
    const Tin* __restrict__ dt, const Tin* __restrict__ x,
    __nv_bfloat16* __restrict__ dt_x,
    int B, int L, int H, int P, int Q)
{
    int p_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blh   = blockIdx.z * gridDim.y + blockIdx.y;  // split: blh может быть > 65535.
    if (p_idx >= P) return;
    if (blh >= B * L * H) return;

    int h = blh % H;
    int l = (blh / H) % L;
    int b = blh / (L * H);
    int t = l / Q;
    int q = l % Q;
    int bh = b * H + h;
    int BH = B * H;

    float dt_v = load_f(dt + ((size_t)b * L + l) * H + h);
    float x_v  = load_f(x  + ((size_t)b * L * H + (size_t)l * H + h) * P + p_idx);
    dt_x[(((size_t)t * BH + bh) * Q + q) * P + p_idx] = __float2bfloat16(dt_v * x_v);
}

extern "C" __global__ void mamba2_compute_dt_x_f32_to_bf16(
    const float* dt, const float* x, __nv_bfloat16* dt_x,
    int B, int L, int H, int P, int Q)
{ compute_dt_x_impl<float>(dt, x, dt_x, B, L, H, P, Q); }

extern "C" __global__ void mamba2_compute_dt_x_f16_to_bf16(
    const __half* dt, const __half* x, __nv_bfloat16* dt_x,
    int B, int L, int H, int P, int Q)
{ compute_dt_x_impl<__half>(dt, x, dt_x, B, L, H, P, Q); }

extern "C" __global__ void mamba2_compute_dt_x_bf16_to_bf16(
    const __nv_bfloat16* dt, const __nv_bfloat16* x, __nv_bfloat16* dt_x,
    int B, int L, int H, int P, int Q)
{ compute_dt_x_impl<__nv_bfloat16>(dt, x, dt_x, B, L, H, P, Q); }

// ─── 4. transpose_bf16 (per-batch) ──────────────────────────────────────────
// Input:  src [BAT, R, C] bf16. Output: dst [BAT, C, R] bf16.
// Один thread = один элемент. Без шаринга (простой).
extern "C" __global__ void mamba2_transpose_bf16(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16*       __restrict__ dst,
    int BAT, int R, int C)
{
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int bat   = blockIdx.z;
    if (c_idx >= C || r_idx >= R || bat >= BAT) return;
    size_t src_off = ((size_t)bat * R + r_idx) * C + c_idx;
    size_t dst_off = ((size_t)bat * C + c_idx) * R + r_idx;
    dst[dst_off] = src[src_off];
}

// ─── 5. apply_decay_mask_to_bf16 ────────────────────────────────────────────
// Per (bh*T, i, j): A_decayed[i,j] = A_intra[i,j] * exp(α_cum[i]-α_cum[j]) * [j≤i].
// Input:  A_intra [BHT, Q, Q] f32, alpha_cum [BHT, Q] f32 (BHT = BH*T).
// Output: A_decayed [BHT, Q, Q] bf16.
extern "C" __global__ void mamba2_apply_decay_mask_to_bf16(
    const float*          __restrict__ A_intra,
    const float*          __restrict__ alpha_cum,
    __nv_bfloat16*        __restrict__ A_decayed,
    int BHT, int Q)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int bht = blockIdx.z;
    if (i >= Q || j >= Q || bht >= BHT) return;

    size_t off = ((size_t)bht * Q + i) * Q + j;
    float v;
    if (j > i) {
        v = 0.0f;
    } else {
        float ai = alpha_cum[(size_t)bht * Q + i];
        float aj = alpha_cum[(size_t)bht * Q + j];
        v = A_intra[off] * __expf(ai - aj);
    }
    A_decayed[off] = __float2bfloat16(v);
}

// ─── 6. col_broadcast_exp_mul_to_bf16 ───────────────────────────────────────
// Per (bh*T, r, c): dst[r,c] = src[r,c] * exp(vec[r]) (если from_end=0)
//                  или    = src[r,c] * exp(vec[Q-1] - vec[r]) (если from_end=1).
// Used: C_QN * exp(α_cum)            (from_end=0, vec = alpha_cum[chunk])
//       dt_x_PQ * exp(α_end - α_cum) (from_end=1, vec = alpha_cum[chunk])
// Шейпы:
//   src [BAT, R, C] bf16, vec [BAT, R] f32, dst [BAT, R, C] bf16.
// Note: для dt_x_PQ (где R=P, C=Q) caller должен сначала транспонировать;
// эта функция применяет коэффициент по R-оси. См. orchestrator.
extern "C" __global__ void mamba2_col_broadcast_exp_mul_bf16(
    const __nv_bfloat16* __restrict__ src,
    const float*          __restrict__ vec,
    __nv_bfloat16*       __restrict__ dst,
    int BAT, int R, int C, int from_end)
{
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int bat   = blockIdx.z;
    if (c_idx >= C || r_idx >= R || bat >= BAT) return;

    float v = load_f(src + ((size_t)bat * R + r_idx) * C + c_idx);
    float a;
    if (from_end) {
        float a_end = vec[(size_t)bat * R + (R - 1)];
        float a_r   = vec[(size_t)bat * R + r_idx];
        a = __expf(a_end - a_r);
    } else {
        a = __expf(vec[(size_t)bat * R + r_idx]);
    }
    dst[((size_t)bat * R + r_idx) * C + c_idx] = __float2bfloat16(v * a);
}

// Версия для row-based scaling по Q-оси (для dt_x_PQ: vec индексируется
// по последней оси, не R). dst[p, q] = src[p, q] * exp(vec[Q-1] - vec[q]).
extern "C" __global__ void mamba2_row_broadcast_exp_mul_bf16(
    const __nv_bfloat16* __restrict__ src,
    const float*          __restrict__ vec,    // [BAT, Q_vec]
    __nv_bfloat16*       __restrict__ dst,
    int BAT, int R, int C, int Q_vec, int from_end)
{
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int bat   = blockIdx.z;
    if (c_idx >= C || r_idx >= R || bat >= BAT) return;
    if (c_idx >= Q_vec) return;

    float v = load_f(src + ((size_t)bat * R + r_idx) * C + c_idx);
    float a;
    if (from_end) {
        float a_end = vec[(size_t)bat * Q_vec + (Q_vec - 1)];
        float a_c   = vec[(size_t)bat * Q_vec + c_idx];
        a = __expf(a_end - a_c);
    } else {
        a = __expf(vec[(size_t)bat * Q_vec + c_idx]);
    }
    dst[((size_t)bat * R + r_idx) * C + c_idx] = __float2bfloat16(v * a);
}

// ─── 7. state_linear_decay_f32 ──────────────────────────────────────────────
// state[bh, p, n] *= exp(alpha_cum[chunk, bh, Q-1]).
// Input/output: state [BH, P, N] f32, alpha_cum [T, BH, Q] f32, chunk idx.
extern "C" __global__ void mamba2_state_linear_decay_f32(
    float*       __restrict__ state,
    const float* __restrict__ alpha_cum,
    int BH, int P, int N, int T_, int Q, int chunk)
{
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int p_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int bh    = blockIdx.z;
    if (n_idx >= N || p_idx >= P || bh >= BH) return;

    float a_end = alpha_cum[((size_t)chunk * BH + bh) * Q + (Q - 1)];
    float decay = __expf(a_end);
    size_t off = ((size_t)bh * P + p_idx) * N + n_idx;
    state[off] *= decay;
}

// ─── 8. add_inplace_f32 ─────────────────────────────────────────────────────
// dst += src (f32). Сliced на 1D-block, элементарный.
extern "C" __global__ void mamba2_add_inplace_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += src[i];
}

// ─── 9. add_yoff_chunk_f32 ──────────────────────────────────────────────────
// Y_intra[chunk, bh, q, p] += Y_off_chunk[bh, q, p].
// Y_intra полный shape: [T, BH, Q, P], Y_off chunk: [BH, Q, P].
extern "C" __global__ void mamba2_add_yoff_chunk_f32(
    float*       __restrict__ Y_intra,
    const float* __restrict__ Y_off_chunk,
    int BH, int T_, int Q, int P, int chunk)
{
    int p_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int bh    = blockIdx.z;
    if (p_idx >= P || q_idx >= Q || bh >= BH) return;

    size_t yi_off   = (((size_t)chunk * BH + bh) * Q + q_idx) * P + p_idx;
    size_t yoff_off = ((size_t)bh * Q + q_idx) * P + p_idx;
    Y_intra[yi_off] += Y_off_chunk[yoff_off];
}

// ─── 10. state_cast_f32_to_bf16 ─────────────────────────────────────────────
// state[BH, P, N] f32 → state_bf16 (тот же layout) bf16. Без транспозы.
// (Для bmm Y_off нужен state как B-операнд в формате [BH, P, N] row-major.)
extern "C" __global__ void mamba2_state_cast_f32_to_bf16(
    const float*    __restrict__ src,
    __nv_bfloat16*  __restrict__ dst,
    size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(src[i]);
}

// ─── 11. post (unpermute + skip-D) ──────────────────────────────────────────
// y_out[b, l, h, p] = Y_intra[t, bh, q, p] + (has_d ? D[h] * x[b, l, h, p] : 0).
// Output dtype Tout (=Tin типов x/D). Y_intra всегда f32, layout (T, BH, Q, P).
template <typename Tout>
__device__ __forceinline__ void post_impl(
    const float* __restrict__ Y_intra,
    const Tout*  __restrict__ x,
    const Tout*  __restrict__ D,
    int has_d,
    Tout*        __restrict__ y_out,
    int B, int L, int H, int P, int Q)
{
    int p_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blh   = blockIdx.z * gridDim.y + blockIdx.y;  // split: blh может быть > 65535.
    if (p_idx >= P) return;
    if (blh >= B * L * H) return;

    int h = blh % H;
    int l = (blh / H) % L;
    int b = blh / (L * H);
    int t = l / Q;
    int q = l % Q;
    int bh = b * H + h;
    int BH = B * H;

    size_t yi_off = (((size_t)t * BH + bh) * Q + q) * P + p_idx;
    size_t yo_off = ((size_t)b * L * H + (size_t)l * H + h) * P + p_idx;
    float val = Y_intra[yi_off];
    if (has_d) {
        float d_val = load_f(D + h);
        float x_val = load_f(x + yo_off);
        val += d_val * x_val;
    }
    store_f(y_out + yo_off, val);
}

extern "C" __global__ void mamba2_post_f32(
    const float* Y_intra, const float* x, const float* D, int has_d,
    float* y_out, int B, int L, int H, int P, int Q)
{ post_impl<float>(Y_intra, x, D, has_d, y_out, B, L, H, P, Q); }

extern "C" __global__ void mamba2_post_f16(
    const float* Y_intra, const __half* x, const __half* D, int has_d,
    __half* y_out, int B, int L, int H, int P, int Q)
{ post_impl<__half>(Y_intra, x, D, has_d, y_out, B, L, H, P, Q); }

extern "C" __global__ void mamba2_post_bf16(
    const float* Y_intra, const __nv_bfloat16* x, const __nv_bfloat16* D, int has_d,
    __nv_bfloat16* y_out, int B, int L, int H, int P, int Q)
{ post_impl<__nv_bfloat16>(Y_intra, x, D, has_d, y_out, B, L, H, P, Q); }
