#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Mamba2 State-Space Duality (SSD) — рекуррентная форма (functional, не chunked).
//
// Отличие от Mamba1 (cu/mamba_scan.cu): A — скаляр на голову (не матрица [D,N]),
// есть структура голов (H голов × P head_dim), B/C — на голову. Это и есть
// "duality" SSM. Chunked-SSD (segment-sum) — будущая оптимизация; здесь —
// корректный рекуррент (эквивалент chunked по результату).
//
// Inputs (row-major):
//   x      [B, L, H, P]   входная последовательность (P = head_dim)
//   dt     [B, L, H]      timestep (>0, после softplus)
//   A      [H]            скалярный decay на голову (обычно < 0)
//   B_in   [B, L, H, N]   selective input projection (N = d_state)
//   C_in   [B, L, H, N]   selective output projection
//   D_skip [H]            optional skip (nullptr ⟹ no skip)
// Output:
//   y      [B, L, H, P]
//
// Per-step:
//   a_t = exp(dt[b,t,h] * A[h])
//   state[h,p,:] = a_t * state[h,p,:] + (dt[b,t,h] * x[b,t,h,p]) * B_in[b,t,h,:]
//   y[b,t,h,p]   = Σ_n C_in[b,t,h,n] * state[h,p,n]  +  D[h]*x[b,t,h,p]
//
// Layout: один block = одна (b, h, p) тройка; block_dim = N (один thread на
// state-dim). Thread держит state[n] в регистре, sequential по L. Reduction
// Σ_n C*state — tree-reduce в shared (N — степень двойки, ≤ 1024).

__device__ __forceinline__ float load_f(const float* p) { return *p; }
__device__ __forceinline__ float load_f(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_f(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void store_f(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_f(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_f(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void mamba2_ssd_impl(
    const T* __restrict__ x,
    const T* __restrict__ dt,
    const T* __restrict__ A,
    const T* __restrict__ B_in,
    const T* __restrict__ C_in,
    const T* __restrict__ D_skip,
    int has_d,
    T* __restrict__ y,
    int B, int L, int H, int P, int N
) {
    int bid = blockIdx.x;
    int p = bid % P;
    int h = (bid / P) % H;
    int b = bid / (P * H);
    int n = threadIdx.x;
    if (n >= N || b >= B) return;

    extern __shared__ float red_sm[];  // [N]

    float state = 0.0f;
    float a_h = load_f(A + h);
    float d_skip_val = has_d ? load_f(D_skip + h) : 0.0f;

    for (int t = 0; t < L; t++) {
        // Скаляры на (b,t,h).
        size_t dt_off = ((size_t)b * L + t) * H + h;
        float dt_t = load_f(dt + dt_off);

        size_t x_off = (((size_t)b * L + t) * H + h) * P + p;
        float x_t = load_f(x + x_off);

        size_t bc_off = (((size_t)b * L + t) * H + h) * N + n;
        float b_tn = load_f(B_in + bc_off);
        float c_tn = load_f(C_in + bc_off);

        float a_t = expf(dt_t * a_h);
        float dBx = dt_t * x_t * b_tn;
        state = a_t * state + dBx;

        float partial = c_tn * state;
        red_sm[n] = partial;
        __syncthreads();
        for (int s = N >> 1; s > 0; s >>= 1) {
            if (n < s) red_sm[n] += red_sm[n + s];
            __syncthreads();
        }
        if (n == 0) {
            float y_val = red_sm[0] + (has_d ? d_skip_val * x_t : 0.0f);
            store_f(y + x_off, y_val);
        }
        __syncthreads();
    }
}

extern "C" __global__ void mamba2_ssd_f32(
    const float* x, const float* dt, const float* A,
    const float* B_in, const float* C_in,
    const float* D_skip, int has_d,
    float* y, int B, int L, int H, int P, int N
) {
    mamba2_ssd_impl<float>(x, dt, A, B_in, C_in, D_skip, has_d, y, B, L, H, P, N);
}

extern "C" __global__ void mamba2_ssd_f16(
    const __half* x, const __half* dt, const __half* A,
    const __half* B_in, const __half* C_in,
    const __half* D_skip, int has_d,
    __half* y, int B, int L, int H, int P, int N
) {
    mamba2_ssd_impl<__half>(x, dt, A, B_in, C_in, D_skip, has_d, y, B, L, H, P, N);
}

extern "C" __global__ void mamba2_ssd_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* dt, const __nv_bfloat16* A,
    const __nv_bfloat16* B_in, const __nv_bfloat16* C_in,
    const __nv_bfloat16* D_skip, int has_d,
    __nv_bfloat16* y, int B, int L, int H, int P, int N
) {
    mamba2_ssd_impl<__nv_bfloat16>(x, dt, A, B_in, C_in, D_skip, has_d, y, B, L, H, P, N);
}
