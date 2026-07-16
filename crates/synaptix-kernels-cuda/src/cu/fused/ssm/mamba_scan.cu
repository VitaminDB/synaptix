#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Selective state-space scan (Mamba S6).
//
// Inputs (row-major):
//   u      [B, L, D]    input sequence
//   delta  [B, L, D]    per-token discretization Δ (> 0)
//   A      [D, N]       state transition (negative; ln_inv_softplus init)
//   B_in   [B, L, N]    selective input projection
//   C_in   [B, L, N]    selective output projection
//   D_skip [D]          optional skip connection (nullptr ⟹ skip)
// Output:
//   y      [B, L, D]
//
// Per-step (forward):
//   delta_a = exp(A[d, :] * delta[b, t, d])           [N]
//   delta_b = delta[b, t, d] * B_in[b, t, :]          [N]
//   h[b, d, :] = delta_a * h[b, d, :] + delta_b * u[b, t, d]
//   y[b, t, d] = sum(C_in[b, t, :] * h[b, d, :])  +  D_skip[d] * u[b, t, d]
//
// Layout: один block = одна (b, d) пара (parallel). Block dim = N (state size,
// обычно 16). Каждый thread держит h[n] в register, sequential loop по L.
// Reduction `sum(C * h)` через warp shuffle (N ≤ 32 ⟹ один warp).

__device__ __forceinline__ float load_f(const float* p) { return *p; }
__device__ __forceinline__ float load_f(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_f(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void store_f(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_f(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_f(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void mamba_scan_impl(
    const T* __restrict__ u,
    const T* __restrict__ delta,
    const T* __restrict__ A,
    const T* __restrict__ B_in,
    const T* __restrict__ C_in,
    const T* __restrict__ D_skip,
    int has_d,
    T* __restrict__ y,
    int B, int L, int D, int N
) {
    int bd = blockIdx.x;
    int b = bd / D;
    int d = bd % D;
    int n = threadIdx.x;
    if (n >= N || b >= B) return;

    float h = 0.0f;
    float a_dn = load_f(A + (size_t)d * N + n);
    float d_skip_val = has_d ? load_f(D_skip + d) : 0.0f;

    __shared__ float y_red;

    for (int t = 0; t < L; t++) {
        size_t u_off = ((size_t)b * L + t) * D + d;
        size_t bc_off = ((size_t)b * L + t) * N + n;
        float u_t = load_f(u + u_off);
        float delta_t = load_f(delta + u_off);
        float b_tn = load_f(B_in + bc_off);
        float c_tn = load_f(C_in + bc_off);

        float delta_a = expf(a_dn * delta_t);
        float delta_b = delta_t * b_tn;
        h = delta_a * h + delta_b * u_t;

        float partial = c_tn * h;
        unsigned int mask = 0xFFFFFFFFu;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            partial += __shfl_down_sync(mask, partial, off, 32);
        }
        if (n == 0) {
            float y_val = partial + (has_d ? d_skip_val * u_t : 0.0f);
            y_red = y_val;
        }
        __syncthreads();
        if (n == 0) {
            store_f(y + u_off, y_red);
        }
        __syncthreads();
    }
}

extern "C" __global__ void mamba_scan_f32(
    const float* u, const float* delta, const float* A,
    const float* B_in, const float* C_in,
    const float* D_skip, int has_d,
    float* y, int B, int L, int D, int N
) {
    mamba_scan_impl<float>(u, delta, A, B_in, C_in, D_skip, has_d, y, B, L, D, N);
}

extern "C" __global__ void mamba_scan_f16(
    const __half* u, const __half* delta, const __half* A,
    const __half* B_in, const __half* C_in,
    const __half* D_skip, int has_d,
    __half* y, int B, int L, int D, int N
) {
    mamba_scan_impl<__half>(u, delta, A, B_in, C_in, D_skip, has_d, y, B, L, D, N);
}

extern "C" __global__ void mamba_scan_bf16(
    const __nv_bfloat16* u, const __nv_bfloat16* delta, const __nv_bfloat16* A,
    const __nv_bfloat16* B_in, const __nv_bfloat16* C_in,
    const __nv_bfloat16* D_skip, int has_d,
    __nv_bfloat16* y, int B, int L, int D, int N
) {
    mamba_scan_impl<__nv_bfloat16>(u, delta, A, B_in, C_in, D_skip, has_d, y, B, L, D, N);
}
