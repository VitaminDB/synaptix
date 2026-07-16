#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define MAX_RANK 8

struct UnaryParams {
    int op_code;
    float scalar_a;
    float scalar_b;
    int rank;
    long long numel;
    long long in_offset;
    int dims[MAX_RANK];
    int in_strides[MAX_RANK];
};

struct BinaryParams {
    int op_code;
    int rank;
    long long numel;
    long long a_offset;
    long long b_offset;
    int dims[MAX_RANK];
    int a_strides[MAX_RANK];
    int b_strides[MAX_RANK];
};

struct CastParams {
    long long numel;
    long long in_offset;
    int rank;
    int dims[MAX_RANK];
    int in_strides[MAX_RANK];
};

__device__ __forceinline__ float apply_unary(int op_code, float x, float a, float b) {
    switch (op_code) {
        case 0:  return -x;
        case 1:  return fabsf(x);
        case 2:  return sqrtf(x);
        case 3:  return x * x;
        case 4:  return 1.0f / x;
        case 5:  return expf(x);
        case 6:  return logf(x);
        case 7:  return sinf(x);
        case 8:  return cosf(x);
        case 9:  return x / (1.0f + expf(-x));
        case 10: {
            float c = sqrtf(2.0f / 3.14159265358979323846f);
            return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
        }
        case 11: return tanhf(x);
        case 12: return fmaxf(a, fminf(b, x));
        case 13: return powf(x, a);
        case 14: return a * x + b;
        case 15: return erff(x);
        case 16: return 1.0f / (1.0f + expf(-x));
        case 17: return x > 0.0f ? x : 0.0f;
        case 18: {
            float r = x > 0.0f ? x : 0.0f;
            return r * r;
        }
        case 19: return x > 0.0f ? x : a * x;
        case 20: return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
        case 21: return x > 0.0f ? 1.0f : 0.0f;
        case 22: return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
        case 23: return rintf(x);
        case 24: return floorf(x);
        case 25: return ceilf(x);
        default: return x;
    }
}

__device__ __forceinline__ float apply_binary(int op_code, float x, float y) {
    switch (op_code) {
        case 0: return x + y;
        case 1: return x - y;
        case 2: return x * y;
        case 3: return x / y;
        case 4: return fmaxf(x, y);
        case 5: return fminf(x, y);
        default: return x;
    }
}

__device__ __forceinline__ long long unravel_idx(
    long long tid, int rank, const int* dims, const int* strides, long long base
) {
    long long rem = tid;
    long long idx = base;
    for (int d = rank - 1; d >= 0; --d) {
        int dim = dims[d];
        int coord = (int)(rem % (long long)dim);
        rem /= (long long)dim;
        idx += (long long)coord * (long long)strides[d];
    }
    return idx;
}

#define UNARY_KERNEL(name, T, to_f32, from_f32) \
extern "C" __global__ void name( \
    const T* __restrict__ in, T* __restrict__ out, UnaryParams p \
) { \
    long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x; \
    if (tid >= p.numel) return; \
    long long in_idx = unravel_idx(tid, p.rank, p.dims, p.in_strides, p.in_offset); \
    float x = to_f32(in[in_idx]); \
    float y = apply_unary(p.op_code, x, p.scalar_a, p.scalar_b); \
    out[tid] = from_f32(y); \
}

#define BINARY_KERNEL(name, T, to_f32, from_f32) \
extern "C" __global__ void name( \
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out, BinaryParams p \
) { \
    long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x; \
    if (tid >= p.numel) return; \
    long long a_idx = unravel_idx(tid, p.rank, p.dims, p.a_strides, p.a_offset); \
    long long b_idx = unravel_idx(tid, p.rank, p.dims, p.b_strides, p.b_offset); \
    float x = to_f32(a[a_idx]); \
    float y = to_f32(b[b_idx]); \
    float z = apply_binary(p.op_code, x, y); \
    out[tid] = from_f32(z); \
}

__device__ __forceinline__ float f32_to_f32(float x) { return x; }
__device__ __forceinline__ float f16_to_f32(__half x) { return __half2float(x); }
__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float   f32_from_f32(float x) { return x; }
__device__ __forceinline__ __half  f16_from_f32(float x) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 bf16_from_f32(float x) { return __float2bfloat16(x); }

UNARY_KERNEL(unary_f32,  float,         f32_to_f32, f32_from_f32)
UNARY_KERNEL(unary_f16,  __half,        f16_to_f32, f16_from_f32)
UNARY_KERNEL(unary_bf16, __nv_bfloat16, bf16_to_f32, bf16_from_f32)

BINARY_KERNEL(binary_f32,  float,         f32_to_f32, f32_from_f32)
BINARY_KERNEL(binary_f16,  __half,        f16_to_f32, f16_from_f32)
BINARY_KERNEL(binary_bf16, __nv_bfloat16, bf16_to_f32, bf16_from_f32)

// ── Быстрые binary-пути (generic strided ~60-90GB/s душил рой LTX/diffusion:
// residual/gate/ada-модуляции — same-shape и [N,D]⊕[D]) ──
// FLAT: a,b,out contiguous same-shape. ROWB: a,out contiguous [N,D], b — одна
// строка [D] (broadcast по внешним осям). 16Б-вектор (8×16bit / 4×f32),
// математика f32 как в generic → бит-в-бит; выравнивание/кратность гейтится в
// Rust-диспетчере (иначе фоллбэк generic).
#define BINARY_FLAT_KERNEL(name, T, VEC_N, to_f32, from_f32) \
extern "C" __global__ void name( \
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out, \
    long long numel, long long a_off, long long b_off, int op_code \
) { \
    a += a_off; b += b_off; \
    long long i = ((long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x) * VEC_N; \
    if (i + VEC_N <= numel) { \
        uint4 av = *reinterpret_cast<const uint4*>(a + i); \
        uint4 bv = *reinterpret_cast<const uint4*>(b + i); \
        T ae[VEC_N], be[VEC_N], oe[VEC_N]; \
        *reinterpret_cast<uint4*>(ae) = av; \
        *reinterpret_cast<uint4*>(be) = bv; \
        _Pragma("unroll") \
        for (int k = 0; k < VEC_N; ++k) oe[k] = from_f32(apply_binary(op_code, to_f32(ae[k]), to_f32(be[k]))); \
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<uint4*>(oe); \
    } else { \
        for (; i < numel; ++i) out[i] = from_f32(apply_binary(op_code, to_f32(a[i]), to_f32(b[i]))); \
    } \
}

#define BINARY_ROWB_KERNEL(name, T, VEC_N, to_f32, from_f32) \
extern "C" __global__ void name( \
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out, \
    long long numel, long long a_off, long long b_off, int d, int op_code \
) { \
    a += a_off; b += b_off; \
    long long i = ((long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x) * VEC_N; \
    if (i + VEC_N <= numel) { \
        int j = (int)(i % (long long)d); \
        uint4 av = *reinterpret_cast<const uint4*>(a + i); \
        uint4 bv = *reinterpret_cast<const uint4*>(b + j); \
        T ae[VEC_N], be[VEC_N], oe[VEC_N]; \
        *reinterpret_cast<uint4*>(ae) = av; \
        *reinterpret_cast<uint4*>(be) = bv; \
        _Pragma("unroll") \
        for (int k = 0; k < VEC_N; ++k) oe[k] = from_f32(apply_binary(op_code, to_f32(ae[k]), to_f32(be[k]))); \
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<uint4*>(oe); \
    } else { \
        for (; i < numel; ++i) out[i] = from_f32(apply_binary(op_code, to_f32(a[i]), to_f32(b[(int)(i % (long long)d)]))); \
    } \
}

BINARY_FLAT_KERNEL(binary_flat_f32,  float,         4, f32_to_f32,  f32_from_f32)
BINARY_FLAT_KERNEL(binary_flat_f16,  __half,        8, f16_to_f32,  f16_from_f32)
BINARY_FLAT_KERNEL(binary_flat_bf16, __nv_bfloat16, 8, bf16_to_f32, bf16_from_f32)
BINARY_ROWB_KERNEL(binary_rowb_f32,  float,         4, f32_to_f32,  f32_from_f32)
BINARY_ROWB_KERNEL(binary_rowb_f16,  __half,        8, f16_to_f32,  f16_from_f32)
BINARY_ROWB_KERNEL(binary_rowb_bf16, __nv_bfloat16, 8, bf16_to_f32, bf16_from_f32)

// COLB: broadcast по ПОСЛЕДНЕЙ оси — a contiguous [..., D], b плотный по
// внешним осям со stride 0 на последней ([..,G,1]→[..,G,D]): один b-скаляр
// на группу из D элементов (gate-умножения [1,T,H,dh]×[1,T,H,1], маски
// [1,T,1]×[1,T,D]). D % VEC_N == 0 → вектор не пересекает группу.
#define BINARY_COLB_KERNEL(name, T, VEC_N, to_f32, from_f32) \
extern "C" __global__ void name( \
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out, \
    long long numel, long long a_off, long long b_off, int d, int op_code \
) { \
    a += a_off; b += b_off; \
    long long i = ((long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x) * VEC_N; \
    if (i + VEC_N <= numel) { \
        float bv = to_f32(b[i / (long long)d]); \
        uint4 av = *reinterpret_cast<const uint4*>(a + i); \
        T ae[VEC_N], oe[VEC_N]; \
        *reinterpret_cast<uint4*>(ae) = av; \
        _Pragma("unroll") \
        for (int k = 0; k < VEC_N; ++k) oe[k] = from_f32(apply_binary(op_code, to_f32(ae[k]), bv)); \
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<uint4*>(oe); \
    } else { \
        for (; i < numel; ++i) out[i] = from_f32(apply_binary(op_code, to_f32(a[i]), to_f32(b[i / (long long)d]))); \
    } \
}

BINARY_COLB_KERNEL(binary_colb_f32,  float,         4, f32_to_f32,  f32_from_f32)
BINARY_COLB_KERNEL(binary_colb_f16,  __half,        8, f16_to_f32,  f16_from_f32)
BINARY_COLB_KERNEL(binary_colb_bf16, __nv_bfloat16, 8, bf16_to_f32, bf16_from_f32)

// FMA (gated-residual): out = x + y*g — промежуточный T-раунд y*g повторяет
// decomposed broadcast_mul→add → бит-в-бит. FLAT: все contiguous same-shape;
// ROWB: g — строка [D] (broadcast по внешним осям, D % VEC_N == 0).
#define FMA_FLAT_KERNEL(name, T, VEC_N, to_f32, from_f32) \
extern "C" __global__ void name( \
    const T* __restrict__ x, const T* __restrict__ y, const T* __restrict__ g, \
    T* __restrict__ out, long long numel, long long x_off, long long y_off, long long g_off \
) { \
    x += x_off; y += y_off; g += g_off; \
    long long i = ((long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x) * VEC_N; \
    if (i + VEC_N <= numel) { \
        uint4 xv = *reinterpret_cast<const uint4*>(x + i); \
        uint4 yv = *reinterpret_cast<const uint4*>(y + i); \
        uint4 gv = *reinterpret_cast<const uint4*>(g + i); \
        T xe[VEC_N], ye[VEC_N], ge[VEC_N], oe[VEC_N]; \
        *reinterpret_cast<uint4*>(xe) = xv; \
        *reinterpret_cast<uint4*>(ye) = yv; \
        *reinterpret_cast<uint4*>(ge) = gv; \
        _Pragma("unroll") \
        for (int k = 0; k < VEC_N; ++k) { \
            float yg = to_f32(from_f32(to_f32(ye[k]) * to_f32(ge[k]))); \
            oe[k] = from_f32(to_f32(xe[k]) + yg); \
        } \
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<uint4*>(oe); \
    } else { \
        for (; i < numel; ++i) { \
            float yg = to_f32(from_f32(to_f32(y[i]) * to_f32(g[i]))); \
            out[i] = from_f32(to_f32(x[i]) + yg); \
        } \
    } \
}

#define FMA_ROWB_KERNEL(name, T, VEC_N, to_f32, from_f32) \
extern "C" __global__ void name( \
    const T* __restrict__ x, const T* __restrict__ y, const T* __restrict__ g, \
    T* __restrict__ out, long long numel, long long x_off, long long y_off, long long g_off, int d \
) { \
    x += x_off; y += y_off; g += g_off; \
    long long i = ((long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x) * VEC_N; \
    if (i + VEC_N <= numel) { \
        int j = (int)(i % (long long)d); \
        uint4 xv = *reinterpret_cast<const uint4*>(x + i); \
        uint4 yv = *reinterpret_cast<const uint4*>(y + i); \
        uint4 gv = *reinterpret_cast<const uint4*>(g + j); \
        T xe[VEC_N], ye[VEC_N], ge[VEC_N], oe[VEC_N]; \
        *reinterpret_cast<uint4*>(xe) = xv; \
        *reinterpret_cast<uint4*>(ye) = yv; \
        *reinterpret_cast<uint4*>(ge) = gv; \
        _Pragma("unroll") \
        for (int k = 0; k < VEC_N; ++k) { \
            float yg = to_f32(from_f32(to_f32(ye[k]) * to_f32(ge[k]))); \
            oe[k] = from_f32(to_f32(xe[k]) + yg); \
        } \
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<uint4*>(oe); \
    } else { \
        for (; i < numel; ++i) { \
            int j = (int)(i % (long long)d); \
            float yg = to_f32(from_f32(to_f32(y[i]) * to_f32(g[j]))); \
            out[i] = from_f32(to_f32(x[i]) + yg); \
        } \
    } \
}

FMA_FLAT_KERNEL(fma_flat_f32,  float,         4, f32_to_f32,  f32_from_f32)
FMA_FLAT_KERNEL(fma_flat_f16,  __half,        8, f16_to_f32,  f16_from_f32)
FMA_FLAT_KERNEL(fma_flat_bf16, __nv_bfloat16, 8, bf16_to_f32, bf16_from_f32)
FMA_ROWB_KERNEL(fma_rowb_f32,  float,         4, f32_to_f32,  f32_from_f32)
FMA_ROWB_KERNEL(fma_rowb_f16,  __half,        8, f16_to_f32,  f16_from_f32)
FMA_ROWB_KERNEL(fma_rowb_bf16, __nv_bfloat16, 8, bf16_to_f32, bf16_from_f32)

// MOD_ROWB (adaLN-модуляция готовой нормы): out = x*(1+s)+sh, s/sh — строки
// [D]. Раунды T после (1+s), после mul и после add повторяют decomposed
// add_scalar→broadcast_mul→broadcast_add → бит-в-бит.
#define MOD_ROWB_KERNEL(name, T, VEC_N, to_f32, from_f32) \
extern "C" __global__ void name( \
    const T* __restrict__ x, const T* __restrict__ s, const T* __restrict__ sh, \
    T* __restrict__ out, long long numel, long long x_off, long long s_off, long long sh_off, int d \
) { \
    x += x_off; s += s_off; sh += sh_off; \
    long long i = ((long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x) * VEC_N; \
    if (i + VEC_N <= numel) { \
        int j = (int)(i % (long long)d); \
        uint4 xv = *reinterpret_cast<const uint4*>(x + i); \
        uint4 sv = *reinterpret_cast<const uint4*>(s + j); \
        uint4 shv = *reinterpret_cast<const uint4*>(sh + j); \
        T xe[VEC_N], se[VEC_N], she[VEC_N], oe[VEC_N]; \
        *reinterpret_cast<uint4*>(xe) = xv; \
        *reinterpret_cast<uint4*>(se) = sv; \
        *reinterpret_cast<uint4*>(she) = shv; \
        _Pragma("unroll") \
        for (int k = 0; k < VEC_N; ++k) { \
            float s1 = to_f32(from_f32(1.0f + to_f32(se[k]))); \
            float t1 = to_f32(from_f32(to_f32(xe[k]) * s1)); \
            oe[k] = from_f32(t1 + to_f32(she[k])); \
        } \
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<uint4*>(oe); \
    } else { \
        for (; i < numel; ++i) { \
            int j = (int)(i % (long long)d); \
            float s1 = to_f32(from_f32(1.0f + to_f32(s[j]))); \
            float t1 = to_f32(from_f32(to_f32(x[i]) * s1)); \
            out[i] = from_f32(t1 + to_f32(sh[j])); \
        } \
    } \
}

MOD_ROWB_KERNEL(mod_rowb_f32,  float,         4, f32_to_f32,  f32_from_f32)
MOD_ROWB_KERNEL(mod_rowb_f16,  __half,        8, f16_to_f32,  f16_from_f32)
MOD_ROWB_KERNEL(mod_rowb_bf16, __nv_bfloat16, 8, bf16_to_f32, bf16_from_f32)

// Flat-unary (in/out contiguous): 16Б-вектор, та же f32-математика → бит-в-бит
// с generic (gelu/sigmoid/affine на [*,16384] душились strided-ядром).
#define UNARY_FLAT_KERNEL(name, T, VEC_N, to_f32, from_f32) \
extern "C" __global__ void name( \
    const T* __restrict__ in, T* __restrict__ out, \
    long long numel, long long in_off, int op_code, float sa, float sb \
) { \
    in += in_off; \
    long long i = ((long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x) * VEC_N; \
    if (i + VEC_N <= numel) { \
        uint4 iv = *reinterpret_cast<const uint4*>(in + i); \
        T ie[VEC_N], oe[VEC_N]; \
        *reinterpret_cast<uint4*>(ie) = iv; \
        _Pragma("unroll") \
        for (int k = 0; k < VEC_N; ++k) oe[k] = from_f32(apply_unary(op_code, to_f32(ie[k]), sa, sb)); \
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<uint4*>(oe); \
    } else { \
        for (; i < numel; ++i) out[i] = from_f32(apply_unary(op_code, to_f32(in[i]), sa, sb)); \
    } \
}

UNARY_FLAT_KERNEL(unary_flat_f32,  float,         4, f32_to_f32,  f32_from_f32)
UNARY_FLAT_KERNEL(unary_flat_f16,  __half,        8, f16_to_f32,  f16_from_f32)
UNARY_FLAT_KERNEL(unary_flat_bf16, __nv_bfloat16, 8, bf16_to_f32, bf16_from_f32)

// ROWS-unary: src strided по внешним осям, ПОСЛЕДНЯЯ ось плотная (stride 1,
// D % VEC == 0, строки выровнены 16Б) → векторное чтение строк + линейная
// запись. Главный клиент — transpose(1,2).contiguous() в attention
// ([B,T,H,D]↔[B,H,T,D]: copy шёл generic-ядром ~60-90GB/s).
struct RowsParams {
    int op_code;
    float scalar_a;
    float scalar_b;
    int rank_outer; // осей БЕЗ последней
    int d;          // последняя ось
    long long numel;
    long long in_offset;
    int dims[MAX_RANK];    // внешние оси
    int strides[MAX_RANK]; // strides внешних осей (в элементах)
};

#define UNARY_ROWS_KERNEL(name, T, VEC_N, to_f32, from_f32) \
extern "C" __global__ void name( \
    const T* __restrict__ in, T* __restrict__ out, RowsParams p \
) { \
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    int nvec = p.d / VEC_N; \
    if (gid >= p.numel / VEC_N) return; \
    long long row = gid / nvec; \
    int col = (int)(gid % nvec) * VEC_N; \
    long long rem = row; \
    long long src = p.in_offset; \
    for (int dd = p.rank_outer - 1; dd >= 0; --dd) { \
        int dim = p.dims[dd]; \
        int coord = (int)(rem % dim); \
        rem /= dim; \
        src += (long long)coord * p.strides[dd]; \
    } \
    uint4 iv = *reinterpret_cast<const uint4*>(in + src + col); \
    T ie[VEC_N], oe[VEC_N]; \
    *reinterpret_cast<uint4*>(ie) = iv; \
    _Pragma("unroll") \
    for (int k = 0; k < VEC_N; ++k) oe[k] = from_f32(apply_unary(p.op_code, to_f32(ie[k]), p.scalar_a, p.scalar_b)); \
    *reinterpret_cast<uint4*>(out + gid * VEC_N) = *reinterpret_cast<uint4*>(oe); \
}

UNARY_ROWS_KERNEL(unary_rows_f32,  float,         4, f32_to_f32,  f32_from_f32)
UNARY_ROWS_KERNEL(unary_rows_f16,  __half,        8, f16_to_f32,  f16_from_f32)
UNARY_ROWS_KERNEL(unary_rows_bf16, __nv_bfloat16, 8, bf16_to_f32, bf16_from_f32)

#define CAST_KERNEL(name, IN, OUT, in_to_f64, f64_to_out) \
extern "C" __global__ void name( \
    const IN* __restrict__ in, OUT* __restrict__ out, CastParams p \
) { \
    long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x; \
    if (tid >= p.numel) return; \
    long long in_idx = unravel_idx(tid, p.rank, p.dims, p.in_strides, p.in_offset); \
    out[tid] = f64_to_out(in_to_f64(in[in_idx])); \
}

__device__ __forceinline__ double f32_to_f64(float x) { return (double)x; }
__device__ __forceinline__ double f64_to_f64(double x) { return x; }
__device__ __forceinline__ double f16_to_f64(__half x) { return (double)__half2float(x); }
__device__ __forceinline__ double bf16_to_f64(__nv_bfloat16 x) { return (double)__bfloat162float(x); }
__device__ __forceinline__ double u32_to_f64(unsigned int x) { return (double)x; }
__device__ __forceinline__ double i64_to_f64(long long x) { return (double)x; }
__device__ __forceinline__ float       f64_to_f32(double x) { return (float)x; }
__device__ __forceinline__ double      f64_to_f64_out(double x) { return x; }
__device__ __forceinline__ __half      f64_to_f16(double x) { return __float2half((float)x); }
__device__ __forceinline__ __nv_bfloat16 f64_to_bf16(double x) { return __float2bfloat16((float)x); }
__device__ __forceinline__ unsigned int f64_to_u32(double x) { return (unsigned int)x; }
__device__ __forceinline__ long long    f64_to_i64(double x) { return (long long)x; }

// Flat-cast (оба contiguous, выравнено 16Б): 8 эл/поток векторно. Маршрут
// in→f32→out бит-эквивалентен generic in→f64→out для пар {f32,f16,bf16}
// (f64/f32 представляют вход точно, раунд один). Главный клиент — BF16↔F16
// касты вокруг квант-GEMM (ff-mid 460MB шёл strided-ядром ~60-90GB/s).
#define CAST_FLAT_KERNEL(name, IN, OUT, in_to_f32, f32_to_out) \
extern "C" __global__ void name( \
    const IN* __restrict__ in, OUT* __restrict__ out, long long numel, long long in_off \
) { \
    in += in_off; \
    long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 8; \
    if (i + 8 <= numel) { \
        IN ie[8]; \
        OUT oe[8]; \
        _Pragma("unroll") \
        for (int v = 0; v < (int)(sizeof(IN) * 8 / 16); ++v) \
            reinterpret_cast<uint4*>(ie)[v] = reinterpret_cast<const uint4*>(in + i)[v]; \
        _Pragma("unroll") \
        for (int k = 0; k < 8; ++k) oe[k] = f32_to_out(in_to_f32(ie[k])); \
        _Pragma("unroll") \
        for (int v = 0; v < (int)(sizeof(OUT) * 8 / 16); ++v) \
            reinterpret_cast<uint4*>(out + i)[v] = reinterpret_cast<uint4*>(oe)[v]; \
    } else { \
        for (; i < numel; ++i) out[i] = f32_to_out(in_to_f32(in[i])); \
    } \
}

CAST_FLAT_KERNEL(cast_flat_f16_bf16, __half,        __nv_bfloat16, f16_to_f32,  bf16_from_f32)
CAST_FLAT_KERNEL(cast_flat_bf16_f16, __nv_bfloat16, __half,        bf16_to_f32, f16_from_f32)
CAST_FLAT_KERNEL(cast_flat_f32_bf16, float,         __nv_bfloat16, f32_to_f32,  bf16_from_f32)
CAST_FLAT_KERNEL(cast_flat_bf16_f32, __nv_bfloat16, float,         bf16_to_f32, f32_from_f32)
CAST_FLAT_KERNEL(cast_flat_f32_f16,  float,         __half,        f32_to_f32,  f16_from_f32)
CAST_FLAT_KERNEL(cast_flat_f16_f32,  __half,        float,         f16_to_f32,  f32_from_f32)

CAST_KERNEL(cast_f32_f16,  float,         __half,        f32_to_f64, f64_to_f16)
CAST_KERNEL(cast_f32_bf16, float,         __nv_bfloat16, f32_to_f64, f64_to_bf16)
CAST_KERNEL(cast_f32_f64,  float,         double,        f32_to_f64, f64_to_f64_out)
CAST_KERNEL(cast_f32_u32,  float,         unsigned int,  f32_to_f64, f64_to_u32)
CAST_KERNEL(cast_f16_f32,  __half,        float,         f16_to_f64, f64_to_f32)
CAST_KERNEL(cast_f16_bf16, __half,        __nv_bfloat16, f16_to_f64, f64_to_bf16)
CAST_KERNEL(cast_bf16_f32, __nv_bfloat16, float,         bf16_to_f64, f64_to_f32)
CAST_KERNEL(cast_bf16_f16, __nv_bfloat16, __half,        bf16_to_f64, f64_to_f16)
CAST_KERNEL(cast_f64_f32,  double,        float,         f64_to_f64, f64_to_f32)
CAST_KERNEL(cast_u32_f32,  unsigned int,  float,         u32_to_f64, f64_to_f32)
CAST_KERNEL(cast_u32_i64,  unsigned int,  long long,     u32_to_f64, f64_to_i64)
CAST_KERNEL(cast_i64_u32,  long long,     unsigned int,  i64_to_f64, f64_to_u32)
