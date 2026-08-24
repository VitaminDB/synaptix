#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define MAX_RANK 8

struct ReduceParams {
    int op_code;
    int rank;
    int n_reduce;
    long long out_numel;
    long long inner_size;
    long long in_offset;
    int dims_out[MAX_RANK];
    int dims_red[MAX_RANK];
    int red_axes[MAX_RANK];
    int strides_in[MAX_RANK];
};

__device__ __forceinline__ float init_value(int op_code) {
    if (op_code == 2 || op_code == 3) return -3.4028235e38f;
    return 0.0f;
}

__device__ __forceinline__ float pass_f32(float x) { return x; }
__device__ __forceinline__ float pass_f16(__half x) { return __half2float(x); }
__device__ __forceinline__ float pass_bf16(__nv_bfloat16 x) { return __bfloat162float(x); }


#define ROWS_BLOCK_MAX 1024

__device__ __forceinline__ float id_f32(float x) { return x; }

__device__ __forceinline__ float rows_init(int op_code) {
    return (op_code == 2) ? -3.4028235e38f : 0.0f;
}

__device__ __forceinline__ float rows_combine(int op_code, float a, float b) {
    return (op_code == 2) ? fmaxf(a, b) : (a + b);
}

__device__ __forceinline__ float rows_block_reduce(int op_code, float acc) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc = rows_combine(op_code, acc, __shfl_xor_sync(0xFFFFFFFFu, acc, off));
    }
    __shared__ float part[ROWS_BLOCK_MAX / 32];
    unsigned lane = threadIdx.x & 31u;
    unsigned warp = threadIdx.x >> 5;
    unsigned nwarps = (blockDim.x + 31u) >> 5;
    if (lane == 0u) part[warp] = acc;
    __syncthreads();
    if (warp == 0u) {
        acc = (lane < nwarps) ? part[lane] : rows_init(op_code);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc = rows_combine(op_code, acc, __shfl_xor_sync(0xFFFFFFFFu, acc, off));
        }
    }
    return acc;
}

#define REDUCE_ROWS(NAME, T, TO_F32, FROM_F32)                                      \
extern "C" __global__ void NAME(                                                    \
    const T* __restrict__ in, T* __restrict__ out,                                  \
    long long inner, long long in_offset, int op_code)                              \
{                                                                                   \
    const T* src = in + in_offset + (long long)blockIdx.x * inner;                  \
    float acc = rows_init(op_code);                                                 \
    for (long long j = (long long)threadIdx.x; j < inner; j += (long long)blockDim.x) { \
        acc = rows_combine(op_code, acc, TO_F32(src[j]));                           \
    }                                                                               \
    acc = rows_block_reduce(op_code, acc);                                          \
    if (threadIdx.x == 0u) {                                                        \
        if (op_code == 1 && inner > 0) acc /= (float)inner;                         \
        out[blockIdx.x] = FROM_F32(acc);                                            \
    }                                                                               \
}

REDUCE_ROWS(reduce_rows_f32, float, pass_f32, id_f32)
REDUCE_ROWS(reduce_rows_f16, __half, pass_f16, __float2half)
REDUCE_ROWS(reduce_rows_bf16, __nv_bfloat16, pass_bf16, __float2bfloat16)

extern "C" __global__ void reduce_f32(
    const float* __restrict__ in, float* __restrict__ out, ReduceParams p
) {
    long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (tid >= p.out_numel) return;
    int coords[MAX_RANK];
    long long rem = tid;
    for (int d = p.rank - 1; d >= 0; --d) {
        coords[d] = (int)(rem % (long long)p.dims_out[d]);
        rem /= (long long)p.dims_out[d];
    }
    float acc = init_value(p.op_code);
    for (long long inner = 0; inner < p.inner_size; ++inner) {
        long long irem = inner;
        for (int j = p.n_reduce - 1; j >= 0; --j) {
            int dim = p.dims_red[j];
            coords[p.red_axes[j]] = (int)(irem % (long long)dim);
            irem /= (long long)dim;
        }
        long long lin = p.in_offset;
        for (int d = 0; d < p.rank; ++d) {
            lin += (long long)coords[d] * (long long)p.strides_in[d];
        }
        float v = pass_f32(in[lin]);
        if (p.op_code == 0 || p.op_code == 1) acc += v;
        else if (p.op_code == 2) acc = fmaxf(acc, v);
    }
    if (p.op_code == 1 && p.inner_size > 0) acc /= (float)p.inner_size;
    out[tid] = acc;
}

extern "C" __global__ void reduce_f16(
    const __half* __restrict__ in, __half* __restrict__ out, ReduceParams p
) {
    long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (tid >= p.out_numel) return;
    int coords[MAX_RANK];
    long long rem = tid;
    for (int d = p.rank - 1; d >= 0; --d) {
        coords[d] = (int)(rem % (long long)p.dims_out[d]);
        rem /= (long long)p.dims_out[d];
    }
    float acc = init_value(p.op_code);
    for (long long inner = 0; inner < p.inner_size; ++inner) {
        long long irem = inner;
        for (int j = p.n_reduce - 1; j >= 0; --j) {
            int dim = p.dims_red[j];
            coords[p.red_axes[j]] = (int)(irem % (long long)dim);
            irem /= (long long)dim;
        }
        long long lin = p.in_offset;
        for (int d = 0; d < p.rank; ++d) {
            lin += (long long)coords[d] * (long long)p.strides_in[d];
        }
        float v = __half2float(in[lin]);
        if (p.op_code == 0 || p.op_code == 1) acc += v;
        else if (p.op_code == 2) acc = fmaxf(acc, v);
    }
    if (p.op_code == 1 && p.inner_size > 0) acc /= (float)p.inner_size;
    out[tid] = __float2half(acc);
}

extern "C" __global__ void reduce_bf16(
    const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out, ReduceParams p
) {
    long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (tid >= p.out_numel) return;
    int coords[MAX_RANK];
    long long rem = tid;
    for (int d = p.rank - 1; d >= 0; --d) {
        coords[d] = (int)(rem % (long long)p.dims_out[d]);
        rem /= (long long)p.dims_out[d];
    }
    float acc = init_value(p.op_code);
    for (long long inner = 0; inner < p.inner_size; ++inner) {
        long long irem = inner;
        for (int j = p.n_reduce - 1; j >= 0; --j) {
            int dim = p.dims_red[j];
            coords[p.red_axes[j]] = (int)(irem % (long long)dim);
            irem /= (long long)dim;
        }
        long long lin = p.in_offset;
        for (int d = 0; d < p.rank; ++d) {
            lin += (long long)coords[d] * (long long)p.strides_in[d];
        }
        float v = __bfloat162float(in[lin]);
        if (p.op_code == 0 || p.op_code == 1) acc += v;
        else if (p.op_code == 2) acc = fmaxf(acc, v);
    }
    if (p.op_code == 1 && p.inner_size > 0) acc /= (float)p.inner_size;
    out[tid] = __float2bfloat16(acc);
}

#define ARGMAX_KERNEL(name, T, to_f32) \
extern "C" __global__ void name( \
    const T* __restrict__ in, unsigned int* __restrict__ out, ReduceParams p \
) { \
    long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x; \
    if (tid >= p.out_numel) return; \
    int coords[MAX_RANK]; \
    long long rem = tid; \
    for (int d = p.rank - 1; d >= 0; --d) { \
        coords[d] = (int)(rem % (long long)p.dims_out[d]); \
        rem /= (long long)p.dims_out[d]; \
    } \
    float best = -3.4028235e38f; \
    unsigned int best_idx = 0; \
    for (long long inner = 0; inner < p.inner_size; ++inner) { \
        long long irem = inner; \
        for (int j = p.n_reduce - 1; j >= 0; --j) { \
            int dim = p.dims_red[j]; \
            coords[p.red_axes[j]] = (int)(irem % (long long)dim); \
            irem /= (long long)dim; \
        } \
        long long lin = p.in_offset; \
        for (int d = 0; d < p.rank; ++d) { \
            lin += (long long)coords[d] * (long long)p.strides_in[d]; \
        } \
        float v = to_f32(in[lin]); \
        if (v > best) { best = v; best_idx = (unsigned int)inner; } \
    } \
    out[tid] = best_idx; \
}

ARGMAX_KERNEL(argmax_f32, float, pass_f32)
ARGMAX_KERNEL(argmax_f16, __half, pass_f16)
ARGMAX_KERNEL(argmax_bf16, __nv_bfloat16, pass_bf16)
