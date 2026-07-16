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
