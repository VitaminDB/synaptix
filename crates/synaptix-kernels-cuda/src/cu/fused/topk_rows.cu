#include <cuda_fp16.h>

// Top-k по строкам небольшой матрицы (роутер MoE: 512 экспертов, 10 слотов).
//
// Строка целиком ложится в shared, дальше k раз берётся максимум блочной
// редукцией и вычёркивается. Смысл не в арифметике, а в том, что на хост
// уезжают k индексов со значениями вместо всей строки логитов: на промпте в
// 35k токенов это 3 МБ вместо 72 МБ на слой, и выбор больше не считается на
// процессоре.
//
//   scores  (rows, cols) f32
//   out_idx (rows, k)    u32
//   out_val (rows, k)    f32

#define TOPK_NEG_INF (__int_as_float(0xFF800000))

extern "C" __global__ void topk_rows_f32(
    const float* __restrict__ scores,
    unsigned int* __restrict__ out_idx,
    float* __restrict__ out_val,
    unsigned int rows,
    unsigned int cols,
    unsigned int k
) {
    extern __shared__ float sh[];
    float* vals = sh;                                   // [cols]
    float* red_v = vals + cols;                         // [blockDim.x]
    unsigned int* red_i = (unsigned int*)(red_v + blockDim.x);

    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    unsigned int tid = threadIdx.x, bs = blockDim.x;

    const float* src = scores + (unsigned long long)row * cols;
    for (unsigned int i = tid; i < cols; i += bs) vals[i] = src[i];
    __syncthreads();

    for (unsigned int slot = 0; slot < k; ++slot) {
        float best = TOPK_NEG_INF;
        unsigned int best_i = 0;
        for (unsigned int i = tid; i < cols; i += bs) {
            float v = vals[i];
            // При равенстве побеждает меньший индекс — так же, как на хосте.
            if (v > best) {
                best = v;
                best_i = i;
            }
        }
        // Сведение argmax: варп шаффлами, затем лидеры варпов — первым
        // варпом (2 барьера на раунд вместо log2(blockDim)). Ничья — меньший
        // индекс, как и прежде.
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, best, o);
            unsigned int oi = __shfl_xor_sync(0xffffffffu, best_i, o);
            if (ov > best || (ov == best && oi < best_i)) {
                best = ov;
                best_i = oi;
            }
        }
        unsigned int lane = tid & 31u, wid = tid >> 5, nw = (bs + 31u) >> 5;
        if (lane == 0) {
            red_v[wid] = best;
            red_i[wid] = best_i;
        }
        __syncthreads();
        if (wid == 0) {
            best = lane < nw ? red_v[lane] : TOPK_NEG_INF;
            best_i = lane < nw ? red_i[lane] : 0xffffffffu;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                float ov = __shfl_xor_sync(0xffffffffu, best, o);
                unsigned int oi = __shfl_xor_sync(0xffffffffu, best_i, o);
                if (ov > best || (ov == best && oi < best_i)) {
                    best = ov;
                    best_i = oi;
                }
            }
            if (lane == 0) {
                red_v[0] = best;
                red_i[0] = best_i;
            }
        }
        __syncthreads();
        if (tid == 0) {
            unsigned long long o = (unsigned long long)row * k + slot;
            out_idx[o] = red_i[0];
            out_val[o] = red_v[0];
            vals[red_i[0]] = TOPK_NEG_INF;
        }
        __syncthreads();
    }
}
