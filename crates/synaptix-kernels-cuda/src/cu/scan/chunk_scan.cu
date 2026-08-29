// Примитивы для chunk-scan оркестратора (chunked gated delta rule).
//
// Самодостаточные f32-ядра (cuBLAS выпилен из synaptix): cumsum по чанку,
// L2-norm со scale, построчное умножение на скаляр, наивный strided-batched GEMM.
// Тяжёлые chunk-aware части делает cu/chunk_fla.cu; здесь — препроцессинг и bmm.

extern "C" {

// out[row, i] = Σ_{j≤i} in[row, j].  Per-chunk cumsum: rows = BH*NC, n = CS.
// Один thread на строку (n мал, ≤ CS).
__global__ void cumsum_lastdim_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    unsigned int rows,
    unsigned int n
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    unsigned long long base = (unsigned long long)row * n;
    float acc = 0.0f;
    for (unsigned int i = 0; i < n; ++i) {
        acc += in[base + i];
        out[base + i] = acc;
    }
}

// out[row, d] = in[row, d] / sqrt(mean_or_sum?) ... L2: / sqrt(Σ_d x² + eps) * scale.
// Один thread на строку. rows = BH*T, dim = HK.
__global__ void l2norm_scale_lastdim_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    unsigned int rows,
    unsigned int dim,
    float scale,
    float eps
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    unsigned long long base = (unsigned long long)row * dim;
    float ss = 0.0f;
    for (unsigned int d = 0; d < dim; ++d) {
        float v = in[base + d];
        ss += v * v;
    }
    float inv = rsqrtf(ss + eps) * scale;
    for (unsigned int d = 0; d < dim; ++d) {
        out[base + d] = in[base + d] * inv;
    }
}

// out[row, d] = in[row, d] * scal[row]. Построчное умножение на скаляр.
__global__ void mul_rowwise_f32(
    const float* __restrict__ in,
    const float* __restrict__ scal,
    float* __restrict__ out,
    unsigned int rows,
    unsigned int dim
) {
    unsigned int row = blockIdx.x;
    unsigned int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows || d >= dim) return;
    unsigned long long idx = (unsigned long long)row * dim + d;
    out[idx] = in[idx] * scal[row];
}

// Наивный strided-batched GEMM: C = alpha * op(A) · op(B) + beta * C.
//   op(A) = (M,K) [transA? A хранится (K,M)];  op(B) = (K,N) [transB? B хранится (N,K)];
//   C = (M,N) row-major. Offsets/strides — в элементах. Один thread на (batch, m, n).
__global__ void bmm_f32(
    const float* __restrict__ A, unsigned int offA,
    const float* __restrict__ B, unsigned int offB,
    float* __restrict__ C, unsigned int offC,
    int transA, int transB,
    unsigned int M, unsigned int N, unsigned int K,
    long long strideA, long long strideB, long long strideC,
    unsigned int batch,
    float alpha, float beta
) {
    unsigned long long total = (unsigned long long)batch * M * N;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int bi = (unsigned int)(idx / ((unsigned long long)M * N));
    unsigned int rem = (unsigned int)(idx % ((unsigned long long)M * N));
    unsigned int m = rem / N;
    unsigned int n = rem % N;

    const float* Ab = A + offA + (long long)bi * strideA;
    const float* Bb = B + offB + (long long)bi * strideB;
    float* Cb = C + offC + (long long)bi * strideC;

    float acc = 0.0f;
    for (unsigned int kk = 0; kk < K; ++kk) {
        float a = transA ? Ab[(unsigned long long)kk * M + m] : Ab[(unsigned long long)m * K + kk];
        float b = transB ? Bb[(unsigned long long)n * K + kk] : Bb[(unsigned long long)kk * N + n];
        acc += a * b;
    }
    unsigned long long cidx = (unsigned long long)m * N + n;
    float prev = (beta != 0.0f) ? Cb[cidx] : 0.0f;
    Cb[cidx] = alpha * acc + beta * prev;
}

// Тайловый вариант того же GEMM: тайл 16×16 через shared, по одному элементу
// C на поток. Наивная версия читала A и B из глобальной памяти на каждый шаг
// K — на формах чанк-скана (64×64×128, сотни матриц в батче) это и было
// главной статьёй расхода линейного внимания.
#define BMM_TILE 16

__global__ void bmm_tiled_f32(
    const float* __restrict__ A, unsigned int offA,
    const float* __restrict__ B, unsigned int offB,
    float* __restrict__ C, unsigned int offC,
    int transA, int transB,
    unsigned int M, unsigned int N, unsigned int K,
    long long strideA, long long strideB, long long strideC,
    unsigned int batch,
    float alpha, float beta
) {
    __shared__ float As[BMM_TILE][BMM_TILE + 1];
    __shared__ float Bs[BMM_TILE][BMM_TILE + 1];

    unsigned int bi = blockIdx.z;
    if (bi >= batch) return;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int m = blockIdx.y * BMM_TILE + ty;
    unsigned int n = blockIdx.x * BMM_TILE + tx;

    const float* Ab = A + offA + (long long)bi * strideA;
    const float* Bb = B + offB + (long long)bi * strideB;
    float* Cb = C + offC + (long long)bi * strideC;

    float acc = 0.0f;
    for (unsigned int k0 = 0; k0 < K; k0 += BMM_TILE) {
        unsigned int ka = k0 + tx;
        As[ty][tx] = (m < M && ka < K)
            ? (transA ? Ab[(unsigned long long)ka * M + m] : Ab[(unsigned long long)m * K + ka])
            : 0.0f;
        unsigned int kb = k0 + ty;
        Bs[ty][tx] = (n < N && kb < K)
            ? (transB ? Bb[(unsigned long long)n * K + kb] : Bb[(unsigned long long)kb * N + n])
            : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BMM_TILE; ++kk) acc += As[ty][kk] * Bs[kk][tx];
        __syncthreads();
    }
    if (m >= M || n >= N) return;
    unsigned long long cidx = (unsigned long long)m * N + n;
    float prev = (beta != 0.0f) ? Cb[cidx] : 0.0f;
    Cb[cidx] = alpha * acc + beta * prev;
}

// Регистровый тайл: блок 64×64 элементов C, поток считает 4×4. Тайл 16×16 с
// одним элементом на поток упирался в чтение shared — на формах чанк-скана
// (64×128×128, десятки матриц в батче) это половина времени линейного
// внимания.
#define BMM2_BM 64
#define BMM2_BN 64
#define BMM2_BK 16
#define BMM2_TM 4
#define BMM2_TN 4

__global__ void bmm_reg_f32(
    const float* __restrict__ A, unsigned int offA,
    const float* __restrict__ B, unsigned int offB,
    float* __restrict__ C, unsigned int offC,
    int transA, int transB,
    unsigned int M, unsigned int N, unsigned int K,
    long long strideA, long long strideB, long long strideC,
    unsigned int batch,
    float alpha, float beta
) {
    __shared__ float As[BMM2_BK][BMM2_BM + 1];
    __shared__ float Bs[BMM2_BK][BMM2_BN + 1];

    unsigned int bi = blockIdx.z;
    if (bi >= batch) return;
    const float* Ab = A + offA + (long long)bi * strideA;
    const float* Bb = B + offB + (long long)bi * strideB;
    float* Cb = C + offC + (long long)bi * strideC;

    unsigned int m0 = blockIdx.y * BMM2_BM;
    unsigned int n0 = blockIdx.x * BMM2_BN;
    unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int tm = (tid / (BMM2_BN / BMM2_TN)) * BMM2_TM;
    unsigned int tn = (tid % (BMM2_BN / BMM2_TN)) * BMM2_TN;

    float acc[BMM2_TM][BMM2_TN];
    #pragma unroll
    for (int i = 0; i < BMM2_TM; ++i)
        #pragma unroll
        for (int j = 0; j < BMM2_TN; ++j) acc[i][j] = 0.0f;

    for (unsigned int k0 = 0; k0 < K; k0 += BMM2_BK) {
        // Каждый из 256 потоков тянет по четыре элемента A и B.
        #pragma unroll
        for (unsigned int rep = 0; rep < (BMM2_BM * BMM2_BK) / 256; ++rep) {
            unsigned int idx = rep * 256 + tid;
            unsigned int kk = idx / BMM2_BM, mm = idx % BMM2_BM;
            unsigned int gm = m0 + mm, gk = k0 + kk;
            As[kk][mm] = (gm < M && gk < K)
                ? (transA ? Ab[(unsigned long long)gk * M + gm] : Ab[(unsigned long long)gm * K + gk])
                : 0.0f;
        }
        #pragma unroll
        for (unsigned int rep = 0; rep < (BMM2_BN * BMM2_BK) / 256; ++rep) {
            unsigned int idx = rep * 256 + tid;
            unsigned int kk = idx / BMM2_BN, nn = idx % BMM2_BN;
            unsigned int gn = n0 + nn, gk = k0 + kk;
            Bs[kk][nn] = (gn < N && gk < K)
                ? (transB ? Bb[(unsigned long long)gn * K + gk] : Bb[(unsigned long long)gk * N + gn])
                : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BMM2_BK; ++kk) {
            float a[BMM2_TM], b[BMM2_TN];
            #pragma unroll
            for (int i = 0; i < BMM2_TM; ++i) a[i] = As[kk][tm + i];
            #pragma unroll
            for (int j = 0; j < BMM2_TN; ++j) b[j] = Bs[kk][tn + j];
            #pragma unroll
            for (int i = 0; i < BMM2_TM; ++i)
                #pragma unroll
                for (int j = 0; j < BMM2_TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < BMM2_TM; ++i) {
        unsigned int m = m0 + tm + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < BMM2_TN; ++j) {
            unsigned int n = n0 + tn + j;
            if (n >= N) continue;
            unsigned long long cidx = (unsigned long long)m * N + n;
            float prev = (beta != 0.0f) ? Cb[cidx] : 0.0f;
            Cb[cidx] = alpha * acc[i][j] + beta * prev;
        }
    }
}

} // extern "C"
