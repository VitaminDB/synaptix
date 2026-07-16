// Истинный F32 GEMM (NN): C[M,N] = A[M,K] @ B[K,N], row-major. На sm_120 нет
// tensor-core пути для f32 → tiled SIMT (CUDA-cores, float-аккумулятор по
// определению), register-blocking 4x4. Bounds-проверки → любые M/N/K. cuBLAS/
// CUTLASS считают f32 так же. Перф не критичен (f32 на GPU редок, compute=F16/BF16).

extern "C" __global__ void __launch_bounds__(256)
    gemm_f32_nn(const float *A, const float *B, float *C, int M, int N, int K) {
  constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
  __shared__ float sa[BM][BK];
  __shared__ float sb[BK][BN];

  const int tid = threadIdx.x;       // 0..255
  const int tx = tid % (BN / TN);    // 0..15
  const int ty = tid / (BN / TN);    // 0..15
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++)
      acc[i][j] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int t = 0; t < (BM * BK) / 256; t++) {
      int idx = tid + t * 256;
      int r = idx / BK, c = idx % BK;
      int gr = block_row + r, gc = k0 + c;
      sa[r][c] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.f;
    }
#pragma unroll
    for (int t = 0; t < (BK * BN) / 256; t++) {
      int idx = tid + t * 256;
      int r = idx / BN, c = idx % BN;
      int gr = k0 + r, gc = block_col + c;
      sb[r][c] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; kk++) {
      float ar[TM], br[TN];
#pragma unroll
      for (int i = 0; i < TM; i++)
        ar[i] = sa[ty * TM + i][kk];
#pragma unroll
      for (int j = 0; j < TN; j++)
        br[j] = sb[kk][tx * TN + j];
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++)
          acc[i][j] += ar[i] * br[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) {
      int r = block_row + ty * TM + i;
      int c = block_col + tx * TN + j;
      if (r < M && c < N)
        C[(size_t)r * N + c] = acc[i][j];
    }
}
