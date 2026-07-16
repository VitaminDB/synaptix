#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))

namespace {

__device__ __forceinline__ int div_ceil(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ __forceinline__ void cp_async_or_zero(unsigned smem_addr,
                                                 const void *gmem, bool valid) {
  if (valid) {
    CP_ASYNC_CG(smem_addr, gmem, 16);
  } else {
    asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};\n" ::"r"(smem_addr),
                 "r"(0));
  }
}

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

// f32-аккумулятор: WMMA store_matrix_sync пишет float-фрагмент во float-стейдж,
// затем конвертим в T и раскладываем в C (bounds-проверка при partial).
template <typename T>
__device__ __forceinline__ void
store_tile(T *C, int N, int M, int gm, int gn, bool bounded,
           wmma::fragment<wmma::accumulator, 16, 16, 16, float> &f,
           float *fstage, int lane) {
  wmma::store_matrix_sync(fstage, f, 16, wmma::mem_row_major);
  __syncwarp();
#pragma unroll
  for (int e = 0; e < 8; e++) {
    int idx = lane * 8 + e;
    int gr = gm + (idx >> 4);
    int gc = gn + (idx & 15);
    if (!bounded || (gr < M && gc < N))
      C[(size_t)gr * N + gc] = from_float<T>(fstage[idx]);
  }
  __syncwarp();
}

template <typename T, const int WMMA_M = 16, const int WMMA_N = 16,
          const int WMMA_K = 16, const int WMMA_TILE_M = 4,
          const int WMMA_TILE_N = 2, const int WARP_TILE_M = 2,
          const int WARP_TILE_N = 4, const int A_PAD = 0, const int B_PAD = 16,
          const int K_STAGE = 3, const bool BLOCK_SWIZZLE = true,
          const bool PARTIAL = false>
__device__ __forceinline__ void
hgemm_wmma_stages_impl(const T *A, const T *B, T *C, int M, int N, int K) {
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
  constexpr int BK = WMMA_K;

  __shared__ T s_a[K_STAGE][BM][BK + A_PAD];
  __shared__ T s_b[K_STAGE][BK][BN + B_PAD];
  __shared__ float fstage[8][256];
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid / 32;
  const int warp_m = warp_id / 2;
  const int warp_n = warp_id % 2;

  int load_smem_a_m = tid / 2;
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;
  int load_smem_b_k = tid / 16;
  int load_smem_b_n = (tid % 16) * 8;
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  if constexpr (!PARTIAL) {
    if (load_gmem_a_m >= M || load_gmem_b_n >= N)
      return;
  }
  const bool a_valid = load_gmem_a_m < M;
  const bool b_valid = load_gmem_b_n < N;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      C_frag[WARP_TILE_M][WARP_TILE_N];
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i)
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j)
      wmma::fill_fragment(C_frag[i][j], 0.0);

  unsigned smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  unsigned smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) {
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    unsigned load_smem_a_ptr =
        (smem_a_base_ptr +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
             sizeof(T));
    unsigned load_smem_b_ptr =
        (smem_b_base_ptr +
         (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
             sizeof(T));
    if constexpr (PARTIAL) {
      cp_async_or_zero(load_smem_a_ptr, &A[load_gmem_a_addr], a_valid);
      cp_async_or_zero(load_smem_b_ptr, &B[load_gmem_b_addr], b_valid);
    } else {
      CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
      CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
  __syncthreads();

#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
    int smem_sel = (k + 1) % K_STAGE;
    int smem_sel_next = k % K_STAGE;

    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    unsigned load_smem_a_ptr =
        (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset +
                            load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
                               sizeof(T));
    unsigned load_smem_b_ptr =
        (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset +
                            load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
                               sizeof(T));
    if constexpr (PARTIAL) {
      cp_async_or_zero(load_smem_a_ptr, &A[load_gmem_a_addr], a_valid);
      cp_async_or_zero(load_smem_b_ptr, &B[load_gmem_b_addr], b_valid);
    } else {
      CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
      CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T,
                   wmma::row_major>
        A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T,
                   wmma::row_major>
        B_frag[WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0],
                             BK + A_PAD);
    }
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n],
                             BN + B_PAD);
    }
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j)
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
  }

  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }

  {
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T,
                     wmma::row_major>
          A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T,
                     wmma::row_major>
          B_frag[WARP_TILE_N];
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[i], &s_a[stage_sel][warp_smem_a_m][0],
                               BK + A_PAD);
      }
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[j], &s_b[stage_sel][0][warp_smem_b_n],
                               BN + B_PAD);
      }
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i)
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
    }
  }

#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i)
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int gm = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int gn = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      store_tile(C, N, M, gm, gn, PARTIAL, C_frag[i][j], fstage[warp_id], lane);
    }
}

} // namespace

#define WMMA_NN_ENTRY(T, NAME, KSTAGE, SWZ, PART)                              \
  extern "C" __global__ void __launch_bounds__(256)                            \
      NAME(const T *A, const T *B, T *C, int M, int N, int K) {                \
    hgemm_wmma_stages_impl<T, 16, 16, 16, 4, 2, 2, 4, 0, 16, KSTAGE, SWZ,      \
                           PART>(A, B, C, M, N, K);                            \
  }

WMMA_NN_ENTRY(__half, gemm_wmma_f16_s3_swz, 3, true, false)
WMMA_NN_ENTRY(__half, gemm_wmma_f16_s2_swz, 2, true, false)
WMMA_NN_ENTRY(__half, gemm_wmma_f16_s4_swz, 4, true, false)
WMMA_NN_ENTRY(__half, gemm_wmma_f16_s3_noswz, 3, false, false)
WMMA_NN_ENTRY(__half, gemm_wmma_f16_part, 3, false, true)
WMMA_NN_ENTRY(__nv_bfloat16, gemm_wmma_bf16_s3_swz, 3, true, false)
WMMA_NN_ENTRY(__nv_bfloat16, gemm_wmma_bf16_s2_swz, 2, true, false)
WMMA_NN_ENTRY(__nv_bfloat16, gemm_wmma_bf16_s4_swz, 4, true, false)
WMMA_NN_ENTRY(__nv_bfloat16, gemm_wmma_bf16_s3_noswz, 3, false, false)
WMMA_NN_ENTRY(__nv_bfloat16, gemm_wmma_bf16_part, 3, false, true)
