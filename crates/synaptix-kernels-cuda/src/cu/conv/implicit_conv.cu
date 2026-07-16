#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))
namespace {

template <typename T>
__device__ __forceinline__ void
mma_m16n8k16(float &d0, float &d1, float &d2, float &d3, unsigned a0, unsigned a1,
             unsigned a2, unsigned a3, unsigned b0, unsigned b1);
template <>
__device__ __forceinline__ void
mma_m16n8k16<__nv_bfloat16>(float &d0, float &d1, float &d2, float &d3,
                            unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                            unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <>
__device__ __forceinline__ void
mma_m16n8k16<__half>(float &d0, float &d1, float &d2, float &d3, unsigned a0,
                     unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                     unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <typename T> __device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <> __device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half(v);
}
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

struct ConvEpi {
  const void *bias;
  const void *residual;
  const void *temb;
  int has_bias, has_residual, has_temb;
  int out_nhwc;
};

__device__ __forceinline__ int div_ceil(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ __forceinline__ void zero_smem16(unsigned smem_addr) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};\n" ::"r"(smem_addr),
               "r"(0));
}

struct ConvParams {
  int M, K, Nb, H, W, C, Kh, Kw, P, Q, sh, sw, ph, pw;
};

template <typename T>
__device__ __forceinline__ void load_a_conv(unsigned smem_addr,
                                            const T *__restrict__ input, int m,
                                            int kg, const ConvParams &cp) {
  bool valid = (m < cp.M) && (kg < cp.K);
  long long addr = 0;
  if (valid) {
    int q = m % cp.Q;
    int mt = m / cp.Q;
    int p = mt % cp.P;
    int nb = mt / cp.P;
    int c = kg % cp.C;
    int kt = kg / cp.C;
    int kw = kt % cp.Kw;
    int kh = kt / cp.Kw;
    int ih = p * cp.sh - cp.ph + kh;
    int iw = q * cp.sw - cp.pw + kw;
    if (ih >= 0 && ih < cp.H && iw >= 0 && iw < cp.W) {
      addr = (((long long)nb * cp.H + ih) * cp.W + iw) * (long long)cp.C + c;
    } else {
      valid = false;
    }
  }
  if (valid) {
    CP_ASYNC_CG(smem_addr, &input[addr], 16);
  } else {
    zero_smem16(smem_addr);
  }
}

template <bool PARTIAL>
__device__ __forceinline__ void load_b(unsigned smem_addr, const void *gmem,
                                       bool valid) {
  if constexpr (PARTIAL) {
    if (valid) {
      CP_ASYNC_CG(smem_addr, gmem, 16);
    } else {
      zero_smem16(smem_addr);
    }
  } else {
    CP_ASYNC_CG(smem_addr, gmem, 16);
  }
}

template <const int kColStride = 16, const int kStep = 8>
__device__ __forceinline__ int swizzle_permuted_j(int i, int j) {
  return (((j >> 3) ^ (i >> 2)) % (kColStride >> 3)) << 3;
}

template <typename T>
__device__ __forceinline__ void write_nchw(T *__restrict__ out, const ConvEpi &epi,
                                            int mm, int nn, float val,
                                            const ConvParams &cp, int N) {
  if (mm >= cp.M || nn >= N) return;
  if (epi.has_bias) val += to_float(((const T *)epi.bias)[nn]);
  if (epi.has_temb) {
    int b = mm / (cp.P * cp.Q);
    val += to_float(((const T *)epi.temb)[(long long)b * N + nn]);
  }
  long long idx;
  if (epi.out_nhwc) {
    idx = (long long)mm * N + nn;
  } else {
    int q = mm % cp.Q;
    int t = mm / cp.Q;
    int p = t % cp.P;
    int b = t / cp.P;
    idx = (((long long)b * N + nn) * cp.P + p) * cp.Q + q;
  }
  if (epi.has_residual) val += to_float(((const T *)epi.residual)[idx]);
  out[idx] = from_float<T>(val);
}

template <typename T, const int MMA_M = 16, const int MMA_N = 8,
          const int MMA_K = 16, const int MMA_TILE_M = 2,
          const int MMA_TILE_N = 4, const int WARP_TILE_M = 4,
          const int WARP_TILE_N = 4, const int WARP_TILE_K = 2,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 3,
          const bool BLOCK_SWIZZLE = true, const bool PARTIAL = false>
__device__ __forceinline__ void
implicit_conv_impl(const T *__restrict__ A, const T *__restrict__ B,
                   T *__restrict__ C, int M, int N, int K, ConvParams cp,
                   ConvEpi epi) {
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K * WARP_TILE_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;
  constexpr int BK = MMA_K;

  extern __shared__ __align__(16) char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);
  T *s_a = smem;
  T *s_b = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BN * (BK + B_PAD);
  constexpr int s_a_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
  constexpr int s_b_mma_k_store_offset = K_STAGE * BN * (BK + B_PAD);

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int warp_m = warp_id % 2;
  const int warp_n = warp_id / 2;

  int load_smem_a_m = tid / 2;
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;
  int load_smem_b_n = tid / 2;
  int load_smem_b_k = (tid % 2 == 0) ? 0 : 8;
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  if constexpr (!PARTIAL) {
    if (load_gmem_a_m >= M || load_gmem_b_n >= N)
      return;
  }
  const bool b_valid = load_gmem_b_n < N;

  float RC[WARP_TILE_M][WARP_TILE_N][4];
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i)
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0.f;
      RC[i][j][1] = 0.f;
      RC[i][j][2] = 0.f;
      RC[i][j][3] = 0.f;
    }

  unsigned smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  unsigned smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) {
    int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k;
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k;

    unsigned load_smem_a_ptr =
        (smem_a_base_ptr +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) +
          swizzle_permuted_j<MMA_K>(load_smem_a_m, load_smem_a_k)) *
             sizeof(T));
    load_a_conv<T>(load_smem_a_ptr, A, load_gmem_a_m, load_gmem_a_k, cp);
    unsigned load_smem_a_mma_k_ptr =
        (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(T) +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) +
          swizzle_permuted_j<MMA_K>(load_smem_a_m, load_smem_a_k)) *
             sizeof(T));
    load_a_conv<T>(load_smem_a_mma_k_ptr, A, load_gmem_a_m, load_gmem_a_k + 16,
                   cp);

    unsigned load_smem_b_ptr =
        (smem_b_base_ptr +
         (k * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) +
          swizzle_permuted_j<MMA_K>(load_smem_b_n, load_smem_b_k)) *
             sizeof(T));
    load_b<PARTIAL>(load_smem_b_ptr, &B[load_gmem_b_addr], b_valid);
    unsigned load_smem_b_mma_k_ptr =
        (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(T) +
         (k * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) +
          swizzle_permuted_j<MMA_K>(load_smem_b_n, load_smem_b_k)) *
             sizeof(T));
    load_b<PARTIAL>(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr + 16], b_valid);
    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
  __syncthreads();

  unsigned RA[2][WARP_TILE_M][4];
  unsigned RB[2][WARP_TILE_N][2];
  int reg_store_idx = 0;
  int reg_load_idx = 1;

  {
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
      int lane_smem_a_k = (lane_id / 16) * 8;
      unsigned lane_smem_a_ptr =
          (smem_a_base_ptr +
           (0 * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) +
            swizzle_permuted_j<MMA_K>(lane_smem_a_m, lane_smem_a_k)) *
               sizeof(T));
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                  lane_smem_a_ptr);
    }
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8;
      unsigned lane_smem_b_ptr =
          (smem_b_base_ptr +
           s_b_mma_k_store_offset * sizeof(T) * (lane_id / 16) +
           (0 * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) +
            swizzle_permuted_j<MMA_K>(lane_smem_b_n, lane_smem_b_k)) *
               sizeof(T));
      LDMATRIX_X4(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                  RB[reg_load_idx][j][0], RB[reg_load_idx][j][1],
                  lane_smem_b_ptr);
    }
  }

#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    reg_store_idx ^= 1;
    reg_load_idx ^= 1;
    int smem_sel = (k + 1) % K_STAGE;
    int smem_sel_next = k % K_STAGE;

    int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k;
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k;

    unsigned load_smem_a_ptr =
        (smem_a_base_ptr +
         (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) +
          swizzle_permuted_j<MMA_K>(load_smem_a_m, load_smem_a_k)) *
             sizeof(T));
    load_a_conv<T>(load_smem_a_ptr, A, load_gmem_a_m, load_gmem_a_k, cp);
    unsigned load_smem_a_mma_k_ptr =
        (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(T) +
         (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) +
          swizzle_permuted_j<MMA_K>(load_smem_a_m, load_smem_a_k)) *
             sizeof(T));
    load_a_conv<T>(load_smem_a_mma_k_ptr, A, load_gmem_a_m, load_gmem_a_k + 16,
                   cp);

    unsigned load_smem_b_ptr =
        (smem_b_base_ptr +
         (smem_sel_next * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) +
          swizzle_permuted_j<MMA_K>(load_smem_b_n, load_smem_b_k)) *
             sizeof(T));
    load_b<PARTIAL>(load_smem_b_ptr, &B[load_gmem_b_addr], b_valid);
    unsigned load_smem_b_mma_k_ptr =
        (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(T) +
         (smem_sel_next * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) +
          swizzle_permuted_j<MMA_K>(load_smem_b_n, load_smem_b_k)) *
             sizeof(T));
    load_b<PARTIAL>(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr + 16], b_valid);
    CP_ASYNC_COMMIT_GROUP();

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
      int lane_smem_a_k = (lane_id / 16) * 8;
      unsigned lane_smem_a_ptr =
          (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(T) +
           (smem_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) +
            swizzle_permuted_j<MMA_K>(lane_smem_a_m, lane_smem_a_k)) *
               sizeof(T));
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                  lane_smem_a_ptr);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j)
        mma_m16n8k16<T>(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3],
                      RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                      RA[reg_load_idx][i][2], RA[reg_load_idx][i][3],
                      RB[reg_load_idx][j][0], RB[reg_load_idx][j][1]);

    reg_store_idx ^= 1;
    reg_load_idx ^= 1;
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j)
        mma_m16n8k16<T>(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3],
                      RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                      RA[reg_load_idx][i][2], RA[reg_load_idx][i][3],
                      RB[reg_load_idx][j][0], RB[reg_load_idx][j][1]);

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    int smem_sel_reg = (smem_sel + 1) % K_STAGE;
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
      int lane_smem_a_k = (lane_id / 16) * 8;
      unsigned lane_smem_a_ptr =
          (smem_a_base_ptr +
           (smem_sel_reg * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) +
            swizzle_permuted_j<MMA_K>(lane_smem_a_m, lane_smem_a_k)) *
               sizeof(T));
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                  lane_smem_a_ptr);
    }
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8;
      unsigned lane_smem_b_ptr =
          (smem_b_base_ptr +
           s_b_mma_k_store_offset * sizeof(T) * (lane_id / 16) +
           (smem_sel_reg * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) +
            swizzle_permuted_j<MMA_K>(lane_smem_b_n, lane_smem_b_k)) *
               sizeof(T));
      LDMATRIX_X4(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                  RB[reg_load_idx][j][0], RB[reg_load_idx][j][1],
                  lane_smem_b_ptr);
    }
  }

  if constexpr ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }

  {
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      reg_store_idx ^= 1;
      reg_load_idx ^= 1;
      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
        int lane_smem_a_k = (lane_id / 16) * 8;
        unsigned lane_smem_a_ptr =
            (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(T) +
             (stage_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) +
              swizzle_permuted_j<MMA_K>(lane_smem_a_m, lane_smem_a_k)) *
                 sizeof(T));
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                    lane_smem_a_ptr);
      }
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i)
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
          mma_m16n8k16<T>(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3],
                        RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                        RA[reg_load_idx][i][2], RA[reg_load_idx][i][3],
                        RB[reg_load_idx][j][0], RB[reg_load_idx][j][1]);

      reg_store_idx ^= 1;
      reg_load_idx ^= 1;
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i)
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
          mma_m16n8k16<T>(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3],
                        RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                        RA[reg_load_idx][i][2], RA[reg_load_idx][i][3],
                        RB[reg_load_idx][j][0], RB[reg_load_idx][j][1]);

      int stage_sel_reg = (stage_sel + 1) % K_STAGE;
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
        int lane_smem_a_k = (lane_id / 16) * 8;
        unsigned lane_smem_a_ptr =
            (smem_a_base_ptr +
             (stage_sel_reg * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) +
              swizzle_permuted_j<MMA_K>(lane_smem_a_m, lane_smem_a_k)) *
                 sizeof(T));
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                    lane_smem_a_ptr);
      }
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
        int lane_smem_b_k = ((lane_id / 8) % 2) * 8;
        unsigned lane_smem_b_ptr =
            (smem_b_base_ptr +
             s_b_mma_k_store_offset * sizeof(T) * (lane_id / 16) +
             (stage_sel_reg * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) +
              swizzle_permuted_j<MMA_K>(lane_smem_b_n, lane_smem_b_k)) *
                 sizeof(T));
        LDMATRIX_X4(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                    RB[reg_load_idx][j][0], RB[reg_load_idx][j][1],
                    lane_smem_b_ptr);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i)
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int gmem_m = by * BM + warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int gmem_n = bx * BN + warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int row = lane_id / 4;
      int col = (lane_id % 4) * 2;
      float *d = RC[i][j];
      write_nchw<T>(C, epi, gmem_m + row, gmem_n + col, d[0], cp, N);
      write_nchw<T>(C, epi, gmem_m + row, gmem_n + col + 1, d[1], cp, N);
      write_nchw<T>(C, epi, gmem_m + row + 8, gmem_n + col, d[2], cp, N);
      write_nchw<T>(C, epi, gmem_m + row + 8, gmem_n + col + 1, d[3], cp, N);
    }
}

} // namespace

#define IMPLICIT_CONV_ENTRY(T, NAME, KSTAGE, SWZ, PART)                        \
  extern "C" __global__ void __launch_bounds__(256) NAME(                      \
      const T *A, const T *B, T *C, int M, int N, int K, int Nb, int H, int W, \
      int Cc, int Kh, int Kw, int P, int Q, int sh, int sw, int ph, int pw,    \
      const T *bias, int has_bias, const T *residual, int has_residual,        \
      const T *temb, int has_temb, int out_nhwc) {                             \
    ConvParams cp{M, K, Nb, H, W, Cc, Kh, Kw, P, Q, sh, sw, ph, pw};           \
    ConvEpi epi{(const void *)bias, (const void *)residual, (const void *)temb,\
                has_bias, has_residual, has_temb, out_nhwc};                   \
    implicit_conv_impl<T, 16, 8, 16, 2, 4, 4, 4, 2, 0, 0, KSTAGE, SWZ, PART>(  \
        A, B, C, M, N, K, cp, epi);                                            \
  }

IMPLICIT_CONV_ENTRY(__nv_bfloat16, implicit_conv_bf16_swz_s3, 3, true, false)
IMPLICIT_CONV_ENTRY(__nv_bfloat16, implicit_conv_bf16_swz_s4, 4, true, false)
IMPLICIT_CONV_ENTRY(__nv_bfloat16, implicit_conv_bf16_part, 3, false, true)
IMPLICIT_CONV_ENTRY(__half, implicit_conv_f16_swz_s3, 3, true, false)
IMPLICIT_CONV_ENTRY(__half, implicit_conv_f16_swz_s4, 4, true, false)
IMPLICIT_CONV_ENTRY(__half, implicit_conv_f16_part, 3, false, true)
