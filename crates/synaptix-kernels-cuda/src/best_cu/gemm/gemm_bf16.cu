#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst),       \
               "l"(src), "n"(bytes))
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr)                                              \
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"    \
               : "=r"(R0), "=r"(R1)                                            \
               : "r"(addr))
namespace {

// Тип-специфичные операции (TN-ядро шарят bf16 и f16: одинаковый pipeline,
// 16-битные ldmatrix/cp.async; различаются только MMA-инструкция и store-cvt).
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

template <typename T> struct Pack2;
template <> struct Pack2<__nv_bfloat16> { using type = __nv_bfloat162; };
template <> struct Pack2<__half> { using type = __half2; };

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

__device__ __forceinline__ int div_ceil(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

template <bool PARTIAL>
__device__ __forceinline__ void load16(unsigned smem_addr, const void *gmem,
                                       bool valid) {
  if constexpr (PARTIAL) {
    if (valid) {
      CP_ASYNC_CG(smem_addr, gmem, 16);
    } else {
      asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};\n" ::"r"(smem_addr),
                   "r"(0));
    }
  } else {
    CP_ASYNC_CG(smem_addr, gmem, 16);
  }
}

// L2::256B prefetch-hint на потоковом чтении W (B-матрица читается ровно 1 раз,
// DRAM-страница открывается на полную 256B-линию — тот же приём, что L2-promotion
// у TMA-дескрипторов, давший +5% в aee00fad). A не метим: он горячий в L2.
#define CP_ASYNC_CG_L2_256(dst, src)                                           \
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" ::"r"(   \
                   dst),                                                       \
               "l"(src))

template <bool PARTIAL>
__device__ __forceinline__ void load16_l2(unsigned smem_addr, const void *gmem,
                                          bool valid) {
  if constexpr (PARTIAL) {
    if (valid) {
      CP_ASYNC_CG_L2_256(smem_addr, gmem);
    } else {
      asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};\n" ::"r"(smem_addr),
                   "r"(0));
    }
  } else {
    CP_ASYNC_CG_L2_256(smem_addr, gmem);
  }
}

template <const int kColStride = 16, const int kStep = 8>
__device__ __forceinline__ int swizzle_permuted_j(int i, int j) {
  return (((j >> 3) ^ (i >> 2)) % (kColStride >> 3)) << 3;
}

// ldk — страйд строк A/B (полный K матрицы); K — длина обрабатываемого
// k-диапазона (= ldk без split-K). ws != nullptr → split-K режим: партиалы
// пишутся в f32-workspace (bias/residual добавит reduce-ядро).
template <typename T, const int MMA_M = 16, const int MMA_N = 8,
          const int MMA_K = 16, const int MMA_TILE_M = 2,
          const int MMA_TILE_N = 4, const int WARP_TILE_M = 4,
          const int WARP_TILE_N = 4, const int WARP_TILE_K = 2,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 3,
          const bool BLOCK_SWIZZLE = true, const bool PARTIAL = false,
          const int SLACK = 1>
__device__ __forceinline__ void
gemm_bf16_impl_ex(const T *__restrict__ A, const T *__restrict__ B,
                  T *__restrict__ C, int M, int N, int K,
                  const T *__restrict__ bias, int has_bias,
                  const T *__restrict__ residual, int has_residual,
                  int ldk, float *__restrict__ ws) {
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K * WARP_TILE_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;
  constexpr int BK = MMA_K;

  // type-erased dynamic smem: одно имя на все инстанциации T (иначе extern
  // __shared__ T smem[] конфликтует при T=bf16 и T=__half).
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
  const int warp_m = warp_id % MMA_TILE_M;
  const int warp_n = warp_id / MMA_TILE_M;

  // g2s-загрузчик: 4 потока на строку (64B contiguous на строку за варп —
  // полные L2-линии вместо 32B-фрагментов), 64 строки за проход.
  static_assert(BM % 64 == 0 && BN % 64 == 0, "g2s рассчитан на тайлы кратные 64");
  const int lrow = tid / 4;
  const int lchunk = tid % 4;
  const int lk = (lchunk % 2) * 8;
  const int lreg_a_off = (lchunk / 2) * s_a_mma_k_store_offset;
  const int lreg_b_off = (lchunk / 2) * s_b_mma_k_store_offset;
  if constexpr (!PARTIAL) {
    if (by * BM + BM - 1 >= M || bx * BN + BN - 1 >= N)
      return;
  }
  bool a_valid_r[BM / 64];
#pragma unroll
  for (int p = 0; p < BM / 64; ++p)
    a_valid_r[p] = by * BM + lrow + p * 64 < M;
  bool b_valid_r[BN / 64];
#pragma unroll
  for (int p = 0; p < BN / 64; ++p)
    b_valid_r[p] = bx * BN + lrow + p * 64 < N;

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

  // Полная g2s-загрузка одной стадии (A+B) в слот slot из k-тайла ktile.
  auto g2s_stage = [&](int slot, int ktile) {
    const int gk = ktile * (BK * WARP_TILE_K) + lchunk * 8;
#pragma unroll
    for (int p = 0; p < BM / 64; ++p) {
      int row = lrow + p * 64;
      unsigned dst =
          (smem_a_base_ptr +
           (lreg_a_off + slot * s_a_stage_offset + row * (BK + A_PAD) +
            swizzle_permuted_j<MMA_K>(row, lk)) *
               sizeof(T));
      load16<PARTIAL>(dst, &A[(size_t)(by * BM + row) * ldk + gk], a_valid_r[p]);
    }
#pragma unroll
    for (int p = 0; p < BN / 64; ++p) {
      int row = lrow + p * 64;
      unsigned dst =
          (smem_b_base_ptr +
           (lreg_b_off + slot * s_b_stage_offset + row * (BK + B_PAD) +
            swizzle_permuted_j<MMA_K>(row, lk)) *
               sizeof(T));
      load16<PARTIAL>(dst, &B[(size_t)(bx * BN + row) * ldk + gk], b_valid_r[p]);
    }
  };

#pragma unroll
  for (int k = 0; k < (K_STAGE - SLACK); ++k) {
    g2s_stage(k, k);
    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - SLACK - 1);
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
    int smem_sel_next = (k + 1 - SLACK) % K_STAGE;

    // wait+барьер в голове итерации: все варпы синхронизируются сразу после
    // общего события памяти, дрейф не ресинкается посреди MMA-конвейера.
    // Глубина K_STAGE-3 до commit'а этой итерации даёт ту же гарантию
    // свежести тайла, что K_STAGE-2 после (для s2 commit обязан случиться
    // раньше wait — оставляем старую позицию ниже).
    // SLACK>1: g2s пишет тайл k+1-SLACK — слот, читанный SLACK итераций
    // назад → достаточно барьера раз в SLACK (писатель не догонит читателя
    // в пределах окна); wait остаётся каждую итерацию (per-warp fence).
    if constexpr (K_STAGE >= 3) {
      CP_ASYNC_WAIT_GROUP(K_STAGE - SLACK - 2);
      if (SLACK == 1 || ((k - (K_STAGE - 1)) % SLACK) == 0)
        __syncthreads();
    }

    g2s_stage(smem_sel_next, k + 1 - SLACK);
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

    // Софт-пайплайн: ldmatrix следующего тайла идут ДО второй MMA-группы
    // (латентность прячется под её 32 MMA), B-half2 — после (прячется под g1
    // следующей итерации). Банки: A-half1-next/B-half1-next → store-банк
    // (потреблён g1), B-half2-next → load-банк (потреблён g2 ниже).
    if constexpr (K_STAGE < 3) {
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads();
    }

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
           (smem_sel_reg * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) +
            swizzle_permuted_j<MMA_K>(lane_smem_b_n, lane_smem_b_k)) *
               sizeof(T));
      LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                  lane_smem_b_ptr);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j)
        mma_m16n8k16<T>(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3],
                      RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                      RA[reg_load_idx][i][2], RA[reg_load_idx][i][3],
                      RB[reg_load_idx][j][0], RB[reg_load_idx][j][1]);

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8;
      unsigned lane_smem_b_ptr =
          (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(T) +
           (smem_sel_reg * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) +
            swizzle_permuted_j<MMA_K>(lane_smem_b_n, lane_smem_b_k)) *
               sizeof(T));
      LDMATRIX_X2(RB[reg_load_idx][j][0], RB[reg_load_idx][j][1],
                  lane_smem_b_ptr);
    }
  }

  if constexpr (SLACK > 1) {
    __syncthreads();
#pragma unroll
    for (int d = 1; d < SLACK; ++d) {
      int t = NUM_K_TILES - SLACK + d;
      g2s_stage(t % K_STAGE, t);
      CP_ASYNC_COMMIT_GROUP();
    }
  }
  if constexpr ((K_STAGE - 2) > 0 || SLACK > 1) {
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
      float bc = has_bias ? to_float(bias[gmem_n + col]) : 0.f;
      float bc1 = (has_bias && gmem_n + col + 1 < N) ? to_float(bias[gmem_n + col + 1]) : 0.f;
      // Пара (col, col+1) пишется одним 32-битным стором на горячем пути
      // (обе колонки в N и адрес выровнен на 4B — всегда при чётном N);
      // иначе скалярный fallback (хвост нечётного N).
      using T2 = typename Pack2<T>::type;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        int gm = gmem_m + row + half * 8;
        if (gm >= M)
          continue;
        size_t r0 = (size_t)gm * N + gmem_n + col;
        float v0 = d[half * 2] + bc;
        float v1 = d[half * 2 + 1] + bc1;
        if (ws != nullptr) {
          // split-K: f32-партиал (streaming — ws не перечитывается этим ядром)
          if (gmem_n + col < N)
            __stcs(&ws[r0], v0);
          if (gmem_n + col + 1 < N)
            __stcs(&ws[r0 + 1], v1);
          continue;
        }
        if (gmem_n + col + 1 < N && (r0 & 1) == 0) {
          if (has_residual) {
            v0 += to_float(residual[r0]);
            v1 += to_float(residual[r0 + 1]);
          }
          T2 out2;
          out2.x = from_float<T>(v0);
          out2.y = from_float<T>(v1);
          // streaming-store (evict-first): C не перечитывается, не даём
          // write-allocate вытеснять горячий B-тайл из L2.
          asm volatile("st.global.cs.b32 [%0], %1;\n" ::"l"(&C[r0]),
                       "r"(*reinterpret_cast<unsigned *>(&out2)));
        } else {
          if (gmem_n + col < N)
            C[r0] = from_float<T>(
                v0 + (has_residual ? to_float(residual[r0]) : 0.f));
          if (gmem_n + col + 1 < N)
            C[r0 + 1] = from_float<T>(
                v1 + (has_residual ? to_float(residual[r0 + 1]) : 0.f));
        }
      }
    }
}

// Обёртка под историческую сигнатуру (без split-K): ldk = K, ws = nullptr.
template <typename T, const int MMA_M = 16, const int MMA_N = 8,
          const int MMA_K = 16, const int MMA_TILE_M = 2,
          const int MMA_TILE_N = 4, const int WARP_TILE_M = 4,
          const int WARP_TILE_N = 4, const int WARP_TILE_K = 2,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 3,
          const bool BLOCK_SWIZZLE = true, const bool PARTIAL = false,
          const int SLACK = 1>
__device__ __forceinline__ void
gemm_bf16_impl(const T *__restrict__ A, const T *__restrict__ B,
               T *__restrict__ C, int M, int N, int K,
               const T *__restrict__ bias, int has_bias,
               const T *__restrict__ residual, int has_residual) {
  gemm_bf16_impl_ex<T, MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,
                    WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD,
                    K_STAGE, BLOCK_SWIZZLE, PARTIAL, SLACK>(
      A, B, C, M, N, K, bias, has_bias, residual, has_residual, K, nullptr);
}

// split-K (s64-тайл): grid.z = индекс k-чанка; каждый блок считает свой
// диапазон K и пишет f32-партиал в ws[z*M*N + ...]. Без растра (z занят).
// Интерьер/край — та же гибрид-схема, что у part-ядер.
template <typename T, const int K_STAGE>
__device__ __forceinline__ void
gemm_splitk_s64_impl(const T *__restrict__ A, const T *__restrict__ B,
                     float *__restrict__ ws, int M, int N, int K, int kchunk) {
  const int k0 = (int)blockIdx.z * kchunk;
  const int len = min(kchunk, K - k0);
  float *wsz = ws + (size_t)blockIdx.z * (size_t)M * (size_t)N;
  if ((int)blockIdx.y * 64 + 64 <= M && (int)blockIdx.x * 64 + 64 <= N)
    gemm_bf16_impl_ex<T, 16, 8, 16, 2, 4, 2, 2, 2, 0, 0, K_STAGE, false,
                      false>(A + k0, B + k0, nullptr, M, N, len, nullptr, 0,
                             nullptr, 0, K, wsz);
  else
    gemm_bf16_impl_ex<T, 16, 8, 16, 2, 4, 2, 2, 2, 0, 0, K_STAGE, false,
                      true>(A + k0, B + k0, nullptr, M, N, len, nullptr, 0,
                            nullptr, 0, K, wsz);
}

// Редьюс split-K: фиксированный порядок суммирования по сплитам (детерминизм)
// + bias/residual + конверсия в T. MN мало (малые M) — 1D-грид хватает.
template <typename T>
__device__ __forceinline__ void
splitk_reduce_impl(const float *__restrict__ ws, T *__restrict__ C,
                   long long mn, int N, int splits,
                   const T *__restrict__ bias, int has_bias,
                   const T *__restrict__ residual, int has_residual) {
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < mn;
       i += stride) {
    float v = 0.f;
    for (int s = 0; s < splits; ++s)
      v += ws[(size_t)s * mn + i];
    if (has_bias)
      v += to_float(bias[(int)(i % N)]);
    if (has_residual)
      v += to_float(residual[i]);
    C[i] = from_float<T>(v);
  }
}

}


// ── k128-конвейер для малых M (схема cuBLAS skinny 32x32_128x2) ──
// Супер-стадия = SUB суб-стадий по 32-K → ОДИН барьер на SUB*32 шагов K
// (у классического пути барьер каждые 32: cp.async других потоков видим
// только через __syncthreads, разредить нельзя — k128 снижает саму частоту).
// Два слота (double-buffer): g2s супер t+1 стартует до потребления супер t.
// Тайл 64×64, варп-грид 2×4, warp-тайл 32×16 — как s64-класс.
template <typename T, const int SUB, const bool PARTIAL,
          const bool RASTER = true, const int MMA_TILE_M = 2,
          const int MMA_TILE_N = 4, const int WARP_TILE_M = 2,
          const int WARP_TILE_N = 2, const int NTHREADS = 256,
          const bool L2P = false>
__device__ __forceinline__ void
gemm_bf16_k128_impl(const T *__restrict__ A, const T *__restrict__ B,
                    T *__restrict__ C, int M, int N, int K,
                    const T *__restrict__ bias, int has_bias,
                    const T *__restrict__ residual, int has_residual,
                    int ldk, float *__restrict__ ws) {
  constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
  constexpr int A_PAD = 0, B_PAD = 0;
  constexpr int ST = 2 * SUB;
  static_assert(MMA_TILE_M * MMA_TILE_N * 32 == NTHREADS, "варп-грид = блок");
  // RASTER=false в split-K: grid.z там — индекс k-чанка, НЕ N-растр.
  const int bx = ((int)RASTER) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K * 2);
  const int T_SUPER = NUM_K_TILES / SUB;
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;
  constexpr int BK = MMA_K;

  extern __shared__ __align__(16) char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);
  T *s_a = smem;
  T *s_b = smem + ST * BM * (BK + A_PAD) * 2;
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BN * (BK + B_PAD);
  constexpr int s_a_mma_k_store_offset = ST * BM * (BK + A_PAD);
  constexpr int s_b_mma_k_store_offset = ST * BN * (BK + B_PAD);

  const int tid = threadIdx.x;
  const int lane_id = tid % 32;
  const int warp_id = tid / 32;
  const int warp_m = warp_id % MMA_TILE_M;
  const int warp_n = warp_id / MMA_TILE_M;

  constexpr int RPP = NTHREADS / 4; // строк за проход g2s (4 потока/строку)
  static_assert(BM % RPP == 0 && BN % RPP == 0, "g2s: тайл кратен RPP");
  const int lrow = tid / 4;
  const int lchunk = tid % 4;
  const int lk = (lchunk % 2) * 8;
  const int lreg_a_off = (lchunk / 2) * s_a_mma_k_store_offset;
  const int lreg_b_off = (lchunk / 2) * s_b_mma_k_store_offset;
  if constexpr (!PARTIAL) {
    if (by * BM + BM - 1 >= M || bx * BN + BN - 1 >= N)
      return;
  }
  bool a_valid_r[BM / RPP];
#pragma unroll
  for (int p = 0; p < BM / RPP; ++p)
    a_valid_r[p] = by * BM + lrow + p * RPP < M;
  bool b_valid_r[BN / RPP];
#pragma unroll
  for (int p = 0; p < BN / RPP; ++p)
    b_valid_r[p] = bx * BN + lrow + p * RPP < N;

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

  auto g2s_stage = [&](int slot, int ktile) {
    const int gk = ktile * (BK * 2) + lchunk * 8;
#pragma unroll
    for (int p = 0; p < BM / RPP; ++p) {
      int row = lrow + p * RPP;
      unsigned dst =
          (smem_a_base_ptr +
           (lreg_a_off + slot * s_a_stage_offset + row * (BK + A_PAD) +
            swizzle_permuted_j<MMA_K>(row, lk)) *
               sizeof(T));
      load16<PARTIAL>(dst, &A[(size_t)(by * BM + row) * ldk + gk], a_valid_r[p]);
    }
#pragma unroll
    for (int p = 0; p < BN / RPP; ++p) {
      int row = lrow + p * RPP;
      unsigned dst =
          (smem_b_base_ptr +
           (lreg_b_off + slot * s_b_stage_offset + row * (BK + B_PAD) +
            swizzle_permuted_j<MMA_K>(row, lk)) *
               sizeof(T));
      if constexpr (L2P)
        load16_l2<PARTIAL>(dst, &B[(size_t)(bx * BN + row) * ldk + gk],
                           b_valid_r[p]);
      else
        load16<PARTIAL>(dst, &B[(size_t)(bx * BN + row) * ldk + gk],
                        b_valid_r[p]);
    }
  };

  unsigned RA[2][WARP_TILE_M][4];
  unsigned RB[2][WARP_TILE_N][2];
  int reg_store_idx = 0;
  int reg_load_idx = 1;

  // A half0 стадии st → RA[reg_store_idx]
  auto ldsm_a_h0 = [&](int st) {
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
      int lane_smem_a_k = (lane_id / 16) * 8;
      unsigned ptr =
          (smem_a_base_ptr +
           (st * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) +
            swizzle_permuted_j<MMA_K>(lane_smem_a_m, lane_smem_a_k)) *
               sizeof(T));
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], ptr);
    }
  };
  // A half2 стадии st → RA[reg_store_idx]
  auto ldsm_a_h2 = [&](int st) {
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
      int lane_smem_a_k = (lane_id / 16) * 8;
      unsigned ptr =
          (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(T) +
           (st * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) +
            swizzle_permuted_j<MMA_K>(lane_smem_a_m, lane_smem_a_k)) *
               sizeof(T));
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], ptr);
    }
  };
  // B обе половины стадии st: half0 → RB[reg_store_idx], half2 → RB[reg_load_idx]
  auto ldsm_b_x4 = [&](int st) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8;
      unsigned ptr =
          (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(T) * (lane_id / 16) +
           (st * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) +
            swizzle_permuted_j<MMA_K>(lane_smem_b_n, lane_smem_b_k)) *
               sizeof(T));
      LDMATRIX_X4(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                  RB[reg_load_idx][j][0], RB[reg_load_idx][j][1], ptr);
    }
  };
  // B half0 стадии st → RB[reg_store_idx] (X2)
  auto ldsm_b_h0 = [&](int st) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8;
      unsigned ptr =
          (smem_b_base_ptr +
           (st * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) +
            swizzle_permuted_j<MMA_K>(lane_smem_b_n, lane_smem_b_k)) *
               sizeof(T));
      LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], ptr);
    }
  };
  // B half2 стадии st → RB[reg_load_idx] (X2, trailing)
  auto ldsm_b_h2 = [&](int st) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8;
      unsigned ptr =
          (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(T) +
           (st * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) +
            swizzle_permuted_j<MMA_K>(lane_smem_b_n, lane_smem_b_k)) *
               sizeof(T));
      LDMATRIX_X2(RB[reg_load_idx][j][0], RB[reg_load_idx][j][1], ptr);
    }
  };
  auto mma_group = [&]() {
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j)
        mma_m16n8k16<T>(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3],
                        RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                        RA[reg_load_idx][i][2], RA[reg_load_idx][i][3],
                        RB[reg_load_idx][j][0], RB[reg_load_idx][j][1]);
  };

  // Пролог: супер-стадия 0 (слот 0), один commit-group, полный wait.
#pragma unroll
  for (int sub = 0; sub < SUB; ++sub)
    g2s_stage(sub, sub);
  CP_ASYNC_COMMIT_GROUP();
  CP_ASYNC_WAIT_GROUP(0);
  __syncthreads();
  ldsm_a_h0(0);
  ldsm_b_x4(0);

  for (int t = 0; t < T_SUPER; ++t) {
    const int slot = (t & 1) * SUB;
    if (t + 1 < T_SUPER) {
      const int nslot = ((t + 1) & 1) * SUB;
#pragma unroll
      for (int sub = 0; sub < SUB; ++sub)
        g2s_stage(nslot + sub, (t + 1) * SUB + sub);
      CP_ASYNC_COMMIT_GROUP();
    }
#pragma unroll
    for (int sub = 0; sub < SUB; ++sub) {
      const int st = slot + sub;
      const bool has_next = sub < SUB - 1;
      reg_store_idx ^= 1;
      reg_load_idx ^= 1;
      ldsm_a_h2(st);
      mma_group(); // half0 стадии st
      reg_store_idx ^= 1;
      reg_load_idx ^= 1;
      if (has_next) {
        ldsm_a_h0(st + 1);
        ldsm_b_h0(st + 1);
      }
      mma_group(); // half2 стадии st
      if (has_next)
        ldsm_b_h2(st + 1);
    }
    if (t + 1 < T_SUPER) {
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads();
      const int nfirst = ((t + 1) & 1) * SUB;
      ldsm_a_h0(nfirst);
      ldsm_b_x4(nfirst);
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
      float bc = has_bias ? to_float(bias[gmem_n + col]) : 0.f;
      float bc1 = (has_bias && gmem_n + col + 1 < N) ? to_float(bias[gmem_n + col + 1]) : 0.f;
      using T2 = typename Pack2<T>::type;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        int gm = gmem_m + row + half * 8;
        if (gm >= M)
          continue;
        size_t r0 = (size_t)gm * N + gmem_n + col;
        float v0 = d[half * 2] + bc;
        float v1 = d[half * 2 + 1] + bc1;
        if (ws != nullptr) {
          if (gmem_n + col < N)
            __stcs(&ws[r0], v0);
          if (gmem_n + col + 1 < N)
            __stcs(&ws[r0 + 1], v1);
          continue;
        }
        if (gmem_n + col + 1 < N && (r0 & 1) == 0) {
          if (has_residual) {
            v0 += to_float(residual[r0]);
            v1 += to_float(residual[r0 + 1]);
          }
          T2 out2;
          out2.x = from_float<T>(v0);
          out2.y = from_float<T>(v1);
          asm volatile("st.global.cs.b32 [%0], %1;\n" ::"l"(&C[r0]),
                       "r"(*reinterpret_cast<unsigned *>(&out2)));
        } else {
          if (gmem_n + col < N)
            C[r0] = from_float<T>(
                v0 + (has_residual ? to_float(residual[r0]) : 0.f));
          if (gmem_n + col + 1 < N)
            C[r0 + 1] = from_float<T>(
                v1 + (has_residual ? to_float(residual[r0 + 1]) : 0.f));
        }
      }
    }
}

// ── m32g: порт конвейера cuBLAS Kernel2 (cutlass_80_wmma 32x32_128x2, SASS:
// LDG.E.LTC128B.128 → reg → STS.128, БЕЗ LDGSTS/LDSM в g-пути) ──
// Геометрия = k128m32 (тайл 32×32, BK-супер 128, 128 потоков, варп-грид 2×2),
// но загрузка через РЕГИСТРОВЫЙ буфер (8×uint4 = 32 рег): LDG t+2 улетают на
// всё время compute(t+1) — глубина in-flight рег-scoreboard'а выше очереди
// cp.async, конвейер не голодает на высоком клоке (их 84 рег vs наши 46).
__device__ __forceinline__ uint4 ldg_l2_256(const void *p) {
  uint4 v;
  asm volatile("ld.global.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

template <typename T, const int SUB, const bool PARTIAL>
__device__ __forceinline__ void
gemm_bf16_m32g_impl(const T *__restrict__ A, const T *__restrict__ B,
                    T *__restrict__ C, int M, int N, int K,
                    const T *__restrict__ bias, int has_bias,
                    const T *__restrict__ residual, int has_residual) {
  constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
  constexpr int MMA_TILE_M = 2, MMA_TILE_N = 2;
  constexpr int WARP_TILE_M = 1, WARP_TILE_N = 2;
  constexpr int NTHREADS = 128;
  constexpr int ST = 2 * SUB;
  const int bx = (int)(blockIdx.z * gridDim.x + blockIdx.x);
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K * 2);
  const int T_SUPER = NUM_K_TILES / SUB;
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;
  constexpr int BK = MMA_K;

  extern __shared__ __align__(16) char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);
  T *s_a = smem;
  T *s_b = smem + ST * BM * BK * 2;
  constexpr int s_a_stage_offset = BM * BK;
  constexpr int s_b_stage_offset = BN * BK;
  constexpr int s_a_mma_k_store_offset = ST * BM * BK;
  constexpr int s_b_mma_k_store_offset = ST * BN * BK;

  const int tid = threadIdx.x;
  const int lane_id = tid % 32;
  const int warp_id = tid / 32;
  const int warp_m = warp_id % MMA_TILE_M;
  const int warp_n = warp_id / MMA_TILE_M;

  constexpr int RPP = NTHREADS / 4;
  static_assert(BM == RPP && BN == RPP, "m32g: тайл = один проход g-загрузки");
  const int lrow = tid / 4;
  const int lchunk = tid % 4;
  const int lk = (lchunk % 2) * 8;
  const int lreg_a_off = (lchunk / 2) * s_a_mma_k_store_offset;
  const int lreg_b_off = (lchunk / 2) * s_b_mma_k_store_offset;
  if constexpr (!PARTIAL) {
    if (by * BM + BM - 1 >= M || bx * BN + BN - 1 >= N)
      return;
  }
  const bool a_valid = by * BM + lrow < M;
  const bool b_valid = bx * BN + lrow < N;

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

  // Рег-буфер супер-стадии: SUB×(A+B) uint4 (SUB=4 → 32 рег — как Kernel2).
  uint4 ra[SUB], rb[SUB];
  auto ldg_super = [&](int tsuper) {
#pragma unroll
    for (int sub = 0; sub < SUB; ++sub) {
      const int gk = (tsuper * SUB + sub) * (BK * 2) + lchunk * 8;
      if constexpr (PARTIAL) {
        ra[sub] = a_valid ? ldg_l2_256(&A[(size_t)(by * BM + lrow) * K + gk])
                          : uint4{0, 0, 0, 0};
        rb[sub] = b_valid ? ldg_l2_256(&B[(size_t)(bx * BN + lrow) * K + gk])
                          : uint4{0, 0, 0, 0};
      } else {
        ra[sub] = ldg_l2_256(&A[(size_t)(by * BM + lrow) * K + gk]);
        rb[sub] = ldg_l2_256(&B[(size_t)(bx * BN + lrow) * K + gk]);
      }
    }
  };
  auto sts_super = [&](int slot0) {
#pragma unroll
    for (int sub = 0; sub < SUB; ++sub) {
      unsigned da =
          (smem_a_base_ptr +
           (lreg_a_off + (slot0 + sub) * s_a_stage_offset + lrow * BK +
            swizzle_permuted_j<MMA_K>(lrow, lk)) *
               sizeof(T));
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(da),
                   "r"(ra[sub].x), "r"(ra[sub].y), "r"(ra[sub].z),
                   "r"(ra[sub].w));
      unsigned db =
          (smem_b_base_ptr +
           (lreg_b_off + (slot0 + sub) * s_b_stage_offset + lrow * BK +
            swizzle_permuted_j<MMA_K>(lrow, lk)) *
               sizeof(T));
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(db),
                   "r"(rb[sub].x), "r"(rb[sub].y), "r"(rb[sub].z),
                   "r"(rb[sub].w));
    }
  };

  unsigned RA[2][WARP_TILE_M][4];
  unsigned RB[2][WARP_TILE_N][2];
  int reg_store_idx = 0;
  int reg_load_idx = 1;

  auto ldsm_a_h0 = [&](int st) {
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int lane_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16;
      int lane_smem_a_k = (lane_id / 16) * 8;
      unsigned ptr = (smem_a_base_ptr +
                      (st * s_a_stage_offset + lane_smem_a_m * BK +
                       swizzle_permuted_j<MMA_K>(lane_smem_a_m, lane_smem_a_k)) *
                          sizeof(T));
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], ptr);
    }
  };
  auto ldsm_a_h2 = [&](int st) {
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int lane_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16;
      int lane_smem_a_k = (lane_id / 16) * 8;
      unsigned ptr = (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(T) +
                      (st * s_a_stage_offset + lane_smem_a_m * BK +
                       swizzle_permuted_j<MMA_K>(lane_smem_a_m, lane_smem_a_k)) *
                          sizeof(T));
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], ptr);
    }
  };
  auto ldsm_b_x4 = [&](int st) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int lane_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N + lane_id % 8;
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8;
      unsigned ptr = (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(T) * (lane_id / 16) +
                      (st * s_b_stage_offset + lane_smem_b_n * BK +
                       swizzle_permuted_j<MMA_K>(lane_smem_b_n, lane_smem_b_k)) *
                          sizeof(T));
      LDMATRIX_X4(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                  RB[reg_load_idx][j][0], RB[reg_load_idx][j][1], ptr);
    }
  };
  auto ldsm_b_h0 = [&](int st) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int lane_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N + lane_id % 8;
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8;
      unsigned ptr = (smem_b_base_ptr +
                      (st * s_b_stage_offset + lane_smem_b_n * BK +
                       swizzle_permuted_j<MMA_K>(lane_smem_b_n, lane_smem_b_k)) *
                          sizeof(T));
      LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], ptr);
    }
  };
  auto ldsm_b_h2 = [&](int st) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int lane_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N + lane_id % 8;
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8;
      unsigned ptr = (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(T) +
                      (st * s_b_stage_offset + lane_smem_b_n * BK +
                       swizzle_permuted_j<MMA_K>(lane_smem_b_n, lane_smem_b_k)) *
                          sizeof(T));
      LDMATRIX_X2(RB[reg_load_idx][j][0], RB[reg_load_idx][j][1], ptr);
    }
  };
  auto mma_group = [&]() {
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j)
        mma_m16n8k16<T>(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3],
                        RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                        RA[reg_load_idx][i][2], RA[reg_load_idx][i][3],
                        RB[reg_load_idx][j][0], RB[reg_load_idx][j][1]);
  };

  // Пролог: стадия 0 рег→smem; стадия t+1 грузится В НАЧАЛЕ compute(t)
  // (схема Kernel2: LDG interleaved с первыми HMMA, ДО барьера — выдача не
  // сериализована цепочкой BAR←STS←долёт LDG; наш вариант «LDG после BAR»
  // давал стоил 1109 сэмплов на первом STS).
  ldg_super(0);
  sts_super(0);
  __syncthreads();
  ldsm_a_h0(0);
  ldsm_b_x4(0);

  for (int t = 0; t < T_SUPER; ++t) {
    const int slot = (t & 1) * SUB;
    if (t + 1 < T_SUPER)
      ldg_super(t + 1);
#pragma unroll
    for (int sub = 0; sub < SUB; ++sub) {
      const int st = slot + sub;
      const bool has_next = sub < SUB - 1;
      reg_store_idx ^= 1;
      reg_load_idx ^= 1;
      ldsm_a_h2(st);
      mma_group();
      reg_store_idx ^= 1;
      reg_load_idx ^= 1;
      if (has_next) {
        ldsm_a_h0(st + 1);
        ldsm_b_h0(st + 1);
      }
      mma_group();
      if (has_next)
        ldsm_b_h2(st + 1);
    }
    if (t + 1 < T_SUPER) {
      const int nslot = ((t + 1) & 1) * SUB;
      sts_super(nslot);
      __syncthreads();
      ldsm_a_h0(nslot);
      ldsm_b_x4(nslot);
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
      float bc = has_bias ? to_float(bias[gmem_n + col]) : 0.f;
      float bc1 = (has_bias && gmem_n + col + 1 < N) ? to_float(bias[gmem_n + col + 1]) : 0.f;
      using T2 = typename Pack2<T>::type;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        int gm = gmem_m + row + half * 8;
        if (gm >= M)
          continue;
        size_t r0 = (size_t)gm * N + gmem_n + col;
        float v0 = d[half * 2] + bc;
        float v1 = d[half * 2 + 1] + bc1;
        if (gmem_n + col + 1 < N && (r0 & 1) == 0) {
          if (has_residual) {
            v0 += to_float(residual[r0]);
            v1 += to_float(residual[r0 + 1]);
          }
          T2 out2;
          out2.x = from_float<T>(v0);
          out2.y = from_float<T>(v1);
          asm volatile("st.global.cs.b32 [%0], %1;\n" ::"l"(&C[r0]),
                       "r"(*reinterpret_cast<unsigned *>(&out2)));
        } else {
          if (gmem_n + col < N)
            C[r0] = from_float<T>(v0 + (has_residual ? to_float(residual[r0]) : 0.f));
          if (gmem_n + col + 1 < N)
            C[r0 + 1] = from_float<T>(
                v1 + (has_residual ? to_float(residual[r0 + 1]) : 0.f));
        }
      }
    }
}

#define GEMM_TN_ENTRY_M32G(T, NAME, SUB)                                       \
  extern "C" __global__ void __launch_bounds__(128)                            \
      NAME(const T *A, const T *B, T *C, int M, int N, int K,                  \
           const T *bias, int has_bias, const T *residual, int has_residual) { \
    const int bx_ = (int)(blockIdx.z * gridDim.x + blockIdx.x);                \
    if ((int)blockIdx.y * 32 + 32 <= M && bx_ * 32 + 32 <= N)                  \
      gemm_bf16_m32g_impl<T, SUB, false>(A, B, C, M, N, K, bias, has_bias,     \
                                         residual, has_residual);              \
    else                                                                       \
      gemm_bf16_m32g_impl<T, SUB, true>(A, B, C, M, N, K, bias, has_bias,      \
                                        residual, has_residual);               \
  }

GEMM_TN_ENTRY_M32G(__nv_bfloat16, gemm_bf16_k128m32g, 4)
GEMM_TN_ENTRY_M32G(__half, gemm_f16tn_k128m32g, 4)

// ── m32v: compute WMMA-стиля Kernel2 — 32-бит LDS mma-фрагментов из row-major
// smem (PAD=8 → питч 272B: на варп 32 уникальных банка, conflict-free), БЕЗ
// ldmatrix. Их цикл = 64×LD.E + 8×LDG + 8×STS + 16×HMMA: мелкие LDS тонким
// слоем между HMMA → LSU не забит залпами LDSM (m32g стоял 750-1109 на STS:
// 12×LDSM/стадию душили долёт LDG). Конвейер загрузки — как m32g.
template <typename T, const int SUB, const bool PARTIAL>
__device__ __forceinline__ void
gemm_bf16_m32v_impl(const T *__restrict__ A, const T *__restrict__ B,
                    T *__restrict__ C, int M, int N, int K,
                    const T *__restrict__ bias, int has_bias,
                    const T *__restrict__ residual, int has_residual) {
  constexpr int BM = 32, BN = 32, BKS = 32 * SUB;
  constexpr int PAD = 8;
  constexpr int LDP = BKS + PAD;
  const int bx = (int)(blockIdx.z * gridDim.x + blockIdx.x);
  const int by = blockIdx.y;
  const int T_SUPER = div_ceil(K, BKS);

  extern __shared__ __align__(16) char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);
  // слот s: A[32][LDP] затем B[32][LDP]; 2 слота.
  constexpr int SLOT = (BM + BN) * LDP;
  T *s_a = smem;
  T *s_b = smem + BM * LDP;

  const int tid = threadIdx.x;
  const int lane_id = tid % 32;
  const int warp_id = tid / 32;
  const int warp_m = warp_id % 2;
  const int warp_n = warp_id / 2;

  const int lrow = tid / 4;
  const int lchunk = tid % 4;
  if constexpr (!PARTIAL) {
    if (by * BM + BM - 1 >= M || bx * BN + BN - 1 >= N)
      return;
  }
  const bool a_valid = by * BM + lrow < M;
  const bool b_valid = bx * BN + lrow < N;

  float RC[2][4];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    RC[j][0] = 0.f;
    RC[j][1] = 0.f;
    RC[j][2] = 0.f;
    RC[j][3] = 0.f;
  }

  uint4 ra[SUB], rb[SUB];
  auto ldg_super = [&](int tsuper) {
#pragma unroll
    for (int sub = 0; sub < SUB; ++sub) {
      const int gk = (tsuper * SUB + sub) * 32 + lchunk * 8;
      if constexpr (PARTIAL) {
        ra[sub] = a_valid ? ldg_l2_256(&A[(size_t)(by * BM + lrow) * K + gk])
                          : uint4{0, 0, 0, 0};
        rb[sub] = b_valid ? ldg_l2_256(&B[(size_t)(bx * BN + lrow) * K + gk])
                          : uint4{0, 0, 0, 0};
      } else {
        ra[sub] = ldg_l2_256(&A[(size_t)(by * BM + lrow) * K + gk]);
        rb[sub] = ldg_l2_256(&B[(size_t)(bx * BN + lrow) * K + gk]);
      }
    }
  };
  auto sts_super = [&](int slot) {
    T *da = s_a + slot * SLOT + lrow * LDP + lchunk * 8;
    T *db = s_b + slot * SLOT + lrow * LDP + lchunk * 8;
#pragma unroll
    for (int sub = 0; sub < SUB; ++sub) {
      *reinterpret_cast<uint4 *>(da + sub * 32) = ra[sub];
      *reinterpret_cast<uint4 *>(db + sub * 32) = rb[sub];
    }
  };

  // mma-фрагменты 32-бит LDS из row-major: A m16k16 (4×LDS.32), B n8k16 (2×).
  const int fa_row = warp_m * 16 + (lane_id >> 2);
  const int f_col = (lane_id & 3) * 2;
  const int fb_row0 = warp_n * 16 + (lane_id >> 2);
  auto compute_slot = [&](int slot) {
    const T *ca = s_a + slot * SLOT;
    const T *cb = s_b + slot * SLOT;
#pragma unroll
    for (int ks = 0; ks < 2 * SUB; ++ks) {
      const int k0 = ks * 16 + f_col;
      unsigned a0 = *reinterpret_cast<const unsigned *>(&ca[fa_row * LDP + k0]);
      unsigned a1 =
          *reinterpret_cast<const unsigned *>(&ca[(fa_row + 8) * LDP + k0]);
      unsigned a2 =
          *reinterpret_cast<const unsigned *>(&ca[fa_row * LDP + k0 + 8]);
      unsigned a3 =
          *reinterpret_cast<const unsigned *>(&ca[(fa_row + 8) * LDP + k0 + 8]);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        unsigned b0 = *reinterpret_cast<const unsigned *>(
            &cb[(fb_row0 + j * 8) * LDP + k0]);
        unsigned b1 = *reinterpret_cast<const unsigned *>(
            &cb[(fb_row0 + j * 8) * LDP + k0 + 8]);
        mma_m16n8k16<T>(RC[j][0], RC[j][1], RC[j][2], RC[j][3], a0, a1, a2, a3,
                        b0, b1);
      }
    }
  };

  ldg_super(0);
  sts_super(0);
  __syncthreads();

  for (int t = 0; t < T_SUPER; ++t) {
    if (t + 1 < T_SUPER)
      ldg_super(t + 1);
    compute_slot(t & 1);
    if (t + 1 < T_SUPER) {
      sts_super((t + 1) & 1);
      __syncthreads();
    }
  }

#pragma unroll
  for (int j = 0; j < 2; ++j) {
    int gmem_m = by * BM + warp_m * 16;
    int gmem_n = bx * BN + warp_n * 16 + j * 8;
    int row = lane_id / 4;
    int col = (lane_id % 4) * 2;
    float *d = RC[j];
    float bc = has_bias ? to_float(bias[gmem_n + col]) : 0.f;
    float bc1 = (has_bias && gmem_n + col + 1 < N) ? to_float(bias[gmem_n + col + 1]) : 0.f;
    using T2 = typename Pack2<T>::type;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      int gm = gmem_m + row + half * 8;
      if (gm >= M)
        continue;
      size_t r0 = (size_t)gm * N + gmem_n + col;
      float v0 = d[half * 2] + bc;
      float v1 = d[half * 2 + 1] + bc1;
      if (gmem_n + col + 1 < N && (r0 & 1) == 0) {
        if (has_residual) {
          v0 += to_float(residual[r0]);
          v1 += to_float(residual[r0 + 1]);
        }
        T2 out2;
        out2.x = from_float<T>(v0);
        out2.y = from_float<T>(v1);
        asm volatile("st.global.cs.b32 [%0], %1;\n" ::"l"(&C[r0]),
                     "r"(*reinterpret_cast<unsigned *>(&out2)));
      } else {
        if (gmem_n + col < N)
          C[r0] = from_float<T>(v0 + (has_residual ? to_float(residual[r0]) : 0.f));
        if (gmem_n + col + 1 < N)
          C[r0 + 1] = from_float<T>(
              v1 + (has_residual ? to_float(residual[r0 + 1]) : 0.f));
      }
    }
  }
}

#define GEMM_TN_ENTRY_M32V(T, NAME, SUB)                                       \
  extern "C" __global__ void __launch_bounds__(128)                            \
      NAME(const T *A, const T *B, T *C, int M, int N, int K,                  \
           const T *bias, int has_bias, const T *residual, int has_residual) { \
    const int bx_ = (int)(blockIdx.z * gridDim.x + blockIdx.x);                \
    if ((int)blockIdx.y * 32 + 32 <= M && bx_ * 32 + 32 <= N)                  \
      gemm_bf16_m32v_impl<T, SUB, false>(A, B, C, M, N, K, bias, has_bias,     \
                                         residual, has_residual);              \
    else                                                                       \
      gemm_bf16_m32v_impl<T, SUB, true>(A, B, C, M, N, K, bias, has_bias,      \
                                        residual, has_residual);               \
  }

GEMM_TN_ENTRY_M32V(__nv_bfloat16, gemm_bf16_k128m32v, 4)
GEMM_TN_ENTRY_M32V(__half, gemm_f16tn_k128m32v, 4)

#define GEMM_TN_ENTRY(T, NAME, KSTAGE, SWZ, PART)                              \
  extern "C" __global__ void __launch_bounds__(256)                            \
      NAME(const T *A, const T *B, T *C, int M, int N, int K,                  \
           const T *bias, int has_bias, const T *residual, int has_residual) { \
    gemm_bf16_impl<T, 16, 8, 16, 2, 4, 4, 4, 2, 0, 0, KSTAGE, SWZ, PART>(      \
        A, B, C, M, N, K, bias, has_bias, residual, has_residual);            \
  }

// part-гибрид: интерьерные блоки (вся плитка в границах) идут по пути без
// OOB-предикатов на каждый cp.async; только краевые блоки платят за PARTIAL.
// Растр включён (SWZ=true): bx = z*gridDim.x + x — N-чанкинг держит B-слайс
// горячим в L2 (критично при N*K*2 > L2, напр. ff_up 128MB).
#define GEMM_TN_ENTRY_PART_HYB(T, NAME, KSTAGE, MTM, MTN, WTM, WTN, MINB)      \
  extern "C" __global__ void __launch_bounds__(256, MINB)                      \
      NAME(const T *A, const T *B, T *C, int M, int N, int K,                  \
           const T *bias, int has_bias, const T *residual, int has_residual) { \
    constexpr int BM_ = 16 * MTM * WTM;                                        \
    constexpr int BN_ = 8 * MTN * WTN;                                         \
    const int bx_ = (int)(blockIdx.z * gridDim.x + blockIdx.x);                \
    if ((int)blockIdx.y * BM_ + BM_ <= M && bx_ * BN_ + BN_ <= N)              \
      gemm_bf16_impl<T, 16, 8, 16, MTM, MTN, WTM, WTN, 2, 0, 0, KSTAGE, true,  \
                     false>(A, B, C, M, N, K, bias, has_bias, residual,        \
                            has_residual);                                     \
    else                                                                       \
      gemm_bf16_impl<T, 16, 8, 16, MTM, MTN, WTM, WTN, 2, 0, 0, KSTAGE, true,  \
                     true>(A, B, C, M, N, K, bias, has_bias, residual,         \
                           has_residual);                                      \
  }

// MINB=2 форсит ptxas в ≤128 рег/поток → 2 блока/SM (smem S2=32KB тоже даёт 2).
#define GEMM_TN_ENTRY_MB2(T, NAME, KSTAGE, SWZ, PART)                          \
  extern "C" __global__ void __launch_bounds__(256, 2)                         \
      NAME(const T *A, const T *B, T *C, int M, int N, int K,                  \
           const T *bias, int has_bias, const T *residual, int has_residual) { \
    gemm_bf16_impl<T, 16, 8, 16, 2, 4, 4, 4, 2, 0, 0, KSTAGE, SWZ, PART>(      \
        A, B, C, M, N, K, bias, has_bias, residual, has_residual);            \
  }

GEMM_TN_ENTRY(__nv_bfloat16, gemm_bf16_swz_s5, 5, true, false)
GEMM_TN_ENTRY(__nv_bfloat16, gemm_bf16_swz_s6, 6, true, false)
GEMM_TN_ENTRY(__half, gemm_f16tn_swz_s5, 5, true, false)
GEMM_TN_ENTRY(__half, gemm_f16tn_swz_s6, 6, true, false)
GEMM_TN_ENTRY(__nv_bfloat16, gemm_bf16_swz_s3, 3, true, false)
GEMM_TN_ENTRY(__nv_bfloat16, gemm_bf16_swz_s4, 4, true, false)
GEMM_TN_ENTRY_PART_HYB(__nv_bfloat16, gemm_bf16_part, 3, 2, 4, 4, 4, 1)
GEMM_TN_ENTRY(__half, gemm_f16tn_swz_s3, 3, true, false)
GEMM_TN_ENTRY(__half, gemm_f16tn_swz_s4, 4, true, false)
GEMM_TN_ENTRY_PART_HYB(__half, gemm_f16tn_part, 3, 2, 4, 4, 4, 1)
GEMM_TN_ENTRY_MB2(__nv_bfloat16, gemm_bf16_swz_s2, 2, true, false)
GEMM_TN_ENTRY_PART_HYB(__nv_bfloat16, gemm_bf16_part_s2, 2, 2, 4, 4, 4, 2)
GEMM_TN_ENTRY_MB2(__half, gemm_f16tn_swz_s2, 2, true, false)
GEMM_TN_ENTRY_PART_HYB(__half, gemm_f16tn_part_s2, 2, 2, 4, 4, 4, 2)
GEMM_TN_ENTRY_PART_HYB(__nv_bfloat16, gemm_bf16_part_s6, 6, 2, 4, 4, 4, 1)
GEMM_TN_ENTRY_PART_HYB(__half, gemm_f16tn_part_s6, 6, 2, 4, 4, 4, 1)

// Малый тайл 64×64 (как cutlass 32x32_128x2 у cuBLAS на M=32): для M=32..192
// у 128-тайла грид 32 CTA (attn) → SM пустые (warps_active 17%, DRAM 20%).
// 64×64 даёт ×2 CTA по N и не жжёт 75% тайла по M. Варп-грид тот же 2×4
// (256 потоков), warp-тайл 32×16, RC=16 рег → высокая occupancy.
#define GEMM_TN_ENTRY_S64(T, NAME, KSTAGE, SWZ, PART)                          \
  extern "C" __global__ void __launch_bounds__(256)                            \
      NAME(const T *A, const T *B, T *C, int M, int N, int K,                  \
           const T *bias, int has_bias, const T *residual, int has_residual) { \
    gemm_bf16_impl<T, 16, 8, 16, 2, 4, 2, 2, 2, 0, 0, KSTAGE, SWZ, PART>(      \
        A, B, C, M, N, K, bias, has_bias, residual, has_residual);            \
  }

GEMM_TN_ENTRY_S64(__nv_bfloat16, gemm_bf16_swz_s64s3, 3, true, false)
GEMM_TN_ENTRY_PART_HYB(__nv_bfloat16, gemm_bf16_part_s64s3, 3, 2, 4, 2, 2, 1)
GEMM_TN_ENTRY_S64(__nv_bfloat16, gemm_bf16_swz_s64s4, 4, true, false)
GEMM_TN_ENTRY_PART_HYB(__nv_bfloat16, gemm_bf16_part_s64s4, 4, 2, 4, 2, 2, 1)
GEMM_TN_ENTRY_S64(__half, gemm_f16tn_swz_s64s3, 3, true, false)
GEMM_TN_ENTRY_PART_HYB(__half, gemm_f16tn_part_s64s3, 3, 2, 4, 2, 2, 1)
GEMM_TN_ENTRY_S64(__half, gemm_f16tn_swz_s64s4, 4, true, false)
GEMM_TN_ENTRY_PART_HYB(__half, gemm_f16tn_part_s64s4, 4, 2, 4, 2, 2, 1)


#define GEMM_TN_ENTRY_K128(T, NAME, SUB)                                       \
  extern "C" __global__ void __launch_bounds__(256)                            \
      NAME(const T *A, const T *B, T *C, int M, int N, int K,                  \
           const T *bias, int has_bias, const T *residual, int has_residual) { \
    const int bx_ = (int)(blockIdx.z * gridDim.x + blockIdx.x);                \
    if ((int)blockIdx.y * 64 + 64 <= M && bx_ * 64 + 64 <= N)                  \
      gemm_bf16_k128_impl<T, SUB, false>(A, B, C, M, N, K, bias, has_bias,     \
                                         residual, has_residual, K, nullptr);  \
    else                                                                       \
      gemm_bf16_k128_impl<T, SUB, true>(A, B, C, M, N, K, bias, has_bias,      \
                                        residual, has_residual, K, nullptr);   \
  }

GEMM_TN_ENTRY_K128(__nv_bfloat16, gemm_bf16_k128, 4)
GEMM_TN_ENTRY_K128(__half, gemm_f16tn_k128, 4)
GEMM_TN_ENTRY_K128(__nv_bfloat16, gemm_bf16_k64, 2)
GEMM_TN_ENTRY_K128(__half, gemm_f16tn_k64, 2)

// warp-grid 2×2 (128 потоков): warp-тайл 32×32 → 8 MMA/группу на варп
// (вдвое больше ILP — лекарство от wait-стоилов 52%), CTA вдвое легче.
#define GEMM_TN_ENTRY_K128W(T, NAME, SUB)                                      \
  extern "C" __global__ void __launch_bounds__(128)                            \
      NAME(const T *A, const T *B, T *C, int M, int N, int K,                  \
           const T *bias, int has_bias, const T *residual, int has_residual) { \
    const int bx_ = (int)(blockIdx.z * gridDim.x + blockIdx.x);                \
    if ((int)blockIdx.y * 64 + 64 <= M && bx_ * 64 + 64 <= N)                  \
      gemm_bf16_k128_impl<T, SUB, false, true, 2, 2, 2, 4, 128>(               \
          A, B, C, M, N, K, bias, has_bias, residual, has_residual, K,         \
          nullptr);                                                            \
    else                                                                       \
      gemm_bf16_k128_impl<T, SUB, true, true, 2, 2, 2, 4, 128>(                \
          A, B, C, M, N, K, bias, has_bias, residual, has_residual, K,         \
          nullptr);                                                            \
  }

GEMM_TN_ENTRY_K128W(__nv_bfloat16, gemm_bf16_k128w, 4)
GEMM_TN_ENTRY_K128W(__half, gemm_f16tn_k128w, 4)
GEMM_TN_ENTRY_K128W(__nv_bfloat16, gemm_bf16_k64w, 2)
GEMM_TN_ENTRY_K128W(__half, gemm_f16tn_k64w, 2)

// Тайл 64×32 (128 потоков, варп-грид 2×2, warp-тайл 32×16): ×2 CTA по N
// (attn: 128 CTA) + 48KB → 2 CTA/SM. BN=32 стал возможен: RPP=NTHREADS/4=32.
#define GEMM_TN_ENTRY_K128N32(T, NAME, SUB)                                    \
  extern "C" __global__ void __launch_bounds__(128)                            \
      NAME(const T *A, const T *B, T *C, int M, int N, int K,                  \
           const T *bias, int has_bias, const T *residual, int has_residual) { \
    const int bx_ = (int)(blockIdx.z * gridDim.x + blockIdx.x);                \
    if ((int)blockIdx.y * 64 + 64 <= M && bx_ * 32 + 32 <= N)                  \
      gemm_bf16_k128_impl<T, SUB, false, true, 2, 2, 2, 2, 128>(               \
          A, B, C, M, N, K, bias, has_bias, residual, has_residual, K,         \
          nullptr);                                                            \
    else                                                                       \
      gemm_bf16_k128_impl<T, SUB, true, true, 2, 2, 2, 2, 128>(                \
          A, B, C, M, N, K, bias, has_bias, residual, has_residual, K,         \
          nullptr);                                                            \
  }

GEMM_TN_ENTRY_K128N32(__nv_bfloat16, gemm_bf16_k128n32, 4)
GEMM_TN_ENTRY_K128N32(__half, gemm_f16tn_k128n32, 4)
GEMM_TN_ENTRY_K128N32(__nv_bfloat16, gemm_bf16_k64n32, 2)
GEMM_TN_ENTRY_K128N32(__half, gemm_f16tn_k64n32, 2)

// Генерик-вход k-класса: тайл/варп-грид/потоки параметрами макроса.
// m32n32 = точная форма cuBLAS-skinny 32x32_128x2: BM=BN=32, 128 потоков,
// варп-грид 2×2, warp-тайл 16×16 — attn M=32 даёт 128 CTA + 3 CTA/SM (32KB).
#define GEMM_TN_ENTRY_KTILE(T, NAME, SUB, MTM, MTN, WTM, WTN, THR)             \
  extern "C" __global__ void __launch_bounds__(THR)                            \
      NAME(const T *A, const T *B, T *C, int M, int N, int K,                  \
           const T *bias, int has_bias, const T *residual, int has_residual) { \
    constexpr int BM_ = 16 * MTM * WTM;                                        \
    constexpr int BN_ = 8 * MTN * WTN;                                         \
    const int bx_ = (int)(blockIdx.z * gridDim.x + blockIdx.x);                \
    if ((int)blockIdx.y * BM_ + BM_ <= M && bx_ * BN_ + BN_ <= N)              \
      gemm_bf16_k128_impl<T, SUB, false, true, MTM, MTN, WTM, WTN, THR>(       \
          A, B, C, M, N, K, bias, has_bias, residual, has_residual, K,         \
          nullptr);                                                            \
    else                                                                       \
      gemm_bf16_k128_impl<T, SUB, true, true, MTM, MTN, WTM, WTN, THR>(        \
          A, B, C, M, N, K, bias, has_bias, residual, has_residual, K,         \
          nullptr);                                                            \
  }

GEMM_TN_ENTRY_KTILE(__nv_bfloat16, gemm_bf16_k128m32, 4, 2, 2, 1, 2, 128)
GEMM_TN_ENTRY_KTILE(__half, gemm_f16tn_k128m32, 4, 2, 2, 1, 2, 128)
GEMM_TN_ENTRY_KTILE(__nv_bfloat16, gemm_bf16_k64m32, 2, 2, 2, 1, 2, 128)
GEMM_TN_ENTRY_KTILE(__half, gemm_f16tn_k64m32, 2, 2, 2, 1, 2, 128)
GEMM_TN_ENTRY_KTILE(__nv_bfloat16, gemm_bf16_k256n32, 8, 2, 2, 2, 2, 128)
GEMM_TN_ENTRY_KTILE(__half, gemm_f16tn_k256n32, 8, 2, 2, 2, 2, 128)

// KTILE + L2::256B prefetch на B (W-стрим): cuBLAS на attn M=32 при той же
// occupancy 13% держит DRAM 75-77% vs наши 73% (ncu 2026-06-05) — добираем
// полосу качеством стрима, не параллелизмом (split-K её РОНЯЛ: 73→67%).
#define GEMM_TN_ENTRY_KTILE_L2(T, NAME, SUB, MTM, MTN, WTM, WTN, THR)          \
  extern "C" __global__ void __launch_bounds__(THR)                            \
      NAME(const T *A, const T *B, T *C, int M, int N, int K,                  \
           const T *bias, int has_bias, const T *residual, int has_residual) { \
    constexpr int BM_ = 16 * MTM * WTM;                                        \
    constexpr int BN_ = 8 * MTN * WTN;                                         \
    const int bx_ = (int)(blockIdx.z * gridDim.x + blockIdx.x);                \
    if ((int)blockIdx.y * BM_ + BM_ <= M && bx_ * BN_ + BN_ <= N)              \
      gemm_bf16_k128_impl<T, SUB, false, true, MTM, MTN, WTM, WTN, THR, true>( \
          A, B, C, M, N, K, bias, has_bias, residual, has_residual, K,         \
          nullptr);                                                            \
    else                                                                       \
      gemm_bf16_k128_impl<T, SUB, true, true, MTM, MTN, WTM, WTN, THR, true>(  \
          A, B, C, M, N, K, bias, has_bias, residual, has_residual, K,         \
          nullptr);                                                            \
  }

GEMM_TN_ENTRY_KTILE_L2(__nv_bfloat16, gemm_bf16_k128m32l2, 4, 2, 2, 1, 2, 128)
GEMM_TN_ENTRY_KTILE_L2(__half, gemm_f16tn_k128m32l2, 4, 2, 2, 1, 2, 128)
GEMM_TN_ENTRY_KTILE_L2(__nv_bfloat16, gemm_bf16_k256m32l2, 8, 2, 2, 1, 2, 128)
GEMM_TN_ENTRY_KTILE_L2(__half, gemm_f16tn_k256m32l2, 8, 2, 2, 1, 2, 128)
GEMM_TN_ENTRY_KTILE_L2(__nv_bfloat16, gemm_bf16_k64m32l2, 2, 2, 2, 1, 2, 128)
GEMM_TN_ENTRY_KTILE_L2(__half, gemm_f16tn_k64m32l2, 2, 2, 2, 1, 2, 128)
// m32 на 64 потоках (2 варпа, warp-тайл 16×32): ×2 MMA/ILP на варп —
// cuBLAS-стиль «меньше варпов, больше регистров» (их 84 рег лучше скейлятся
// клоком; наш остаток attn-32 на горячих клоках).
GEMM_TN_ENTRY_KTILE_L2(__nv_bfloat16, gemm_bf16_k128m32w, 4, 2, 1, 1, 4, 64)
GEMM_TN_ENTRY_KTILE_L2(__half, gemm_f16tn_k128m32w, 4, 2, 1, 1, 4, 64)

#define GEMM_TN_ENTRY_K128_SPLITK(T, NAME, SUB)                                \
  extern "C" __global__ void __launch_bounds__(256)                            \
      NAME(const T *A, const T *B, float *ws, int M, int N, int K,             \
           int kchunk) {                                                       \
    const int k0 = (int)blockIdx.z * kchunk;                                   \
    const int len = min(kchunk, K - k0);                                       \
    float *wsz = ws + (size_t)blockIdx.z * (size_t)M * (size_t)N;              \
    if ((int)blockIdx.y * 64 + 64 <= M && (int)blockIdx.x * 64 + 64 <= N)      \
      gemm_bf16_k128_impl<T, SUB, false, false>(A + k0, B + k0, nullptr, M, N, \
                                                len, nullptr, 0, nullptr, 0,   \
                                                K, wsz);                       \
    else                                                                       \
      gemm_bf16_k128_impl<T, SUB, true, false>(A + k0, B + k0, nullptr, M, N,  \
                                               len, nullptr, 0, nullptr, 0,    \
                                               K, wsz);                        \
  }

GEMM_TN_ENTRY_K128_SPLITK(__nv_bfloat16, gemm_bf16_splitk_k128, 4)
GEMM_TN_ENTRY_K128_SPLITK(__half, gemm_f16tn_splitk_k128, 4)

// split-K на генерик-тайле k-класса (m32 и т.п.): grid.z = k-чанк, RASTER=false.
// m32-сплит: attn M=32 даёт 128 CTA — половина слотов SM (smem-лимит 3 CTA/SM,
// occupancy 13%, DRAM 73%); BM=32 расти нельзя (W перечитывался бы) → CTA
// добираем по K. splits=2 → 256 CTA (заполняет 246 слотов), ws +4MB к 34MB.
#define GEMM_TN_ENTRY_KTILE_SPLITK(T, NAME, SUB, MTM, MTN, WTM, WTN, THR)      \
  extern "C" __global__ void __launch_bounds__(THR)                            \
      NAME(const T *A, const T *B, float *ws, int M, int N, int K,             \
           int kchunk) {                                                       \
    constexpr int BM_ = 16 * MTM * WTM;                                        \
    constexpr int BN_ = 8 * MTN * WTN;                                         \
    const int k0 = (int)blockIdx.z * kchunk;                                   \
    const int len = min(kchunk, K - k0);                                       \
    float *wsz = ws + (size_t)blockIdx.z * (size_t)M * (size_t)N;              \
    if ((int)blockIdx.y * BM_ + BM_ <= M && (int)blockIdx.x * BN_ + BN_ <= N)  \
      gemm_bf16_k128_impl<T, SUB, false, false, MTM, MTN, WTM, WTN, THR>(      \
          A + k0, B + k0, nullptr, M, N, len, nullptr, 0, nullptr, 0, K, wsz); \
    else                                                                       \
      gemm_bf16_k128_impl<T, SUB, true, false, MTM, MTN, WTM, WTN, THR>(       \
          A + k0, B + k0, nullptr, M, N, len, nullptr, 0, nullptr, 0, K, wsz); \
  }

GEMM_TN_ENTRY_KTILE_SPLITK(__nv_bfloat16, gemm_bf16_splitk_k128m32, 4, 2, 2, 1, 2, 128)
GEMM_TN_ENTRY_KTILE_SPLITK(__half, gemm_f16tn_splitk_k128m32, 4, 2, 2, 1, 2, 128)

#define GEMM_TN_ENTRY_SPLITK(T, NAME, KSTAGE)                                  \
  extern "C" __global__ void __launch_bounds__(256)                            \
      NAME(const T *A, const T *B, float *ws, int M, int N, int K,             \
           int kchunk) {                                                       \
    gemm_splitk_s64_impl<T, KSTAGE>(A, B, ws, M, N, K, kchunk);                \
  }

#define GEMM_TN_ENTRY_SPLITK_REDUCE(T, NAME)                                   \
  extern "C" __global__ void __launch_bounds__(256)                            \
      NAME(const float *ws, T *C, long long mn, int N, int splits,             \
           const T *bias, int has_bias, const T *residual,                     \
           int has_residual) {                                                 \
    splitk_reduce_impl<T>(ws, C, mn, N, splits, bias, has_bias, residual,      \
                          has_residual);                                       \
  }

GEMM_TN_ENTRY_S64(__nv_bfloat16, gemm_bf16_swz_s64s8, 8, true, false)
GEMM_TN_ENTRY_PART_HYB(__nv_bfloat16, gemm_bf16_part_s64s8, 8, 2, 4, 2, 2, 1)
GEMM_TN_ENTRY_S64(__half, gemm_f16tn_swz_s64s8, 8, true, false)
GEMM_TN_ENTRY_PART_HYB(__half, gemm_f16tn_part_s64s8, 8, 2, 4, 2, 2, 1)

GEMM_TN_ENTRY_S64(__nv_bfloat16, gemm_bf16_swz_s64s6, 6, true, false)
GEMM_TN_ENTRY_PART_HYB(__nv_bfloat16, gemm_bf16_part_s64s6, 6, 2, 4, 2, 2, 1)
GEMM_TN_ENTRY_S64(__half, gemm_f16tn_swz_s64s6, 6, true, false)
GEMM_TN_ENTRY_PART_HYB(__half, gemm_f16tn_part_s64s6, 6, 2, 4, 2, 2, 1)

// split-K на генерик-тайле gemm_bf16_impl_ex (b256-сплит = рецепт cuBLAS
// ff_down-256 «128x256_32x3 split-5»: их грид (8,4,5)=160 CTA, smem 73.73KB =
// наша b256s3-геометрия 256×128 + сплит по K; SM у них 91%).
#define GEMM_TN_ENTRY_SPLITK_CFG(T, NAME, KSTAGE, MTM, MTN, WTM, WTN)          \
  extern "C" __global__ void __launch_bounds__(256)                            \
      NAME(const T *A, const T *B, float *ws, int M, int N, int K,             \
           int kchunk) {                                                       \
    constexpr int BM_ = 16 * MTM * WTM;                                        \
    constexpr int BN_ = 8 * MTN * WTN;                                         \
    const int k0 = (int)blockIdx.z * kchunk;                                   \
    const int len = min(kchunk, K - k0);                                       \
    float *wsz = ws + (size_t)blockIdx.z * (size_t)M * (size_t)N;              \
    if ((int)blockIdx.y * BM_ + BM_ <= M && (int)blockIdx.x * BN_ + BN_ <= N)  \
      gemm_bf16_impl_ex<T, 16, 8, 16, MTM, MTN, WTM, WTN, 2, 0, 0, KSTAGE,     \
                        false, false>(A + k0, B + k0, nullptr, M, N, len,      \
                                      nullptr, 0, nullptr, 0, K, wsz);         \
    else                                                                       \
      gemm_bf16_impl_ex<T, 16, 8, 16, MTM, MTN, WTM, WTN, 2, 0, 0, KSTAGE,     \
                        false, true>(A + k0, B + k0, nullptr, M, N, len,       \
                                     nullptr, 0, nullptr, 0, K, wsz);          \
  }

GEMM_TN_ENTRY_SPLITK_CFG(__nv_bfloat16, gemm_bf16_splitk_b256s3, 3, 4, 2, 4, 8)
GEMM_TN_ENTRY_SPLITK_CFG(__half, gemm_f16tn_splitk_b256s3, 3, 4, 2, 4, 8)

GEMM_TN_ENTRY_SPLITK(__nv_bfloat16, gemm_bf16_splitk_s64s3, 3)
GEMM_TN_ENTRY_SPLITK(__half, gemm_f16tn_splitk_s64s3, 3)
GEMM_TN_ENTRY_SPLITK(__nv_bfloat16, gemm_bf16_splitk_s64s6, 6)
GEMM_TN_ENTRY_SPLITK(__half, gemm_f16tn_splitk_s64s6, 6)
GEMM_TN_ENTRY_SPLITK(__nv_bfloat16, gemm_bf16_splitk_s64s8, 8)
GEMM_TN_ENTRY_SPLITK(__half, gemm_f16tn_splitk_s64s8, 8)
GEMM_TN_ENTRY_SPLITK_REDUCE(__nv_bfloat16, gemm_bf16_splitk_reduce)
GEMM_TN_ENTRY_SPLITK_REDUCE(__half, gemm_f16tn_splitk_reduce)

// Большой тайл 256×128 (как cutlass 256x128_32x3 у cuBLAS): −25% L2-байт/FLOP →
// меньше энергии → выше DVFS-клок. Warp-грид 4×2, warp-тайл 64×64, RC=128 рег.
#define GEMM_TN_ENTRY_B256(T, NAME, KSTAGE, SWZ, PART)                         \
  extern "C" __global__ void __launch_bounds__(256)                            \
      NAME(const T *A, const T *B, T *C, int M, int N, int K,                  \
           const T *bias, int has_bias, const T *residual, int has_residual) { \
    gemm_bf16_impl<T, 16, 8, 16, 4, 2, 4, 8, 2, 0, 0, KSTAGE, SWZ, PART>(      \
        A, B, C, M, N, K, bias, has_bias, residual, has_residual);            \
  }

// Транспонированный крупный тайл 128×256 (рецепт cuBLAS ff_up-4992:
// cutlass 256x128_32x3 в C^T-ориентации = 128(M)×256(N), 2496 ровных CTA на
// M=4992 против наших 19.5 рядов 256×128; широкий N-тайл реюзит A на N=16384).
// Варп-грид 2×4, варп-тайл 64×64, smem s3 = 73.73KB (как у них).
#define GEMM_TN_ENTRY_B256T(T, NAME, KSTAGE, SWZ, PART)                        \
  extern "C" __global__ void __launch_bounds__(256)                            \
      NAME(const T *A, const T *B, T *C, int M, int N, int K,                  \
           const T *bias, int has_bias, const T *residual, int has_residual) { \
    gemm_bf16_impl<T, 16, 8, 16, 2, 4, 4, 8, 2, 0, 0, KSTAGE, SWZ, PART>(      \
        A, B, C, M, N, K, bias, has_bias, residual, has_residual);            \
  }

GEMM_TN_ENTRY_B256T(__nv_bfloat16, gemm_bf16_swz_b256ts3, 3, true, false)
GEMM_TN_ENTRY_PART_HYB(__nv_bfloat16, gemm_bf16_part_b256ts3, 3, 2, 4, 4, 8, 1)
GEMM_TN_ENTRY_B256T(__nv_bfloat16, gemm_bf16_swz_b256ts4, 4, true, false)
GEMM_TN_ENTRY_PART_HYB(__nv_bfloat16, gemm_bf16_part_b256ts4, 4, 2, 4, 4, 8, 1)
GEMM_TN_ENTRY_B256T(__half, gemm_f16tn_swz_b256ts3, 3, true, false)
GEMM_TN_ENTRY_PART_HYB(__half, gemm_f16tn_part_b256ts3, 3, 2, 4, 4, 8, 1)
GEMM_TN_ENTRY_B256T(__half, gemm_f16tn_swz_b256ts4, 4, true, false)
GEMM_TN_ENTRY_PART_HYB(__half, gemm_f16tn_part_b256ts4, 4, 2, 4, 4, 8, 1)

GEMM_TN_ENTRY_B256(__nv_bfloat16, gemm_bf16_swz_b256s3, 3, true, false)
GEMM_TN_ENTRY_PART_HYB(__nv_bfloat16, gemm_bf16_part_b256s3, 3, 4, 2, 4, 8, 1)
GEMM_TN_ENTRY_B256(__nv_bfloat16, gemm_bf16_swz_b256s4, 4, true, false)
GEMM_TN_ENTRY_PART_HYB(__nv_bfloat16, gemm_bf16_part_b256s4, 4, 4, 2, 4, 8, 1)
GEMM_TN_ENTRY_B256(__half, gemm_f16tn_swz_b256s3, 3, true, false)
GEMM_TN_ENTRY_PART_HYB(__half, gemm_f16tn_part_b256s3, 3, 4, 2, 4, 8, 1)
GEMM_TN_ENTRY_B256(__half, gemm_f16tn_swz_b256s4, 4, true, false)
GEMM_TN_ENTRY_PART_HYB(__half, gemm_f16tn_part_b256s4, 4, 4, 2, 4, 8, 1)

// ===================== TMA+mbarrier (порт mxfp8-rot) =====================
// bf16/f16 TMA+mbarrier GEMM малых M — порт рецепта mxfp8-rot/nvfp4 (sm_120a):
// TMA shared::cta (НЕ cluster: C7506 глушит setmaxnreg) + mbarrier-конвейер +
// fused-producer(tid 0) + ротация пар k16-блоков (байтовая геометрия идентична
// mxfp8: 32Б на kk-блок, XOR pp*64) + staged-эпилог. Цель: снять потолок
// cp.async-класса (block-wide __syncthreads на каждую стадию) на attn 32-128.
// C[M,N] T = A[M,K] @ B[N,K]ᵀ, f32-аккум, OOB по M закрывает TMA zero-fill.

__device__ __forceinline__ unsigned int bt_swz128(unsigned int off) {
    return off ^ (((off >> 7) & 7u) << 4);
}

template <int NREG> __device__ __forceinline__ void bt_setmaxnreg_dec() {
    if constexpr (NREG > 0)
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(NREG));
}
template <int NREG> __device__ __forceinline__ void bt_setmaxnreg_inc() {
    if constexpr (NREG > 0)
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NREG));
}

template <typename T>
__device__ __forceinline__ void bt_mma(float *C, const unsigned *A, const unsigned *B);
template <>
__device__ __forceinline__ void bt_mma<__nv_bfloat16>(float *C, const unsigned *A,
                                                      const unsigned *B) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}
template <>
__device__ __forceinline__ void bt_mma<__half>(float *C, const unsigned *A,
                                               const unsigned *B) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}

template <typename T> __device__ __forceinline__ unsigned int bt_pack2(float a, float b);
template <> __device__ __forceinline__ unsigned int bt_pack2<__nv_bfloat16>(float a, float b) {
    __nv_bfloat162 v = __float22bfloat162_rn({a, b});
    return *reinterpret_cast<unsigned int *>(&v);
}
template <> __device__ __forceinline__ unsigned int bt_pack2<__half>(float a, float b) {
    __half2 v = __float22half2_rn({a, b});
    return *reinterpret_cast<unsigned int *>(&v);
}

__device__ __host__ constexpr unsigned int bt_cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1u) / b;
}

template <typename T, unsigned int BM, unsigned int BN, unsigned int WM,
          unsigned int WN, unsigned int STAGES, unsigned int PROD_W = 0u,
          int RDEC = 0, int RINC = 0>
__device__ __forceinline__ void bf16_tma_device(
    const void* __restrict__ a_desc,
    const void* __restrict__ b_desc,
    T*          __restrict__ out,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int raster_gr)
{
    constexpr unsigned int BKB = 128u;  // байт стадии по K = 64 bf16-элемента
    constexpr unsigned int KE = 64u;
    constexpr unsigned int THREADS = WM * WN * 32u;
    constexpr unsigned int WARP_M = BM / WM;
    constexpr unsigned int WARP_N = BN / WN;
    constexpr unsigned int MA = WARP_M / 16u;
    constexpr unsigned int NB = WARP_N / 8u;
    constexpr unsigned int A_SZ = BM * BKB;
    constexpr unsigned int B_SZ = BN * BKB;
    constexpr unsigned int STAGE = A_SZ + B_SZ;
    constexpr unsigned int TX = STAGE;
    constexpr unsigned int B_OFF = A_SZ;
    constexpr unsigned int BAR_OFF = STAGES * STAGE;

    extern __shared__ __align__(128) unsigned char smem[];
    unsigned int sbase = (unsigned int)__cvta_generic_to_shared(smem);
    #define BT_MFULL(b)  (sbase + BAR_OFF + (b) * 8u)
    #define BT_MEMPTY(b) (sbase + BAR_OFF + STAGES * 8u + (b) * 8u)

    unsigned int tid  = threadIdx.x;
    unsigned int warp = tid >> 5;
    unsigned int lane = tid & 31u;
    unsigned int num_kt = K / KE;
    constexpr unsigned int CONS_WARPS = WM * WN;
    bool is_prod = PROD_W != 0u && warp >= CONS_WARPS;

    // L2-растр рецепта rot; tiles_m = ceil (малые M: 1 полоса, TMA нулит OOB).
    unsigned int tiles_m = bt_cdiv(M, BM);
    unsigned int bid = blockIdx.x;
    unsigned int sup = bid / (tiles_m * raster_gr);
    unsigned int rem = bid % (tiles_m * raster_gr);
    unsigned int block_m0 = (rem / raster_gr) * BM;
    unsigned int block_n0 = (sup * raster_gr + rem % raster_gr) * BN;

    if (tid == 0) {
        #pragma unroll
        for (unsigned int s = 0; s < STAGES; s++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(BT_MFULL(s)));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                         :: "r"(BT_MEMPTY(s)), "r"(THREADS));
        }
    }
    __syncthreads();

    auto issue_chunk = [&](unsigned int g) {
        unsigned int buf = g % STAGES;
        unsigned int pass = g / STAGES;
        unsigned int fa = BT_MFULL(buf);
        if (pass > 0) {
            unsigned int ph = (pass - 1u) & 1u;
            asm volatile(
              "{\n.reg .pred p;\nBWE_%=:\n"
              "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
              "@!p bra BWE_%=;\n}\n" :: "r"(BT_MEMPTY(buf)), "r"(ph) : "memory");
        }
        unsigned int kb = g * BKB;
        unsigned long long st;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n"
                     : "=l"(st) : "r"(fa), "r"(TX));
        asm volatile(
          "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
          " [%0], [%1, {%2, %3}], [%4];\n"
          :: "r"(sbase + buf * STAGE), "l"((unsigned long long)a_desc),
             "r"(kb), "r"(block_m0), "r"(fa) : "memory");
        asm volatile(
          "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
          " [%0], [%1, {%2, %3}], [%4];\n"
          :: "r"(sbase + buf * STAGE + B_OFF), "l"((unsigned long long)b_desc),
             "r"(kb), "r"(block_n0), "r"(fa) : "memory");
    };

    if constexpr (PROD_W != 0u) {
        if (is_prod) {
            bt_setmaxnreg_dec<RDEC>();
            if (warp == CONS_WARPS && lane == 0) {
                for (unsigned int g = 0; g < num_kt; g++)
                    issue_chunk(g);
            }
            return;
        }
        bt_setmaxnreg_inc<RINC>();
    } else {
        if (tid == 0) {
            unsigned int pre = (STAGES - 1u < num_kt) ? STAGES - 1u : num_kt;
            for (unsigned int g = 0; g < pre; g++)
                issue_chunk(g);
        }
    }

    unsigned int wm = warp / WN;
    unsigned int wn = warp % WN;
    unsigned int a_lane = bt_swz128((wm * WARP_M + (lane & 15u)) * BKB + (lane >> 4) * 16u);
    unsigned int b_lane = bt_swz128((wn * WARP_N + (lane & 7u)) * BKB + (lane >> 3) * 16u);

    unsigned int aR[2][MA][2][4];
    unsigned int bR[2][NB][2][2];
    float acc[MA][NB][4];
    #pragma unroll
    for (unsigned int m = 0; m < MA; m++)
        #pragma unroll
        for (unsigned int n = 0; n < NB; n++) {
            acc[m][n][0] = 0.f; acc[m][n][1] = 0.f; acc[m][n][2] = 0.f; acc[m][n][3] = 0.f;
        }

    // pp = пара k16-блоков (kk = 2pp, 2pp+1; 32Б каждый — байт-в-байт структура
    // mxfp8-rot). B: один x4-ldmatrix покрывает оба kk пары (XOR pp*64).
    auto load_pair = [&](unsigned int buf, unsigned int pp,
                         unsigned int (&a)[MA][2][4], unsigned int (&b)[NB][2][2]) {
        unsigned int abase = sbase + buf * STAGE;
        unsigned int bbase = abase + B_OFF;
        #pragma unroll
        for (unsigned int m = 0; m < MA; m++) {
            #pragma unroll
            for (unsigned int d = 0; d < 2; d++) {
                unsigned int kk = pp * 2u + d;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a[m][d][0]), "=r"(a[m][d][1]), "=r"(a[m][d][2]), "=r"(a[m][d][3])
                    : "r"((abase + m * (16u * BKB) + a_lane) ^ (kk * 32u)));
            }
        }
        #pragma unroll
        for (unsigned int n = 0; n < NB; n++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(b[n][0][0]), "=r"(b[n][0][1]), "=r"(b[n][1][0]), "=r"(b[n][1][1])
                : "r"((bbase + n * (8u * BKB) + b_lane) ^ (pp * 64u)));
        }
    };
    auto gemm_pair = [&](const unsigned int (&a)[MA][2][4],
                         const unsigned int (&b)[NB][2][2]) {
        #pragma unroll
        for (unsigned int d = 0; d < 2; d++)
            #pragma unroll
            for (unsigned int m = 0; m < MA; m++)
                #pragma unroll
                for (unsigned int n = 0; n < NB; n++)
                    bt_mma<T>(acc[m][n], a[m][d], b[n][d]);
    };

    asm volatile(
      "{\n.reg .pred p;\nBWF0_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra BWF0_%=;\n}\n" :: "r"(BT_MFULL(0)), "r"(0u) : "memory");
    load_pair(0u, 0u, aR[0], bR[0]);

    unsigned int c = 0;
    for (; c + 1u < num_kt; c++) {
        unsigned int buf  = c % STAGES;
        unsigned int nbuf = (c + 1u) % STAGES;
        unsigned int nph  = ((c + 1u) / STAGES) & 1u;
        load_pair(buf, 1u, aR[1], bR[1]);
        gemm_pair(aR[0], bR[0]);
        unsigned long long st;
        asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
                     : "=l"(st) : "r"(BT_MEMPTY(buf)) : "memory");
        if constexpr (PROD_W == 0u) {
            if (tid == 0) {
                unsigned int g = c + STAGES - 1u;
                if (g < num_kt)
                    issue_chunk(g);
            }
        }
        asm volatile(
          "{\n.reg .pred p;\nBWF_%=:\n"
          "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
          "@!p bra BWF_%=;\n}\n" :: "r"(BT_MFULL(nbuf)), "r"(nph) : "memory");
        load_pair(nbuf, 0u, aR[0], bR[0]);
        gemm_pair(aR[1], bR[1]);
    }
    {
        unsigned int buf = c % STAGES;
        load_pair(buf, 1u, aR[1], bR[1]);
        gemm_pair(aR[0], bR[0]);
        unsigned long long st;
        asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
                     : "=l"(st) : "r"(BT_MEMPTY(buf)) : "memory");
        gemm_pair(aR[1], bR[1]);
    }

    // staged-эпилог: smem свободен; M-гард на финальном STG (TMA-нули строк >= M
    // в C не пишем). N кратно BN (гард диспетча).
    static_assert(BM * BN * 2u <= STAGES * STAGE, "bf16 TMA: эпилог не влезает");
    asm volatile("bar.sync 7, %0;\n" :: "r"(THREADS) : "memory");
    #pragma unroll
    for (unsigned int m = 0; m < MA; m++) {
        unsigned int row = wm * WARP_M + m * 16u + (lane >> 2);
        #pragma unroll
        for (unsigned int n = 0; n < NB; n++) {
            unsigned int col = wn * WARP_N + n * 8u + (lane & 3u) * 2u;
            unsigned int lo = bt_pack2<T>(acc[m][n][0], acc[m][n][1]);
            unsigned int hi = bt_pack2<T>(acc[m][n][2], acc[m][n][3]);
            asm volatile("st.shared.u32 [%0], %1;\n"
                :: "r"(sbase + (row * BN + col) * 2u), "r"(lo));
            asm volatile("st.shared.u32 [%0], %1;\n"
                :: "r"(sbase + ((row + 8u) * BN + col) * 2u), "r"(hi));
        }
    }
    asm volatile("bar.sync 7, %0;\n" :: "r"(THREADS) : "memory");
    constexpr unsigned int EP_CHUNKS = (BM * BN * 2u) / 16u;
    constexpr unsigned int ROW_CHUNKS = (BN * 2u) / 16u;
    for (unsigned int s = tid; s < EP_CHUNKS; s += THREADS) {
        unsigned int row   = s / ROW_CHUNKS;
        if (block_m0 + row >= M)
            continue;
        unsigned int n_off = (s % ROW_CHUNKS) * 8u;
        unsigned int v0, v1, v2, v3;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "r"(sbase + s * 16u));
        T* dst = out + (size_t)(block_m0 + row) * N + block_n0 + n_off;
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};\n"
            :: "l"(dst), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
    }
    #undef BT_MFULL
    #undef BT_MEMPTY
}

#define BT_ENTRY(T, NAME, BM, BN, WM, WN, STAGES)                              \
  extern "C" __global__ void __launch_bounds__(WM * WN * 32, 1)                \
      NAME(const void *a_desc, const void *b_desc, T *out, unsigned int M,     \
           unsigned int N, unsigned int K, unsigned int raster_gr) {           \
    bf16_tma_device<T, BM, BN, WM, WN, STAGES>(a_desc, b_desc, out, M, N, K,   \
                                               raster_gr);                     \
  }

BT_ENTRY(__nv_bfloat16, gn_bf16_tma_64x64_s3, 64, 64, 2, 2, 3)
BT_ENTRY(__nv_bfloat16, gn_bf16_tma_64x64_s5, 64, 64, 2, 2, 5)
BT_ENTRY(__half, gn_f16_tma_64x64_s3, 64, 64, 2, 2, 3)
BT_ENTRY(__half, gn_f16_tma_64x64_s5, 64, 64, 2, 2, 5)

// Выделенный producer-warpgroup (drot-схема mxfp8: fused-tid0 голодал;
// setmaxnreg 24/240 работает после ::cta-фикса). 128 консьюмеров + 128
// продьюсеров (warpgroup-гранулярность setmaxnreg), активен tid 128.
#define BT_ENTRY_D(T, NAME, BM, BN, WM, WN, STAGES)                            \
  extern "C" __global__ void __launch_bounds__(WM * WN * 32 + 128, 1)          \
      NAME(const void *a_desc, const void *b_desc, T *out, unsigned int M,     \
           unsigned int N, unsigned int K, unsigned int raster_gr) {           \
    bf16_tma_device<T, BM, BN, WM, WN, STAGES, 4u, 24, 240>(                   \
        a_desc, b_desc, out, M, N, K, raster_gr);                              \
  }

BT_ENTRY_D(__nv_bfloat16, gn_bf16_tma_64x64_s3d, 64, 64, 2, 2, 3)
BT_ENTRY_D(__nv_bfloat16, gn_bf16_tma_64x64_s5d, 64, 64, 2, 2, 5)
BT_ENTRY_D(__half, gn_f16_tma_64x64_s3d, 64, 64, 2, 2, 3)
BT_ENTRY_D(__half, gn_f16_tma_64x64_s5d, 64, 64, 2, 2, 5)

// Тайл 64×32: ×2 CTA по N (attn 32-128: было 64 CTA на 82 SM) + 2 CTA/SM.
BT_ENTRY(__nv_bfloat16, gn_bf16_tma_64x32_s3, 64, 32, 2, 2, 3)
BT_ENTRY(__nv_bfloat16, gn_bf16_tma_64x32_s5, 64, 32, 2, 2, 5)
BT_ENTRY(__half, gn_f16_tma_64x32_s3, 64, 32, 2, 2, 3)
BT_ENTRY(__half, gn_f16_tma_64x32_s5, 64, 32, 2, 2, 5)

// Тайл 32×32 (форма cuBLAS-skinny, M=32 без холостых строк): стадия 8KB →
// s5 = 40KB (2 CTA/SM), s8 = 64KB.
BT_ENTRY(__nv_bfloat16, gn_bf16_tma_32x32_s5, 32, 32, 2, 2, 5)
BT_ENTRY(__nv_bfloat16, gn_bf16_tma_32x32_s8, 32, 32, 2, 2, 8)
BT_ENTRY(__half, gn_f16_tma_32x32_s5, 32, 32, 2, 2, 5)
BT_ENTRY(__half, gn_f16_tma_32x32_s8, 32, 32, 2, 2, 8)
// 32×32 + дед-продьюсер (s5d-схема, на 256-классе дала 69/75/79→76/98/83).
BT_ENTRY_D(__nv_bfloat16, gn_bf16_tma_32x32_s5d, 32, 32, 2, 2, 5)
BT_ENTRY_D(__half, gn_f16_tma_32x32_s5d, 32, 32, 2, 2, 5)
// Крупные тайлы (зеркало cuBLAS ff_down-256: 256 потоков, smem 73.7KB, 222 рег,
// SM 91% vs наши 76 у s5d-64×64): стадия (64+128)*128 = 24.6KB → s3 = 73.8KB,
// s4 = 98.4KB (влезает в optin 101376 ровно).
BT_ENTRY_D(__nv_bfloat16, gn_bf16_tma_64x128_s3d, 64, 128, 2, 2, 3)
BT_ENTRY_D(__nv_bfloat16, gn_bf16_tma_64x128_s4d, 64, 128, 2, 2, 4)
BT_ENTRY_D(__nv_bfloat16, gn_bf16_tma_128x64_s3d, 128, 64, 2, 2, 3)
BT_ENTRY_D(__nv_bfloat16, gn_bf16_tma_128x64_s4d, 128, 64, 2, 2, 4)
BT_ENTRY_D(__half, gn_f16_tma_64x128_s3d, 64, 128, 2, 2, 3)
BT_ENTRY_D(__half, gn_f16_tma_64x128_s4d, 64, 128, 2, 2, 4)
BT_ENTRY_D(__half, gn_f16_tma_128x64_s3d, 128, 64, 2, 2, 3)
BT_ENTRY_D(__half, gn_f16_tma_128x64_s4d, 128, 64, 2, 2, 4)
