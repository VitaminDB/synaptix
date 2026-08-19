#include <cuda_fp16.h>
#include <cuda_fp8.h>

// MXFP8-KV flash-decode v2 (sm_120a): GQA-групповое split-K ядро для малых Tq
// (decode Tq=1, MTP-verify Tq=2..8). Отличия от скалярного flash_decode.cu:
//
//   • один блок = (bi, h_kv, subgroup, ti, split): GROUP query-голов одной
//     KV-головы считаются вместе — KV-сегмент читается из DRAM один раз на
//     группу (скалярное ядро читало его n_rep раз, по числу q-голов);
//   • K/V читаются uint4-загрузками (16 байт), деквант E4M3 — аппаратным
//     cvt.rn.f16x2.e4m3x2 (__nv_cvt_fp8x2_to_halfraw2), а не ветвистой
//     побайтовой распаковкой (~10 инструкций/байт);
//   • E8M0-скейлы тайла предекодируются в float в smem одним проходом
//     (скалярное ядро читало v_scale из DRAM на каждую пару (j,d));
//   • Q заранее умножается на attn_scale и лежит в smem как half2 —
//     Q·K-дот идёт HFMA2-цепочками по 8 с float-слиянием на 32-блок
//     (полуслово не переполняется: |q·scale| ≲ 1, |k| ≤ 448, суммы ≤ 4k);
//   • V-аккумулятор — float-регистры по d-отображению (поток владеет
//     элементами d = tid + o·128), между тайлами домножается на alpha —
//     никакой кросс-поточной редукции в конце не нужно.
//
// Раскладки и семантика совпадают со скалярным ядром: q (B,NH,Tq,D), k/v —
// E4M3-байты (B,NKV,T,D) c физическим T-шагом t_stride, k/v_scale — E8M0
// (B,NKV,T,D/32), partial'ы (B·NH·Tq·split_k, D) + (m,l) ненормализованные,
// merge — v2-копия с SPLIT_K_MAX 64. Причинная маска: q_pos = Tkv-Tq+ti.
//
// NVRTC: без <math.h>; -inf через __int_as_float (как flash_decode.cu).

#define FD2_NEG_INF (__int_as_float(0xFF800000))
#define FD2_TILE 128
#define FD2_BLOCK 128
#define FD2_SPLIT_K_MAX 64

__device__ __forceinline__ bool fd2_is_finite(float x) {
  return (__float_as_int(x) & 0x7F800000) != 0x7F800000;
}

__device__ __forceinline__ float fd2_ldq(const float* p) { return *p; }
__device__ __forceinline__ float fd2_ldq(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float fd2_ldq(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void fd2_st(float* p, float v) { *p = v; }
__device__ __forceinline__ void fd2_st(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void fd2_st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// 2×E4M3 (little-endian пара байт) → half2 одной cvt-инструкцией.
__device__ __forceinline__ __half2 fd2_cvt2(unsigned short two) {
  return __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)two, __NV_E4M3));
}

// E8M0: байт b → 2^(b-127), пол 1e-12 (совпадает с append-квантом и CPU).
__device__ __forceinline__ float fd2_e8m0(unsigned char b) {
  return fmaxf(__uint_as_float(((unsigned)b) << 23), 1e-12f);
}

__device__ __forceinline__ float fd2_reduce_max(float val, float* warp_red, int tid) {
  unsigned int mask = 0xFFFFFFFFu;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) val = fmaxf(val, __shfl_down_sync(mask, val, off, 32));
  int warp = tid >> 5, lane = tid & 31;
  if (lane == 0) warp_red[warp] = val;
  __syncthreads();
  float r = FD2_NEG_INF;
  if (warp == 0) {
    r = (lane < FD2_BLOCK / 32) ? warp_red[lane] : FD2_NEG_INF;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) r = fmaxf(r, __shfl_down_sync(mask, r, off, 32));
  }
  return r;  // валиден в lane 0 warp 0
}

__device__ __forceinline__ float fd2_reduce_sum(float val, float* warp_red, int tid) {
  unsigned int mask = 0xFFFFFFFFu;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(mask, val, off, 32);
  int warp = tid >> 5, lane = tid & 31;
  if (lane == 0) warp_red[warp] = val;
  __syncthreads();
  float r = 0.0f;
  if (warp == 0) {
    r = (lane < FD2_BLOCK / 32) ? warp_red[lane] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) r += __shfl_down_sync(mask, r, off, 32);
  }
  return r;  // валиден в lane 0 warp 0
}

// Q·K-дот одного 32-блока: 8 uint = 32 байта K, q — half2 из smem (broadcast).
// Две независимые HFMA2-цепочки по 8, float-слияние, умножение на sv снаружи.
__device__ __forceinline__ float fd2_dot_block(
    const unsigned int* kw,           // 8 распакованных uint (32 байта K)
    const __half2* q_pair) {          // q_sh + blk*16: 16 half2 блока
  __half2 acc0 = __half2(__half2_raw{0, 0});
  __half2 acc1 = acc0;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    unsigned int u = kw[i];
    acc0 = __hfma2(fd2_cvt2((unsigned short)(u & 0xFFFFu)), q_pair[i * 2], acc0);
    acc1 = __hfma2(fd2_cvt2((unsigned short)(u >> 16)), q_pair[i * 2 + 1], acc1);
  }
  float2 f0 = __half22float2(acc0);
  float2 f1 = __half22float2(acc1);
  float lo = (f0.x + f0.y) + (f1.x + f1.y);
  acc0 = __half2(__half2_raw{0, 0});
  acc1 = acc0;
  #pragma unroll
  for (int i = 4; i < 8; i++) {
    unsigned int u = kw[i];
    acc0 = __hfma2(fd2_cvt2((unsigned short)(u & 0xFFFFu)), q_pair[i * 2], acc0);
    acc1 = __hfma2(fd2_cvt2((unsigned short)(u >> 16)), q_pair[i * 2 + 1], acc1);
  }
  f0 = __half22float2(acc0);
  f1 = __half22float2(acc1);
  return lo + (f0.x + f0.y) + (f1.x + f1.y);
}

template <typename T, int GROUP, int DHEAD>
__device__ __forceinline__ void fd2_split_impl(
    const T* __restrict__ q,
    const unsigned char* __restrict__ k, const unsigned char* __restrict__ v,
    const unsigned char* __restrict__ k_scale, const unsigned char* __restrict__ v_scale,
    float* __restrict__ partial_acc, float* __restrict__ partial_m, float* __restrict__ partial_l,
    int B, int NH, int NKV, int Tq, int Tkv,
    float scale, int causal, int split_k, int t_stride) {
  constexpr int NB = DHEAD / 32;
  constexpr int NOWN = DHEAD / FD2_BLOCK;  // элементов V-акка на поток (1 или 2)
  int n_rep = NH / NKV;
  int nsub = n_rep / GROUP;

  long gx = blockIdx.x;  // ((bi·NKV + h_kv)·nsub + sub)·Tq + ti
  int ti = (int)(gx % Tq);
  int sub = (int)((gx / Tq) % nsub);
  int h_kv = (int)((gx / ((long)Tq * nsub)) % NKV);
  int bi = (int)(gx / ((long)Tq * nsub * NKV));
  int split_id = (int)blockIdx.y;
  if (bi >= B || split_id >= split_k) return;
  int h0 = h_kv * n_rep + sub * GROUP;

  int tid = threadIdx.x;
  int q_pos = (Tkv >= Tq) ? (Tkv - Tq + ti) : ti;
  long kv_stride = (t_stride > 0) ? (long)t_stride : (long)Tkv;
  long kv_head_base = (long)(bi * NKV + h_kv) * kv_stride;  // в T-строках

  int seg = (Tkv + split_k - 1) / split_k;
  int kv_start = split_id * seg;
  int kv_end_unb = kv_start + seg;
  int kv_end = (kv_end_unb < Tkv) ? kv_end_unb : Tkv;

  // ── dynamic smem: q_sh | p_sh | ksc_sh | vsc_sh ──
  extern __shared__ unsigned char fd2_smem[];
  __half2* q_sh = reinterpret_cast<__half2*>(fd2_smem);            // [GROUP][DHEAD/2]
  float* p_sh = reinterpret_cast<float*>(q_sh + GROUP * (DHEAD / 2));  // [GROUP][TILE]
  float* ksc_sh = p_sh + GROUP * FD2_TILE;                         // [NB][TILE]
  float* vsc_sh = ksc_sh + NB * FD2_TILE;                          // [NB][TILE]
  __shared__ float warp_red[32];
  __shared__ float run_m[GROUP], run_l[GROUP], alpha_sh[GROUP], mnew_sh[GROUP];

  auto partial_idx = [&](int g) {
    return (((long)bi * NH + (h0 + g)) * Tq + ti) * (long)split_k + split_id;
  };

  // Пустой сегмент → partial = (-inf, 0, 0) на каждую голову группы.
  if (kv_start >= kv_end) {
    #pragma unroll
    for (int g = 0; g < GROUP; g++) {
      long pi = partial_idx(g);
      if (tid == 0) { partial_m[pi] = FD2_NEG_INF; partial_l[pi] = 0.0f; }
      for (int d = tid; d < DHEAD; d += FD2_BLOCK) partial_acc[pi * DHEAD + d] = 0.0f;
    }
    return;
  }

  // Q группы → smem как half2, attn_scale вплавлен (дот тогда сразу = score).
  for (int i = tid; i < GROUP * DHEAD; i += FD2_BLOCK) {
    int g = i / DHEAD, d = i % DHEAD;
    const T* q_row = q + (((long)(bi * NH + h0 + g) * Tq + ti) * DHEAD);
    float qv = fd2_ldq(q_row + d) * scale;
    reinterpret_cast<__half*>(q_sh)[g * DHEAD + d] = __float2half(qv);
  }
  if (tid < GROUP) { run_m[tid] = FD2_NEG_INF; run_l[tid] = 0.0f; }

  float acc[GROUP][NOWN];
  #pragma unroll
  for (int g = 0; g < GROUP; g++)
    #pragma unroll
    for (int o = 0; o < NOWN; o++) acc[g][o] = 0.0f;
  __syncthreads();

  for (int tile_base = kv_start; tile_base < kv_end; tile_base += FD2_TILE) {
    int tile_count = kv_end - tile_base;
    if (tile_count > FD2_TILE) tile_count = FD2_TILE;

    // Stage A: E8M0-скейлы тайла → float в smem (layout [blk][token] —
    // broadcast-чтения и в K-, и в V-фазе).
    for (int i = tid; i < tile_count * NB; i += FD2_BLOCK) {
      int t = i / NB, blk = i % NB;
      long srow = (kv_head_base + tile_base + t) * NB;
      ksc_sh[blk * FD2_TILE + t] = fd2_e8m0(k_scale[srow + blk]);
      vsc_sh[blk * FD2_TILE + t] = fd2_e8m0(v_scale[srow + blk]);
    }
    __syncthreads();

    // Stage B: поток ↔ токен, скоры всех GROUP голов за один проход по K-строке.
    float dot[GROUP];
    #pragma unroll
    for (int g = 0; g < GROUP; g++) dot[g] = FD2_NEG_INF;
    if (tid < tile_count) {
      int kv_t = tile_base + tid;
      if (!(causal && kv_t > q_pos)) {
        const uint4* k_row = reinterpret_cast<const uint4*>(k + (kv_head_base + kv_t) * DHEAD);
        #pragma unroll
        for (int g = 0; g < GROUP; g++) dot[g] = 0.0f;
        #pragma unroll
        for (int blk = 0; blk < NB; blk++) {
          uint4 w0 = k_row[blk * 2];
          uint4 w1 = k_row[blk * 2 + 1];
          unsigned int kw[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
          float sv = ksc_sh[blk * FD2_TILE + tid];
          #pragma unroll
          for (int g = 0; g < GROUP; g++)
            dot[g] = fmaf(fd2_dot_block(kw, q_sh + g * (DHEAD / 2) + blk * 16), sv, dot[g]);
        }
      }
    }
    #pragma unroll
    for (int g = 0; g < GROUP; g++) p_sh[g * FD2_TILE + tid] = dot[g];
    __syncthreads();

    // Stage C: online-softmax по каждой голове (s → p, m/l/alpha).
    #pragma unroll
    for (int g = 0; g < GROUP; g++) {
      float m_tile = fd2_reduce_max(p_sh[g * FD2_TILE + tid], warp_red, tid);
      if (tid == 0) {
        float m_curr = run_m[g];
        float m_new = fmaxf(m_curr, m_tile);
        float a;
        if (!fd2_is_finite(m_curr)) a = 0.0f;
        else if (!fd2_is_finite(m_new)) a = 1.0f;
        else a = __expf(m_curr - m_new);
        mnew_sh[g] = m_new;
        alpha_sh[g] = a;
      }
      __syncthreads();
      float m_new = mnew_sh[g];
      float sj = p_sh[g * FD2_TILE + tid];
      float p = (!fd2_is_finite(m_new) || sj == FD2_NEG_INF) ? 0.0f : __expf(sj - m_new);
      p_sh[g * FD2_TILE + tid] = p;
      float p_sum = fd2_reduce_sum(p, warp_red, tid);
      if (tid == 0) {
        run_l[g] = run_l[g] * alpha_sh[g] + p_sum;
        run_m[g] = m_new;
      }
      __syncthreads();
    }

    // Stage D: V-фаза. Поток владеет d = tid + o·128; alpha-домножение
    // регистров, затем проход по токенам тайла: деквант пары байт одной cvt,
    // скейл общий на группу, FFMA на каждую голову.
    #pragma unroll
    for (int g = 0; g < GROUP; g++) {
      float a = alpha_sh[g];
      #pragma unroll
      for (int o = 0; o < NOWN; o++) acc[g][o] *= a;
    }
    for (int j = 0; j < tile_count; j++) {
      const unsigned char* v_row = v + (kv_head_base + tile_base + j) * DHEAD;
      float p[GROUP];
      #pragma unroll
      for (int g = 0; g < GROUP; g++) p[g] = p_sh[g * FD2_TILE + j];
      float vf[NOWN];
      if (NOWN == 2) {
        unsigned short two =
            (unsigned short)(v_row[tid] | ((unsigned short)v_row[tid + FD2_BLOCK] << 8));
        float2 f = __half22float2(fd2_cvt2(two));
        vf[0] = f.x * vsc_sh[(tid / 32) * FD2_TILE + j];
        vf[NOWN - 1] = f.y * vsc_sh[((tid + FD2_BLOCK) / 32) * FD2_TILE + j];
      } else {
        float f = __half2float(__low2half(fd2_cvt2((unsigned short)v_row[tid])));
        vf[0] = f * vsc_sh[(tid / 32) * FD2_TILE + j];
      }
      #pragma unroll
      for (int g = 0; g < GROUP; g++)
        #pragma unroll
        for (int o = 0; o < NOWN; o++) acc[g][o] = fmaf(p[g], vf[o], acc[g][o]);
    }
    __syncthreads();
  }

  // Ненормализованные partial'ы: m/l из tid 0, acc — прямой поэлементный сброс
  // (d-отображение регистров покрывает весь DHEAD без редукции).
  #pragma unroll
  for (int g = 0; g < GROUP; g++) {
    long pi = partial_idx(g);
    if (tid == 0) { partial_m[pi] = run_m[g]; partial_l[pi] = run_l[g]; }
    #pragma unroll
    for (int o = 0; o < NOWN; o++)
      partial_acc[pi * DHEAD + tid + o * FD2_BLOCK] = acc[g][o];
  }
}

// Merge v2: как flash_decode_merge_impl, но SPLIT_K_MAX 64.
template <typename T>
__device__ __forceinline__ void fd2_merge_impl(
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    T* __restrict__ out,
    int B, int NH, int Tq, int D, int split_k) {
  long row = blockIdx.x;  // 0 .. B·NH·Tq
  int ti = (int)(row % Tq);
  int h = (int)((row / Tq) % NH);
  int bi = (int)(row / ((long)Tq * NH));
  if (bi >= B) return;

  int tid = threadIdx.x, bs = blockDim.x;
  long base_idx = (((long)bi * NH + h) * Tq + ti) * split_k;

  __shared__ float m_max_sh, l_global_sh, corr_sh[FD2_SPLIT_K_MAX];

  if (tid == 0) {
    float m_max = FD2_NEG_INF;
    for (int i = 0; i < split_k; i++) {
      float mi = partial_m[base_idx + i];
      if (mi > m_max) m_max = mi;
    }
    m_max_sh = m_max;
    float l_sum = 0.0f;
    bool m_finite = fd2_is_finite(m_max);
    for (int i = 0; i < split_k; i++) {
      float mi = partial_m[base_idx + i];
      float li = partial_l[base_idx + i];
      float c = (!m_finite || !fd2_is_finite(mi)) ? 0.0f : __expf(mi - m_max);
      corr_sh[i] = c;
      l_sum += li * c;
    }
    l_global_sh = l_sum;
  }
  __syncthreads();

  float inv = (l_global_sh > 0.0f) ? (1.0f / l_global_sh) : 0.0f;
  T* out_row = out + (((long)(bi * NH + h) * Tq + ti) * D);
  for (int d = tid; d < D; d += bs) {
    float acc = 0.0f;
    for (int i = 0; i < split_k; i++) {
      acc += partial_acc[(base_idx + i) * D + d] * corr_sh[i];
    }
    fd2_st(out_row + d, acc * inv);
  }
}

extern "C" {

// Имена: fd2_<qdtype>_g<GROUP>_d<DHEAD>[_dev]. Tkv у _dev — device-скаляр
// (*Tkv_ptr), launch config от него не зависит (валидно под CUDA-graph).
#define FD2_SPLIT(NAME, T, GROUP, DHEAD)                                            \
  __global__ void NAME(                                                             \
      const T* q, const unsigned char* k, const unsigned char* v,                   \
      const unsigned char* k_scale, const unsigned char* v_scale,                   \
      float* partial_acc, float* partial_m, float* partial_l,                       \
      int B, int NH, int NKV, int Tq, int Tkv,                                      \
      float scale, int causal, int split_k, int t_stride) {                         \
    fd2_split_impl<T, GROUP, DHEAD>(q, k, v, k_scale, v_scale,                      \
        partial_acc, partial_m, partial_l,                                          \
        B, NH, NKV, Tq, Tkv, scale, causal, split_k, t_stride);                     \
  }                                                                                 \
  __global__ void NAME##_dev(                                                       \
      const T* q, const unsigned char* k, const unsigned char* v,                   \
      const unsigned char* k_scale, const unsigned char* v_scale,                   \
      float* partial_acc, float* partial_m, float* partial_l,                       \
      int B, int NH, int NKV, int Tq, const int* Tkv_ptr,                           \
      float scale, int causal, int split_k, int t_stride) {                         \
    fd2_split_impl<T, GROUP, DHEAD>(q, k, v, k_scale, v_scale,                      \
        partial_acc, partial_m, partial_l,                                          \
        B, NH, NKV, Tq, *Tkv_ptr, scale, causal, split_k, t_stride);                \
  }

FD2_SPLIT(fd2_f16_g1_d128, __half, 1, 128)
FD2_SPLIT(fd2_f16_g2_d128, __half, 2, 128)
FD2_SPLIT(fd2_f16_g4_d128, __half, 4, 128)
FD2_SPLIT(fd2_f16_g6_d128, __half, 6, 128)
FD2_SPLIT(fd2_f16_g1_d256, __half, 1, 256)
FD2_SPLIT(fd2_f16_g2_d256, __half, 2, 256)
FD2_SPLIT(fd2_f16_g4_d256, __half, 4, 256)
FD2_SPLIT(fd2_f16_g6_d256, __half, 6, 256)
FD2_SPLIT(fd2_bf16_g1_d128, __nv_bfloat16, 1, 128)
FD2_SPLIT(fd2_bf16_g2_d128, __nv_bfloat16, 2, 128)
FD2_SPLIT(fd2_bf16_g4_d128, __nv_bfloat16, 4, 128)
FD2_SPLIT(fd2_bf16_g6_d128, __nv_bfloat16, 6, 128)
FD2_SPLIT(fd2_bf16_g1_d256, __nv_bfloat16, 1, 256)
FD2_SPLIT(fd2_bf16_g2_d256, __nv_bfloat16, 2, 256)
FD2_SPLIT(fd2_bf16_g4_d256, __nv_bfloat16, 4, 256)
FD2_SPLIT(fd2_bf16_g6_d256, __nv_bfloat16, 6, 256)

__global__ void fd2_merge_f16(
    const float* partial_acc, const float* partial_m, const float* partial_l,
    __half* out, int B, int NH, int Tq, int D, int split_k) {
  fd2_merge_impl<__half>(partial_acc, partial_m, partial_l, out, B, NH, Tq, D, split_k);
}
__global__ void fd2_merge_bf16(
    const float* partial_acc, const float* partial_m, const float* partial_l,
    __nv_bfloat16* out, int B, int NH, int Tq, int D, int split_k) {
  fd2_merge_impl<__nv_bfloat16>(partial_acc, partial_m, partial_l, out, B, NH, Tq, D, split_k);
}

}  // extern "C"
