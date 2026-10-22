#include <cuda_fp16.h>
#include <cuda_bf16.h>
// Chunk-FLA helper-ядра для Gated DeltaNet prefill (chunked linear attention).
//
// Портировано из ai-quant/src/kernels/cu_kernels/chunk_fla.cu (валидировано
// bit-exact в проде Qwen3.6). Алгоритм соответствует `torch_chunk_gated_delta_rule`
// из transformers (Qwen3.5/3.6). Все ядра — F32 internal, row-major.
//
// Чанковая форма: последовательность делится на чанки CS (=64). Внутри чанка —
// параллельная intra-attn (closed-form cumulative product), между чанками —
// рекуррентность по state. Эти ядра покрывают chunk-aware element-wise части;
// батчевые GEMM делает оркестратор (см. src/scan/chunk_scan.rs).
//
// Состав:
//   compute_chunk_attn_f32               — intra-chunk attn + decay_mask
//   state_update_decay_f32               — state *= exp(g_last) (по g_last)
//   mul_decay_mask_f32                   — attn_intra *= decay_mask (BH,CS,CS)
//   sub_inplace_f32 / add_inplace_f32    — a ∓= b
//   scale_by_exp_diff_f32                — q_scaled / k_cumdecay / k_decayed
//   sub_chunk_f32                        — value_proc[:,ci] -= v_prime
//   mul_decay_mask_chunk_f32             — attn_intra *= decay_mask[:,ci]
//   scale_k_decayed_chunk_f32            — k_decayed = k[:,ci]*exp(g_last-g_cumsum)
//   state_decay_from_gcumsum_chunk_f32   — state *= exp(g_cumsum[:,ci,CS-1])

extern "C" {

// ─────────────────────────────────────────────────────────────────────────────
// compute_chunk_attn_f32 — intra-chunk attn + decay_mask.
//
//   decay_mask[i,j] = exp(g[i] - g[j])  if j ≤ i else 0
//   attn[i,j]       = -(k_beta[i,:] · key[j,:]) * decay_mask[i,j]  if j < i else 0
//   attn[i,:i]     += Σ_{l<i} attn[i,l] * attn[l,:i]   (closed-form cumprod)
//   attn[i,i]      += 1
//
// Grid: (BH, NC, 1). Block: (CS). Shared: g_sm[CS] + attn_sm[CS*CS].
// ─────────────────────────────────────────────────────────────────────────────
__global__ void compute_chunk_attn_f32(
    const float* __restrict__ k_beta,    // (BH, NC, CS, HK)
    const float* __restrict__ key,       // (BH, NC, CS, HK)
    const float* __restrict__ g_cumsum,  // (BH, NC, CS)
    float* __restrict__ attn_out,        // (BH, NC, CS, CS)
    float* __restrict__ decay_mask_out,  // (BH, NC, CS, CS)
    unsigned int bh,
    unsigned int nc,
    unsigned int cs,
    unsigned int hk
) {
    unsigned int b = blockIdx.x;
    unsigned int c = blockIdx.y;
    unsigned int tid = threadIdx.x;  // 0..cs-1
    if (b >= bh || c >= nc) return;

    extern __shared__ float shm[];
    float* g_sm = shm;          // [CS]
    float* attn_sm = shm + cs;  // [CS * CS]

    unsigned long long base_kv = (unsigned long long)b * nc * cs * hk + (unsigned long long)c * cs * hk;
    unsigned long long base_g  = (unsigned long long)b * nc * cs + (unsigned long long)c * cs;
    unsigned long long base_attn = (unsigned long long)b * nc * cs * cs + (unsigned long long)c * cs * cs;

    if (tid < cs) {
        g_sm[tid] = g_cumsum[base_g + tid];
    }
    __syncthreads();

    // Шаг 1: decay_mask + attn инициализация. Thread tid отвечает за строку i=tid.
    unsigned int i = tid;
    float g_i = g_sm[i];
    for (unsigned int j = 0; j < cs; ++j) {
        float dm = (j <= i) ? __expf(g_i - g_sm[j]) : 0.0f;
        decay_mask_out[base_attn + i * cs + j] = dm;

        float a;
        if (j < i) {
            float acc = 0.0f;
            for (unsigned int d = 0; d < hk; ++d) {
                acc += k_beta[base_kv + i * hk + d] * key[base_kv + j * hk + d];
            }
            a = -acc * dm;
        } else {
            a = 0.0f;  // upper triangle + diag
        }
        attn_sm[i * cs + j] = a;
    }
    __syncthreads();

    // Шаг 2: closed-form cumulative product.
    // RACE-ФИКС: поток j читает attn_sm[row*cs + l] (l<row) — это строка `row`,
    // которую ПАРАЛЛЕЛЬНО пишет поток l (тоже <row) в этой же итерации. fla-эталон
    // использует СТАРЫЕ значения строки (`.clone()`). При cs=64 (2 варпа, не
    // lockstep) гонка → недетерминизм + неверная UT-инверсия. Разделяем фазы:
    // (1) все читают старое и копят new_val в регистр, (2) барьер, (3) все пишут.
    for (unsigned int row = 1; row < cs; ++row) {
        float new_val = 0.0f;
        bool active = (tid < row);
        if (active) {
            unsigned int j = tid;
            float acc = 0.0f;
            for (unsigned int l = 0; l < row; ++l) {
                acc += attn_sm[row * cs + l] * attn_sm[l * cs + j];
            }
            new_val = attn_sm[row * cs + j] + acc;
        }
        __syncthreads();
        if (active) {
            attn_sm[row * cs + tid] = new_val;
        }
        __syncthreads();
    }

    // Шаг 3: attn += I.
    if (tid < cs) {
        attn_sm[tid * cs + tid] += 1.0f;
    }
    __syncthreads();

    for (unsigned int j = 0; j < cs; ++j) {
        attn_out[base_attn + tid * cs + j] = attn_sm[tid * cs + j];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_chunk_attn_t64_f32 — то же для CS=64, HK%32==0, блок 256 потоков.
// Скалярные произведения k_beta·keyᵀ — тайлами по 32 столбца через shared
// (коалесцированно, поток считает 4×4 выхода), порядок суммирования по d —
// прежний, поэтому результат побитово совпадает с compute_chunk_attn_f32.
// Там поток читал свою строку из global без коалесцирования и перечитывал
// её для каждого j: 3.3 мс на слой куска 2048 у Qwen3.8-27B (44 % скана).
// Подстановка (шаг 2) — та же, тем же порядком.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void __launch_bounds__(256) compute_chunk_attn_t64_f32(
    const float* __restrict__ k_beta,    // (BH, NC, 64, HK)
    const float* __restrict__ key,       // (BH, NC, 64, HK)
    const float* __restrict__ g_cumsum,  // (BH, NC, 64)
    float* __restrict__ attn_out,        // (BH, NC, 64, 64)
    float* __restrict__ decay_mask_out,  // (BH, NC, 64, 64)
    unsigned int bh,
    unsigned int nc,
    unsigned int hk
) {
    const unsigned int CS = 64, TK = 32, LD = 33, ALD = 65;
    unsigned int b = blockIdx.x;
    unsigned int c = blockIdx.y;
    unsigned int tid = threadIdx.x;
    if (b >= bh || c >= nc) return;

    __shared__ float g_sm[64];
    __shared__ float kb_s[64 * 33];
    __shared__ float key_s[64 * 33];
    __shared__ float attn_sm[64 * 65];

    unsigned long long base_kv = ((unsigned long long)b * nc + c) * CS * hk;
    unsigned long long base_g  = ((unsigned long long)b * nc + c) * CS;
    unsigned long long base_attn = ((unsigned long long)b * nc + c) * CS * CS;

    if (tid < CS) g_sm[tid] = g_cumsum[base_g + tid];

    // поток: строки ty + 16·r, столбцы tx + 16·q (r, q = 0..3)
    unsigned int ty = tid >> 4, tx = tid & 15;
    float acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int q = 0; q < 4; ++q) acc[r][q] = 0.0f;

    for (unsigned int k0 = 0; k0 < hk; k0 += TK) {
        __syncthreads();
        // 64×32 каждой матрицы: 2048 чисел, по 8 на поток, строки подряд
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            unsigned int e = p * 256 + tid;
            unsigned int row = e >> 5, col = e & 31;
            kb_s[row * LD + col]  = k_beta[base_kv + (unsigned long long)row * hk + k0 + col];
            key_s[row * LD + col] = key[base_kv + (unsigned long long)row * hk + k0 + col];
        }
        __syncthreads();
        #pragma unroll 8
        for (unsigned int d = 0; d < TK; ++d) {
            float a[4], bb[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) a[r] = kb_s[(ty + 16 * r) * LD + d];
            #pragma unroll
            for (int q = 0; q < 4; ++q) bb[q] = key_s[(tx + 16 * q) * LD + d];
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[r][q] += a[r] * bb[q];
        }
    }

    // decay_mask и строго нижний attn
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        unsigned int i = ty + 16 * r;
        float g_i = g_sm[i];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            unsigned int j = tx + 16 * q;
            float dm = (j <= i) ? __expf(g_i - g_sm[j]) : 0.0f;
            decay_mask_out[base_attn + i * CS + j] = dm;
            attn_sm[i * ALD + j] = (j < i) ? -acc[r][q] * dm : 0.0f;
        }
    }
    __syncthreads();

    // Шаг 2: подстановка, как в compute_chunk_attn_f32 (старые значения строки).
    for (unsigned int row = 1; row < CS; ++row) {
        float new_val = 0.0f;
        bool active = (tid < row);
        if (active) {
            unsigned int j = tid;
            float s = 0.0f;
            for (unsigned int l = 0; l < row; ++l) {
                s += attn_sm[row * ALD + l] * attn_sm[l * ALD + j];
            }
            new_val = attn_sm[row * ALD + j] + s;
        }
        __syncthreads();
        if (active) attn_sm[row * ALD + tid] = new_val;
        __syncthreads();
    }
    if (tid < CS) attn_sm[tid * ALD + tid] += 1.0f;
    __syncthreads();

    #pragma unroll
    for (int p = 0; p < 16; ++p) {
        unsigned int e = p * 256 + tid;
        unsigned int i = e >> 6, j = e & 63;
        attn_out[base_attn + e] = attn_sm[i * ALD + j];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// state_update_decay_f32 — state[b,:,:] *= exp(g_last[b]). Grid (BH, HK, *).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void state_update_decay_f32(
    float* __restrict__ state,           // (BH, HK, HV)
    const float* __restrict__ g_last,    // (BH,)
    unsigned int bh,
    unsigned int hk,
    unsigned int hv
) {
    unsigned int b = blockIdx.x;
    unsigned int k = blockIdx.y;
    unsigned int v = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= bh || k >= hk || v >= hv) return;
    float decay = __expf(g_last[b]);
    unsigned long long idx = (unsigned long long)b * hk * hv + (unsigned long long)k * hv + v;
    state[idx] = state[idx] * decay;
}

// ─────────────────────────────────────────────────────────────────────────────
// Element-wise helpers.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void mul_decay_mask_f32(
    float* __restrict__ attn_intra,           // (BH, CS, CS) in/out
    const float* __restrict__ decay_mask_i,   // (BH, CS, CS) for chunk i
    unsigned int bh,
    unsigned int cs
) {
    unsigned int b = blockIdx.x;
    unsigned int i = blockIdx.y;
    unsigned int j = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= bh || i >= cs || j >= cs) return;
    unsigned long long idx = (unsigned long long)b * cs * cs + (unsigned long long)i * cs + j;
    attn_intra[idx] = attn_intra[idx] * decay_mask_i[idx];
}

__global__ void sub_inplace_f32(float* __restrict__ a, const float* __restrict__ b, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] = a[idx] - b[idx];
}

__global__ void add_inplace_f32(float* __restrict__ a, const float* __restrict__ b, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] = a[idx] + b[idx];
}

// ─────────────────────────────────────────────────────────────────────────────
// scale_by_exp_diff_f32 — common scaling.
//   mode=0 → factor = exp(vec_g[row])                 (q_scaled, k_cumdecay_input)
//   mode=1 → factor = exp(scalar_g[row/cs_in] - vec_g[row])  (k_decayed)
// row кладём в grid_dim.x (без лимита 65535).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void scale_by_exp_diff_f32(
    float* __restrict__ out,             // (total_rows, d)
    const float* __restrict__ in,
    const float* __restrict__ scalar_g,  // (BH * NC,) — может быть == vec_g для mode=0
    const float* __restrict__ vec_g,     // (total_rows,)
    unsigned int total_rows,
    unsigned int d,
    unsigned int cs_in,
    unsigned int mode
) {
    unsigned int row = blockIdx.x;
    unsigned int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= total_rows || col >= d) return;
    float gr = vec_g[row];
    float factor;
    if (mode == 0) {
        factor = __expf(gr);
    } else {
        float gl = scalar_g[row / cs_in];
        factor = __expf(gl - gr);
    }
    unsigned long long idx = (unsigned long long)row * d + col;
    out[idx] = in[idx] * factor;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunk-aware element-wise: работа на (BH, NC, CS, D) без per-BH host-loop.
// ─────────────────────────────────────────────────────────────────────────────

// value_proc[:, ci, :, :] -= v_prime[:].
__global__ void sub_chunk_f32(
    float* __restrict__ value_proc,     // (BH, NC, CS, HV)
    const float* __restrict__ v_prime,  // (BH, CS, HV)
    unsigned int bh,
    unsigned int nc,
    unsigned int cs,
    unsigned int hv,
    unsigned int chunk_idx
) {
    unsigned int b = blockIdx.x;
    unsigned int t = blockIdx.y;
    unsigned int d = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= bh || t >= cs || d >= hv) return;
    unsigned long long off_vp = (unsigned long long)b * nc * cs * hv
                              + (unsigned long long)chunk_idx * cs * hv
                              + (unsigned long long)t * hv + d;
    unsigned long long off_vpt = (unsigned long long)b * cs * hv
                               + (unsigned long long)t * hv + d;
    value_proc[off_vp] = value_proc[off_vp] - v_prime[off_vpt];
}

// attn_intra[bh,:,:] *= decay_mask[bh, ci, :, :].
__global__ void mul_decay_mask_chunk_f32(
    float* __restrict__ attn_intra,           // (BH, CS, CS)
    const float* __restrict__ decay_mask,     // (BH, NC, CS, CS)
    unsigned int bh,
    unsigned int nc,
    unsigned int cs,
    unsigned int chunk_idx
) {
    unsigned int b = blockIdx.x;
    unsigned int i = blockIdx.y;
    unsigned int j = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= bh || i >= cs || j >= cs) return;
    unsigned long long off_a = (unsigned long long)b * cs * cs
                             + (unsigned long long)i * cs + j;
    unsigned long long off_d = (unsigned long long)b * nc * cs * cs
                             + (unsigned long long)chunk_idx * cs * cs
                             + (unsigned long long)i * cs + j;
    attn_intra[off_a] = attn_intra[off_a] * decay_mask[off_d];
}

// k_decayed[bh,:,:] = k[bh, ci, :, :] * exp(g_last[bh] - g_cumsum[bh, ci, :]).
// k_decayed[bh, c, :, :] = k[bh, c, :, :] * exp(g_last[bh,c] - g_cumsum[bh,c,:])
// сразу по всем чанкам: от состояния этот шаг не зависит, а поштучно он стоил
// запуска на чанк при сетке в BH блоков.
__global__ void scale_k_decayed_all_f32(
    float* __restrict__ k_decayed_out,        // (BH, NC, CS, HK)
    const float* __restrict__ k,              // (BH, NC, CS, HK)
    const float* __restrict__ g_cumsum,       // (BH, NC, CS)
    unsigned int bh,
    unsigned int nc,
    unsigned int cs,
    unsigned int hk
) {
    unsigned int row = blockIdx.x;
    unsigned int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= bh * nc * cs || d >= hk) return;
    unsigned int t = row % cs;
    unsigned int rest = row / cs;
    unsigned long long off_g = (unsigned long long)rest * cs;
    float factor = __expf(g_cumsum[off_g + cs - 1] - g_cumsum[off_g + t]);
    unsigned long long off = (unsigned long long)row * hk + d;
    k_decayed_out[off] = k[off] * factor;
}

// Поэлементное a *= b по всему буферу.
__global__ void mul_inplace_f32(
    float* __restrict__ a,
    const float* __restrict__ b,
    unsigned long long n
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = a[i] * b[i];
}

__global__ void scale_k_decayed_chunk_f32(
    float* __restrict__ k_decayed_out,        // (BH, CS, HK)
    const float* __restrict__ k,              // (BH, NC, CS, HK)
    const float* __restrict__ g_cumsum,       // (BH, NC, CS)
    unsigned int bh,
    unsigned int nc,
    unsigned int cs,
    unsigned int hk,
    unsigned int chunk_idx
) {
    unsigned int b = blockIdx.x;
    unsigned int t = blockIdx.y;
    unsigned int d = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= bh || t >= cs || d >= hk) return;
    unsigned long long off_g = (unsigned long long)b * nc * cs
                             + (unsigned long long)chunk_idx * cs;
    float g_last = g_cumsum[off_g + cs - 1];
    float g_t = g_cumsum[off_g + t];
    float factor = __expf(g_last - g_t);
    unsigned long long off_k = (unsigned long long)b * nc * cs * hk
                             + (unsigned long long)chunk_idx * cs * hk
                             + (unsigned long long)t * hk + d;
    unsigned long long off_kd = (unsigned long long)b * cs * hk
                              + (unsigned long long)t * hk + d;
    k_decayed_out[off_kd] = k[off_k] * factor;
}

// state[b,:,:] *= exp(g_cumsum[b, ci, CS-1]). Полностью device-only.
__global__ void state_decay_from_gcumsum_chunk_f32(
    float* __restrict__ state,                // (BH, HK, HV)
    const float* __restrict__ g_cumsum,       // (BH, NC, CS)
    unsigned int bh,
    unsigned int nc,
    unsigned int cs,
    unsigned int hk,
    unsigned int hv,
    unsigned int chunk_idx
) {
    unsigned int b = blockIdx.x;
    unsigned int k = blockIdx.y;
    unsigned int v = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= bh || k >= hk || v >= hv) return;
    unsigned long long off_g = (unsigned long long)b * nc * cs
                             + (unsigned long long)chunk_idx * cs + cs - 1;
    float decay = __expf(g_cumsum[off_g]);
    unsigned long long idx = (unsigned long long)b * hk * hv
                           + (unsigned long long)k * hv + v;
    state[idx] = state[idx] * decay;
}

} // extern "C"

// Весь главный цикл чанк-скана одним ядром.
//
// Шесть отдельных запусков на чанк (два умножения на состояние, внутричанковое
// внимание, затухание состояния и его пополнение) шли цепочкой с сеткой в BH
// блоков, а состояние между ними уезжало в глобальную память. Здесь блок берёт
// одну голову и полосу выходных каналов, держит состояние в shared и проходит
// по всем чанкам сам: полосы независимы, потому что и выход, и пополнение
// состояния касаются только своих каналов.
//
//   q_scaled/k_cumdecay/k_decayed (BH, NC, CS, HK)
//   value_proc                    (BH, NC, CS, HV)
//   attn (внутричанковое, с маской) (BH, NC, CS, CS)
//   g_cumsum                      (BH, NC, CS)
//   state                         (BH, HK, HV)   in/out
//   out                           (BH, NC·CS, HV)
//
// Формы жёсткие: CS=64, HK=128, полоса HV=64. Каждый поток считает не один
// выход, а тайл 4×4 (у пополнения состояния 8×4): иначе на каждое умножение
// приходится своё чтение, и ядро выходит вдвое медленнее цепочки из bmm.

#define GCS_CS 64
#define GCS_HK 128
// Полоса выходных каналов на блок. 32 вместо 64: скан по чанкам у головы
// последовательный, и время ядра — это время одного блока; при 64 сетка
// (48 голов × 2) занимала карту меньше чем одной волной.
#ifndef GCS_TILE
#define GCS_TILE 32
#endif
#define GCS_BLOCK (GCS_TILE * 4)
#define GCS_TC (GCS_TILE / 4)   // групп столбцов по четыре

extern "C" __global__ __launch_bounds__(GCS_BLOCK) void gdn_chunk_scan_f32(
    const float* __restrict__ q_scaled,
    const float* __restrict__ k_cumdecay,
    const float* __restrict__ k_decayed,
    const float* __restrict__ value_proc,
    const float* __restrict__ attn,
    const float* __restrict__ g_cumsum,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int nc,
    unsigned int hv
) {
    __shared__ float st[GCS_HK][GCS_TILE];
    __shared__ float vp[GCS_CS][GCS_TILE];

    const unsigned int b = blockIdx.x;
    const unsigned int j0 = blockIdx.y * GCS_TILE;
    // Сетка потоков 16 × GCS_TC: строки по четыре, столбцы по четыре.
    const unsigned int rj = (threadIdx.x % GCS_TC) * 4u;
    const unsigned int ri = (threadIdx.x / GCS_TC) * 4u;
    // Для загрузок и пополнения состояния — раскладка «строка × 64 канала».
    const unsigned int lx = threadIdx.x & (GCS_TILE - 1);
    const unsigned int ly = threadIdx.x / GCS_TILE;

    const unsigned long long st_base = (unsigned long long)b * GCS_HK * hv + j0 + lx;
    for (unsigned int d = ly; d < GCS_HK; d += GCS_BLOCK / GCS_TILE) {
        st[d][lx] = state[st_base + (unsigned long long)d * hv];
    }
    __syncthreads();

    const unsigned long long qk_head = (unsigned long long)b * nc * GCS_CS * GCS_HK;
    const unsigned long long v_head = (unsigned long long)b * nc * GCS_CS * hv;
    const unsigned long long a_head = (unsigned long long)b * nc * GCS_CS * GCS_CS;
    const unsigned long long g_head = (unsigned long long)b * nc * GCS_CS;

    for (unsigned int ci = 0; ci < nc; ++ci) {
        const float* kcd = k_cumdecay + qk_head + (unsigned long long)ci * GCS_CS * GCS_HK;
        const float* qsc = q_scaled + qk_head + (unsigned long long)ci * GCS_CS * GCS_HK;
        const float* kdec = k_decayed + qk_head + (unsigned long long)ci * GCS_CS * GCS_HK;
        const float* vsrc = value_proc + v_head + (unsigned long long)ci * GCS_CS * hv;
        const float* am = attn + a_head + (unsigned long long)ci * GCS_CS * GCS_CS;
        const float* gc = g_cumsum + g_head + (unsigned long long)ci * GCS_CS;

        // vp = value_proc − k_cumdecay @ state.
        {
            float acc[4][4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
            for (int d = 0; d < GCS_HK; ++d) {
                float kv[4], sv[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) kv[i] = kcd[(unsigned long long)(ri + i) * GCS_HK + d];
                #pragma unroll
                for (int j = 0; j < 4; ++j) sv[j] = st[d][rj + j];
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += kv[i] * sv[j];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    vp[ri + i][rj + j] =
                        vsrc[(unsigned long long)(ri + i) * hv + j0 + rj + j] - acc[i][j];
                }
        }
        __syncthreads();

        // out = q_scaled @ state + attn @ vp.
        {
            float acc[4][4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
            for (int d = 0; d < GCS_HK; ++d) {
                float qv[4], sv[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) qv[i] = qsc[(unsigned long long)(ri + i) * GCS_HK + d];
                #pragma unroll
                for (int j = 0; j < 4; ++j) sv[j] = st[d][rj + j];
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += qv[i] * sv[j];
            }
            for (int u = 0; u < GCS_CS; ++u) {
                float av[4], vv[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) av[i] = am[(unsigned long long)(ri + i) * GCS_CS + u];
                #pragma unroll
                for (int j = 0; j < 4; ++j) vv[j] = vp[u][rj + j];
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * vv[j];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    out[v_head + (unsigned long long)(ci * GCS_CS + ri + i) * hv + j0 + rj + j] =
                        acc[i][j];
                }
        }
        __syncthreads();

        // state = state·exp(g_last) + k_decayed^T @ vp.
        {
            const float decay = __expf(gc[GCS_CS - 1]);
            const unsigned int d0 = (threadIdx.x / GCS_TC) * 8u;
            float acc[8][4];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = st[d0 + i][rj + j] * decay;
            for (int t = 0; t < GCS_CS; ++t) {
                float kv[8], vv[4];
                #pragma unroll
                for (int i = 0; i < 8; ++i) kv[i] = kdec[(unsigned long long)t * GCS_HK + d0 + i];
                #pragma unroll
                for (int j = 0; j < 4; ++j) vv[j] = vp[t][rj + j];
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += kv[i] * vv[j];
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) st[d0 + i][rj + j] = acc[i][j];
        }
        __syncthreads();
    }

    for (unsigned int d = ly; d < GCS_HK; d += GCS_BLOCK / GCS_TILE) {
        state[st_base + (unsigned long long)d * hv] = st[d][lx];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// gdn_prep_fused_* — вся подготовка чанк-скана одним проходом (CS=64, HK=HV=128).
// Блок — голова × чанк. Заменяет цепочку prep_scatter → l2norm(q), l2norm(k)
// → k·β, v·β → cumsum(g) → q·e^g, kβ·e^g → k·e^(g_last−g): семь проходов по
// буферам в f32 (1.5 мс на слой куска 2048 у Qwen3.8-27B). Арифметика и
// порядок сумм — те же (норма строки и cumsum — последовательно одним
// потоком), выходы побитово совпадают с прежней цепочкой.
//
//   post_conv (T_in, conv_dim) — [Q | K | V] после свёртки и SiLU, T = f16/bf16/f32
//   a, b (T_in, num_v) f16; dt_bias, a_log (num_v) f32
//   выходы (num_v, T_pad, ·) f32: q_n, q_sc, k_n, k_beta, kcdi, k_dec, v_beta; g_cum (num_v, T_pad)
//   строки t ≥ T_in — нули (как у prep_scatter с alloc_zeros), g = β = 0.
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float gp_ld(const float* p) { return *p; }
__device__ __forceinline__ float gp_ld(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float gp_ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

template <typename T>
__device__ __forceinline__ void gdn_prep_fused_impl(
    const T* __restrict__ post_conv,
    const __half* __restrict__ b_f16, const __half* __restrict__ a_f16,
    const float* __restrict__ dt_bias, const float* __restrict__ a_log,
    float* __restrict__ q_n, float* __restrict__ q_sc,
    float* __restrict__ k_n, float* __restrict__ k_beta,
    float* __restrict__ kcdi, float* __restrict__ k_dec,
    float* __restrict__ v_beta, float* __restrict__ g_cum,
    unsigned int T_in, unsigned int T_pad, unsigned int num_v, unsigned int n_rep,
    unsigned int key_dim, float q_scale, float eps) {
  const unsigned int CS = 64, HD = 128, LD = 129;
  __shared__ float rows[64 * 129];
  __shared__ float g_sm[64], gc_sm[64], beta_sm[64], inv_sm[64];

  const unsigned int hi = blockIdx.x;
  const unsigned int c = blockIdx.y;
  const unsigned int tid = threadIdx.x;   // 0..127
  const unsigned int h_in = hi / n_rep;
  const unsigned int conv_dim = 2u * key_dim + num_v * HD;
  const unsigned int t0 = c * CS;

  // g, β (как prep_scatter), затем cumsum одним потоком (как cumsum_lastdim).
  if (tid < CS) {
    unsigned int t = t0 + tid;
    float g = 0.0f, be = 0.0f;
    if (t < T_in) {
      float bv = __half2float(b_f16[t * num_v + hi]);
      be = 1.0f / (1.0f + __expf(-bv));
      float av = __half2float(a_f16[t * num_v + hi]);
      float dt = av + dt_bias[hi];
      float softplus_dt = __logf(1.0f + __expf(dt));
      float a_log_neg_exp = -__expf(a_log[hi]);
      g = softplus_dt * a_log_neg_exp;
    }
    g_sm[tid] = g;
    beta_sm[tid] = be;
  }
  __syncthreads();
  if (tid == 0) {
    float acc = 0.0f;
    for (unsigned int i = 0; i < CS; ++i) {
      acc += g_sm[i];
      gc_sm[i] = acc;
    }
  }
  __syncthreads();
  const unsigned long long row0 = (unsigned long long)hi * T_pad + t0;
  if (tid < CS) g_cum[row0 + tid] = gc_sm[tid];
  const float g_last = gc_sm[CS - 1];

  // ── Q: строки чанка в shared, норма строки — одним потоком по порядку d.
  for (unsigned int r = 0; r < CS; ++r) {
    unsigned int t = t0 + r;
    rows[r * LD + tid] = (t < T_in) ? gp_ld(post_conv + (unsigned long long)t * conv_dim + h_in * HD + tid) : 0.0f;
  }
  __syncthreads();
  if (tid < CS) {
    float ss = 0.0f;
    for (unsigned int d = 0; d < HD; ++d) {
      float v = rows[tid * LD + d];
      ss += v * v;
    }
    inv_sm[tid] = rsqrtf(ss + eps) * q_scale;
  }
  __syncthreads();
  for (unsigned int r = 0; r < CS; ++r) {
    unsigned long long o = (row0 + r) * HD + tid;
    float qn = rows[r * LD + tid] * inv_sm[r];
    q_n[o] = qn;
    q_sc[o] = qn * __expf(gc_sm[r]);
  }
  __syncthreads();

  // ── K: k_n, k·β, kβ·e^g, k·e^(g_last−g).
  for (unsigned int r = 0; r < CS; ++r) {
    unsigned int t = t0 + r;
    rows[r * LD + tid] = (t < T_in) ? gp_ld(post_conv + (unsigned long long)t * conv_dim + key_dim + h_in * HD + tid) : 0.0f;
  }
  __syncthreads();
  if (tid < CS) {
    float ss = 0.0f;
    for (unsigned int d = 0; d < HD; ++d) {
      float v = rows[tid * LD + d];
      ss += v * v;
    }
    inv_sm[tid] = rsqrtf(ss + eps) * 1.0f;
  }
  __syncthreads();
  for (unsigned int r = 0; r < CS; ++r) {
    unsigned long long o = (row0 + r) * HD + tid;
    float kn = rows[r * LD + tid] * inv_sm[r];
    float kb = kn * beta_sm[r];
    k_n[o] = kn;
    k_beta[o] = kb;
    kcdi[o] = kb * __expf(gc_sm[r]);
    k_dec[o] = kn * __expf(g_last - gc_sm[r]);
  }

  // ── V: v·β прямо из post_conv.
  for (unsigned int r = 0; r < CS; ++r) {
    unsigned int t = t0 + r;
    float v = (t < T_in) ? gp_ld(post_conv + (unsigned long long)t * conv_dim + 2u * key_dim + hi * HD + tid) : 0.0f;
    v_beta[(row0 + r) * HD + tid] = v * beta_sm[r];
  }
}

#define GDN_PREP_FUSED(NAME, T) \
extern "C" __global__ void __launch_bounds__(128) NAME( \
    const T* post_conv, const __half* b_f16, const __half* a_f16, \
    const float* dt_bias, const float* a_log, \
    float* q_n, float* q_sc, float* k_n, float* k_beta, float* kcdi, float* k_dec, \
    float* v_beta, float* g_cum, \
    unsigned int T_in, unsigned int T_pad, unsigned int num_v, unsigned int n_rep, \
    unsigned int key_dim, float q_scale, float eps) { \
  gdn_prep_fused_impl<T>(post_conv, b_f16, a_f16, dt_bias, a_log, q_n, q_sc, k_n, k_beta, \
      kcdi, k_dec, v_beta, g_cum, T_in, T_pad, num_v, n_rep, key_dim, q_scale, eps); \
}
GDN_PREP_FUSED(gdn_prep_fused_f16, __half)
GDN_PREP_FUSED(gdn_prep_fused_bf16, __nv_bfloat16)
GDN_PREP_FUSED(gdn_prep_fused_f32, float)
