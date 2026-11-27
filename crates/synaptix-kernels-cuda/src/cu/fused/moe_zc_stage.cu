// Кэш экспертов MoE целиком на карте (zero-copy декод), политика «часов».
//
// Состояние слоя на карте:
//   tab_gu/tab_dn [E] u64 — адрес слота VRAM или адрес блоба в закреплённой
//                           RAM с меткой `| 1`;
//   host_gu/host_dn [E] u64 — адреса блобов в RAM (без метки);
//   owner [S] i32 — эксперт в слоте (-1 — пусто), used [S] u8 — бит обращения,
//   clock [1] u32 — стрелка, stats [2] u32 — (попадания, выборы).
//
// zc_lru (один поток): отмечает слоты, выбранные шагом; каждому промаху
// отдаёт слот по часовой стрелке (слот с битом обращения получает второй
// шанс), переписывает таблицы (вытесненный — обратно на RAM) и пишет, что
// копировать. zc_copy: широкими загрузками (~46 ГБ/с) RAM → слот. На выходе
// eff_* [pairs] — адреса по парам для индексных GEMV. Хост не участвует.
//
// Двухблобные форматы (NVFP4, MXFP8 — packed и scales раздельно): слот —
// [gu_p | gu_s | dn_p | dn_s], gu_bytes/dn_bytes — размеры packed,
// gu_s_bytes/dn_s_bytes — масштабов. tab_* хранят адрес packed (слот или
// RAM с меткой), масштабы резидента — сразу за packed в слоте, масштабы в
// RAM — host_*_s. Выходы eff_*_s и источники копии cp_src_*_s пишутся при
// eff_gu_s != 0 / cp_src_gu_s != 0. У одноблобных все *_s нулевые — путь
// прежний.

struct ZcArgs {
    unsigned long long idx, tab_gu, tab_dn, host_gu, host_dn, owner, used, clock, stats;
    unsigned long long eff_gu, eff_dn, cp_src_gu, cp_src_dn, cp_dst;
    unsigned long long slot_base, slot_bytes, gu_bytes, dn_bytes;
    unsigned int pairs, experts, slots, _pad;
    // Маршрутизация на карте (router_w != 0): logits = W[E,H]·x[t,H] (W — f32,
    // x — f16), top-k → idx[t·k], softmax по top-k → wts[t·k].
    unsigned long long router_w, x, logits, wts;
    unsigned int rows, k, H, _pad2;
    // Двухблобные форматы (см. шапку). Только в конец: раскладка = ZcLaunch.
    unsigned long long host_gu_s, host_dn_s, eff_gu_s, eff_dn_s, cp_src_gu_s, cp_src_dn_s;
    unsigned long long gu_s_bytes, dn_s_bytes;
};

// Смещение down в слоте: за packed и масштабами gate_up.
__device__ __forceinline__ unsigned long long zc_dn_off(const ZcArgs& a) { return a.gu_bytes + a.gu_s_bytes; }

// Адреса масштабов пары `p` эксперта `e`: `slot_gu` — адрес gate_up в слоте
// (0 — эксперт читается из RAM).
__device__ __forceinline__ void zc_eff_s(const ZcArgs& a, unsigned int p, unsigned int e, unsigned long long slot_gu) {
    if (a.eff_gu_s == 0ull) return;
    unsigned long long* egs = (unsigned long long*)a.eff_gu_s;
    unsigned long long* eds = (unsigned long long*)a.eff_dn_s;
    if (slot_gu != 0ull) {
        egs[p] = slot_gu + a.gu_bytes;
        eds[p] = slot_gu + zc_dn_off(a) + a.dn_bytes;
    } else {
        egs[p] = ((const unsigned long long*)a.host_gu_s)[e];
        eds[p] = ((const unsigned long long*)a.host_dn_s)[e];
    }
}

// Источники копии масштабов промаха (RAM).
__device__ __forceinline__ void zc_cp_s(const ZcArgs& a, unsigned int p, unsigned int e) {
    if (a.cp_src_gu_s == 0ull) return;
    ((unsigned long long*)a.cp_src_gu_s)[p] = ((const unsigned long long*)a.host_gu_s)[e];
    ((unsigned long long*)a.cp_src_dn_s)[p] = ((const unsigned long long*)a.host_dn_s)[e];
}

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Строка роутера на варп; порядок fmaf — как у mma_gemv_f32 (вектор из 4 на
// лейн, шаг 32 вектора): выбор экспертов совпадает с прежним путём.
// grid = (ceil(E/8), rows), block = 256.
extern "C" __global__ void zc_router(ZcArgs a) {
    const unsigned e = blockIdx.x * 8u + (threadIdx.x >> 5);
    const unsigned t = blockIdx.y;
    if (e >= a.experts) return;
    const unsigned lane = threadIdx.x & 31u;
    const float4* w4 = reinterpret_cast<const float4*>((const float*)a.router_w + (size_t)e * a.H);
    const __half2* x2 = reinterpret_cast<const __half2*>((const __half*)a.x + (size_t)t * a.H);
    float acc = 0.f;
    const unsigned k4 = a.H >> 2;
    for (unsigned i = lane; i < k4; i += 32u) {
        const float4 wv = w4[i];
        const float2 xa = __half22float2(x2[2 * i]);
        const float2 xb = __half22float2(x2[2 * i + 1]);
        acc = fmaf(wv.x, xa.x, acc);
        acc = fmaf(wv.y, xa.y, acc);
        acc = fmaf(wv.z, xb.x, acc);
        acc = fmaf(wv.w, xb.y, acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) ((float*)a.logits)[(size_t)t * a.experts + e] = acc;
}

extern "C" __global__ void zc_router_bf16(ZcArgs a) {
    const unsigned e = blockIdx.x * 8u + (threadIdx.x >> 5);
    const unsigned t = blockIdx.y;
    if (e >= a.experts) return;
    const unsigned lane = threadIdx.x & 31u;
    const float4* w4 = reinterpret_cast<const float4*>((const float*)a.router_w + (size_t)e * a.H);
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>((const __nv_bfloat16*)a.x + (size_t)t * a.H);
    float acc = 0.f;
    const unsigned k4 = a.H >> 2;
    for (unsigned i = lane; i < k4; i += 32u) {
        const float4 wv = w4[i];
        const float2 xa = __bfloat1622float2(x2[2 * i]);
        const float2 xb = __bfloat1622float2(x2[2 * i + 1]);
        acc = fmaf(wv.x, xa.x, acc);
        acc = fmaf(wv.y, xa.y, acc);
        acc = fmaf(wv.z, xb.x, acc);
        acc = fmaf(wv.w, xb.y, acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) ((float*)a.logits)[(size_t)t * a.experts + e] = acc;
}

// top-k строки логитов и softmax по выбранным. grid = rows, block = 256,
// E ≤ 1024. Ничья — меньший индекс.
extern "C" __global__ void zc_topk(ZcArgs a) {
    __shared__ float vals[1024];
    __shared__ float red_v[8];
    __shared__ unsigned red_i[8];
    __shared__ float top_v[64];
    const unsigned t = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    const unsigned lane = tid & 31u, wid = tid >> 5, nw = bs >> 5;
    const float NEG = __int_as_float(0xFF800000);
    const float* src = (const float*)a.logits + (size_t)t * a.experts;
    for (unsigned i = tid; i < a.experts; i += bs) vals[i] = src[i];
    __syncthreads();
    for (unsigned slot = 0; slot < a.k; ++slot) {
        float best = NEG;
        unsigned bi = 0xffffffffu;
        for (unsigned i = tid; i < a.experts; i += bs) {
            const float v = vals[i];
            if (v > best || bi == 0xffffffffu) { best = v; bi = i; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const unsigned oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) { red_v[wid] = best; red_i[wid] = bi; }
        __syncthreads();
        if (wid == 0) {
            best = lane < nw ? red_v[lane] : NEG;
            bi = lane < nw ? red_i[lane] : 0xffffffffu;
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, best, o);
                const unsigned oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
            }
            if (lane == 0) {
                ((unsigned*)a.idx)[t * a.k + slot] = bi;
                top_v[slot] = best;
                vals[bi] = NEG;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        // Значения идут по убыванию: максимум — первый.
        const float m = top_v[0];
        float sum = 0.f;
        for (unsigned j = 0; j < a.k; ++j) sum += __expf(top_v[j] - m);
        for (unsigned j = 0; j < a.k; ++j) ((float*)a.wts)[t * a.k + j] = __expf(top_v[j] - m) / sum;
    }
}

extern "C" __global__ void zc_lru(ZcArgs a) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const unsigned int* idx = (const unsigned int*)a.idx;
    unsigned long long* tg = (unsigned long long*)a.tab_gu;
    unsigned long long* td = (unsigned long long*)a.tab_dn;
    const unsigned long long* hg = (const unsigned long long*)a.host_gu;
    const unsigned long long* hd = (const unsigned long long*)a.host_dn;
    int* owner = (int*)a.owner;
    unsigned char* used = (unsigned char*)a.used;
    unsigned int* clock = (unsigned int*)a.clock;
    unsigned int* stats = (unsigned int*)a.stats;
    unsigned long long* eg = (unsigned long long*)a.eff_gu;
    unsigned long long* ed = (unsigned long long*)a.eff_dn;
    unsigned long long* csg = (unsigned long long*)a.cp_src_gu;
    unsigned long long* csd = (unsigned long long*)a.cp_src_dn;
    unsigned long long* cdst = (unsigned long long*)a.cp_dst;
    if ((a._pad & 1u) != 0u) {
        // Упреждающая подкачка (второй стрим, до шага слоя): предсказанные
        // эксперты — резидентам бит обращения, не более `_pad2` промахов
        // поднимаются в слоты по часам. Таблицы eff и статистика не нужны.
        for (unsigned int p = 0; p < a.pairs; ++p) {
            cdst[p] = 0ull;
            unsigned int e = idx[p];
            if (e >= a.experts) continue;
            unsigned long long g = tg[e];
            if ((g & 1ull) == 0ull) {
                unsigned int s = (unsigned int)((g - a.slot_base) / a.slot_bytes);
                if (used[s] == 0) used[s] = 1;
            }
        }
        unsigned int budget = a._pad2;
        for (unsigned int p = 0; p < a.pairs && budget > 0u; ++p) {
            unsigned int e = idx[p];
            if (e >= a.experts) continue;
            if ((tg[e] & 1ull) == 0ull) continue;       // уже резидент (или поднят выше)
            unsigned int s = 0;
            bool found = false;
            for (unsigned int it = 0; it < 2u * a.slots; ++it) {
                s = clock[0] % a.slots;
                clock[0] = clock[0] + 1u;
                if (owner[s] >= 0 && used[s] >= 1) { used[s] = 0; continue; }
                found = true;
                break;
            }
            if (!found) break;
            int old = owner[s];
            if (old >= 0) {
                tg[old] = hg[old] | 1ull;
                td[old] = hd[old] | 1ull;
            }
            owner[s] = (int)e;
            used[s] = 1;
            unsigned long long dst = a.slot_base + (unsigned long long)s * a.slot_bytes;
            tg[e] = dst;
            td[e] = dst + zc_dn_off(a);
            csg[p] = hg[e];
            csd[p] = hd[e];
            zc_cp_s(a, p, e);
            cdst[p] = dst;
            --budget;
        }
        stats[2] += a.pairs;
        return;
    }
    // 1) слоты экспертов шага — неприкосновенны в этом шаге.
    for (unsigned int p = 0; p < a.pairs; ++p) {
        unsigned int e = idx[p];
        if (e >= a.experts) continue;
        unsigned long long g = tg[e];
        if ((g & 1ull) == 0ull) used[(g - a.slot_base) / a.slot_bytes] = 2;
    }
    unsigned int hits = 0;
    for (unsigned int p = 0; p < a.pairs; ++p) {
        cdst[p] = 0ull;
        unsigned int e = idx[p];
        if (e >= a.experts) {
            eg[p] = eg[0]; ed[p] = ed[0];
            if (a.eff_gu_s != 0ull) {
                ((unsigned long long*)a.eff_gu_s)[p] = ((unsigned long long*)a.eff_gu_s)[0];
                ((unsigned long long*)a.eff_dn_s)[p] = ((unsigned long long*)a.eff_dn_s)[0];
            }
            continue;
        }
        unsigned long long g = tg[e];
        if ((g & 1ull) == 0ull) {
            eg[p] = g;
            ed[p] = td[e];
            zc_eff_s(a, p, e, g);
            // повтор эксперта в шаге (у пары токенов) — не попадание сверх первого
            bool dup = false;
            for (unsigned int q = 0; q < p; ++q) dup |= (idx[q] == e);
            if (!dup) ++hits;
            continue;
        }
        // 2) промах: слот по часовой стрелке.
        unsigned int s = 0;
        bool found = false;
        for (unsigned int it = 0; it < 3u * a.slots; ++it) {
            s = clock[0] % a.slots;
            clock[0] = clock[0] + 1u;
            if (used[s] == 2) continue;         // занят шагом
            if (owner[s] >= 0 && used[s] == 1) { used[s] = 0; continue; }  // второй шанс
            found = true;
            break;
        }
        if (!found) {
            // все слоты заняты шагом (слотов меньше пар) — читаем из RAM как есть
            eg[p] = g & ~1ull;
            ed[p] = td[e] & ~1ull;
            zc_eff_s(a, p, e, 0ull);
            continue;
        }
        int old = owner[s];
        if (old >= 0) {
            tg[old] = hg[old] | 1ull;
            td[old] = hd[old] | 1ull;
        }
        owner[s] = (int)e;
        used[s] = 2;
        unsigned long long dst = a.slot_base + (unsigned long long)s * a.slot_bytes;
        tg[e] = dst;
        td[e] = dst + zc_dn_off(a);
        eg[p] = dst;
        ed[p] = dst + zc_dn_off(a);
        zc_eff_s(a, p, e, dst);
        csg[p] = hg[e];
        csd[p] = hd[e];
        zc_cp_s(a, p, e);
        cdst[p] = dst;
    }
    // 3) отметки шага → обычный бит обращения.
    for (unsigned int p = 0; p < a.pairs; ++p) {
        unsigned int e = idx[p];
        if (e >= a.experts) continue;
        unsigned long long g = tg[e];
        if ((g & 1ull) == 0ull) used[(g - a.slot_base) / a.slot_bytes] = 1;
    }
    stats[0] += hits;
    stats[1] += a.pairs;
}

// zc_lru для шага декода (pairs ≤ 32) варпом: проверка резидентности, отметки
// и адреса попаданий — по лейну на пару (одна глубина обращений к памяти
// вместо цикла), последовательно идёт только выбор слотов под промахи.
// Семантика — как у zc_lru. block = 32.
extern "C" __global__ void zc_lru_w(ZcArgs a) {
    __shared__ unsigned int e_sm[32];
    const unsigned int lane = threadIdx.x;
    if (blockIdx.x != 0 || lane >= 32u) return;
    const unsigned int* idx = (const unsigned int*)a.idx;
    unsigned long long* tg = (unsigned long long*)a.tab_gu;
    unsigned long long* td = (unsigned long long*)a.tab_dn;
    const unsigned long long* hg = (const unsigned long long*)a.host_gu;
    const unsigned long long* hd = (const unsigned long long*)a.host_dn;
    int* owner = (int*)a.owner;
    unsigned char* used = (unsigned char*)a.used;
    unsigned int* clock = (unsigned int*)a.clock;
    unsigned int* stats = (unsigned int*)a.stats;
    unsigned long long* eg = (unsigned long long*)a.eff_gu;
    unsigned long long* ed = (unsigned long long*)a.eff_dn;
    unsigned long long* csg = (unsigned long long*)a.cp_src_gu;
    unsigned long long* csd = (unsigned long long*)a.cp_src_dn;
    unsigned long long* cdst = (unsigned long long*)a.cp_dst;

    const bool active = lane < a.pairs;
    const unsigned int e = active ? idx[lane] : 0xffffffffu;
    const bool valid = active && e < a.experts;
    const unsigned long long g = valid ? tg[e] : 1ull;
    const bool res = valid && (g & 1ull) == 0ull;
    if (res) {
        used[(g - a.slot_base) / a.slot_bytes] = 2;
        eg[lane] = g;
        ed[lane] = td[e];
        zc_eff_s(a, lane, e, g);
    } else if (active && !valid) {
        eg[lane] = hg[0];
        ed[lane] = hd[0];
        zc_eff_s(a, lane, 0u, 0ull);
    }
    if (active) cdst[lane] = 0ull;
    const unsigned int same = __match_any_sync(0xffffffffu, e);
    const bool first = (unsigned int)(__ffs(same) - 1) == lane;
    const unsigned int hitmask = __ballot_sync(0xffffffffu, res && first);
    unsigned int mm = __ballot_sync(0xffffffffu, valid && !res);
    e_sm[lane] = e;
    __syncwarp();
    if (lane == 0u) {
        while (mm) {
            const unsigned int p = (unsigned int)(__ffs(mm) - 1);
            mm &= mm - 1u;
            const unsigned int em = e_sm[p];
            const unsigned long long gm = tg[em];
            if ((gm & 1ull) == 0ull) {          // тот же эксперт уже поднят парой раньше
                eg[p] = gm;
                ed[p] = td[em];
                zc_eff_s(a, p, em, gm);
                continue;
            }
            unsigned int s = 0;
            bool found = false;
            for (unsigned int it = 0; it < 3u * a.slots; ++it) {
                s = clock[0] % a.slots;
                clock[0] = clock[0] + 1u;
                if (used[s] == 2) continue;
                if (owner[s] >= 0 && used[s] == 1) { used[s] = 0; continue; }
                found = true;
                break;
            }
            if (!found) {
                eg[p] = gm & ~1ull;
                ed[p] = td[em] & ~1ull;
                zc_eff_s(a, p, em, 0ull);
                continue;
            }
            const int old = owner[s];
            if (old >= 0) {
                tg[old] = hg[old] | 1ull;
                td[old] = hd[old] | 1ull;
            }
            owner[s] = (int)em;
            used[s] = 2;
            const unsigned long long dst = a.slot_base + (unsigned long long)s * a.slot_bytes;
            tg[em] = dst;
            td[em] = dst + zc_dn_off(a);
            eg[p] = dst;
            ed[p] = dst + zc_dn_off(a);
            zc_eff_s(a, p, em, dst);
            csg[p] = hg[em];
            csd[p] = hd[em];
            zc_cp_s(a, p, em);
            cdst[p] = dst;
        }
        stats[0] += (unsigned int)__popc(hitmask);
        stats[1] += a.pairs;
    }
    __syncwarp();
    if (valid) {
        const unsigned long long g2 = tg[e];
        if ((g2 & 1ull) == 0ull) used[(g2 - a.slot_base) / a.slot_bytes] = 1;
    }
}

extern "C" __global__ void zc_copy(ZcArgs a) {
    const unsigned int p = blockIdx.x;
    if (p >= a.pairs) return;
    const unsigned long long dst = ((const unsigned long long*)a.cp_dst)[p];
    if (dst == 0ull) return;
    const uint4* sg = (const uint4*)((const unsigned long long*)a.cp_src_gu)[p];
    const uint4* sd = (const uint4*)((const unsigned long long*)a.cp_src_dn)[p];
    uint4* dg = (uint4*)dst;
    uint4* dd = (uint4*)(dst + zc_dn_off(a));
    const unsigned long long ng = a.gu_bytes / 16ull, nd = a.dn_bytes / 16ull;
    const unsigned long long stride = (unsigned long long)gridDim.y * blockDim.x;
    const unsigned long long i0 = (unsigned long long)blockIdx.y * blockDim.x + threadIdx.x;
    for (unsigned long long i = i0; i < ng; i += stride) dg[i] = sg[i];
    for (unsigned long long i = i0; i < nd; i += stride) dd[i] = sd[i];
    if (a.cp_src_gu_s != 0ull) {
        // Масштабы: сразу за packed своей матрицы в слоте.
        const uint4* sgs = (const uint4*)((const unsigned long long*)a.cp_src_gu_s)[p];
        const uint4* sds = (const uint4*)((const unsigned long long*)a.cp_src_dn_s)[p];
        uint4* dgs = (uint4*)(dst + a.gu_bytes);
        uint4* dds = (uint4*)(dst + zc_dn_off(a) + a.dn_bytes);
        const unsigned long long ngs = a.gu_s_bytes / 16ull, nds = a.dn_s_bytes / 16ull;
        for (unsigned long long i = i0; i < ngs; i += stride) dgs[i] = sgs[i];
        for (unsigned long long i = i0; i < nds; i += stride) dds[i] = sds[i];
    }
}
