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

struct ZcArgs {
    unsigned long long idx, tab_gu, tab_dn, host_gu, host_dn, owner, used, clock, stats;
    unsigned long long eff_gu, eff_dn, cp_src_gu, cp_src_dn, cp_dst;
    unsigned long long slot_base, slot_bytes, gu_bytes, dn_bytes;
    unsigned int pairs, experts, slots, _pad;
};

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
        if (e >= a.experts) { eg[p] = eg[0]; ed[p] = ed[0]; continue; }
        unsigned long long g = tg[e];
        if ((g & 1ull) == 0ull) {
            eg[p] = g;
            ed[p] = td[e];
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
        td[e] = dst + a.gu_bytes;
        eg[p] = dst;
        ed[p] = dst + a.gu_bytes;
        csg[p] = hg[e];
        csd[p] = hd[e];
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

extern "C" __global__ void zc_copy(ZcArgs a) {
    const unsigned int p = blockIdx.x;
    if (p >= a.pairs) return;
    const unsigned long long dst = ((const unsigned long long*)a.cp_dst)[p];
    if (dst == 0ull) return;
    const uint4* sg = (const uint4*)((const unsigned long long*)a.cp_src_gu)[p];
    const uint4* sd = (const uint4*)((const unsigned long long*)a.cp_src_dn)[p];
    uint4* dg = (uint4*)dst;
    uint4* dd = (uint4*)(dst + a.gu_bytes);
    const unsigned long long ng = a.gu_bytes / 16ull, nd = a.dn_bytes / 16ull;
    const unsigned long long stride = (unsigned long long)gridDim.y * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.y * blockDim.x + threadIdx.x; i < ng; i += stride) dg[i] = sg[i];
    for (unsigned long long i = (unsigned long long)blockIdx.y * blockDim.x + threadIdx.x; i < nd; i += stride) dd[i] = sd[i];
}
