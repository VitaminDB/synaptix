// Top-k по широким строкам: выбор блоков индексатором QSA (512 из тысяч).
//
// Строка в shared уже не помещается, а k слишком велико для «k раз найти
// максимум», поэтому порог ищется radix-select'ом по битам float: значения
// неотрицательны (идут после relu), и их битовое представление монотонно.
// Три прохода по 8 бит дают порог с точностью до 24 старших бит, дальше
// элементы выше порога забираются целиком, а равные добираются до k.
//
//   scores (rows, cols) f32
//   valid  (rows)       u32 — сколько первых столбцов действительны
//   out    (rows, k)    u32 — индексы, незанятые слоты помечены 0xFFFFFFFF

#define TW_MISSING 0xFFFFFFFFu
#define TW_BINS 256

// Атомик в гистограмму с агрегацией по варпу: оценки после relu почти все
// одного порядка, старший байт у них общий, и 32 лейна били бы в одну
// ячейку shared по очереди. Лейны с одинаковым бином складываются в одну
// запись. Хвост цикла: неактивные лейны передают бин TW_BINS (не пишется).
__device__ __forceinline__ void tw_hist_add(unsigned int* hist, unsigned int bin) {
    unsigned int peers = __match_any_sync(0xFFFFFFFFu, bin);
    unsigned int lane = threadIdx.x & 31u;
    if (bin < TW_BINS && lane == (unsigned int)(__ffs(peers) - 1)) atomicAdd(&hist[bin], (unsigned int)__popc(peers));
}

extern "C" __global__ void topk_wide_f32(
    const float* __restrict__ scores,
    const unsigned int* __restrict__ valid,
    unsigned int* __restrict__ out,
    unsigned int rows,
    unsigned int cols,
    unsigned int k,
    unsigned int recent
) {
    __shared__ unsigned int hist[TW_BINS];
    __shared__ unsigned int shared_prefix;
    __shared__ unsigned int shared_bin;
    __shared__ unsigned int taken;
    __shared__ unsigned int cnt_above[1024];
    __shared__ unsigned int cnt_equal[1024];
    __shared__ unsigned int cnt_skip[1024];

    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    unsigned int tid = threadIdx.x, bs = blockDim.x;
    const float* src = scores + (unsigned long long)row * cols;
    unsigned int* dst = out + (unsigned long long)row * k;
    unsigned int n = valid[row];
    if (n > cols) n = cols;

    // Меньше k кандидатов — берём всех, хвост слотов помечаем.
    if (n <= k) {
        for (unsigned int i = tid; i < k; i += bs) dst[i] = (i < n) ? i : TW_MISSING;
        return;
    }

    // Первый проход: гистограмма по старшим восьми битам.
    for (unsigned int b = tid; b < TW_BINS; b += bs) hist[b] = 0;
    __syncthreads();
    for (unsigned int base = 0; base < n; base += bs) {
        unsigned int i = base + tid;
        unsigned int bin = TW_BINS;
        if (i < n) bin = __float_as_uint(src[i]) >> 24;
        tw_hist_add(hist, bin);
    }
    __syncthreads();
    if (tid == 0) {
        unsigned int acc = 0, bin = 0;
        for (int b = TW_BINS - 1; b >= 0; --b) {
            unsigned int c = hist[b];
            if (acc + c >= k) {
                bin = (unsigned int)b;
                break;
            }
            acc += c;
        }
        shared_bin = bin;
        shared_prefix = acc;
    }
    __syncthreads();
    unsigned int hi_bin = shared_bin;
    unsigned int above = shared_prefix;

    // Второй проход: уточняем порог внутри найденного бина.
    for (unsigned int b = tid; b < TW_BINS; b += bs) hist[b] = 0;
    __syncthreads();
    for (unsigned int base = 0; base < n; base += bs) {
        unsigned int i = base + tid;
        unsigned int bin = TW_BINS;
        if (i < n) {
            unsigned int bits = __float_as_uint(src[i]);
            if ((bits >> 24) == hi_bin) bin = (bits >> 16) & 0xFFu;
        }
        tw_hist_add(hist, bin);
    }
    __syncthreads();
    if (tid == 0) {
        unsigned int acc = above, bin = 0;
        for (int b = TW_BINS - 1; b >= 0; --b) {
            unsigned int c = hist[b];
            if (acc + c >= k) {
                bin = (unsigned int)b;
                break;
            }
            acc += c;
        }
        shared_bin = (hi_bin << 8) | bin;
        shared_prefix = acc;
    }
    __syncthreads();
    unsigned int hi16 = shared_bin;
    unsigned int above16 = shared_prefix;

    // Третий проход: ещё восемь бит — порог становится точным до 2^-24
    // относительной величины, и «равными» остаются лишь по-настоящему близкие
    // значения.
    for (unsigned int b = tid; b < TW_BINS; b += bs) hist[b] = 0;
    __syncthreads();
    for (unsigned int base = 0; base < n; base += bs) {
        unsigned int i = base + tid;
        unsigned int bin = TW_BINS;
        if (i < n) {
            unsigned int bits = __float_as_uint(src[i]);
            if ((bits >> 16) == hi16) bin = (bits >> 8) & 0xFFu;
        }
        tw_hist_add(hist, bin);
    }
    __syncthreads();
    if (tid == 0) {
        unsigned int acc = above16, bin = 0;
        for (int b = TW_BINS - 1; b >= 0; --b) {
            unsigned int c = hist[b];
            if (acc + c >= k) {
                bin = (unsigned int)b;
                break;
            }
            acc += c;
        }
        shared_bin = (hi16 << 8) | bin;
        taken = 0;
    }
    __syncthreads();
    unsigned int thresh = shared_bin;   // старшие 24 бита порога

    // Отбор детерминирован: кто вошёл и в каком порядке, от запуска к запуску
    // не меняется. Прежний вариант раздавал слоты атомиком — при равных
    // оценках (после relu их много: нулевые блоки) набор выбранных блоков
    // зависел от гонки нитей, и ответ модели на длинном контексте плавал.
    // Здесь: всё, что строго выше порога, входит; равные порогу добираются
    // до k по индексу (с начала строки или, при `recent`, с конца); выход —
    // по возрастанию индекса.
    // Нить ведёт сплошной отрезок строки, слоты — по префиксным суммам.
    unsigned int seg = (n + bs - 1u) / bs;
    unsigned int lo = tid * seg;
    unsigned int hi = lo + seg < n ? lo + seg : n;
    unsigned int ca = 0, ce = 0;
    for (unsigned int i = lo; i < hi; ++i) {
        unsigned int bits = __float_as_uint(src[i]) >> 8;
        ca += bits > thresh;
        ce += bits == thresh;
    }
    cnt_above[tid] = ca;
    cnt_equal[tid] = ce;
    __syncthreads();
    if (tid == 0) {
        unsigned int total_above = 0;
        for (unsigned int t = 0; t < bs; ++t) total_above += cnt_above[t];
        unsigned int need = total_above < k ? k - total_above : 0u;   // сколько равных добрать
        // Квота равных по нитям: с начала строки (наименьшие индексы — как у
        // torch.topk) либо с конца (самые свежие блоки, `recent`).
        if (recent) {
            for (int t = (int)bs - 1; t >= 0; --t) {
                unsigned int e = cnt_equal[t];
                unsigned int take = need < e ? need : e;
                cnt_skip[t] = e - take;
                cnt_equal[t] = take;
                need -= take;
            }
        } else {
            for (unsigned int t = 0; t < bs; ++t) {
                unsigned int e = cnt_equal[t];
                unsigned int take = need < e ? need : e;
                cnt_skip[t] = 0u;
                cnt_equal[t] = take;
                need -= take;
            }
        }
        unsigned int slot = 0;
        for (unsigned int t = 0; t < bs; ++t) {
            unsigned int a = cnt_above[t];
            cnt_above[t] = slot;                          // первый слот нити
            slot += a + cnt_equal[t];
        }
        taken = slot < k ? slot : k;
    }
    __syncthreads();
    {
        unsigned int slot = cnt_above[tid];
        unsigned int skip = cnt_skip[tid];
        unsigned int quota = cnt_equal[tid];
        for (unsigned int i = lo; i < hi; ++i) {
            unsigned int bits = __float_as_uint(src[i]) >> 8;
            bool in = bits > thresh;
            if (bits == thresh) {
                if (skip > 0u) --skip;
                else if (quota > 0u) { in = true; --quota; }
            }
            if (in) {
                if (slot < k) dst[slot] = i;
                ++slot;
            }
        }
    }
    __syncthreads();

    if (taken < k) {
        for (unsigned int i = tid + taken; i < k; i += bs) dst[i] = TW_MISSING;
    }
}
