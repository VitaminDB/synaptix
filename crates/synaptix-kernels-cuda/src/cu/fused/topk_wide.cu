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

extern "C" __global__ void topk_wide_f32(
    const float* __restrict__ scores,
    const unsigned int* __restrict__ valid,
    unsigned int* __restrict__ out,
    unsigned int rows,
    unsigned int cols,
    unsigned int k
) {
    __shared__ unsigned int hist[TW_BINS];
    __shared__ unsigned int shared_prefix;
    __shared__ unsigned int shared_bin;
    __shared__ unsigned int taken;

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
    for (unsigned int i = tid; i < n; i += bs) {
        unsigned int bits = __float_as_uint(src[i]);
        atomicAdd(&hist[bits >> 24], 1u);
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
    for (unsigned int i = tid; i < n; i += bs) {
        unsigned int bits = __float_as_uint(src[i]);
        if ((bits >> 24) == hi_bin) atomicAdd(&hist[(bits >> 16) & 0xFFu], 1u);
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
    for (unsigned int i = tid; i < n; i += bs) {
        unsigned int bits = __float_as_uint(src[i]);
        if ((bits >> 16) == hi16) atomicAdd(&hist[(bits >> 8) & 0xFFu], 1u);
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

    // Всё, что строго выше порога, входит наверняка: таких меньше k.
    for (unsigned int i = tid; i < n; i += bs) {
        unsigned int bits = __float_as_uint(src[i]) >> 8;
        if (bits > thresh) {
            unsigned int slot = atomicAdd(&taken, 1u);
            if (slot < k) dst[slot] = i;
        }
    }
    __syncthreads();

    // Равные порогу добираются до k — их порядок не определён, как и у
    // отбора на процессоре при равных значениях.
    for (unsigned int i = tid; i < n; i += bs) {
        if (taken >= k) break;
        unsigned int bits = __float_as_uint(src[i]) >> 8;
        if (bits == thresh) {
            unsigned int slot = atomicAdd(&taken, 1u);
            if (slot < k) dst[slot] = i;
        }
    }
    __syncthreads();

    if (taken < k) {
        for (unsigned int i = tid + taken; i < k; i += bs) dst[i] = TW_MISSING;
    }
}
