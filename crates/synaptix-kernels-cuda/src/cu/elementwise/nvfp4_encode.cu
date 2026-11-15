// Кодировщик NVFP4 с выбором масштабов и поколоночной компенсацией ошибки
// (GPTQ внутри блока). Повторяет CPU-эталон `transcode::nvfp4_encode_rows`
// бит в бит: модуль собирается с `--fmad=false`, порядок операций и сумм
// тот же, коды E4M3 берутся по таблице порогов, посчитанной на хосте
// CPU-кодировщиком (`thr[c]` — наименьшее x > 0, у которого код ≥ c).
//
// Блок — одна строка, 128 потоков, участок 128 столбцов за раз.
//   u == nullptr: группы по 16 независимы — масштабы выбираются параллельно
//                 (поток на группу), значения кодируются параллельно;
//   u != nullptr: width == 128, группы и столбцы по очереди: ошибка столбца
//                 i переносится на столбцы j > i (`w[j] -= err·u[i,j]/d[j]`).
//
// Выход: packed [n, width/2] (младший полубайт — чётный столбец),
// scales [n, width/16] (коды E4M3 построчно), deq [n, width] (f32).

#define NV_THREADS 128

__device__ __forceinline__ unsigned char e2m1_code(float x) {
    unsigned char sign = signbit(x) ? 8 : 0;
    float a = fabsf(x);
    unsigned char idx;
    if (a >= 5.0f) idx = 7;
    else if (a >= 3.5f) idx = 6;
    else if (a >= 2.5f) idx = 5;
    else if (a >= 1.75f) idx = 4;
    else if (a >= 1.25f) idx = 3;
    else if (a >= 0.75f) idx = 2;
    else if (a >= 0.25f) idx = 1;
    else idx = 0;
    return sign | idx;
}

__device__ __forceinline__ float e2m1_val(unsigned char c, float s) {
    const float tab[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float sg = (c & 8) ? -1.0f : 1.0f;
    return __fmul_rn(__fmul_rn(tab[c & 7], s), sg);
}

// Код E4M3 положительного x (как `encode_e4m3`): число порогов ≤ x.
__device__ __forceinline__ int e4m3_code(float x, const float* thr) {
    int lo = 0, hi = 0x7e; // ответ в [0, 0x7e]
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (x >= thr[mid]) lo = mid; else hi = mid - 1;
    }
    return lo;
}

// Выбор кода масштаба группы (16 значений `w`, важность `m` или nullptr).
__device__ int select_scale(const float* w, const float* m, const float* thr, const float* dec, int search, int noclip) {
    float amax = 0.0f;
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(w[i]));
    int base = e4m3_code(amax > 0.0f ? __fdiv_rn(amax, 6.0f) : 1e-9f, thr);
    int sb = base;
    if (search && amax > 0.0f) {
        double best = 1.7976931348623157e308;
        int lo = base - 2 < 1 ? 1 : base - 2;
        int hi = base + 2 > 0x7e ? 0x7e : base + 2;
        const float keep = 1.0f - 1e-6f;
        for (int c = lo; c <= hi; ++c) {
            float sc = dec[c];
            if (noclip && __fmul_rn(6.0f, sc) < __fmul_rn(amax, keep) && c < 0x7e) continue;
            double e = -0.0;
            for (int i = 0; i < 16; ++i) {
                float q = e2m1_val(e2m1_code(__fdiv_rn(w[i], sc)), sc);
                double dv = (double)__fsub_rn(w[i], q);
                e = __dadd_rn(e, __dmul_rn(__dmul_rn(dv, dv), m ? (double)m[i] : 1.0));
            }
            if (e < best) { best = e; sb = c; }
        }
    }
    return sb;
}

extern "C" __global__ void nvfp4_encode_rows(
    const float* __restrict__ win, int width,
    const float* __restrict__ u, const float* __restrict__ d, const float* __restrict__ wimp,
    const float* __restrict__ thr, const float* __restrict__ dec,
    int search, int noclip,
    unsigned char* __restrict__ packed, unsigned char* __restrict__ scales, float* __restrict__ deq)
{
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int ng = width / 16;
    __shared__ float sw[NV_THREADS];
    __shared__ unsigned char codes[NV_THREADS];
    __shared__ int sbs[NV_THREADS / 16];
    __shared__ float err_s;
    const float* row = win + (size_t)r * width;
    for (int cs = 0; cs < width; cs += NV_THREADS) {
        const int cols = width - cs < NV_THREADS ? width - cs : NV_THREADS;
        const int groups = cols / 16;
        if (t < cols) sw[t] = row[cs + t];
        __syncthreads();
        if (u == nullptr) {
            if (t < groups) sbs[t] = select_scale(&sw[t * 16], wimp ? wimp + cs + t * 16 : nullptr, thr, dec, search, noclip);
            __syncthreads();
            if (t < cols) {
                int g = t / 16;
                float sq = dec[sbs[g]];
                if (sq == 0.0f) sq = 1e-9f;
                unsigned char c = e2m1_code(__fdiv_rn(sw[t], sq));
                codes[t] = c;
                deq[(size_t)r * width + cs + t] = e2m1_val(c, sq);
            }
        } else {
            for (int g = 0; g < groups; ++g) {
                if (t == 0) sbs[g] = select_scale(&sw[g * 16], wimp ? wimp + cs + g * 16 : nullptr, thr, dec, search, noclip);
                __syncthreads();
                float sq = dec[sbs[g]];
                if (sq == 0.0f) sq = 1e-9f;
                for (int i = g * 16; i < g * 16 + 16; ++i) {
                    if (t == i) {
                        float wi = sw[i];
                        unsigned char c = e2m1_code(__fdiv_rn(wi, sq));
                        float v = e2m1_val(c, sq);
                        codes[i] = c;
                        deq[(size_t)r * width + cs + i] = v;
                        float di = d ? d[i] : 1.0f;
                        err_s = __fdiv_rn(__fmul_rn(__fsub_rn(wi, v), di), u[(size_t)i * width + i]);
                    }
                    __syncthreads();
                    float err = err_s;
                    if (err != 0.0f && t > i && t < cols) {
                        float ui = u[(size_t)i * width + t];
                        float delta = d ? __fdiv_rn(__fmul_rn(err, ui), d[t]) : __fmul_rn(err, ui);
                        sw[t] = __fsub_rn(sw[t], delta);
                    }
                    __syncthreads();
                }
            }
        }
        __syncthreads();
        if (t < cols / 2) packed[(size_t)r * (width / 2) + cs / 2 + t] = codes[2 * t] | (codes[2 * t + 1] << 4);
        if (t < groups) scales[(size_t)r * ng + cs / 16 + t] = (unsigned char)sbs[t];
        __syncthreads();
    }
}
