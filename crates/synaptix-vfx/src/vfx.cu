// synaptix-vfx: ядра видеофильтров (повышение разрешения, интерполяция кадров).
//
// Активации — NHWC f16, каналы в памяти с шагом Cs (кратно 16), лишние каналы — нули.
// Свёртка K×K (K = 1 | 3, шаг 1, отступ K/2) — неявный GEMM на тензорных ядрах (mma m16n8k16,
// f16 × f16 → f32): блок 256 потоков считает 2×64 пикселя × 64 канала выхода, вход с ореолом и веса
// одного отвода — в общей памяти. Вход — конкатенация до 4 тензоров (к первому можно прибавить in0b
// и применить активацию при загрузке).
// Эпилог: bias, остаток, активация, pixel shuffle ×r, выход f16 NHWC или RGBA8 (с билинейной базой).

#include <cuda_fp16.h>

#define TW 64
#define THREADS 256
#define MAX_IN 4

struct ConvArgs {
    const __half* in[MAX_IN];  // входы конкатенации [H,W,cs[i]]
    const __half* in0b;  // nullptr или как in[0] — складывается с ним при загрузке
    const __half* w;     // [K*K][cout_pad][cin_pad], cin_pad = сумма cs
    const float* bias;   // [cout_pad]
    const __half* res;   // nullptr или остаток в раскладке выхода до shuffle ([H,W,cout_pad])
    const float* alpha;  // PReLU по каналам (после shuffle) или nullptr
    void* out;           // f16 NHWC [H*r, W*r, cso] или RGBA8 [H*r, W*r, 4]
    const unsigned char* base;  // RGBA8 исходник для билинейной базы (out_kind 1) или nullptr
    int cs[MAX_IN];      // каналов каждого входа в конкатенации (кратно 16), 0 — входа нет
    int ps[MAX_IN];      // шаг пикселя входа в половинках (≥ cs; срез каналов — указатель со сдвигом)
    int h, w_, cout_pad, cso, k;
    int in_act;          // активация входа in[0] при загрузке (коды как act)
    int act;             // 0 нет, 1 relu, 2 lrelu(slope), 3 prelu скаляр (slope), 4 prelu по каналам, 5 silu, 6 sigmoid,
                         // 7 mish, 8 внимание SPAB: (v + res) * (sigmoid(v) − 0,5), 9 ворота: res * sigmoid(v)
    float slope;
    int res_mode;        // 0 нет, 1 до активации, 2 после
    int shuffle;         // r: 1 — без pixel shuffle
    int out_kind;        // 0 f16, 1 RGBA8 (значения 0..1 → 0..255)
    int base_mode;       // 0 билинейная база, 1 ближайший пиксель исходника (повтор по r×r)
    int base_w, base_h;  // размер исходника базы
    float out_scale;     // множитель значения перед записью (1)
    int wbuf;            // буферов весов отвода: 2 — следующий грузится во время счёта, 1 — по очереди
};

__device__ __forceinline__ void cp_async16(void* smem, const void* g, bool valid) {
    unsigned a = (unsigned)__cvta_generic_to_shared(smem);
    int n = valid ? 16 : 0;  // 0 — заполнить нулями
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(a), "l"(g), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n"); }

__device__ __forceinline__ float act_fn(float v, int act, float slope, const float* alpha, int c) {
    switch (act) {
    case 1: return v > 0.f ? v : 0.f;
    case 2: return v > 0.f ? v : v * slope;
    case 3: return v > 0.f ? v : v * slope;
    case 4: return v > 0.f ? v : v * alpha[c];
    case 5: return v / (1.f + __expf(-v));
    case 6: return 1.f / (1.f + __expf(-v));
    case 7: { float sp = v > 20.f ? v : log1pf(__expf(v)); return v * tanhf(sp); }
    default: return v;
    }
}

__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

// Накопление в f16 (на GeForce тензорные ядра с f16-накоплением вдвое быстрее): c — два half2.
__device__ __forceinline__ void mma16816h(unsigned* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma16816(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Билинейная выборка RGBA8 (align_corners = false), канал c, значение 0..1.
__device__ __forceinline__ float bilinear_u8(const unsigned char* src, int sw, int sh, float fx, float fy, int c) {
    fx = fmaxf(fx, 0.f);
    fy = fmaxf(fy, 0.f);
    int x0 = min((int)fx, sw - 1), y0 = min((int)fy, sh - 1);
    int x1 = min(x0 + 1, sw - 1), y1 = min(y0 + 1, sh - 1);
    float ax = fx - x0, ay = fy - y0;
    float a = src[(y0 * sw + x0) * 4 + c], b = src[(y0 * sw + x1) * 4 + c];
    float d = src[(y1 * sw + x0) * 4 + c], e = src[(y1 * sw + x1) * 4 + c];
    return ((a + (b - a) * ax) * (1.f - ay) + (d + (e - d) * ax) * ay) * (1.f / 255.f);
}

// NB — каналов выхода на блок, TH — строк пикселей на блок (2 или 4): блок TH×64 пикселей, у варпа
// MT = TH/2 m-тайлов по 16 пикселей подряд в строке — фрагменты весов переиспользуются MT раз.
// Веса отвода — в двух буферах: следующий отвод грузится cp.async, пока считается текущий.
template <int NB, int TH, bool H16>
__device__ __forceinline__ void conv_body(const ConvArgs& a) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int MT = TH / 2;
    const int P = a.k / 2;
    const int HP = TH + 2 * P, WP = TW + 2 * P;
    int cin = 0;
    for (int i = 0; i < MAX_IN; ++i) cin += a.cs[i];
    const int SI = cin + 8;  // шаг пикселя в общей памяти (+16 байт — без конфликтов банков)
    __half* sIn = (__half*)smem;
    __half* sW = sIn + HP * WP * SI;

    const int x0 = blockIdx.x * TW, y0 = blockIdx.y * TH, n0 = blockIdx.z * NB;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    // вход с ореолом: по 8 половинок (16 байт); индексы ведутся шагами — без делений в цикле
    const int chunks = cin / 8;
    const bool plain = !a.in0b && !a.in_act;
    {
        const int total = HP * WP;
        const int step_px = THREADS / chunks, step_ch = THREADS % chunks;
        int px = tid / chunks, ch = tid % chunks;
        int r = px / WP, c = px % WP;
        while (px < total) {
            int gy = y0 - P + r, gx = x0 - P + c;
            __half* dst = sIn + px * SI + ch * 8;
            bool in = gy >= 0 && gy < a.h && gx >= 0 && gx < a.w_;
            size_t pix = in ? (size_t)gy * a.w_ + gx : 0;
            int k = ch * 8, src = 0;
            while (src < MAX_IN - 1 && k >= a.cs[src]) { k -= a.cs[src]; ++src; }
            const __half* g = a.in[src] + pix * a.ps[src] + k;
            if (plain || src != 0) {
                cp_async16(dst, g, in);
            } else {
                uint4 v = make_uint4(0, 0, 0, 0);
                if (in) {
                    v = *(const uint4*)g;
                    if (a.in0b) {
                        uint4 u = *(const uint4*)(a.in0b + pix * a.ps[0] + k);
                        __half2* pv = (__half2*)&v;
                        const __half2* pu = (const __half2*)&u;
                        for (int j = 0; j < 4; ++j) pv[j] = __hadd2(pv[j], pu[j]);
                    }
                    if (a.in_act) {
                        __half* ph = (__half*)&v;
                        for (int j = 0; j < 8; ++j) ph[j] = __float2half(act_fn(__half2float(ph[j]), a.in_act, a.slope, a.alpha, 0));
                    }
                }
                *(uint4*)dst = v;
            }
            // следующий кусок: +THREADS в плоском индексе
            ch += step_ch;
            int dpx = step_px;
            if (ch >= chunks) { ch -= chunks; ++dpx; }
            px += dpx;
            c += dpx;
            while (c >= WP) { c -= WP; ++r; }
        }
        cp_async_commit();
    }

    constexpr int NT = NB / 8;  // n-тайлов варпа
    float acc[MT][NT][4];
    unsigned acch[MT][NT][2];  // f16-накопление (H16): пары половинок
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < NT; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][i][j] = 0.f;
            acch[m][i][0] = acch[m][i][1] = 0u;
        }

    // пиксели варпа: MT·16 подряд в строке тайла
    constexpr int WPR = 4 / MT;  // варпов на строку
    const int wrow = warp / WPR, wcol = (warp % WPR) * 16 * MT;

    const int taps = a.k * a.k;
    const int wstep_n = THREADS / chunks, wstep_ch = THREADS % chunks;
    const int wn0 = tid / chunks, wch0 = tid % chunks;
    auto load_w = [&](int t, __half* dst) {
        const __half* wt = a.w + (size_t)t * a.cout_pad * cin;
        int n = wn0, ch = wch0;
        while (n < NB) {
            bool ok = n0 + n < a.cout_pad;
            cp_async16(dst + n * SI + ch * 8, wt + (size_t)(ok ? n0 + n : 0) * cin + ch * 8, ok);
            ch += wstep_ch;
            n += wstep_n;
            if (ch >= chunks) { ch -= chunks; ++n; }
        }
        cp_async_commit();
    };
    // счёт одного отвода (dy, dx) с весами в cur
    auto tap = [&](const __half* cur, int dy, int dx) {
        for (int k0 = 0; k0 < cin; k0 += 16) {
            unsigned af[MT][4];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                int p = lane & 15;
                const __half* ap = sIn + ((wrow + dy) * WP + (wcol + m * 16 + p + dx)) * SI + k0 + (lane >> 4) * 8;
                ldsm_x4(af[m][0], af[m][1], af[m][2], af[m][3], ap);
            }
#pragma unroll
            for (int j = 0; j < NT / 2; ++j) {
                unsigned b0, b1, b2, b3;
                int n = j * 16 + (lane & 7) + (lane >> 4) * 8;
                int kk = k0 + ((lane >> 3) & 1) * 8;
                ldsm_x4(b0, b1, b2, b3, cur + n * SI + kk);
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    if constexpr (H16) {
                        mma16816h(acch[m][2 * j], af[m][0], af[m][1], af[m][2], af[m][3], b0, b1);
                        mma16816h(acch[m][2 * j + 1], af[m][0], af[m][1], af[m][2], af[m][3], b2, b3);
                    } else {
                        mma16816(acc[m][2 * j], af[m][0], af[m][1], af[m][2], af[m][3], b0, b1);
                        mma16816(acc[m][2 * j + 1], af[m][0], af[m][1], af[m][2], af[m][3], b2, b3);
                    }
                }
            }
        }
    };

    if (a.wbuf == 0) {
        // большое ядро: веса целой строки отводов разом — барьер на строку, а не на отвод
        for (int dy = 0; dy < a.k; ++dy) {
            if (dy > 0) __syncthreads();  // прошлая строка дочитана
            for (int dx = 0; dx < a.k; ++dx) load_w(dy * a.k + dx, sW + dx * NB * SI);
            cp_async_wait0();
            __syncthreads();
            for (int dx = 0; dx < a.k; ++dx) tap(sW + dx * NB * SI, dy, dx);
        }
    } else {
        const bool dbl = a.wbuf == 2;
        load_w(0, sW);
        for (int t = 0; t < taps; ++t) {
            const int dy = t / a.k, dx = t % a.k;
            __half* cur = dbl ? sW + (t & 1) * NB * SI : sW;
            if (!dbl && t > 0) load_w(t, sW);  // один буфер: прошлый отвод дочитан (барьер в конце шага)
            cp_async_wait0();
            __syncthreads();  // вход и отвод t на месте, буфер t+1 дочитан на прошлом шаге
            if (dbl && t + 1 < taps) load_w(t + 1, sW + ((t + 1) & 1) * NB * SI);
            tap(cur, dy, dx);
            __syncthreads();  // текущий буфер дочитан — на шаге t+1 в него пойдёт отвод t+2
        }
    }
    if constexpr (H16) {
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                float2 lo = __half22float2(*(__half2*)&acch[m][i][0]);
                float2 hi = __half22float2(*(__half2*)&acch[m][i][1]);
                acc[m][i][0] = lo.x; acc[m][i][1] = lo.y; acc[m][i][2] = hi.x; acc[m][i][3] = hi.y;
            }
    }

    // эпилог: всё, что зависит от канала, — один раз на канал; без shuffle — без делений
    const int r = a.shuffle;
    const int oh = a.h * r, ow = a.w_ * r;
    const bool res_in_act = a.act == 8 || a.act == 9;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int nb = n0 + nt * 8 + (lane & 3) * 2;  // пара каналов nb, nb+1
        if (nb >= a.cout_pad) continue;
        const float b0v = a.bias[nb], b1v = nb + 1 < a.cout_pad ? a.bias[nb + 1] : 0.f;
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int half_ = 0; half_ < 2; ++half_) {
            int p = wrow * TW + wcol + m * 16 + (lane >> 2) + half_ * 8;
            int y = y0 + p / TW, x = x0 + p % TW;
            if (y >= a.h || x >= a.w_) continue;
            const size_t pix = (size_t)y * a.w_ + x;
            float v0 = acc[m][nt][half_ * 2] + b0v, v1 = acc[m][nt][half_ * 2 + 1] + b1v;
            float r0 = 0.f, r1 = 0.f;
            if (a.res_mode) {
                float2 rr = __half22float2(*(const __half2*)(a.res + pix * a.cout_pad + nb));
                r0 = rr.x; r1 = rr.y;
            }
            if (a.res_mode == 1 && !res_in_act) { v0 += r0; v1 += r1; }
            if (a.act == 8) {
                v0 = (v0 + r0) * (1.f / (1.f + __expf(-v0)) - 0.5f);
                v1 = (v1 + r1) * (1.f / (1.f + __expf(-v1)) - 0.5f);
            } else if (a.act == 9) {
                v0 = r0 / (1.f + __expf(-v0));
                v1 = r1 / (1.f + __expf(-v1));
            } else if (a.act) {
                int c0 = r == 1 ? nb : nb / (r * r), c1 = r == 1 ? nb + 1 : (nb + 1) / (r * r);
                v0 = act_fn(v0, a.act, a.slope, a.alpha, c0);
                v1 = act_fn(v1, a.act, a.slope, a.alpha, c1);
            }
            if (a.res_mode == 2 && !res_in_act) { v0 += r0; v1 += r1; }
            v0 *= a.out_scale;
            v1 *= a.out_scale;
            if (a.out_kind == 0 && r == 1) {
                // без shuffle: пара каналов — одна запись half2
                if (nb + 1 < a.cso) *(__half2*)((__half*)a.out + pix * a.cso + nb) = __floats2half2_rn(v0, v1);
                else if (nb < a.cso) ((__half*)a.out)[pix * a.cso + nb] = __float2half(v0);
                continue;
            }
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int n = nb + e;
                if (n >= a.cout_pad) continue;
                const float v = e ? v1 : v0;
                const int rr2 = r * r;
                const int c = n / rr2, sub = n - c * rr2;
                const int yy = y * r + sub / r, xx = x * r + sub % r;
                if (a.out_kind == 0) {
                    if (c < a.cso) ((__half*)a.out)[((size_t)yy * ow + xx) * a.cso + c] = __float2half(v);
                } else if (c < 3) {
                    float o = v;
                    if (a.base && a.base_mode == 1) {
                        o += a.base[((size_t)min(y, a.base_h - 1) * a.base_w + min(x, a.base_w - 1)) * 4 + c] * (1.f / 255.f);
                    } else if (a.base) {
                        float fx = (xx + 0.5f) * ((float)a.base_w / ow) - 0.5f;
                        float fy = (yy + 0.5f) * ((float)a.base_h / oh) - 0.5f;
                        o += bilinear_u8(a.base, a.base_w, a.base_h, fx, fy, c);
                    }
                    unsigned char* op = (unsigned char*)a.out + ((size_t)yy * ow + xx) * 4;
                    op[c] = (unsigned char)__float2int_rn(fminf(fmaxf(o, 0.f), 1.f) * 255.f);
                    if (c == 0) op[3] = 255;
                }
            }
        }
    }
}


#define CONV_K(NB, TH) \
    extern "C" __global__ void __launch_bounds__(THREADS) conv_mma##NB##_##TH(ConvArgs a) { conv_body<NB, TH, false>(a); } \
    extern "C" __global__ void __launch_bounds__(THREADS) conv_mmah##NB##_##TH(ConvArgs a) { conv_body<NB, TH, true>(a); }
CONV_K(16, 2) CONV_K(32, 2) CONV_K(48, 2) CONV_K(64, 2)
CONV_K(16, 4) CONV_K(32, 4) CONV_K(48, 4) CONV_K(64, 4)

// RGBA8 → NHWC f16 [H,W,cs]: каналы 0..2 = значение·scale + off[c], остальные — нули.
extern "C" __global__ void rgba_to_nhwc(const unsigned char* src, __half* dst, int n, int cs, float scale, float o0, float o1, float o2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uchar4 s = ((const uchar4*)src)[i];
    __half* d = dst + (size_t)i * cs;
    // первые 8 каналов одной записью, остальные — нули по 16 байт
    __half h[8];
    h[0] = __float2half(s.x * scale + o0);
    h[1] = __float2half(s.y * scale + o1);
    h[2] = __float2half(s.z * scale + o2);
    for (int c = 3; c < 8; ++c) h[c] = __float2half(0.f);
    *(uint4*)d = *(uint4*)h;
    for (int c = 8; c < cs; c += 8) *(uint4*)(d + c) = make_uint4(0, 0, 0, 0);
}

// Копия каналов [c0, c1) из src в dst (одинаковый шаг пикселя ps), n пикселей.
extern "C" __global__ void copy_channels(const __half* src, __half* dst, int n, int ps, int c0, int c1) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int w = c1 - c0;
    if (i >= (size_t)n * w) return;
    size_t p = i / w;
    int c = c0 + (int)(i % w);
    dst[p * ps + c] = src[p * ps + c];
}

// LayerNorm по каналам каждого пикселя (как в RealPLKSR): x [n, cs], каналов c, eps; веса gamma/beta.
extern "C" __global__ void layer_norm_px(const __half* x, __half* y, const float* g, const float* b, int n, int cs, int c, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const __half* p = x + (size_t)i * cs;
    __half* q = y + (size_t)i * cs;
    if (c == 64 && cs == 64) {
        // частый случай RealPLKSR: пиксель целиком в регистры восемью чтениями по 16 байт
        float v[64];
        for (int j = 0; j < 8; ++j) {
            uint4 u = ((const uint4*)p)[j];
            const __half* h = (const __half*)&u;
            for (int k = 0; k < 8; ++k) v[j * 8 + k] = __half2float(h[k]);
        }
        float m = 0.f, s2 = 0.f;
        for (int k = 0; k < 64; ++k) m += v[k];
        m *= (1.f / 64.f);
        for (int k = 0; k < 64; ++k) { float d = v[k] - m; s2 += d * d; }
        float r = rsqrtf(s2 * (1.f / 64.f) + eps);
        for (int j = 0; j < 8; ++j) {
            __half h[8];
            for (int k = 0; k < 8; ++k) h[k] = __float2half((v[j * 8 + k] - m) * r * g[j * 8 + k] + b[j * 8 + k]);
            ((uint4*)q)[j] = *(uint4*)h;
        }
        return;
    }
    float m = 0.f, v = 0.f;
    for (int k = 0; k < c; ++k) m += __half2float(p[k]);
    m /= c;
    for (int k = 0; k < c; ++k) { float d = __half2float(p[k]) - m; v += d * d; }
    float r = rsqrtf(v / c + eps);
    for (int k = 0; k < c; ++k) q[k] = __float2half((__half2float(p[k]) - m) * r * g[k] + b[k]);
}

// GroupNorm, шаг 1: суммы и суммы квадратов по группам (stats [groups][2], обнулены заранее).
extern "C" __global__ void group_stats(const __half* x, float* stats, int n, int cs, int c, int groups) {
    extern __shared__ float sh[];  // [groups][2]
    for (int i = threadIdx.x; i < groups * 2; i += blockDim.x) sh[i] = 0.f;
    __syncthreads();
    int gc = c / groups;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)n * c; i += (size_t)gridDim.x * blockDim.x) {
        size_t p = i / c;
        int k = (int)(i % c);
        float v = __half2float(x[p * cs + k]);
        atomicAdd(&sh[(k / gc) * 2], v);
        atomicAdd(&sh[(k / gc) * 2 + 1], v * v);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < groups * 2; i += blockDim.x) atomicAdd(&stats[i], sh[i]);
}

// GroupNorm, шаг 2: y = (x − mean)/sqrt(var + eps)·gamma + beta + skip.
extern "C" __global__ void group_apply(const __half* x, const __half* skip, __half* y, const float* stats, const float* g, const float* b, int n, int cs, int c, int groups, float eps) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * c) return;
    size_t p = i / c;
    int k = (int)(i % c);
    int gi = k / (c / groups);
    float cnt = (float)n * (c / groups);
    float m = stats[gi * 2] / cnt;
    float var = fmaxf(stats[gi * 2 + 1] / cnt - m * m, 0.f);
    float v = (__half2float(x[p * cs + k]) - m) * rsqrtf(var + eps) * g[k] + b[k];
    if (skip) v += __half2float(skip[p * cs + k]);
    y[p * cs + k] = __float2half(v);
}

// RealPLKSR перед DySample: x = признаки (каналов nc = 3·r²) + вход, повторённый по r² (repeat_interleave).
extern "C" __global__ void add_repeat_rgba(const __half* f, const unsigned char* src, __half* out, int n, int cs, int nc, int r2) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * cs) return;
    size_t p = i / cs;
    int c = (int)(i % cs);
    float v = 0.f;
    if (c < nc) v = __half2float(f[p * cs + c]) + src[p * 4 + c / r2] * (1.f / 255.f);
    out[p * cs + c] = __float2half(v);
}

// DySample (×r, G групп) + конечная свёртка 1×1 в RGB → RGBA8 [H·r, W·r].
// x [H,W,xs] (nc = 3·r² каналов, по nc/G на группу), os [H,W,oss]: первые 2·G·r² — offset (с bias),
// следующие 2·G·r² — scope (без bias); смещение = offset·σ(scope)·½ + init. Канал смещения
// (xy·G + g)·r² + s, s — подпиксель (i·r + j). Выборка — билинейная, координаты прижаты к краю.
extern "C" __global__ void dysample_rgba(const __half* x, int xs, const __half* os, int oss, const float* init,
                                         const float* ew, const float* eb, unsigned char* out, int h, int w, int r, int G) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y;
    int ow = w * r;
    if (X >= ow || Y >= h * r) return;
    int px = X / r, py = Y / r, s = (Y % r) * r + (X % r);
    int r2 = r * r, nc = 3 * r2, cg = nc / G, no = 2 * G * r2;
    const __half* op = os + ((size_t)py * w + px) * oss;
    float acc[3] = {eb[0], eb[1], eb[2]};
    for (int g = 0; g < G; ++g) {
        int cx = (0 * G + g) * r2 + s, cy = (1 * G + g) * r2 + s;
        float ox = __half2float(op[cx]) / (1.f + __expf(-__half2float(op[no + cx]))) * 0.5f + init[cx];
        float oy = __half2float(op[cy]) / (1.f + __expf(-__half2float(op[no + cy]))) * 0.5f + init[cy];
        float fx = fminf(fmaxf(px + ox, 0.f), (float)(w - 1));
        float fy = fminf(fmaxf(py + oy, 0.f), (float)(h - 1));
        int x0 = (int)fx, y0 = (int)fy, x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
        float ax = fx - x0, ay = fy - y0;
        const __half* p00 = x + ((size_t)y0 * w + x0) * xs;
        const __half* p01 = x + ((size_t)y0 * w + x1) * xs;
        const __half* p10 = x + ((size_t)y1 * w + x0) * xs;
        const __half* p11 = x + ((size_t)y1 * w + x1) * xs;
        for (int k = 0; k < cg; ++k) {
            int c = g * cg + k;
            float v = (__half2float(p00[c]) * (1.f - ax) + __half2float(p01[c]) * ax) * (1.f - ay)
                    + (__half2float(p10[c]) * (1.f - ax) + __half2float(p11[c]) * ax) * ay;
            for (int o = 0; o < 3; ++o) acc[o] += ew[o * nc + c] * v;
        }
    }
    unsigned char* q = out + ((size_t)Y * ow + X) * 4;
    for (int o = 0; o < 3; ++o) q[o] = (unsigned char)__float2int_rn(fminf(fmaxf(acc[o], 0.f), 1.f) * 255.f);
    q[3] = 255;
}
