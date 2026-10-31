// synaptix-vfx: ядра видеофильтров (повышение разрешения, интерполяция кадров).
//
// Активации — NHWC f16, каналы в памяти с шагом Cs (кратно 16), лишние каналы — нули.
// Свёртка K×K (K = 1 | 3, шаг 1, отступ K/2) — неявный GEMM на тензорных ядрах (mma m16n8k16,
// f16 × f16 → f32): блок 256 потоков считает 2×64 пикселя × 64 канала выхода, вход с ореолом и веса
// одного отвода — в общей памяти. Вход — конкатенация in0 (+ in0b, складывается при загрузке) и in1.
// Эпилог: bias, остаток, активация, pixel shuffle ×r, выход f16 NHWC или RGBA8 (с билинейной базой).

#include <cuda_fp16.h>

#define TH 2
#define TW 64
#define NB 64
#define THREADS 256

struct ConvArgs {
    const __half* in0;   // [H,W,cs0]
    const __half* in0b;  // nullptr или как in0 — складывается с ним при загрузке
    const __half* in1;   // nullptr или [H,W,cs1] — каналы после in0
    const __half* w;     // [K*K][cout_pad][cin_pad], cin_pad = cs0 + cs1
    const float* bias;   // [cout_pad]
    const __half* res;   // nullptr или остаток в раскладке выхода до shuffle ([H,W,cout_pad])
    const float* alpha;  // PReLU по каналам (после shuffle) или nullptr
    void* out;           // f16 NHWC [H*r, W*r, cso] или RGBA8 [H*r, W*r, 4]
    const unsigned char* base;  // RGBA8 исходник для билинейной базы (out_kind 1) или nullptr
    int h, w_, cs0, cs1, cout_pad, cso, k;
    int act;             // 0 нет, 1 relu, 2 lrelu(slope), 3 prelu скаляр (slope), 4 prelu по каналам, 5 silu, 6 sigmoid
    float slope;
    int res_mode;        // 0 нет, 1 до активации, 2 после
    int shuffle;         // r: 1 — без pixel shuffle
    int out_kind;        // 0 f16, 1 RGBA8 (значения 0..1 → 0..255)
    int base_w, base_h;  // размер исходника базы
    float out_scale;     // множитель значения перед записью (1)
};

__device__ __forceinline__ float act_fn(float v, int act, float slope, const float* alpha, int c) {
    switch (act) {
    case 1: return v > 0.f ? v : 0.f;
    case 2: return v > 0.f ? v : v * slope;
    case 3: return v > 0.f ? v : v * slope;
    case 4: return v > 0.f ? v : v * alpha[c];
    case 5: return v / (1.f + __expf(-v));
    case 6: return 1.f / (1.f + __expf(-v));
    default: return v;
    }
}

__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
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

extern "C" __global__ void __launch_bounds__(THREADS) conv_mma(ConvArgs a) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int P = a.k / 2;
    const int HP = TH + 2 * P, WP = TW + 2 * P;
    const int cin = a.cs0 + a.cs1;
    const int SI = cin + 8;  // шаг пикселя в общей памяти (+16 байт — без конфликтов банков)
    __half* sIn = (__half*)smem;
    __half* sW = sIn + HP * WP * SI;

    const int x0 = blockIdx.x * TW, y0 = blockIdx.y * TH, n0 = blockIdx.z * NB;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    // вход с ореолом: по 8 половинок (16 байт)
    const int chunks = cin / 8;
    for (int i = tid; i < HP * WP * chunks; i += THREADS) {
        int ch = i % chunks, px = i / chunks;
        int r = px / WP, c = px % WP;
        int gy = y0 - P + r, gx = x0 - P + c;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (gy >= 0 && gy < a.h && gx >= 0 && gx < a.w_) {
            size_t pix = (size_t)gy * a.w_ + gx;
            int k = ch * 8;
            if (k < a.cs0) {
                v = *(const uint4*)(a.in0 + pix * a.cs0 + k);
                if (a.in0b) {
                    uint4 u = *(const uint4*)(a.in0b + pix * a.cs0 + k);
                    __half2* pv = (__half2*)&v;
                    const __half2* pu = (const __half2*)&u;
                    for (int j = 0; j < 4; ++j) pv[j] = __hadd2(pv[j], pu[j]);
                }
            } else {
                v = *(const uint4*)(a.in1 + pix * a.cs1 + (k - a.cs0));
            }
        }
        *(uint4*)(sIn + px * SI + ch * 8) = v;
    }

    float acc[8][4];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

    // пиксели варпа: 16 подряд в строке тайла
    const int wrow = warp / 4, wcol = (warp % 4) * 16;

    for (int t = 0; t < a.k * a.k; ++t) {
        const int dy = t / a.k, dx = t % a.k;
        __syncthreads();  // прошлый отвод дочитан
        const __half* wt = a.w + (size_t)t * a.cout_pad * cin;
        for (int i = tid; i < NB * chunks; i += THREADS) {
            int ch = i % chunks, n = i / chunks;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (n0 + n < a.cout_pad) v = *(const uint4*)(wt + (size_t)(n0 + n) * cin + ch * 8);
            *(uint4*)(sW + n * SI + ch * 8) = v;
        }
        __syncthreads();
        for (int k0 = 0; k0 < cin; k0 += 16) {
            unsigned a0, a1, a2, a3;
            {
                int p = lane & 15;
                const __half* ap = sIn + ((wrow + dy) * WP + (wcol + p + dx)) * SI + k0 + (lane >> 4) * 8;
                ldsm_x4(a0, a1, a2, a3, ap);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                unsigned b0, b1, b2, b3;
                int n = j * 16 + (lane & 7) + (lane >> 4) * 8;
                int kk = k0 + ((lane >> 3) & 1) * 8;
                ldsm_x4(b0, b1, b2, b3, sW + n * SI + kk);
                mma16816(acc[2 * j], a0, a1, a2, a3, b0, b1);
                mma16816(acc[2 * j + 1], a0, a1, a2, a3, b2, b3);
            }
        }
    }

    // эпилог
    const int r = a.shuffle;
    const int oh = a.h * r, ow = a.w_ * r;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int half_ = 0; half_ < 2; ++half_) {
            int p = wrow * TW + wcol + (lane >> 2) + half_ * 8;
            int y = y0 + p / TW, x = x0 + p % TW;
            if (y >= a.h || x >= a.w_) continue;
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                int n = n0 + nt * 8 + (lane & 3) * 2 + e;
                if (n >= a.cout_pad) continue;
                float v = acc[nt][half_ * 2 + e] + a.bias[n];
                if (a.res_mode == 1) v += __half2float(a.res[((size_t)y * a.w_ + x) * a.cout_pad + n]);
                int c = n / (r * r), sub = n % (r * r);
                int yy = y * r + sub / r, xx = x * r + sub % r;
                v = act_fn(v, a.act, a.slope, a.alpha, c);
                if (a.res_mode == 2) v += __half2float(a.res[((size_t)y * a.w_ + x) * a.cout_pad + n]);
                v *= a.out_scale;
                if (a.out_kind == 0) {
                    if (c < a.cso) ((__half*)a.out)[((size_t)yy * ow + xx) * a.cso + c] = __float2half(v);
                } else if (c < 3) {
                    if (a.base) {
                        float fx = (xx + 0.5f) * ((float)a.base_w / ow) - 0.5f;
                        float fy = (yy + 0.5f) * ((float)a.base_h / oh) - 0.5f;
                        v += bilinear_u8(a.base, a.base_w, a.base_h, fx, fy, c);
                    }
                    unsigned char* o = (unsigned char*)a.out + ((size_t)yy * ow + xx) * 4;
                    o[c] = (unsigned char)__float2int_rn(fminf(fmaxf(v, 0.f), 1.f) * 255.f);
                    if (c == 0) o[3] = 255;
                }
            }
        }
    }
}

// RGBA8 → NHWC f16 [H,W,cs]: каналы 0..2 = значение/255, остальные — нули.
extern "C" __global__ void rgba_to_nhwc(const unsigned char* src, __half* dst, int n, int cs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned char* s = src + (size_t)i * 4;
    __half* d = dst + (size_t)i * cs;
    d[0] = __float2half(s[0] * (1.f / 255.f));
    d[1] = __float2half(s[1] * (1.f / 255.f));
    d[2] = __float2half(s[2] * (1.f / 255.f));
    for (int c = 3; c < cs; ++c) d[c] = __float2half(0.f);
}
