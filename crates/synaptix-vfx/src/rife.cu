// synaptix-vfx: ядра RIFE (IFNet 4.2x) — f32, NCHW, пакет 1. Поток и маска считаются в f32: в f16
// смещения в пикселях на 4K теряют точность.

// Свёртка K×K со страйдом: out[co] = act(conv·beta[co] + res[co]) или act(conv + b[co]).
// Поток считает пиксель для CG каналов выхода (blockIdx.z — группа каналов), веса группы — в общей памяти.
#define CG 8
extern "C" __global__ void conv_f32(const float* in, int ci, int h, int w, const float* wt, const float* b, const float* beta,
                                    const float* res, float* out, int co, int ho, int wo, int k, int stride, int pad, float slope, int act) {
    extern __shared__ float sw[];  // [CG][ci][k][k]
    const int c0 = blockIdx.z * CG;
    const int nw = ci * k * k;
    for (int i = threadIdx.x; i < CG * nw; i += blockDim.x) {
        int g = i / nw, r = i % nw;
        sw[i] = (c0 + g < co) ? wt[(size_t)(c0 + g) * nw + r] : 0.f;
    }
    __syncthreads();
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= ho * wo) return;
    int oy = p / wo, ox = p % wo;
    float acc[CG];
#pragma unroll
    for (int g = 0; g < CG; ++g) acc[g] = 0.f;
    for (int c = 0; c < ci; ++c) {
        const float* ip = in + (size_t)c * h * w;
        for (int ky = 0; ky < k; ++ky) {
            int iy = oy * stride - pad + ky;
            if (iy < 0 || iy >= h) continue;
            for (int kx = 0; kx < k; ++kx) {
                int ix = ox * stride - pad + kx;
                if (ix < 0 || ix >= w) continue;
                float v = ip[iy * w + ix];
                int wi = (c * k + ky) * k + kx;
#pragma unroll
                for (int g = 0; g < CG; ++g) acc[g] += v * sw[g * nw + wi];
            }
        }
    }
#pragma unroll
    for (int g = 0; g < CG; ++g) {
        int o = c0 + g;
        if (o >= co) break;
        float v = acc[g] + b[o];
        if (beta) v = v * beta[o];  // ResConv: conv(x)·beta + x
        if (res) v += res[(size_t)o * ho * wo + p];
        if (act) v = v > 0.f ? v : v * slope;
        out[(size_t)o * ho * wo + p] = v;
    }
}

// ConvTranspose2d k=4, s=2, p=1: in [ci,h,w], wt [ci][co][4][4] → out [co, 2h, 2w]; при shuffle=1 —
// сразу PixelShuffle(2): канал o → (o/4, 2·oy + (o%4)/2, 2·ox + o%2), выход [co/4, 4h, 4w].
// Поток — пиксель × 4 канала выхода (blockIdx.z — группа), веса группы — в общей памяти.
extern "C" __global__ void deconv4_f32(const float* in, int ci, int h, int w, const float* wt, const float* b, float* out, int co, int shuffle) {
    extern __shared__ float sw[];  // [ci][4][16]
    int ho = 2 * h, wo = 2 * w;
    int o0 = blockIdx.z * 4;
    for (int i = threadIdx.x; i < ci * 64; i += blockDim.x) {
        int c = i / 64, g = (i / 16) % 4, t = i % 16;
        sw[i] = (o0 + g < co) ? wt[((size_t)c * co + o0 + g) * 16 + t] : 0.f;
    }
    __syncthreads();
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= ho * wo) return;
    int oy = p / wo, ox = p % wo;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    // вклад: iy = (oy + 1 − ky)/2 при чётном числителе
    for (int ky = (oy + 1) & 1; ky < 4; ky += 2) {
        int iy = (oy + 1 - ky) / 2;
        if (iy < 0 || iy >= h) continue;
        for (int kx = (ox + 1) & 1; kx < 4; kx += 2) {
            int ix = (ox + 1 - kx) / 2;
            if (ix < 0 || ix >= w) continue;
            int t = ky * 4 + kx;
            for (int c = 0; c < ci; ++c) {
                float v = in[((size_t)c * h + iy) * w + ix];
#pragma unroll
                for (int g = 0; g < 4; ++g) acc[g] += v * sw[(c * 4 + g) * 16 + t];
            }
        }
    }
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        int o = o0 + g;
        if (o >= co) break;
        float v = acc[g] + b[o];
        if (shuffle) {
            int c = o / 4, sy = (o % 4) / 2, sx = o % 2;
            out[((size_t)c * (2 * ho) + 2 * oy + sy) * (2 * wo) + 2 * ox + sx] = v;
        } else {
            out[(size_t)o * ho * wo + p] = v;
        }
    }
}

// Билинейный ресайз как F.interpolate(align_corners=False): каналы [c0, c0+nc) src [*, hs, ws] →
// dst [*, hd, wd] с канала dc; значение ×mul; add — прибавить к dst.
extern "C" __global__ void resize_f32(const float* src, int hs, int ws, int c0, int nc, float* dst, int hd, int wd, int dc, float mul, int add) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.z;
    if (p >= hd * wd || c >= nc) return;
    int y = p / wd, x = p % wd;
    float fy = fmaxf((y + 0.5f) * ((float)hs / hd) - 0.5f, 0.f);
    float fx = fmaxf((x + 0.5f) * ((float)ws / wd) - 0.5f, 0.f);
    int y0 = min((int)fy, hs - 1), x0 = min((int)fx, ws - 1);
    int y1 = min(y0 + 1, hs - 1), x1 = min(x0 + 1, ws - 1);
    float ay = fy - y0, ax = fx - x0;
    const float* s = src + (size_t)(c0 + c) * hs * ws;
    float v = ((s[y0 * ws + x0] * (1.f - ax) + s[y0 * ws + x1] * ax) * (1.f - ay) + (s[y1 * ws + x0] * (1.f - ax) + s[y1 * ws + x1] * ax) * ay) * mul;
    float* d = dst + (size_t)(dc + c) * hd * wd + p;
    *d = add ? *d + v : v;
}

// Обратный warp (grid_sample bilinear, border, align_corners=True): dst[c](x) = src[c](x + flow).
// flow — [2, h, w] (x, y в пикселях) со сдвигом fo каналов в тензоре потока; dst — с канала dc.
extern "C" __global__ void warp_f32(const float* src, int nc, const float* flow, int fo, float* dst, int dc, int h, int w) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= h * w) return;
    int y = p / w, x = p % w;
    float fx = fminf(fmaxf(x + flow[(size_t)fo * h * w + p], 0.f), (float)(w - 1));
    float fy = fminf(fmaxf(y + flow[(size_t)(fo + 1) * h * w + p], 0.f), (float)(h - 1));
    int x0 = (int)fx, y0 = (int)fy, x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    float ax = fx - x0, ay = fy - y0;
    for (int c = 0; c < nc; ++c) {
        const float* s = src + (size_t)c * h * w;
        float v = (s[y0 * w + x0] * (1.f - ax) + s[y0 * w + x1] * ax) * (1.f - ay) + (s[y1 * w + x0] * (1.f - ax) + s[y1 * w + x1] * ax) * ay;
        dst[(size_t)(dc + c) * h * w + p] = v;
    }
}

extern "C" __global__ void fill_f32(float* dst, int n, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = v;
}

// RGBA8 [h, w] → [3, ph, pw] (0..1), за краем — нули (F.pad).
extern "C" __global__ void rgba_to_chw(const unsigned char* src, int w, int h, float* dst, int pw, int ph) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pw * ph) return;
    int y = p / pw, x = p % pw;
    bool in = x < w && y < h;
    for (int c = 0; c < 3; ++c) dst[(size_t)c * pw * ph + p] = in ? src[((size_t)y * w + x) * 4 + c] * (1.f / 255.f) : 0.f;
}

// Итог: warped0·σ(mask) + warped1·(1 − σ(mask)) → RGBA8 [h, w] (обрезка паддинга).
extern "C" __global__ void blend_rgba(const float* w0, const float* w1, const float* mask, int pw, int ph, unsigned char* out, int w, int h) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= w * h) return;
    int y = p / w, x = p % w;
    int q = y * pw + x;
    float m = 1.f / (1.f + __expf(-mask[q]));
    unsigned char* o = out + (size_t)p * 4;
    for (int c = 0; c < 3; ++c) {
        float v = w0[(size_t)c * pw * ph + q] * m + w1[(size_t)c * pw * ph + q] * (1.f - m);
        o[c] = (unsigned char)__float2int_rn(fminf(fmaxf(v, 0.f), 1.f) * 255.f);
    }
    o[3] = 255;
}

// Разница двух кадров (смена сцены): сумма |Δ| яркости по сетке с шагом step → atomicAdd в acc.
extern "C" __global__ void frame_diff(const unsigned char* a, const unsigned char* b, int w, int h, int step, float* acc) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x, gy = blockIdx.y;
    int x = gx * step, y = gy * step;
    if (x >= w || y >= h) return;
    const unsigned char* pa = a + ((size_t)y * w + x) * 4;
    const unsigned char* pb = b + ((size_t)y * w + x) * 4;
    float la = 0.299f * pa[0] + 0.587f * pa[1] + 0.114f * pa[2];
    float lb = 0.299f * pb[0] + 0.587f * pb[1] + 0.114f * pb[2];
    atomicAdd(acc, fabsf(la - lb));
}
