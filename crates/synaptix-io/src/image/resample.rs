//! Ресэмплинг uint8 бит в бит как PIL (`Image.resize`) и torchvision
//! (`tvF.resize(antialias=True)` на uint8 — бикубика с антиалиасом у
//! процессоров Qwen2-VL/Qwen3-VL, картинок и кадров видео). Башня зрения
//! чувствительна к пикселям, поэтому целочисленные ядра повторены до бита.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    /// `LANCZOS` (окно 3).
    Lanczos3,
    /// `BICUBIC` (a = −0.5), с антиалиасом при уменьшении.
    Bicubic,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Lanczos3 => 3.0,
            Filter::Bicubic => 2.0,
        }
    }

    fn eval(self, x: f64) -> f64 {
        match self {
            Filter::Lanczos3 => {
                let sinc = |v: f64| {
                    if v == 0.0 {
                        1.0
                    } else {
                        let p = std::f64::consts::PI * v;
                        p.sin() / p
                    }
                };
                if x.abs() < 3.0 {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            }
            Filter::Bicubic => {
                let a = -0.5;
                let x = x.abs();
                if x < 1.0 {
                    ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * a
                } else {
                    0.0
                }
            }
        }
    }
}

/// Чья целочисленная арифметика: картинка у пайплайнов идёт в uint8, и
/// результат ресайза — тоже uint8. Башня зрения чувствительна к пикселям
/// (0,7 % разницы после ресайза дают 20 % разницы эмбеддингов), поэтому ядра
/// повторены до бита.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// PIL (`Image.resize`, у `VaeImageProcessor`): веса ×2²², int32.
    Pil,
    /// torchvision на uint8 (`tvF.resize(antialias=True)`, у процессора
    /// Qwen2-VL): веса int16, точность — наибольшая, при которой вес < 2¹⁵.
    Torchvision,
}

/// Коэффициенты одной оси: точность (бит) и для каждого выходного индекса —
/// начало окна во входе и целые веса (схема `precompute_coeffs` PIL).
pub fn axis_coeffs(inp: usize, out: usize, filter: Filter, prec: Precision) -> (u32, Vec<(usize, Vec<i64>)>) {
    let scale = inp as f64 / out as f64;
    let fscale = scale.max(1.0);
    let support = filter.support() * fscale;
    let ss = 1.0 / fscale;
    let float: Vec<(usize, Vec<f64>)> = (0..out)
        .map(|o| {
            let center = (o as f64 + 0.5) * scale;
            // (int) у PIL — отсечение к нулю; аргумент тут ≥ −support.
            let xmin = ((center - support + 0.5) as isize).max(0) as usize;
            let xmax = ((center + support + 0.5) as isize).min(inp as isize).max(xmin as isize) as usize;
            let mut k: Vec<f64> = (0..xmax - xmin).map(|x| filter.eval((x as f64 + xmin as f64 - center + 0.5) * ss)).collect();
            let sum: f64 = k.iter().sum();
            if sum != 0.0 {
                for v in &mut k {
                    *v /= sum;
                }
            }
            (xmin, k)
        })
        .collect();
    let bits = match prec {
        Precision::Pil => 22,
        Precision::Torchvision => {
            let wmax = float.iter().flat_map(|(_, k)| k.iter().copied()).fold(f64::MIN, f64::max);
            let mut p = 0u32;
            while p < 22 && ((0.5 + wmax * (1u64 << (p + 1)) as f64) as i64) < (1 << 15) {
                p += 1;
            }
            p
        }
    };
    let q = (1u64 << bits) as f64;
    let ints = float
        .into_iter()
        .map(|(x0, k)| (x0, k.into_iter().map(|v| if v < 0.0 { (-0.5 + v * q) as i64 } else { (0.5 + v * q) as i64 }).collect()))
        .collect();
    (bits, ints)
}

/// Один проход по оси (`axis` 2 — ширина, 1 — высота) над `[C, H, W]` u8.
fn resample_pass(src: &[u8], c: usize, h: usize, w: usize, n: usize, axis: usize, filter: Filter, prec: Precision) -> Vec<u8> {
    let (inp, oh, ow) = if axis == 2 { (w, h, n) } else { (h, n, w) };
    let (bits, coeffs) = axis_coeffs(inp, n, filter, prec);
    let half = 1i64 << (bits.max(1) - 1);
    let mut dst = vec![0u8; c * oh * ow];
    for ch in 0..c {
        let plane = &src[ch * h * w..(ch + 1) * h * w];
        let out = &mut dst[ch * oh * ow..(ch + 1) * oh * ow];
        for y in 0..oh {
            for x in 0..ow {
                let (i0, k) = if axis == 2 { &coeffs[x] } else { &coeffs[y] };
                let mut acc = half;
                for (j, kv) in k.iter().enumerate() {
                    let px = if axis == 2 { plane[y * w + i0 + j] } else { plane[(i0 + j) * w + x] };
                    acc += px as i64 * kv;
                }
                out[y * ow + x] = (acc >> bits).clamp(0, 255) as u8;
            }
        }
    }
    dst
}

/// Ресэмплинг `[C, H, W]` u8 → `[C, nh, nw]` u8: сначала по ширине, потом по
/// высоте; ось без изменения размера пропускается (как PIL и torchvision).
pub fn resize_u8(src: &[u8], c: usize, h: usize, w: usize, nh: usize, nw: usize, filter: Filter, prec: Precision) -> Vec<u8> {
    let mut cur = src.to_vec();
    let mut cw = w;
    if nw != w {
        cur = resample_pass(&cur, c, h, w, nw, 2, filter, prec);
        cw = nw;
    }
    if nh != h {
        cur = resample_pass(&cur, c, h, cw, nh, 1, filter, prec);
    }
    cur
}
