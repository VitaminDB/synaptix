//! Подготовка картинок как у пайплайнов diffusers и процессора Qwen2-VL:
//! размеры (`calculate_dimensions`, `smart_resize`), ресэмплинг бит в бит как
//! PIL (Lanczos у `VaeImageProcessor`) и torchvision на uint8 (бикубика с
//! антиалиасом у процессора Qwen2-VL), нарезка на патчи башни зрения.
//!
//! Всё на CPU: картинки на входе — `[3, H, W]` в [0, 1], ресайз — в uint8.

use synaptix_core::{device::Device, dtype::DType, error::Result, error::SynaptixError, tensor::Tensor};

use crate::config::{ProcessorConfig, VisionConfig};

/// Площадь картинки для VL-энкодера у 2509/2511 (`CONDITION_IMAGE_SIZE`).
pub const CONDITION_AREA: usize = 384 * 384;
/// Площадь картинки для VAE и размера выхода (`VAE_IMAGE_SIZE`).
pub const VAE_AREA: usize = 1024 * 1024;

/// `round()` Python: половина — к чётному.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

/// `calculate_dimensions(target_area, ratio)` пайплайнов: стороны под
/// площадь с сохранением пропорций, округлённые до кратного 32.
pub fn calculate_dimensions(target_area: usize, ratio: f64) -> (usize, usize) {
    let width = (target_area as f64 * ratio).sqrt();
    let height = width / ratio;
    let w = round_half_even(width / 32.0) * 32.0;
    let h = round_half_even(height / 32.0) * 32.0;
    ((w as usize).max(32), (h as usize).max(32))
}

/// `smart_resize` процессора Qwen2-VL: стороны кратны `factor` (28), площадь
/// в пределах `[min_pixels, max_pixels]`. → `(h, w)`.
pub fn smart_resize(h: usize, w: usize, factor: usize, min_pixels: usize, max_pixels: usize) -> (usize, usize) {
    let f = factor as f64;
    let (hf, wf) = (h as f64, w as f64);
    let mut hb = (round_half_even(hf / f) * f) as usize;
    let mut wb = (round_half_even(wf / f) * f) as usize;
    if hb * wb > max_pixels {
        let beta = ((hf * wf) / max_pixels as f64).sqrt();
        hb = factor.max(((hf / beta / f).floor() * f) as usize);
        wb = factor.max(((wf / beta / f).floor() * f) as usize);
    } else if hb * wb < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        hb = ((hf * beta / f).ceil() * f) as usize;
        wb = ((wf * beta / f).ceil() * f) as usize;
    }
    (hb.max(factor), wb.max(factor))
}

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
fn axis_coeffs(inp: usize, out: usize, filter: Filter, prec: Precision) -> (u32, Vec<(usize, Vec<i64>)>) {
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

/// Картинка `[3, H, W]` (любое устройство/тип) → `(u8, H, W)`: как
/// `to_pil_image` — значения [0, 1] округляются до 1/255.
pub fn image_u8(image: &Tensor) -> Result<(Vec<u8>, usize, usize)> {
    let d = image.dims().to_vec();
    if d.len() != 3 || d[0] < 3 {
        return Err(SynaptixError::Other(format!("картинка должна быть [3, H, W], пришло {d:?}")));
    }
    let rgb = if d[0] == 3 { image.clone() } else { image.narrow(0, 0, 3)?.contiguous()? };
    let v = rgb.to_device(Device::Cpu)?.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    Ok((v.iter().map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8).collect(), d[1], d[2]))
}

/// Картинка `[3, H, W]` → `[3, nh, nw]` F32 в [0, 1] на CPU: Lanczos PIL,
/// как `VaeImageProcessor.resize`/`preprocess` у пайплайнов.
pub fn resize_image(image: &Tensor, nw: usize, nh: usize) -> Result<Tensor> {
    let (v, h, w) = image_u8(image)?;
    let out = resize_u8(&v, 3, h, w, nh, nw, Filter::Lanczos3, Precision::Pil);
    Tensor::from_vec(out.into_iter().map(|b| b as f32 / 255.0).collect::<Vec<f32>>(), (3, nh, nw), Device::Cpu)
}

/// Патчи для башни зрения: `[N, 3·T·P·P]` в порядке блоков слияния и сетка
/// `(t, h, w)` в патчах.
pub struct VisionPatches {
    pub patches: Tensor,
    pub grid: (usize, usize, usize),
}

impl VisionPatches {
    /// Токенов LLM после слияния 2×2.
    pub fn tokens(&self, merge: usize) -> usize {
        self.grid.0 * self.grid.1 * self.grid.2 / (merge * merge)
    }
}

/// Процессор Qwen2-VL: `smart_resize` (бикубика с антиалиасом), нормировка
/// CLIP, нарезка `[C, T, P, P]` на патч с повтором кадра по времени.
pub fn vision_patches(image: &Tensor, vc: &VisionConfig, pc: &ProcessorConfig) -> Result<VisionPatches> {
    let (v, h, w) = image_u8(image)?;
    let (nh, nw) = smart_resize(h, w, vc.factor(), pc.min_pixels, pc.max_pixels);
    let u8s = resize_u8(&v, 3, h, w, nh, nw, Filter::Bicubic, Precision::Torchvision);
    // rescale 1/255 и нормировка CLIP — в F32, как `rescale_and_normalize`.
    let mut px = vec![0f32; 3 * nh * nw];
    for c in 0..3 {
        let (m, s) = (pc.mean[c], pc.std[c]);
        for (o, b) in px[c * nh * nw..(c + 1) * nh * nw].iter_mut().zip(&u8s[c * nh * nw..(c + 1) * nh * nw]) {
            *o = (*b as f32 * (1.0 / 255.0) - m) / s;
        }
    }
    let (p, m, tps) = (vc.patch_size, vc.merge_size, vc.temporal_patch_size);
    let (gh, gw) = (nh / p, nw / p);
    let feat = 3 * tps * p * p;
    let mut out = vec![0f32; gh * gw * feat];
    let mut token = 0usize;
    for bh in 0..gh / m {
        for bw in 0..gw / m {
            for mh in 0..m {
                for mw in 0..m {
                    let (ph, pw) = (bh * m + mh, bw * m + mw);
                    let base = token * feat;
                    let mut k = 0usize;
                    for c in 0..3 {
                        for _t in 0..tps {
                            for y in 0..p {
                                let row = (c * nh + ph * p + y) * nw + pw * p;
                                out[base + k..base + k + p].copy_from_slice(&px[row..row + p]);
                                k += p;
                            }
                        }
                    }
                    token += 1;
                }
            }
        }
    }
    Ok(VisionPatches { patches: Tensor::from_vec(out, (gh * gw, feat), Device::Cpu)?, grid: (1, gh, gw) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_like_pipeline() {
        // calculate_dimensions(1024*1024, 1.0) → (1024, 1024)
        assert_eq!(calculate_dimensions(VAE_AREA, 1.0), (1024, 1024));
        // 384² → 384
        assert_eq!(calculate_dimensions(CONDITION_AREA, 1.0), (384, 384));
        // 4:3 → width sqrt(1024²·4/3)=1182.4 → 1184; height 886.8 → 896
        assert_eq!(calculate_dimensions(VAE_AREA, 4.0 / 3.0), (1184, 896));
    }

    #[test]
    fn smart_resize_rounds() {
        assert_eq!(smart_resize(384, 384, 28, 3136, 12845056), (392, 392));
        assert_eq!(smart_resize(1024, 1024, 28, 3136, 12845056), (1036, 1036));
        assert_eq!(smart_resize(10, 10, 28, 3136, 12845056), (56, 56));
    }

    #[test]
    fn resize_identity_and_constant() {
        let src = vec![64u8; 3 * 10 * 12];
        for (f, p) in [(Filter::Lanczos3, Precision::Pil), (Filter::Bicubic, Precision::Torchvision)] {
            assert!(resize_u8(&src, 3, 10, 12, 7, 5, f, p).iter().all(|&v| v == 64));
            assert!(resize_u8(&src, 3, 10, 12, 21, 30, f, p).iter().all(|&v| v == 64));
            assert_eq!(resize_u8(&src, 3, 10, 12, 10, 12, f, p), src);
        }
    }

    #[test]
    fn torchvision_precision_is_int16() {
        // Увеличение 384 → 392: наибольший вес ≈ 0,99 → точность 14 бит.
        let (bits, k) = axis_coeffs(384, 392, Filter::Bicubic, Precision::Torchvision);
        assert!(bits <= 15, "{bits}");
        assert!(k.iter().flat_map(|(_, v)| v.iter()).all(|&v| v.abs() < (1 << 15)));
        let (bits, _) = axis_coeffs(384, 392, Filter::Bicubic, Precision::Pil);
        assert_eq!(bits, 22);
    }

    #[test]
    fn half_even() {
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(3.5), 4.0);
        assert_eq!(round_half_even(2.4), 2.0);
    }
}
