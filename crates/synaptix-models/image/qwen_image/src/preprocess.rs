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

pub use synaptix_io::image::resample::{resize_u8, Filter, Precision};

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
    use synaptix_io::image::resample::axis_coeffs;

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
