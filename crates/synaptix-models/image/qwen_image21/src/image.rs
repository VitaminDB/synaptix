//! Картинки Qwen-Image 2.1 — RGBA в uint8, как у пайплайна diffusers (PIL):
//! ресайз Lanczos с премультипликацией альфы (`Image.resize` у режима RGBA
//! идёт через RGBa), композит на белом для башни зрения (`paste` с маской
//! альфы), патчи процессора Qwen3-VL (бикубика torchvision на uint8) и
//! тензоры для VAE (`[4, H, W]` в [0, 1]).
//!
//! Всё на CPU. Ресэмплинг повторяет целочисленную арифметику PIL/torchvision
//! бит в бит ([`synaptix_image_qwen::preprocess`]): башня зрения чувствительна
//! к пикселям.

use synaptix_core::{device::Device, dtype::DType, error::Result, error::SynaptixError, tensor::Tensor};
use synaptix_image_qwen::preprocess::{resize_u8, smart_resize, Filter, Precision};

use crate::config::{ProcessorConfig, Qwen3VlVisionConfig};

/// Картинка RGBA8 построчно (HWC).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbaImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// `MULDIV255` PIL: `(a·b + 128)` с поправкой на деление на 255.
#[inline]
fn muldiv255(a: u32, b: u32) -> u8 {
    let t = a * b + 128;
    (((t >> 8) + t) >> 8) as u8
}

/// `DIV255` PIL.
#[inline]
fn div255(a: u32) -> u8 {
    let t = a + 128;
    (((t >> 8) + t) >> 8) as u8
}

impl RgbaImage {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self> {
        if data.len() != width * height * 4 {
            return Err(SynaptixError::Other(format!(
                "RgbaImage: {width}×{height} требует {} байт, пришло {}",
                width * height * 4,
                data.len()
            )));
        }
        Ok(Self { width, height, data })
    }

    /// Из тензора `[3, H, W]` (альфа 255) или `[4, H, W]` в [0, 1]: значения
    /// округляются до 1/255, как `to_pil_image`.
    pub fn from_tensor(t: &Tensor) -> Result<Self> {
        let d = t.dims().to_vec();
        let (c, h, w) = match d.as_slice() {
            [c, h, w] => (*c, *h, *w),
            [1, c, h, w] => (*c, *h, *w),
            _ => return Err(SynaptixError::Other(format!("картинка должна быть [3|4, H, W], пришло {d:?}"))),
        };
        if c != 3 && c != 4 {
            return Err(SynaptixError::Other(format!("картинка: ожидалось 3 или 4 канала, пришло {c}")));
        }
        let v = t.to_device(Device::Cpu)?.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let plane = h * w;
        let q = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
        let mut data = Vec::with_capacity(plane * 4);
        for i in 0..plane {
            data.push(q(v[i]));
            data.push(q(v[plane + i]));
            data.push(q(v[2 * plane + i]));
            data.push(if c == 4 { q(v[3 * plane + i]) } else { 255 });
        }
        Ok(Self { width: w, height: h, data })
    }

    pub fn has_alpha(&self) -> bool {
        self.data.chunks_exact(4).any(|p| p[3] != 255)
    }

    /// Порог «в картинке есть прозрачность» для выхода VAE: у непрозрачных
    /// генераций альфа шумит в 244…255, у прозрачных фон уходит в 0.
    pub const OPAQUE_ALPHA: u8 = 204;

    /// Есть ли настоящая прозрачность (альфа ниже [`Self::OPAQUE_ALPHA`]).
    pub fn has_transparency(&self) -> bool {
        self.data.chunks_exact(4).any(|p| p[3] < Self::OPAQUE_ALPHA)
    }

    /// `[4, H, W]` F32 в [0, 1] на CPU (для VAE: `preprocess` → `2x − 1` делает
    /// вызывающий).
    pub fn to_tensor(&self) -> Result<Tensor> {
        let plane = self.width * self.height;
        let mut v = vec![0f32; plane * 4];
        for (i, p) in self.data.chunks_exact(4).enumerate() {
            for c in 0..4 {
                v[c * plane + i] = p[c] as f32 / 255.0;
            }
        }
        Tensor::from_vec(v, (4, self.height, self.width), Device::Cpu)
    }

    /// `Image.resize((w, h), LANCZOS)` для режима RGBA: PIL премультиплицирует
    /// (RGBa), ресэмплит четыре канала и возвращает обратно.
    pub fn resize_lanczos(&self, nw: usize, nh: usize) -> Self {
        if nw == self.width && nh == self.height {
            return self.clone();
        }
        let plane = self.width * self.height;
        // RGBA → RGBa, планарно.
        let mut chw = vec![0u8; plane * 4];
        for (i, p) in self.data.chunks_exact(4).enumerate() {
            let a = p[3] as u32;
            chw[i] = muldiv255(p[0] as u32, a);
            chw[plane + i] = muldiv255(p[1] as u32, a);
            chw[2 * plane + i] = muldiv255(p[2] as u32, a);
            chw[3 * plane + i] = p[3];
        }
        let out = resize_u8(&chw, 4, self.height, self.width, nh, nw, Filter::Lanczos3, Precision::Pil);
        let np = nw * nh;
        let mut data = Vec::with_capacity(np * 4);
        for i in 0..np {
            let a = out[3 * np + i] as u32;
            // `rgba2rgbA` PIL: `255·v / a` без округления.
            let un = |v: u8| -> u8 {
                if a == 255 || a == 0 {
                    v
                } else {
                    ((255 * v as u32) / a).min(255) as u8
                }
            };
            data.push(un(out[i]));
            data.push(un(out[np + i]));
            data.push(un(out[2 * np + i]));
            data.push(a as u8);
        }
        Self { width: nw, height: nh, data }
    }

    /// Композит на белом (`white.paste(img, mask=alpha)`): RGB8 планарно `[3, H, W]`.
    pub fn composite_white_chw(&self) -> Vec<u8> {
        let plane = self.width * self.height;
        let mut out = vec![0u8; plane * 3];
        for (i, p) in self.data.chunks_exact(4).enumerate() {
            let a = p[3] as u32;
            for c in 0..3 {
                out[c * plane + i] = div255(255 * (255 - a) + p[c] as u32 * a);
            }
        }
        out
    }
}

/// Патчи башни зрения одной картинки: `[N, 3·T·P·P]` в порядке блоков
/// слияния и сетка `(t, h, w)` в патчах.
pub struct VisionPatches {
    pub patches: Tensor,
    pub grid: (usize, usize, usize),
}

impl VisionPatches {
    /// Токенов LLM после слияния.
    pub fn tokens(&self, merge: usize) -> usize {
        self.grid.0 * self.grid.1 * self.grid.2 / (merge * merge)
    }

    /// Сетка в токенах LLM `(h, w)`.
    pub fn llm_grid(&self, merge: usize) -> (usize, usize) {
        (self.grid.1 / merge, self.grid.2 / merge)
    }
}

/// Процессор Qwen3-VL над композитом на белом: `smart_resize` (стороны кратны
/// 32, площадь в `[min, max]`), бикубика torchvision с антиалиасом на uint8,
/// нормировка `(x/255 − 0.5)/0.5`, нарезка `[C, T, P, P]` с повтором кадра.
pub fn vision_patches(image: &RgbaImage, vc: &Qwen3VlVisionConfig, pc: &ProcessorConfig) -> Result<VisionPatches> {
    let rgb = image.composite_white_chw();
    let (h, w) = (image.height, image.width);
    let (nh, nw) = smart_resize(h, w, vc.factor(), pc.min_pixels, pc.max_pixels);
    let u8s = resize_u8(&rgb, 3, h, w, nh, nw, Filter::Bicubic, Precision::Torchvision);
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

    fn checker(w: usize, h: usize) -> RgbaImage {
        let mut d = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let a = if x < w / 2 { 255 } else { 128 };
                d.extend_from_slice(&[(x * 7 % 256) as u8, (y * 11 % 256) as u8, 200, a]);
            }
        }
        RgbaImage::new(w, h, d).unwrap()
    }

    #[test]
    fn muldiv_matches_pil() {
        // (a·b + 128)/255 с точностью PIL: 200·128/255 = 100,4 → 100.
        assert_eq!(muldiv255(200, 128), 100);
        assert_eq!(muldiv255(255, 255), 255);
        assert_eq!(muldiv255(37, 0), 0);
        assert_eq!(div255(255 * 127 + 0 * 128), 127);
    }

    #[test]
    fn resize_keeps_opaque_and_size() {
        let img = checker(40, 24);
        let r = img.resize_lanczos(20, 12);
        assert_eq!((r.width, r.height), (20, 12));
        assert!(r.data.chunks_exact(4).take(5).all(|p| p[3] == 255));
        // Без ресайза — та же картинка.
        assert_eq!(img.resize_lanczos(40, 24), img);
        // Однотонная непрозрачная картинка не меняется при любом ресайзе.
        let flat = RgbaImage::new(10, 10, vec![[9u8, 99, 199, 255]; 100].concat()).unwrap();
        let f2 = flat.resize_lanczos(7, 13);
        assert!(f2.data.chunks_exact(4).all(|p| p == [9, 99, 199, 255]));
    }

    #[test]
    fn composite_white_uses_alpha() {
        let img = RgbaImage::new(2, 1, vec![0, 0, 0, 255, 0, 0, 0, 0]).unwrap();
        let rgb = img.composite_white_chw();
        // Непрозрачный чёрный остаётся чёрным, прозрачный — белым.
        assert_eq!(rgb, vec![0, 255, 0, 255, 0, 255]);
        let half = RgbaImage::new(1, 1, vec![0, 0, 0, 128]).unwrap();
        assert_eq!(half.composite_white_chw()[0], div255(255 * 127));
    }

    #[test]
    fn tensor_roundtrip() {
        synaptix_kernels_cpu::ensure_registered();
        let img = checker(6, 4);
        let t = img.to_tensor().unwrap();
        assert_eq!(t.dims(), &[4, 4, 6]);
        let back = RgbaImage::from_tensor(&t).unwrap();
        assert_eq!(back, img);
        assert!(img.has_alpha());
        assert!(img.has_transparency());
        let noisy = RgbaImage::new(1, 2, vec![1, 2, 3, 250, 4, 5, 6, 255]).unwrap();
        assert!(noisy.has_alpha() && !noisy.has_transparency());
        let rgb = t.narrow(0, 0, 3).unwrap().contiguous().unwrap();
        assert!(!RgbaImage::from_tensor(&rgb).unwrap().has_alpha());
    }

    #[test]
    fn patches_layout() {
        synaptix_kernels_cpu::ensure_registered();
        let vc = Qwen3VlVisionConfig {
            depth: 1,
            hidden: 8,
            num_heads: 1,
            intermediate: 8,
            in_channels: 3,
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            out_hidden: 8,
            num_position_embeddings: 4,
            deepstack_indexes: vec![],
            layer_norm_eps: 1e-6,
        };
        let pc = ProcessorConfig { min_pixels: 32 * 32, max_pixels: 1 << 30, ..Default::default() };
        let img = RgbaImage::new(64, 32, vec![255u8; 64 * 32 * 4]).unwrap();
        let p = vision_patches(&img, &vc, &pc).unwrap();
        assert_eq!(p.grid, (1, 2, 4));
        assert_eq!(p.tokens(2), 2);
        assert_eq!(p.llm_grid(2), (1, 2));
        assert_eq!(p.patches.dims(), &[8, 3 * 2 * 16 * 16]);
        // Белый → (1 − 0,5)/0,5 = 1.
        let v = p.patches.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| (x - 1.0).abs() < 1e-6));
    }
}
