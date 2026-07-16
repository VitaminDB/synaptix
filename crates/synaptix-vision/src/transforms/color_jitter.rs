use rand::Rng;

use crate::error::Result;
use crate::image_buf::RgbImage;

#[derive(Debug, Clone, Copy)]
pub struct ColorJitterConfig {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
}

pub fn color_jitter<R: Rng>(img: &RgbImage, cfg: &ColorJitterConfig, rng: &mut R) -> Result<RgbImage> {
    let src = img.to_hwc();
    let b = sample_factor(cfg.brightness, rng);
    let c = sample_factor(cfg.contrast, rng);
    let s = sample_factor(cfg.saturation, rng);
    let mut out = src.clone();
    let mut global_mean = 0.0f32;
    for y in 0..src.height {
        for x in 0..src.width {
            for ch in 0..src.channels {
                global_mean += src.pixel(x, y, ch);
            }
        }
    }
    global_mean /= (src.width * src.height * src.channels) as f32;
    for y in 0..src.height {
        for x in 0..src.width {
            let mut gray = 0.0f32;
            for ch in 0..src.channels {
                gray += src.pixel(x, y, ch);
            }
            gray /= src.channels as f32;
            for ch in 0..src.channels {
                let mut v = src.pixel(x, y, ch);
                v *= b;
                v = (v - global_mean) * c + global_mean;
                v = (v - gray) * s + gray;
                out.set_pixel(x, y, ch, v.clamp(0.0, 1.0));
            }
        }
    }
    Ok(out)
}

fn sample_factor<R: Rng>(strength: f32, rng: &mut R) -> f32 {
    if strength <= 0.0 {
        return 1.0;
    }
    let lo = (1.0 - strength).max(0.0);
    let hi = 1.0 + strength;
    rng.gen_range(lo..=hi)
}
