use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::config::VisionConfig;

#[derive(Debug, Clone, Copy)]
pub struct ImageGrid {
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

impl ImageGrid {
    pub fn patches(&self) -> usize {
        self.t * self.h * self.w
    }
    pub fn tokens(&self, merge: usize) -> usize {
        self.patches() / (merge * merge)
    }
}

pub struct PreparedImage {
    pub patches: Tensor,
    pub grid: ImageGrid,
}

#[derive(Debug, Clone, Copy)]
pub struct PreprocessLimits {
    pub min_pixels: usize,
    pub max_pixels: usize,
}

impl Default for PreprocessLimits {
    fn default() -> Self {
        Self {
            min_pixels: 256 * 256,
            max_pixels: 1024 * 1024,
        }
    }
}

pub fn smart_resize(
    h: usize,
    w: usize,
    factor: usize,
    limits: PreprocessLimits,
) -> (usize, usize) {
    let round_to = |v: f64| -> usize {
        let r = (v / factor as f64).round() as usize;
        r.max(1) * factor
    };
    let mut hb = round_to(h as f64);
    let mut wb = round_to(w as f64);
    let area = hb * wb;
    if area > limits.max_pixels {
        let beta = ((h * w) as f64 / limits.max_pixels as f64).sqrt();
        hb = (((h as f64 / beta) / factor as f64).floor() as usize).max(1) * factor;
        wb = (((w as f64 / beta) / factor as f64).floor() as usize).max(1) * factor;
    } else if area < limits.min_pixels {
        let beta = (limits.min_pixels as f64 / (h * w) as f64).sqrt();
        hb = (((h as f64 * beta) / factor as f64).ceil() as usize).max(1) * factor;
        wb = (((w as f64 * beta) / factor as f64).ceil() as usize).max(1) * factor;
    }
    (hb, wb)
}

pub fn patchify(
    chw: &[f32],
    c: usize,
    h: usize,
    w: usize,
    cfg: &VisionConfig,
) -> (Vec<f32>, ImageGrid) {
    let p = cfg.patch_size;
    let m = cfg.spatial_merge_size;
    let tps = cfg.temporal_patch_size;
    let gh = h / p;
    let gw = w / p;
    let feat = c * tps * p * p;
    let n = gh * gw;
    let mut out = vec![0f32; n * feat];

    let mut token = 0usize;
    for bh in 0..gh / m {
        for bw in 0..gw / m {
            for mh in 0..m {
                for mw in 0..m {
                    let ph = bh * m + mh;
                    let pw = bw * m + mw;
                    let base = token * feat;
                    let mut k = 0usize;
                    for ci in 0..c {
                        for _t in 0..tps {
                            for y in 0..p {
                                let row = (ci * h + ph * p + y) * w + pw * p;
                                out[base + k..base + k + p]
                                    .copy_from_slice(&chw[row..row + p]);
                                k += p;
                            }
                        }
                    }
                    token += 1;
                }
            }
        }
    }
    (out, ImageGrid { t: 1, h: gh, w: gw })
}

pub fn prepare_image(
    path: impl AsRef<std::path::Path>,
    cfg: &VisionConfig,
    limits: PreprocessLimits,
    device: Device,
) -> Result<PreparedImage, PreprocessError> {
    let img = synaptix_io::image::png::load_image(path, Device::Cpu)
        .map_err(|e| PreprocessError::Load(e.to_string()))?;
    prepare_tensor(&img, cfg, limits, device)
}

pub fn prepare_tensor(
    chw: &Tensor,
    cfg: &VisionConfig,
    limits: PreprocessLimits,
    device: Device,
) -> Result<PreparedImage, PreprocessError> {
    let dims = chw.dims();
    if dims.len() != 3 || dims[0] < 3 {
        return Err(PreprocessError::Shape(format!(
            "ожидался [C>=3, H, W], получено {dims:?}"
        )));
    }
    let (h, w) = (dims[1], dims[2]);
    let (nh, nw) = smart_resize(h, w, cfg.size_factor(), limits);
    let rgb = if dims[0] == 3 {
        chw.clone()
    } else {
        chw.narrow(0, 0, 3)
            .and_then(|t| t.contiguous())
            .map_err(|e| PreprocessError::Shape(e.to_string()))?
    };
    let resized = if (nh, nw) == (h, w) {
        rgb
    } else {
        synaptix_io::image::augment::resize_bilinear(&rgb, nh, nw)
            .map_err(|e| PreprocessError::Load(e.to_string()))?
    };
    let mut flat = resized
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| PreprocessError::Shape(e.to_string()))?;
    for v in flat.iter_mut() {
        *v = (*v - 0.5) / 0.5;
    }
    let (patches, grid) = patchify(&flat, 3, nh, nw, cfg);
    let n = grid.patches();
    let tensor = Tensor::from_vec(patches, vec![n, cfg.patch_features()], device)
        .map_err(|e| PreprocessError::Shape(e.to_string()))?;
    Ok(PreparedImage {
        patches: tensor,
        grid,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum PreprocessError {
    #[error("image load: {0}")]
    Load(String),
    #[error("image shape: {0}")]
    Shape(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VisionConfig {
        VisionConfig::default()
    }

    #[test]
    fn smart_resize_rounds_to_factor() {
        let (h, w) = smart_resize(700, 500, 32, PreprocessLimits::default());
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        assert!(h * w <= 1024 * 1024);
    }

    #[test]
    fn smart_resize_upscales_tiny_images() {
        let (h, w) = smart_resize(40, 40, 32, PreprocessLimits::default());
        assert!(h * w >= 256 * 256);
        assert_eq!(h % 32, 0);
    }

    #[test]
    fn smart_resize_downscales_huge_images() {
        let (h, w) = smart_resize(8000, 6000, 32, PreprocessLimits::default());
        assert!(h * w <= 1024 * 1024);
    }

    #[test]
    fn patchify_groups_merge_blocks_consecutively() {
        let c = cfg();
        let (p, m) = (c.patch_size, c.spatial_merge_size);
        let (h, w) = (p * m * 2, p * m * 2);
        let mut chw = vec![0f32; 3 * h * w];
        for (i, v) in chw.iter_mut().enumerate() {
            *v = i as f32;
        }
        let (out, grid) = patchify(&chw, 3, h, w, &c);
        assert_eq!(grid.h, h / p);
        assert_eq!(grid.w, w / p);
        assert_eq!(out.len(), grid.patches() * c.patch_features());

        let feat = c.patch_features();
        let first_px = |token: usize| out[token * feat];
        assert_eq!(first_px(0), 0.0);
        assert_eq!(first_px(1), (p * 1) as f32);
        assert_eq!(first_px(2), (p * w) as f32);
        assert_eq!(first_px(3), (p * w + p) as f32);
        assert_eq!(first_px(4), (p * m) as f32);
    }

    #[test]
    fn patchify_repeats_temporal_slice() {
        let c = cfg();
        let (p, m) = (c.patch_size, c.spatial_merge_size);
        let (h, w) = (p * m, p * m);
        let chw: Vec<f32> = (0..3 * h * w).map(|i| i as f32).collect();
        let (out, _) = patchify(&chw, 3, h, w, &c);
        let per_slice = p * p;
        assert_eq!(out[0], out[per_slice]);
        assert_eq!(out[per_slice - 1], out[2 * per_slice - 1]);
    }
}
