//! Препроцесс картинки Inkling-2 (как `Inkling2ImageProcessor` transformers).
//!
//! Картинка не масштабируется: она дополняется справа и снизу до кратного
//! клетке (40 px = 2×2 патча по 20 px) значением −1 по шкале 0…255, затем
//! `x/255`, нормировка `(x − 0.5)/0.5`. Одна клетка — один токен текста.
//! Прозрачность накладывается на шахматку 204/153 (клетка 8 px), как у
//! `convert_to_rgb` оригинала. Сторона больше `max_side` (по описанию модели
//! 4096) или сверх бюджета токенов уменьшается с сохранением пропорций —
//! это уже наше правило, у оригинала его нет.

use crate::config::VisionConfig;

#[derive(Debug, thiserror::Error)]
pub enum PreprocessError {
    #[error("картинка: {0}")]
    Load(String),
}

/// Картинка, разрезанная на патчи в порядке башни.
pub struct PreparedImage {
    /// `[rows·cols·m², p·p·3]`: клетки построчно, внутри клетки — патчи 2×2
    /// построчно, патч — (строка, столбец, канал).
    pub patches: Vec<f32>,
    /// Сетка клеток (= токенов текста).
    pub rows: usize,
    pub cols: usize,
    /// Размер картинки после возможного уменьшения (высота, ширина).
    pub height: usize,
    pub width: usize,
    /// Исходный размер.
    pub orig_height: usize,
    pub orig_width: usize,
}

impl PreparedImage {
    pub fn tokens(&self) -> usize {
        self.rows * self.cols
    }
}

/// Ограничения размера: сторона и число токенов (клеток) на картинку.
#[derive(Debug, Clone, Copy)]
pub struct ImageLimits {
    pub max_side: usize,
    pub max_tokens: usize,
}

impl Default for ImageLimits {
    fn default() -> Self {
        // 4096 px по стороне — предел из описания модели; 103×103 клеток.
        Self { max_side: 4096, max_tokens: usize::MAX }
    }
}

/// Размер после ограничений (h, w): пропорции сохраняются, меньше не бывает 1 px.
pub fn fit_size(h: usize, w: usize, cell: usize, lim: ImageLimits) -> (usize, usize) {
    let mut s = 1.0f64;
    let side = h.max(w) as f64;
    if side > lim.max_side as f64 {
        s = lim.max_side as f64 / side;
    }
    let tokens = |s: f64| {
        let nh = ((h as f64 * s).round() as usize).max(1);
        let nw = ((w as f64 * s).round() as usize).max(1);
        nh.div_ceil(cell) * nw.div_ceil(cell)
    };
    while tokens(s) > lim.max_tokens && s > 1e-3 {
        s *= 0.98;
    }
    if s >= 1.0 {
        return (h, w);
    }
    (((h as f64 * s).round() as usize).max(1), ((w as f64 * s).round() as usize).max(1))
}

/// Прочитать файл (PNG/JPEG/WebP) и подготовить.
pub fn load_image(path: impl AsRef<std::path::Path>, cfg: &VisionConfig, lim: ImageLimits) -> Result<PreparedImage, PreprocessError> {
    let path = path.as_ref();
    let img = image::ImageReader::open(path)
        .map_err(|e| PreprocessError::Load(format!("{}: {e}", path.display())))?
        .with_guessed_format()
        .map_err(|e| PreprocessError::Load(format!("{}: {e}", path.display())))?
        .decode()
        .map_err(|e| PreprocessError::Load(format!("{}: {e}", path.display())))?;
    Ok(prepare_dynamic(img, cfg, lim))
}

/// Подготовить декодированную картинку.
pub fn prepare_dynamic(img: image::DynamicImage, cfg: &VisionConfig, lim: ImageLimits) -> PreparedImage {
    let (ow, oh) = (img.width() as usize, img.height() as usize);
    let alpha = img.color().has_alpha();
    let (nh, nw) = fit_size(oh, ow, cfg.cell(), lim);
    let img = if (nh, nw) != (oh, ow) {
        img.resize_exact(nw as u32, nh as u32, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    let rgb = if alpha { composite_checkerboard(&img.to_rgba8()) } else { img.to_rgb8().into_raw() };
    let mut p = prepare_rgb8(&rgb, nh, nw, cfg);
    p.orig_height = oh;
    p.orig_width = ow;
    p
}

/// RGBA → RGB поверх шахматки 204/153 с клеткой 8 px (целочисленно, как оригинал).
fn composite_checkerboard(rgba: &image::RgbaImage) -> Vec<u8> {
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    let raw = rgba.as_raw();
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let bg: u32 = if (y / 8 + x / 8) % 2 == 0 { 204 } else { 153 };
            let i = (y * w + x) * 4;
            let a = raw[i + 3] as u32;
            for c in 0..3 {
                out[(y * w + x) * 3 + c] = ((raw[i + c] as u32 * a + bg * (255 - a) + 127) / 255) as u8;
            }
        }
    }
    out
}

/// RGB8 `[h, w, 3]` → патчи башни.
pub fn prepare_rgb8(rgb: &[u8], h: usize, w: usize, cfg: &VisionConfig) -> PreparedImage {
    let cell = cfg.cell();
    let (p, m) = (cfg.native_patch_size, cfg.adapter_merge_size);
    let rows = h.div_ceil(cell).max(1);
    let cols = w.div_ceil(cell).max(1);
    let pd = p * p * 3;
    // Паддинг −1 по шкале 0…255 после rescale/normalize.
    let pad = (-1.0f32 / 255.0 - 0.5) / 0.5;
    let mut lut = [0f32; 256];
    for (v, o) in lut.iter_mut().enumerate() {
        *o = (v as f32 * (1.0 / 255.0) - 0.5) / 0.5;
    }
    let mut patches = vec![0f32; rows * cols * m * m * pd];
    let mut idx = 0usize;
    for r in 0..rows {
        for c in 0..cols {
            for mh in 0..m {
                for mw in 0..m {
                    let y0 = r * cell + mh * p;
                    let x0 = c * cell + mw * p;
                    for py in 0..p {
                        let y = y0 + py;
                        for px in 0..p {
                            let x = x0 + px;
                            for ch in 0..3 {
                                patches[idx] = if y < h && x < w { lut[rgb[(y * w + x) * 3 + ch] as usize] } else { pad };
                                idx += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    PreparedImage { patches, rows, cols, height: h, width: w, orig_height: h, orig_width: w }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VisionConfig {
        serde_json::from_str(
            r#"{"hidden_size":1024,"num_hidden_layers":23,"num_attention_heads":16,"native_patch_size":20,
                "adapter_merge_size":2,"posemb_grid_size":32,"projection_hidden_dim":4096,"text_hidden_size":2304}"#,
        )
        .unwrap()
    }

    #[test]
    fn grid_and_padding() {
        let c = cfg();
        let (h, w) = (41, 85);
        let rgb = vec![255u8; h * w * 3];
        let p = prepare_rgb8(&rgb, h, w, &c);
        assert_eq!((p.rows, p.cols), (2, 3));
        assert_eq!(p.patches.len(), 6 * 4 * 1200);
        // клетка (0,0), патч (0,0), пиксель (0,0) — белый = 1
        assert!((p.patches[0] - 1.0).abs() < 1e-6);
        // клетка (1,0) — строки 40..80, у картинки есть только строка 40
        let cell10 = 3 * 4 * 1200;
        assert!((p.patches[cell10] - 1.0).abs() < 1e-6);
        assert!((p.patches[cell10 + 20 * 3] + 1.0078431).abs() < 1e-5);
    }

    #[test]
    fn fit_keeps_ratio() {
        assert_eq!(fit_size(300, 4096, 40, ImageLimits::default()), (300, 4096));
        assert_eq!(fit_size(8192, 4096, 40, ImageLimits::default()), (4096, 2048));
        let (h, w) = fit_size(4096, 4096, 40, ImageLimits { max_side: 4096, max_tokens: 2500 });
        assert!(h.div_ceil(40) * w.div_ceil(40) <= 2500 && h == w);
    }
}
