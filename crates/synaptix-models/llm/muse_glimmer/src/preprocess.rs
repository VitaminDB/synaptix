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

pub fn smart_resize(h: usize, w: usize, unit: usize, max_tokens: usize) -> (usize, usize) {
    let ideal_h = h as f64 / unit as f64;
    let ideal_w = w as f64 / unit as f64;
    let ratio = if ideal_h > 0.0 { ideal_w / ideal_h } else { 1.0 };
    let (ideal_h, ideal_w) = if ideal_h * ideal_w > max_tokens as f64 {
        let ih = (max_tokens as f64 / ratio).sqrt();
        (ih, ih * ratio)
    } else {
        (ideal_h, ideal_w)
    };
    let mut candidates = Vec::new();
    for gh in [ideal_h.floor() as isize, ideal_h.ceil() as isize] {
        for gw in [ideal_w.floor() as isize, ideal_w.ceil() as isize] {
            if gh >= 1 && gw >= 1 && (gh * gw) as usize <= max_tokens {
                let c = (gh as usize, gw as usize);
                if !candidates.contains(&c) {
                    candidates.push(c);
                }
            }
        }
    }
    if candidates.is_empty() {
        candidates.push((
            (ideal_h.round() as usize).max(1),
            (ideal_w.round() as usize).max(1),
        ));
    }
    let target_ratio = h as f64 / w as f64;
    let (gh, gw) = candidates
        .into_iter()
        .min_by(|a, b| {
            let da = (a.0 as f64 / a.1 as f64 - target_ratio).abs();
            let db = (b.0 as f64 / b.1 as f64 - target_ratio).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    (gh * unit, gw * unit)
}

pub fn patchify(chw: &[f32], c: usize, h: usize, w: usize, cfg: &VisionConfig) -> (Vec<f32>, ImageGrid) {
    let p = cfg.patch_size;
    let tps = cfg.patch_temporal;
    let gh = h / p;
    let gw = w / p;
    let feat = tps * c * p * p;
    let n = gh * gw;
    let mut out = vec![0f32; n * feat];
    for ph in 0..gh {
        for pw in 0..gw {
            let token = ph * gw + pw;
            let base = token * feat;
            let mut k = 0usize;
            for _t in 0..tps {
                for ci in 0..c {
                    for y in 0..p {
                        let row = (ci * h + ph * p + y) * w + pw * p;
                        out[base + k..base + k + p].copy_from_slice(&chw[row..row + p]);
                        k += p;
                    }
                }
            }
        }
    }
    (out, ImageGrid { t: 1, h: gh, w: gw })
}

pub fn prepare_image(
    path: impl AsRef<std::path::Path>,
    cfg: &VisionConfig,
    device: Device,
) -> Result<PreparedImage, PreprocessError> {
    let img = synaptix_io::image::png::load_image(path, Device::Cpu)
        .map_err(|e| PreprocessError::Load(e.to_string()))?;
    prepare_tensor(&img, cfg, device)
}

pub fn prepare_tensor(
    chw: &Tensor,
    cfg: &VisionConfig,
    device: Device,
) -> Result<PreparedImage, PreprocessError> {
    let dims = chw.dims();
    if dims.len() != 3 || dims[0] < 3 {
        return Err(PreprocessError::Shape(format!(
            "ожидался [C>=3, H, W], получено {dims:?}"
        )));
    }
    let (h, w) = (dims[1], dims[2]);
    let unit = cfg.patch_size * cfg.merge_size;
    let (nh, nw) = smart_resize(h, w, unit, cfg.max_image_tokens);
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
    Ok(PreparedImage { patches: tensor, grid })
}

pub struct PreparedVideo {
    pub patches: Tensor,
    pub grid: ImageGrid,
    pub group_timestamps: Vec<f32>,
}

pub const VIDEO_TARGET_FPS: f32 = 2.0;
pub const VIDEO_MAX_FRAMES: usize = 96;
pub const VIDEO_MAX_FRAME_TOKENS: usize = 144;

pub fn sample_frame_indices(total: usize, src_fps: f32, tps: usize) -> Vec<usize> {
    if total == 0 {
        return Vec::new();
    }
    let by_fps = ((total as f32) * VIDEO_TARGET_FPS / src_fps.max(0.01)) as usize;
    let mut n = by_fps.min(VIDEO_MAX_FRAMES).min(total);
    n = ((n / tps) * tps).max(tps);
    n = n.min(total);
    if n <= 1 {
        return vec![0];
    }
    (0..n)
        .map(|k| ((k as f64) * (total - 1) as f64 / (n - 1) as f64) as usize)
        .collect()
}

pub fn patchify_video(frames: &[Vec<f32>], h: usize, w: usize, cfg: &VisionConfig) -> (Vec<f32>, ImageGrid) {
    let p = cfg.patch_size;
    let tps = cfg.patch_temporal;
    let c = 3usize;
    let gh = h / p;
    let gw = w / p;
    let grid_t = frames.len().div_ceil(tps);
    let feat = tps * c * p * p;
    let mut out = vec![0f32; grid_t * gh * gw * feat];
    for gt in 0..grid_t {
        for ph in 0..gh {
            for pw in 0..gw {
                let token = (gt * gh + ph) * gw + pw;
                let base = token * feat;
                let mut k = 0usize;
                for f in 0..tps {
                    let fi = (gt * tps + f).min(frames.len() - 1);
                    let chw = &frames[fi];
                    for ci in 0..c {
                        for y in 0..p {
                            let row = (ci * h + ph * p + y) * w + pw * p;
                            out[base + k..base + k + p].copy_from_slice(&chw[row..row + p]);
                            k += p;
                        }
                    }
                }
            }
        }
    }
    (out, ImageGrid { t: grid_t, h: gh, w: gw })
}

fn probe_video(path: &std::path::Path) -> Result<(usize, usize, f32, usize), PreprocessError> {
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets",
            "-of", "csv=p=0",
        ])
        .arg(path)
        .output()
        .map_err(|e| PreprocessError::Load(format!("ffprobe: {e}")))?;
    if !out.status.success() {
        return Err(PreprocessError::Load(format!(
            "ffprobe: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().next().unwrap_or("");
    let parts: Vec<&str> = line.trim().split(',').collect();
    if parts.len() < 4 {
        return Err(PreprocessError::Load(format!("ffprobe вывод: {line}")));
    }
    let w: usize = parts[0].parse().map_err(|_| PreprocessError::Load("width".into()))?;
    let h: usize = parts[1].parse().map_err(|_| PreprocessError::Load("height".into()))?;
    let fps = match parts[2].split_once('/') {
        Some((n, d)) => {
            let n: f32 = n.parse().unwrap_or(0.0);
            let d: f32 = d.parse().unwrap_or(1.0);
            if d > 0.0 { n / d } else { 25.0 }
        }
        None => parts[2].parse().unwrap_or(25.0),
    };
    let total: usize = parts[3].parse().map_err(|_| PreprocessError::Load("nb_read_packets".into()))?;
    Ok((w, h, fps, total))
}

fn decode_frames(
    path: &std::path::Path,
    indices: &[usize],
    w: usize,
    h: usize,
) -> Result<Vec<Vec<u8>>, PreprocessError> {
    let select: Vec<String> = indices.iter().map(|i| format!("eq(n\\,{i})")).collect();
    let vf = format!("select='{}'", select.join("+"));
    let out = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-vf", &vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .map_err(|e| PreprocessError::Load(format!("ffmpeg: {e}")))?;
    if !out.status.success() {
        return Err(PreprocessError::Load(format!(
            "ffmpeg: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let frame_bytes = w * h * 3;
    let got = out.stdout.len() / frame_bytes;
    if got == 0 || out.stdout.len() % frame_bytes != 0 {
        return Err(PreprocessError::Load(format!(
            "ffmpeg rawvideo: {} байт не кратно кадру {frame_bytes}",
            out.stdout.len()
        )));
    }
    Ok(out.stdout.chunks(frame_bytes).map(|c| c.to_vec()).collect())
}

pub fn prepare_video(
    path: impl AsRef<std::path::Path>,
    cfg: &VisionConfig,
    device: Device,
) -> Result<PreparedVideo, PreprocessError> {
    let path = path.as_ref();
    let (w, h, src_fps, total) = probe_video(path)?;
    if total == 0 {
        return Err(PreprocessError::Load("видео без кадров".into()));
    }
    let mut indices = sample_frame_indices(total, src_fps, cfg.patch_temporal);
    let raw = decode_frames(path, &indices, w, h)?;
    indices.truncate(raw.len());

    let unit = cfg.patch_size * cfg.merge_size;
    let (nh, nw) = smart_resize(h, w, unit, VIDEO_MAX_FRAME_TOKENS);

    let mut frames: Vec<Vec<f32>> = Vec::with_capacity(raw.len());
    for rgb in &raw {
        let mut chw = vec![0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                let src = (y * w + x) * 3;
                chw[y * w + x] = rgb[src] as f32 / 255.0;
                chw[h * w + y * w + x] = rgb[src + 1] as f32 / 255.0;
                chw[2 * h * w + y * w + x] = rgb[src + 2] as f32 / 255.0;
            }
        }
        let t = Tensor::from_vec(chw, vec![3, h, w], Device::Cpu)
            .map_err(|e| PreprocessError::Shape(e.to_string()))?;
        let resized = if (nh, nw) == (h, w) {
            t
        } else {
            synaptix_io::image::augment::resize_bilinear(&t, nh, nw)
                .map_err(|e| PreprocessError::Load(e.to_string()))?
        };
        let mut flat = resized
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| PreprocessError::Shape(e.to_string()))?;
        for v in flat.iter_mut() {
            *v = (*v - 0.5) / 0.5;
        }
        frames.push(flat);
    }

    let (patches, grid) = patchify_video(&frames, nh, nw, cfg);
    let group_timestamps: Vec<f32> = (0..grid.t)
        .map(|g| {
            let fi = (g * cfg.patch_temporal).min(indices.len() - 1);
            indices[fi] as f32 / src_fps.max(0.01)
        })
        .collect();
    let n = grid.patches();
    let tensor = Tensor::from_vec(patches, vec![n, cfg.patch_features()], device)
        .map_err(|e| PreprocessError::Shape(e.to_string()))?;
    Ok(PreparedVideo { patches: tensor, grid, group_timestamps })
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
        VisionConfig {
            hidden_size: 1536,
            num_hidden_layers: 50,
            intermediate_size: 8960,
            num_attention_heads: 16,
            patch_size: 14,
            patch_temporal: 2,
            merge_size: 2,
            pos_emb_side: 32,
            layer_norm_eps: 1e-5,
            full_layers: (0..50).map(|i| (i + 1) % 4 == 0 || i == 49).collect(),
            rope_theta: 10_000.0,
            out_hidden_size: 6144,
            projector_hidden_size: 4096,
            max_image_tokens: 4096,
        }
    }

    #[test]
    fn smart_resize_keeps_aspect_under_cap() {
        let (h, w) = smart_resize(700, 500, 28, 4096);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        assert!((h / 28) * (w / 28) <= 4096);
        let r_in = 700.0 / 500.0;
        let r_out = h as f64 / w as f64;
        assert!((r_in - r_out).abs() / r_in < 0.1);
    }

    #[test]
    fn smart_resize_downscales_huge() {
        let (h, w) = smart_resize(8000, 8000, 28, 4096);
        assert!((h / 28) * (w / 28) <= 4096);
    }

    #[test]
    fn patchify_is_raster_and_temporal_channel_major() {
        let c = cfg();
        let p = c.patch_size;
        let (h, w) = (p * 2, p * 2);
        let chw: Vec<f32> = (0..3 * h * w).map(|i| i as f32).collect();
        let (out, grid) = patchify(&chw, 3, h, w, &c);
        assert_eq!(grid.h, 2);
        assert_eq!(grid.w, 2);
        let feat = c.patch_features();
        assert_eq!(out.len(), 4 * feat);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1 * feat], p as f32);
        assert_eq!(out[2 * feat], (p * w) as f32);
        let per_frame = 3 * p * p;
        assert_eq!(out[0], out[per_frame]);
        assert_eq!(out[per_frame - 1], out[2 * per_frame - 1]);
        let ch1 = p * p;
        assert_eq!(out[ch1], (h * w) as f32);
    }
}
