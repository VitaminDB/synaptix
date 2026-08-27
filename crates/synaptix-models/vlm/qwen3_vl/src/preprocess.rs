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

// ─────────────────────────────── Видео ───────────────────────────────

/// Лимиты сэмплинга видео под бюджет контекста чата.
///
/// HF-процессор Qwen3-VL берёт 2 fps и до сотен кадров при общем бюджете
/// ~24k токенов; в чате с KV-рингом на 12k таких токенов нет, поэтому
/// потолок кадров и токенов здесь скромнее, а бюджет делится между
/// группами кадров (группа = `temporal_patch_size` соседних кадров → один
/// набор патчей).
#[derive(Debug, Clone, Copy)]
pub struct VideoLimits {
    /// Целевая частота сэмплинга, кадров/с.
    pub target_fps: f32,
    /// Потолок кадров; длинное видео режется равномерно по всей длине.
    pub max_frames: usize,
    /// Потолок vision-токенов на всё видео.
    pub max_total_tokens: usize,
    /// Нижняя планка токенов на группу кадров — чтобы долгое видео не
    /// схлопнулось в неразличимые миниатюры (тогда лучше меньше кадров).
    pub min_group_tokens: usize,
}

impl Default for VideoLimits {
    fn default() -> Self {
        Self {
            target_fps: 2.0,
            max_frames: 64,
            max_total_tokens: 4096,
            min_group_tokens: 64,
        }
    }
}

pub struct PreparedVideo {
    /// `[t·h·w, patch_features]` — патчи всех групп кадров подряд.
    pub patches: Tensor,
    pub grid: ImageGrid,
    /// Таймкод первого кадра каждой группы, секунды — идёт в промпт
    /// (`<{t:.1} seconds>` перед блоком группы).
    pub group_timestamps: Vec<f32>,
}

/// Индексы кадров для сэмплинга: по `target_fps`, не больше `max_frames`,
/// кратно `tps` (группа кадров = один temporal-патч), равномерно по длине.
pub fn sample_frame_indices(
    total: usize,
    src_fps: f32,
    limits: &VideoLimits,
    tps: usize,
) -> Vec<usize> {
    if total == 0 {
        return Vec::new();
    }
    let tps = tps.max(1);
    let by_fps = ((total as f32) * limits.target_fps / src_fps.max(0.01)) as usize;
    let mut n = by_fps.min(limits.max_frames.max(tps)).min(total);
    n = ((n / tps) * tps).max(tps);
    n = n.min(total);
    if n <= 1 {
        return vec![0];
    }
    (0..n)
        .map(|k| ((k as f64) * (total - 1) as f64 / (n - 1) as f64) as usize)
        .collect()
}

/// Патчи видео: кадры (CHW, уже нормализованные) группами по
/// `temporal_patch_size`; порядок токенов внутри группы — как у
/// [`patchify`] (по merge-блокам, иначе merger склеит не те 2×2), порядок
/// признаков в патче — `(c, t, y, x)`, как у весов `patch_embed`.
/// Неполная последняя группа добивается повтором последнего кадра.
pub fn patchify_video(
    frames: &[Vec<f32>],
    c: usize,
    h: usize,
    w: usize,
    cfg: &VisionConfig,
) -> (Vec<f32>, ImageGrid) {
    let p = cfg.patch_size;
    let m = cfg.spatial_merge_size;
    let tps = cfg.temporal_patch_size.max(1);
    let gh = h / p;
    let gw = w / p;
    let grid_t = frames.len().div_ceil(tps).max(1);
    let feat = c * tps * p * p;
    let per_group = gh * gw;
    let mut out = vec![0f32; grid_t * per_group * feat];
    let mut token = 0usize;
    for gt in 0..grid_t {
        for bh in 0..gh / m {
            for bw in 0..gw / m {
                for mh in 0..m {
                    for mw in 0..m {
                        let ph = bh * m + mh;
                        let pw = bw * m + mw;
                        let base = token * feat;
                        let mut k = 0usize;
                        for ci in 0..c {
                            for t in 0..tps {
                                let fi = (gt * tps + t).min(frames.len().saturating_sub(1));
                                let chw = &frames[fi];
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
    }
    (out, ImageGrid { t: grid_t, h: gh, w: gw })
}

/// `(w, h, fps, frames)` первого видеопотока через ffprobe.
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
    let total: usize = parts[3]
        .parse()
        .map_err(|_| PreprocessError::Load("nb_read_packets".into()))?;
    Ok((w, h, fps, total))
}

/// Выбранные кадры одним проходом ffmpeg (`select=eq(n,i)+…`), уже
/// уменьшенные до `nh×nw` — так час видео не гоняется через память в
/// исходном разрешении.
fn decode_frames(
    path: &std::path::Path,
    indices: &[usize],
    nw: usize,
    nh: usize,
) -> Result<Vec<Vec<u8>>, PreprocessError> {
    let select: Vec<String> = indices.iter().map(|i| format!("eq(n\\,{i})")).collect();
    let vf = format!("select='{}',scale={nw}:{nh}:flags=bilinear", select.join("+"));
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
    let frame_bytes = nw * nh * 3;
    if out.stdout.is_empty() || out.stdout.len() % frame_bytes != 0 {
        return Err(PreprocessError::Load(format!(
            "ffmpeg rawvideo: {} байт не кратно кадру {frame_bytes}",
            out.stdout.len()
        )));
    }
    Ok(out.stdout.chunks(frame_bytes).map(|c| c.to_vec()).collect())
}

/// Видео → патчи башни: сэмплинг кадров, общий для всех кадров
/// `smart_resize` под бюджет токенов на группу, нормализация как у
/// картинок, [`patchify_video`].
pub fn prepare_video(
    path: impl AsRef<std::path::Path>,
    cfg: &VisionConfig,
    limits: VideoLimits,
    device: Device,
) -> Result<PreparedVideo, PreprocessError> {
    let path = path.as_ref();
    let (w, h, src_fps, total) = probe_video(path)?;
    if total == 0 || w == 0 || h == 0 {
        return Err(PreprocessError::Load("видео без кадров".into()));
    }
    let tps = cfg.temporal_patch_size.max(1);
    let indices = sample_frame_indices(total, src_fps, &limits, tps);
    let groups = indices.len().div_ceil(tps).max(1);

    // Бюджет токенов делится между группами; один merged-токен = unit² px.
    let unit = cfg.size_factor();
    let per_group = (limits.max_total_tokens / groups).max(limits.min_group_tokens).max(1);
    let px = PreprocessLimits {
        max_pixels: per_group * unit * unit,
        min_pixels: (limits.min_group_tokens * unit * unit).min(per_group * unit * unit),
    };
    let (nh, nw) = smart_resize(h, w, unit, px);

    let raw = decode_frames(path, &indices, nw, nh)?;
    let got = raw.len().min(indices.len());
    let mut frames: Vec<Vec<f32>> = Vec::with_capacity(got);
    for rgb in raw.iter().take(got) {
        let mut chw = vec![0f32; 3 * nh * nw];
        for y in 0..nh {
            for x in 0..nw {
                let src = (y * nw + x) * 3;
                for ci in 0..3 {
                    chw[ci * nh * nw + y * nw + x] = (rgb[src + ci] as f32 / 255.0 - 0.5) / 0.5;
                }
            }
        }
        frames.push(chw);
    }
    let (patches, grid) = patchify_video(&frames, 3, nh, nw, cfg);
    let group_timestamps: Vec<f32> = (0..grid.t)
        .map(|g| {
            let fi = (g * tps).min(got.saturating_sub(1));
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
        VisionConfig::default()
    }

    /// Картинка — это «видео» из одного кадра, повторённого на temporal-ось:
    /// patchify_video на паре одинаковых кадров обязан совпасть с patchify.
    #[test]
    fn patchify_video_matches_image_patchify_on_repeated_frame() {
        let c = cfg();
        let (h, w) = (64usize, 96usize);
        let frame: Vec<f32> = (0..3 * h * w).map(|i| (i % 97) as f32 / 97.0).collect();
        let (img, g_img) = patchify(&frame, 3, h, w, &c);
        let (vid, g_vid) = patchify_video(&[frame.clone(), frame.clone()], 3, h, w, &c);
        assert_eq!((g_vid.t, g_vid.h, g_vid.w), (1, g_img.h, g_img.w));
        assert_eq!(img, vid);
        // Три кадра → две группы, последняя добита повтором третьего.
        let (vid3, g3) = patchify_video(&[frame.clone(), frame.clone(), frame.clone()], 3, h, w, &c);
        assert_eq!(g3.t, 2);
        assert_eq!(vid3.len(), 2 * img.len());
        assert_eq!(&vid3[..img.len()], &img[..]);
        assert_eq!(&vid3[img.len()..], &img[..]);
    }

    #[test]
    fn sample_frame_indices_respects_fps_cap_and_group_size() {
        let lim = VideoLimits { target_fps: 2.0, max_frames: 64, ..VideoLimits::default() };
        // 2:11 при 25 fps ≈ 3275 кадров → 2 fps даёт 262, режем до 64.
        let idx = sample_frame_indices(3275, 25.0, &lim, 2);
        assert_eq!(idx.len(), 64);
        assert_eq!(idx[0], 0);
        assert_eq!(*idx.last().unwrap(), 3274);
        assert!(idx.windows(2).all(|p| p[0] < p[1]));
        // Короткий ролик: 10 кадров при 10 fps → 2 кадра (кратно tps).
        let idx = sample_frame_indices(10, 10.0, &lim, 2);
        assert_eq!(idx.len(), 2);
        // Один кадр — один индекс.
        assert_eq!(sample_frame_indices(1, 30.0, &lim, 2), vec![0]);
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
