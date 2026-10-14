//! Препроцессор картинок MiniCPM-V 4.6/4.7 — порт `MiniCPMV4_6ImageProcessor`.
//!
//! Картинка превращается в миниатюру (ресайз под `scale_resolution`² пикселей
//! с сохранением пропорций) и, если она крупнее, — в сетку срезов
//! (`get_sliced_grid`: до `max_slice_nums` кусков, сетка по log-отношению
//! сторон). Все стороны кратны `patch·4` (56 px): два последовательных
//! объединения 2×2. Ресайз — бикубический с антиалиасингом, как
//! `torchvision.transforms.functional.resize(antialias=True)`; нормализация
//! `(x − 0.5)/0.5`; каждый кроп — патчи `[h·w, 3·14·14]` в растровом порядке,
//! признаки патча в порядке `(c, y, x)`.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::config::MiniCpmConfig;

/// Сетка одного кропа в патчах.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropGrid {
    pub h: usize,
    pub w: usize,
}

impl CropGrid {
    pub fn patches(&self) -> usize {
        self.h * self.w
    }

    /// Токенов LLM после объединения `merge_factor × merge_factor`.
    pub fn tokens(&self, merge_factor: usize) -> usize {
        self.patches() / (merge_factor * merge_factor)
    }
}

/// Раскладка картинки: миниатюра и (если есть) `rows × cols` срезов одной
/// сетки.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageLayout {
    pub thumb: CropGrid,
    pub slice: Option<CropGrid>,
    pub rows: usize,
    pub cols: usize,
}

impl ImageLayout {
    /// Кропы в порядке подачи в башню и в промпт: миниатюра, затем срезы
    /// построчно.
    pub fn crops(&self) -> Vec<CropGrid> {
        let mut v = vec![self.thumb];
        if let Some(s) = self.slice {
            v.extend(std::iter::repeat_n(s, self.rows * self.cols));
        }
        v
    }

    pub fn has_slices(&self) -> bool {
        self.slice.is_some() && self.rows > 0 && self.cols > 0
    }

    /// Всего строк эмбеддингов (токенов-заполнителей) картинки.
    pub fn tokens(&self, merge_factor: usize) -> usize {
        self.crops().iter().map(|c| c.tokens(merge_factor)).sum()
    }
}

pub struct PreparedImage {
    /// Патчи каждого кропа `[h·w, 588]` (F32 на `device`), в порядке
    /// [`ImageLayout::crops`].
    pub crops: Vec<Tensor>,
    pub layout: ImageLayout,
}

#[derive(Debug, Clone, Copy)]
pub struct PreprocessLimits {
    /// Наибольшее число срезов (`max_slice_nums`).
    pub max_slice_nums: usize,
    /// Целевая сторона миниатюры и среза (`scale_resolution`).
    pub scale_resolution: usize,
    /// Резать ли крупные картинки на срезы.
    pub slice_mode: bool,
}

impl Default for PreprocessLimits {
    fn default() -> Self {
        Self { max_slice_nums: 9, scale_resolution: 448, slice_mode: true }
    }
}

/// `round` Python: к ближайшему, половинки — к чётному.
fn py_round(v: f64) -> f64 {
    v.round_ties_even()
}

/// `ensure_divide`: ближайшее кратное `divisor`, не меньше `divisor`.
pub fn ensure_divide(length: f64, divisor: usize) -> usize {
    let d = divisor as f64;
    let r = py_round(length / d) * d;
    (r.max(d)) as usize
}

/// `find_best_resize`: размер `(h, w)`, вписанный в `scale_resolution²`
/// пикселей с сохранением пропорций (только вниз, если не `allow_upscale`),
/// стороны кратны `patch·4`.
pub fn find_best_resize(
    (height, width): (f64, f64),
    scale_resolution: usize,
    patch_size: usize,
    allow_upscale: bool,
) -> (usize, usize) {
    let sr = scale_resolution as f64;
    let (mut h, mut w) = (height, width);
    if h * w > sr * sr || allow_upscale {
        let aspect = w / h;
        h = (sr / aspect.sqrt()).trunc();
        w = (h * aspect).trunc();
    }
    let best_w = ensure_divide(w, patch_size * 4);
    let best_h = ensure_divide(h, patch_size * 4);
    (best_h, best_w)
}

/// `get_refine_size`: размер увеличенной картинки под сетку срезов
/// `grid = (rows, cols)`, каждый срез — `find_best_resize` своей доли.
pub fn get_refine_size(
    (height, width): (usize, usize),
    (grid_y, grid_x): (usize, usize),
    scale_resolution: usize,
    patch_size: usize,
    allow_upscale: bool,
) -> (usize, usize) {
    let refine_w = ensure_divide(width as f64, grid_x);
    let refine_h = ensure_divide(height as f64, grid_y);
    let (best_h, best_w) = find_best_resize(
        (refine_h as f64 / grid_y as f64, refine_w as f64 / grid_x as f64),
        scale_resolution,
        patch_size,
        allow_upscale,
    );
    (best_h * grid_y, best_w * grid_x)
}

/// `get_sliced_grid`: сетка срезов `(rows, cols)` или `None`, если картинка
/// не крупнее `scale_resolution²`.
pub fn get_sliced_grid(
    (height, width): (usize, usize),
    max_slice_nums: usize,
    scale_resolution: usize,
) -> Option<(usize, usize)> {
    let (h, w) = (height as f64, width as f64);
    let log_ratio = (w / h).ln();
    let ratio = w * h / (scale_resolution as f64 * scale_resolution as f64);
    let multiple = (ratio.ceil() as usize).min(max_slice_nums);
    if multiple <= 1 {
        return None;
    }
    let mut best = (1usize, 1usize);
    let mut min_err = f64::INFINITY;
    for num_slices in [multiple - 1, multiple, multiple + 1] {
        if num_slices == 1 || num_slices > max_slice_nums {
            continue;
        }
        for rows in 1..=num_slices {
            if num_slices % rows != 0 {
                continue;
            }
            let cols = num_slices / rows;
            let err = (log_ratio - (cols as f64 / rows as f64).ln()).abs();
            if err < min_err {
                best = (rows, cols);
                min_err = err;
            } else if err == min_err && rows > best.0 {
                best = (rows, cols);
            }
        }
    }
    Some(best)
}

/// Раскладка картинки `h×w` пикселей без самих пикселей — для подсчёта
/// токенов и заполнителя.
pub fn plan_layout(
    (height, width): (usize, usize),
    cfg: &MiniCpmConfig,
    limits: PreprocessLimits,
) -> ImageLayout {
    let p = cfg.patch_size;
    let grid = if limits.slice_mode {
        get_sliced_grid((height, width), limits.max_slice_nums, limits.scale_resolution)
    } else {
        None
    };
    let (sh, sw) = find_best_resize(
        (height as f64, width as f64),
        limits.scale_resolution,
        p,
        grid.is_none(),
    );
    let thumb = CropGrid { h: sh / p, w: sw / p };
    match grid {
        None => ImageLayout { thumb, slice: None, rows: 0, cols: 0 },
        Some((rows, cols)) => {
            let (rh, rw) =
                get_refine_size((height, width), (rows, cols), limits.scale_resolution, p, true);
            let (ph, pw) = (rh / rows, rw / cols);
            ImageLayout { thumb, slice: Some(CropGrid { h: ph / p, w: pw / p }), rows, cols }
        }
    }
}

// ─────────────────────────── Бикубический ресайз ───────────────────────────

/// Ядро бикубики PIL/torchvision, `a = −0.5`.
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Веса фильтра по одной оси: для каждого выходного индекса — начало окна и
/// нормированные веса (антиалиасинг: при уменьшении опора растягивается в
/// `scale` раз, как в `_upsample_bicubic2d_aa` torchvision / PIL).
fn aa_weights(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let inv = 1.0 / filter_scale;
    (0..out_size)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5).floor().max(0.0)) as usize;
            let xmax = ((center + support + 0.5).floor()).min(in_size as f64) as usize;
            let xmax = xmax.max(xmin + 1);
            let mut w: Vec<f64> = (xmin..xmax)
                .map(|j| bicubic((j as f64 + 0.5 - center) * inv))
                .collect();
            let s: f64 = w.iter().sum();
            if s != 0.0 {
                w.iter_mut().for_each(|v| *v /= s);
            }
            (xmin, w)
        })
        .collect()
}

/// Бикубический ресайз с антиалиасингом `[C, H, W]` → `[C, nh, nw]` (f32).
/// HF-процессор ресайзит uint8-картинку, и torchvision возвращает uint8:
/// результат округляется к сетке `1/255` и обрезается в `0..1` — так гасится
/// и выброс бикубики за диапазон.
pub fn resize_bicubic_aa(chw: &[f32], c: usize, h: usize, w: usize, nh: usize, nw: usize) -> Vec<f32> {
    if (nh, nw) == (h, w) {
        return chw.to_vec();
    }
    let out = resize_bicubic_aa_f32(chw, c, h, w, nh, nw);
    out.into_iter()
        .map(|v| ((v * 255.0).round_ties_even().clamp(0.0, 255.0)) / 255.0)
        .collect()
}

/// То же без квантования к uint8 (float-путь).
pub fn resize_bicubic_aa_f32(chw: &[f32], c: usize, h: usize, w: usize, nh: usize, nw: usize) -> Vec<f32> {
    if (nh, nw) == (h, w) {
        return chw.to_vec();
    }
    // Сначала по ширине (строки короче — лучше кэш), затем по высоте.
    let ww = aa_weights(w, nw);
    let mut tmp = vec![0f32; c * h * nw];
    for ci in 0..c {
        for y in 0..h {
            let row = &chw[(ci * h + y) * w..(ci * h + y + 1) * w];
            let out = &mut tmp[(ci * h + y) * nw..(ci * h + y + 1) * nw];
            for (x, (x0, ws)) in ww.iter().enumerate() {
                let mut acc = 0f64;
                for (k, wk) in ws.iter().enumerate() {
                    acc += *wk * row[x0 + k] as f64;
                }
                out[x] = acc as f32;
            }
        }
    }
    let wh = aa_weights(h, nh);
    let mut out = vec![0f32; c * nh * nw];
    for ci in 0..c {
        for (y, (y0, ws)) in wh.iter().enumerate() {
            let dst = &mut out[(ci * nh + y) * nw..(ci * nh + y + 1) * nw];
            for (k, wk) in ws.iter().enumerate() {
                let src = &tmp[(ci * h + y0 + k) * nw..(ci * h + y0 + k + 1) * nw];
                for x in 0..nw {
                    dst[x] += (*wk * src[x] as f64) as f32;
                }
            }
        }
    }
    out
}

/// Патчи `[gh·gw, C·p·p]` из `[C, H, W]`: растровый порядок патчей, внутри
/// патча `(c, y, x)` — так читает их `Conv2d(stride=patch)` HF.
pub fn patchify(chw: &[f32], c: usize, h: usize, w: usize, p: usize) -> (Vec<f32>, CropGrid) {
    let (gh, gw) = (h / p, w / p);
    let feat = c * p * p;
    let mut out = vec![0f32; gh * gw * feat];
    for ph in 0..gh {
        for pw in 0..gw {
            let base = (ph * gw + pw) * feat;
            let mut k = 0usize;
            for ci in 0..c {
                for y in 0..p {
                    let row = (ci * h + ph * p + y) * w + pw * p;
                    out[base + k..base + k + p].copy_from_slice(&chw[row..row + p]);
                    k += p;
                }
            }
        }
    }
    (out, CropGrid { h: gh, w: gw })
}

/// Вырезать кусок `[C, ph, pw]` из `[C, H, W]` с угла `(y0, x0)`.
fn crop(chw: &[f32], c: usize, h: usize, w: usize, y0: usize, x0: usize, ph: usize, pw: usize) -> Vec<f32> {
    let mut out = vec![0f32; c * ph * pw];
    for ci in 0..c {
        for y in 0..ph {
            let src = (ci * h + y0 + y) * w + x0;
            let dst = (ci * ph + y) * pw;
            out[dst..dst + pw].copy_from_slice(&chw[src..src + pw]);
        }
    }
    out
}

pub fn prepare_image(
    path: impl AsRef<std::path::Path>,
    cfg: &MiniCpmConfig,
    limits: PreprocessLimits,
    device: Device,
) -> Result<PreparedImage, PreprocessError> {
    let img = synaptix_io::image::png::load_image(path, Device::Cpu)
        .map_err(|e| PreprocessError::Load(e.to_string()))?;
    prepare_tensor(&img, cfg, limits, device)
}

/// Картинка `[C ≥ 3, H, W]` F32 в `0..1` → кропы-патчи и раскладка.
pub fn prepare_tensor(
    chw: &Tensor,
    cfg: &MiniCpmConfig,
    limits: PreprocessLimits,
    device: Device,
) -> Result<PreparedImage, PreprocessError> {
    let dims = chw.dims();
    if dims.len() != 3 || dims[0] < 3 {
        return Err(PreprocessError::Shape(format!("ожидался [C>=3, H, W], получено {dims:?}")));
    }
    let (h, w) = (dims[1], dims[2]);
    if h == 0 || w == 0 {
        return Err(PreprocessError::Shape("пустая картинка".into()));
    }
    let rgb = if dims[0] == 3 {
        chw.clone()
    } else {
        chw.narrow(0, 0, 3)
            .and_then(|t| t.contiguous())
            .map_err(|e| PreprocessError::Shape(e.to_string()))?
    };
    let src = rgb
        .to_device(Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| PreprocessError::Shape(e.to_string()))?;
    let p = cfg.patch_size;
    let layout = plan_layout((h, w), cfg, limits);

    let normalize = |v: &mut [f32]| v.iter_mut().for_each(|x| *x = (*x - 0.5) / 0.5);
    let to_tensor = |mut px: Vec<f32>, ph: usize, pw: usize| -> Result<Tensor, PreprocessError> {
        normalize(&mut px);
        let (patches, grid) = patchify(&px, 3, ph, pw, p);
        Tensor::from_vec(patches, vec![grid.patches(), cfg.patch_features()], device)
            .map_err(|e| PreprocessError::Shape(e.to_string()))
    };

    let (sh, sw) = (layout.thumb.h * p, layout.thumb.w * p);
    let mut crops = vec![to_tensor(resize_bicubic_aa(&src, 3, h, w, sh, sw), sh, sw)?];
    if let Some(s) = layout.slice {
        let (ph, pw) = (s.h * p, s.w * p);
        let (rh, rw) = (ph * layout.rows, pw * layout.cols);
        let refined = resize_bicubic_aa(&src, 3, h, w, rh, rw);
        for r in 0..layout.rows {
            for c in 0..layout.cols {
                let piece = crop(&refined, 3, rh, rw, r * ph, c * pw, ph, pw);
                crops.push(to_tensor(piece, ph, pw)?);
            }
        }
    }
    Ok(PreparedImage { crops, layout })
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

    fn cfg() -> MiniCpmConfig {
        MiniCpmConfig::from_hf_bytes(crate::config::tests::sample().as_bytes()).unwrap()
    }

    #[test]
    fn ensure_divide_rounds_like_python() {
        assert_eq!(ensure_divide(100.0, 56), 112);
        assert_eq!(ensure_divide(83.0, 56), 56);
        assert_eq!(ensure_divide(10.0, 56), 56);
        // 84/56 = 1.5 → к чётному (2)
        assert_eq!(ensure_divide(84.0, 56), 112);
        // 140/56 = 2.5 → 2
        assert_eq!(ensure_divide(140.0, 56), 112);
    }

    #[test]
    fn small_image_no_slices_upscaled() {
        // 200×300 (h×w): 60000 < 448² → без срезов, allow_upscale=true:
        // aspect 1.5, h = int(448/√1.5) = 365, w = int(365·1.5) = 547 →
        // 365/56 = 6.52 → 7·56 = 392; 547/56 = 9.77 → 10·56 = 560.
        let l = plan_layout((200, 300), &cfg(), PreprocessLimits::default());
        assert_eq!(l.thumb, CropGrid { h: 28, w: 40 });
        assert!(l.slice.is_none());
        assert_eq!(l.tokens(4), 28 * 40 / 16);
        assert_eq!(l.crops().len(), 1);
    }

    #[test]
    fn square_448_is_single_crop() {
        let l = plan_layout((448, 448), &cfg(), PreprocessLimits::default());
        // 448² = sr² → не больше → allow_upscale ветка: h = 448, w = 448 → 8×8·56
        assert_eq!(l.thumb, CropGrid { h: 32, w: 32 });
        assert!(l.slice.is_none());
    }

    #[test]
    fn large_image_slices_grid() {
        // 1080×1920: ratio = 2073600/200704 = 10.33 → multiple = min(11, 9) = 9;
        // кандидаты 8, 9, 10→(>9 пропуск). log_ratio = ln(1.777) = 0.575.
        // n=8: (1,8) 2.08, (2,4) 0.118, (4,2) |0.575+0.693|, (8,1) …
        // n=9: (1,9) 2.197, (3,3) 0.575, (9,1) → min — (2,4) с 0.118.
        let l = plan_layout((1080, 1920), &cfg(), PreprocessLimits::default());
        assert_eq!((l.rows, l.cols), (2, 4));
        // миниатюра: 1080·1920 > sr² → h = int(448/√1.777) = 336, w = 597
        // → 336/56 = 6 → 336; 597/56 = 10.66 → 11·56 = 616.
        assert_eq!(l.thumb, CropGrid { h: 24, w: 44 });
        // refine: refine_w = ensure_divide(1920, 4) = 1920, refine_h = ensure_divide(1080, 2) = 1080
        // → find_best_resize((540, 480)): 540·480 > sr² → aspect 0.888,
        // h = int(448/√0.888) = int(475.2) = 475, w = int(475·0.888) = 422
        // → 475/56 = 8.48 → 8·56 = 448; 422/56 = 7.53 → 8·56 = 448.
        assert_eq!(l.slice, Some(CropGrid { h: 32, w: 32 }));
        assert_eq!(l.crops().len(), 9);
        assert_eq!(l.tokens(4), 24 * 44 / 16 + 8 * 64);
    }

    #[test]
    fn sliced_grid_prefers_more_rows_on_tie() {
        // Квадрат 1000×1000: ratio = 4.98 → multiple 5; кандидаты 4, 5, 6.
        // log_ratio = 0: (2,2) err 0 — при n=4; n=6: (2,3) ln1.5, (3,2) ln1.5 — хуже.
        assert_eq!(get_sliced_grid((1000, 1000), 9, 448), Some((2, 2)));
        assert_eq!(get_sliced_grid((400, 400), 9, 448), None);
        // Ровно 448² + 1 → ratio чуть больше 1 → multiple 2: (1,2) и (2,1) равны
        // по ошибке при квадрате → больше строк: (2,1).
        assert_eq!(get_sliced_grid((449, 448), 9, 448), Some((2, 1)));
    }

    #[test]
    fn aa_weights_identity_and_sum() {
        for (i, (x0, w)) in aa_weights(10, 10).iter().enumerate() {
            let s: f64 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-9);
            // центр окна — сам пиксель
            let peak = w.iter().cloned().fold(f64::MIN, f64::max);
            let at = w.iter().position(|v| *v == peak).unwrap();
            assert_eq!(x0 + at, i);
        }
        let img: Vec<f32> = (0..3 * 10 * 10).map(|v| v as f32).collect();
        let same = resize_bicubic_aa(&img, 3, 10, 10, 10, 10);
        assert_eq!(same, img);
    }

    #[test]
    fn downscale_constant_stays_constant() {
        let img = vec![0.25f32; 3 * 97 * 131];
        let out = resize_bicubic_aa_f32(&img, 3, 97, 131, 56, 112);
        assert_eq!(out.len(), 3 * 56 * 112);
        for v in out {
            assert!((v - 0.25).abs() < 1e-5);
        }
        let up = resize_bicubic_aa_f32(&vec![0.75f32; 3 * 20 * 30], 3, 20, 30, 56, 112);
        for v in up {
            assert!((v - 0.75).abs() < 1e-5);
        }
        // uint8-путь: сетка 1/255 и диапазон 0..1
        let q = resize_bicubic_aa(&vec![0.2f32; 3 * 20 * 30], 3, 20, 30, 56, 112);
        for v in q {
            assert!((v - 51.0 / 255.0).abs() < 1e-6);
        }
    }

    #[test]
    fn patchify_order_is_c_y_x() {
        let (c, h, w, p) = (3usize, 4usize, 6usize, 2usize);
        let img: Vec<f32> = (0..c * h * w).map(|v| v as f32).collect();
        let (patches, grid) = patchify(&img, c, h, w, p);
        assert_eq!(grid, CropGrid { h: 2, w: 3 });
        assert_eq!(patches.len(), 6 * 12);
        // патч (1, 2): c=0 → строки y=2,3, x=4,5
        let pt = &patches[(1 * 3 + 2) * 12..(1 * 3 + 2 + 1) * 12];
        assert_eq!(&pt[0..4], &[16.0, 17.0, 22.0, 23.0]);
        assert_eq!(pt[4], 24.0 + 16.0);
    }

    #[test]
    fn prepare_tensor_builds_crops() {
        synaptix_kernels_cpu::ensure_registered();
        let (h, w) = (300usize, 500usize);
        let data: Vec<f32> = (0..3 * h * w).map(|i| (i % 255) as f32 / 255.0).collect();
        let img = Tensor::from_vec(data, vec![3, h, w], Device::Cpu).unwrap();
        let c = cfg();
        let prep = prepare_tensor(&img, &c, PreprocessLimits::default(), Device::Cpu).unwrap();
        let expect = plan_layout((h, w), &c, PreprocessLimits::default());
        assert_eq!(prep.layout, expect);
        assert_eq!(prep.crops.len(), expect.crops().len());
        for (t, g) in prep.crops.iter().zip(expect.crops()) {
            assert_eq!(t.dims(), &[g.patches(), 588]);
        }
        let v = prep.crops[0].flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| (-1.0..=1.0).contains(x)));
    }
}
