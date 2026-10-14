//! Картинка одним вызовом: семейство модели — по `model_index.json`
//! бандла, умолчания (шаги, guidance, размер) — как у `synaptix imagine`.
//! Прогресс по шагам денойза; `progress` возвращает `false` — отмена.

use std::path::{Path, PathBuf};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

type R<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Sdxl,
    Flux,
    Flux2,
    QwenImage,
    QwenImage21,
}

/// Запрос; `None` — умолчание модели.
#[derive(Debug, Clone, Default)]
pub struct ImageRequest {
    pub prompt: String,
    pub negative: Option<String>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub steps: Option<usize>,
    pub guidance: Option<f32>,
    pub seed: u64,
    /// Референсы (FLUX.2, Qwen-Image Edit/2.1): правка по картинкам.
    pub images: Vec<PathBuf>,
    /// Исходная картинка для img2img (SDXL, FLUX.1, FLUX.2).
    pub init: Option<PathBuf>,
    /// Сила изменения исходной картинки, 0..1.
    pub strength: Option<f32>,
    /// Квант трансформера: `nvfp4`, `mxfp8`, пусто — плотный.
    pub quant: Option<String>,
}

pub struct ImageOutput {
    /// `[C,H,W]` F32 0..1 (C = 3 или 4).
    pub image: Tensor,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// Подпись модели и параметров для журнала.
    pub label: String,
}

fn model_index(model: &Path) -> String {
    if model.is_file() {
        return synaptix_bundle::Bundle::open(model)
            .ok()
            .and_then(|b| b.read_file("model_index.json").ok().map(|c| c.into_owned()))
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .unwrap_or_default();
    }
    std::fs::read_to_string(model.join("model_index.json")).unwrap_or_default()
}

/// Семейство модели по пайплайну в `model_index.json`.
pub fn detect(model: &Path) -> R<Family> {
    let index = model_index(model);
    let class = serde_json::from_str::<serde_json::Value>(&index)
        .ok()
        .and_then(|v| v.get("_class_name").and_then(|c| c.as_str()).map(str::to_string))
        .unwrap_or_default();
    Ok(if class.contains("QwenImage21") {
        Family::QwenImage21
    } else if class.starts_with("QwenImage") {
        Family::QwenImage
    } else if class.contains("Flux2") {
        Family::Flux2
    } else if class.contains("Flux") {
        Family::Flux
    } else if class.contains("StableDiffusionXL") {
        Family::Sdxl
    } else {
        return Err(format!("{}: неизвестный пайплайн «{class}» (ожидались SDXL, FLUX.1, FLUX.2, Qwen-Image, Qwen-Image 2.1)", model.display()));
    })
}

fn quant_of(req: &ImageRequest) -> R<DType> {
    quant_dtype(req.quant.as_deref())
}

/// Квант трансформера по названию; пусто — плотный BF16.
pub(crate) fn quant_dtype(q: Option<&str>) -> R<DType> {
    match q.unwrap_or("").to_lowercase().as_str() {
        "" | "none" | "dense" | "bf16" | "f16" | "f32" | "auto" => Ok(DType::BF16),
        "nvfp4" => Ok(DType::NVFP4),
        "mxfp8" | "fp8" => Ok(DType::MXFP8),
        other => Err(format!("неизвестный квант {other} (nvfp4|mxfp8)")),
    }
}

pub(crate) fn load_rgb(path: &Path) -> R<Tensor> {
    synaptix_io::image::load_image(path, Device::Cpu).map_err(|e| format!("{}: {e}", path.display()))
}

pub(crate) fn image_size(t: &Tensor) -> (usize, usize) {
    let d = t.dims();
    (d[2], d[1])
}

/// Сгенерировать картинку моделью `model` (бандл `.syn` или каталог diffusers):
/// стадии [`crate::image_stages`] подряд.
pub fn generate(model: &Path, device: Device, req: &ImageRequest, progress: &mut dyn FnMut(usize, usize) -> bool) -> R<ImageOutput> {
    // Квантованный трансформер — и внимание в FP8 (e4m3 QKᵀ и P·V).
    let _fp8 = synaptix_core::precision::AttentionFp8Guard::new(quant_of(req).map(|q| q.is_quantized()).unwrap_or(false));
    if let Some(s) = req.strength {
        if !(0.0..=1.0).contains(&s) {
            return Err(format!("strength вне [0, 1]: {s}"));
        }
    }
    let m = crate::image_stages::open(model, device, req.quant.as_deref())?;
    let family = m.family;
    // Картинки запроса: у FLUX.2 и Qwen-Image — референсы (Qwen видит их и в описании).
    let qwen = matches!(family, Family::QwenImage | Family::QwenImage21);
    let cond_images: &[PathBuf] = if qwen { &req.images } else { &[] };
    let cond = m.encode(&req.prompt, req.negative.as_deref(), cond_images, None)?;
    let refs = match family {
        Family::Flux2 | Family::QwenImage | Family::QwenImage21 if !req.images.is_empty() => Some(m.references(&req.images)?),
        Family::Sdxl | Family::Flux if !req.images.is_empty() => return Err(format!("{}: референсы не поддерживаются (есть у FLUX.2 и Qwen-Image)", m.name())),
        _ => None,
    };
    let size = (req.width.unwrap_or(0), req.height.unwrap_or(0));
    let latent = match &req.init {
        Some(p) => m.image_latent(p, size, true)?,
        None => crate::image_stages::Latent::Size(size.0, size.1),
    };
    let opts = crate::image_stages::SampleOpts { steps: req.steps, guidance: req.guidance, seed: req.seed, denoise: req.strength };
    let lat = m.sample(&cond, Some(&latent), refs.as_ref(), &opts, progress)?;
    drop((cond, refs, latent));
    let image = m.decode(&lat)?;
    let crate::image_stages::Latent::Sampled { width, height, steps, label, .. } = lat else { unreachable!() };
    Ok(ImageOutput { image, width, height, steps, label })
}

/// Сохранить результат в PNG.
pub fn save_png(image: &Tensor, path: &Path) -> R<()> {
    synaptix_io::image::save_image(image, path).map_err(err)
}

