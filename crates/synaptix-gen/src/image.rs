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
    match req.quant.as_deref().unwrap_or("").to_lowercase().as_str() {
        "" | "none" | "dense" | "bf16" | "f16" | "f32" | "auto" => Ok(DType::BF16),
        "nvfp4" => Ok(DType::NVFP4),
        "mxfp8" | "fp8" => Ok(DType::MXFP8),
        other => Err(format!("неизвестный квант {other} (nvfp4|mxfp8)")),
    }
}

fn load_rgb(path: &Path) -> R<Tensor> {
    synaptix_io::image::load_image(path, Device::Cpu).map_err(|e| format!("{}: {e}", path.display()))
}

fn image_size(t: &Tensor) -> (usize, usize) {
    let d = t.dims();
    (d[2], d[1])
}

fn steps_or(requested: Option<usize>, default: usize) -> usize {
    requested.filter(|&v| v > 0).unwrap_or(default)
}

fn size_or(req: &ImageRequest, fallback: (usize, usize), snap: impl Fn(usize) -> usize) -> (usize, usize) {
    (snap(req.width.filter(|&v| v > 0).unwrap_or(fallback.0)), snap(req.height.filter(|&v| v > 0).unwrap_or(fallback.1)))
}

/// Сгенерировать картинку моделью `model` (бандл `.syn` или каталог diffusers).
pub fn generate(model: &Path, device: Device, req: &ImageRequest, progress: &mut dyn FnMut(usize, usize) -> bool) -> R<ImageOutput> {
    // Квантованный трансформер — и внимание в FP8 (e4m3 QKᵀ и P·V).
    let _fp8 = synaptix_core::precision::AttentionFp8Guard::new(quant_of(req).map(|q| q.is_quantized()).unwrap_or(false));
    if !model.exists() {
        return Err(format!("модели нет: {}", model.display()));
    }
    if let Some(s) = req.strength {
        if !(0.0..=1.0).contains(&s) {
            return Err(format!("strength вне [0, 1]: {s}"));
        }
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    match detect(model)? {
        Family::Sdxl => sdxl(model, device, req, progress),
        Family::Flux => flux(model, device, req, progress),
        Family::Flux2 => flux2(model, device, req, progress),
        Family::QwenImage => qwen(model, device, req, progress),
        Family::QwenImage21 => qwen21(model, device, req, progress),
    }
}

/// Сохранить результат в PNG.
pub fn save_png(image: &Tensor, path: &Path) -> R<()> {
    synaptix_io::image::save_image(image, path).map_err(err)
}

fn sdxl(model: &Path, dev: Device, req: &ImageRequest, progress: &mut dyn FnMut(usize, usize) -> bool) -> R<ImageOutput> {
    use synaptix_image_sdxl::stages::snap_side;
    use synaptix_image_sdxl::{SdxlCheckpoint, SdxlSampleParams};
    let quant = quant_of(req)?;
    let ck = SdxlCheckpoint::open(model, dev).map_err(err)?;
    let init_src = req.init.as_deref().map(load_rgb).transpose()?;
    let (width, height) = size_or(req, init_src.as_ref().map(image_size).unwrap_or((1024, 1024)), snap_side);
    let steps = steps_or(req.steps, 30);
    let guidance = req.guidance.unwrap_or(5.0);
    let denoise = if init_src.is_some() { req.strength.unwrap_or(0.6) } else { 1.0 };
    let cond = ck.encode_prompt(&req.prompt, req.negative.as_deref().unwrap_or("")).map_err(err)?;
    let init = match &init_src {
        Some(img) => Some(ck.encode_image(&synaptix_io::image::fit_image(img, width, height, true).map_err(err)?).map_err(err)?),
        None => None,
    };
    let unet = ck.load_unet(quant).map_err(err)?;
    let p = SdxlSampleParams { width, height, steps, guidance, seed: req.seed, denoise };
    let lat = ck.sample(&unet, &cond, init.as_ref(), &p, &mut |i, n| progress(i, n)).map_err(err)?;
    drop(unet);
    let image = ck.decode(&lat).map_err(err)?;
    Ok(ImageOutput { image, width, height, steps, label: format!("SDXL {width}×{height}, {steps} шагов, cfg {guidance}") })
}

fn flux(model: &Path, dev: Device, req: &ImageRequest, progress: &mut dyn FnMut(usize, usize) -> bool) -> R<ImageOutput> {
    use synaptix_image_flux::model::snap_side;
    use synaptix_image_flux::{FluxModel, OffloadMode, SampleParams};
    let quant = quant_of(req)?;
    let compute = if !dev.is_cuda() {
        DType::F32
    } else if quant.is_quantized() {
        DType::F16
    } else {
        DType::BF16
    };
    let quant = if dev.is_cuda() { quant } else { compute };
    let m = FluxModel::open(model, dev, compute, quant).map_err(err)?.with_offload(OffloadMode::Auto);
    let distilled = m.guidance_distilled();
    let seq = m.default_max_seq_len();
    let init_src = req.init.as_deref().map(load_rgb).transpose()?;
    let (width, height) = size_or(req, init_src.as_ref().map(image_size).unwrap_or((1024, 1024)), snap_side);
    let steps = steps_or(req.steps, if distilled { 28 } else { 4 });
    let guidance = req.guidance.unwrap_or(3.5);
    let denoise = if init_src.is_some() { req.strength.unwrap_or(0.75) } else { 1.0 };
    let cond = m.encode_prompt(&req.prompt, seq).map_err(err)?;
    let init = match &init_src {
        Some(img) => Some(m.encode_image(&synaptix_io::image::fit_image(img, width, height, true).map_err(err)?).map_err(err)?),
        None => None,
    };
    let t = m.load_transformer(FluxModel::tokens_for(width, height, seq)).map_err(err)?;
    let p = SampleParams { width, height, steps, guidance, seed: req.seed, denoise };
    let lat = m.sample(&t, &cond, init.as_ref(), &p, &mut |i, n| progress(i, n)).map_err(err)?;
    drop(t);
    let image = m.decode(&lat).map_err(err)?;
    Ok(ImageOutput { image, width, height, steps, label: format!("FLUX.1 {width}×{height}, {steps} шагов, guidance {guidance}") })
}

fn flux2(model: &Path, dev: Device, req: &ImageRequest, progress: &mut dyn FnMut(usize, usize) -> bool) -> R<ImageOutput> {
    use synaptix_image_flux2::model::{reference_size, snap_side};
    use synaptix_image_flux2::{Flux2Model, MemoryMode, SampleParams};
    if req.images.len() > 10 {
        return Err(format!("FLUX.2 принимает до 10 референсов, передано {}", req.images.len()));
    }
    let quant = quant_of(req)?;
    let m = Flux2Model::open(model, dev, DType::BF16, quant).map_err(err)?.with_memory(MemoryMode::Auto);
    let variant = m.variant();
    let mut references = Vec::with_capacity(req.images.len());
    for p in &req.images {
        let img = load_rgb(p)?;
        let (w, h) = image_size(&img);
        let (rw, rh) = reference_size(w, h);
        references.push(synaptix_io::image::fit_image(&img, rw, rh, true).map_err(err)?);
    }
    let init_src = req.init.as_deref().map(load_rgb).transpose()?;
    let fallback = init_src.as_ref().map(image_size).or(references.first().map(image_size)).unwrap_or((1024, 1024));
    let (width, height) = size_or(req, fallback, snap_side);
    let steps = steps_or(req.steps, variant.default_steps());
    let guidance = req.guidance.unwrap_or(variant.default_guidance());
    let denoise = if init_src.is_some() { req.strength.unwrap_or(0.75) } else { 1.0 };
    let cond = m.encode_prompt(&req.prompt, variant.uses_cfg(guidance)).map_err(err)?;
    let refs = if references.is_empty() { None } else { Some(m.encode_references(&references).map_err(err)?) };
    let init = match &init_src {
        Some(img) => Some(m.encode_image(&synaptix_io::image::fit_image(img, width, height, true).map_err(err)?).map_err(err)?),
        None => None,
    };
    let ref_tokens = refs.as_ref().map(|r| r.num_tokens()).unwrap_or(0);
    let t = m.load_transformer(Flux2Model::tokens_for(width, height, ref_tokens)).map_err(err)?;
    let p = SampleParams { width, height, steps, guidance, seed: req.seed, denoise };
    let lat = m.sample(&t, &cond, refs.as_ref(), init.as_ref(), &p, &mut |i, n| progress(i, n)).map_err(err)?;
    drop(t);
    let image = m.decode(&lat).map_err(err)?;
    Ok(ImageOutput { image, width, height, steps, label: format!("FLUX.2 {variant:?} {width}×{height}, {steps} шагов, референсов {}", references.len()) })
}

fn qwen(model: &Path, dev: Device, req: &ImageRequest, progress: &mut dyn FnMut(usize, usize) -> bool) -> R<ImageOutput> {
    use synaptix_image_qwen::model::snap_side;
    use synaptix_image_qwen::{MemoryMode, QwenImageModel, SampleParams};
    let quant = quant_of(req)?;
    let m = QwenImageModel::open(model, dev, DType::BF16, quant).map_err(err)?.with_memory(MemoryMode::Auto);
    let variant = m.variant();
    if req.images.len() > variant.max_images() {
        return Err(format!("{} принимает картинок: {}, передано {}", variant.label(), variant.max_images(), req.images.len()));
    }
    if variant.is_edit() && req.images.is_empty() {
        return Err(format!("{} правит картинку — нужна исходная картинка", variant.label()));
    }
    let images: Vec<Tensor> = req.images.iter().map(|p| load_rgb(p)).collect::<R<_>>()?;
    let sizes: Vec<(usize, usize)> = images.iter().map(image_size).collect();
    let (width, height) = size_or(req, m.default_size(&sizes), snap_side);
    let steps = steps_or(req.steps, variant.default_steps());
    let cfg = req.guidance.unwrap_or(4.0);
    let negative = req.negative.clone().unwrap_or_else(|| " ".to_string());
    let negative = (!negative.is_empty() && cfg > 1.0).then_some(negative);
    let cond = m.encode_prompt(&req.prompt, negative.as_deref(), &images).map_err(err)?;
    let refs = if images.is_empty() { None } else { Some(m.encode_references(&images).map_err(err)?) };
    let ref_tokens = refs.as_ref().map(|r| r.num_tokens()).unwrap_or(0);
    let txt = cond.embeds.dims()[1].max(cond.negative.as_ref().map(|n| n.dims()[1]).unwrap_or(0));
    let t = m.load_transformer(QwenImageModel::tokens_for(width, height, ref_tokens, txt)).map_err(err)?;
    let p = SampleParams { width, height, steps, cfg, seed: req.seed };
    let lat = m.sample(&t, &cond, refs.as_ref(), &p, &mut |i, n| progress(i, n)).map_err(err)?;
    drop(t);
    let image = m.decode(&lat).map_err(err)?;
    Ok(ImageOutput { image, width, height, steps, label: format!("{} {width}×{height}, {steps} шагов", variant.label()) })
}

fn qwen21(model: &Path, dev: Device, req: &ImageRequest, progress: &mut dyn FnMut(usize, usize) -> bool) -> R<ImageOutput> {
    use synaptix_image_qwen21::model::{snap_side, DEFAULT_RESOLUTION, DEFAULT_STEPS};
    use synaptix_image_qwen21::{MemoryMode, QwenImage21Model, RgbaImage, SampleParams};
    let quant = quant_of(req)?;
    let m = QwenImage21Model::open(model, dev, DType::BF16, quant).map_err(err)?.with_memory(MemoryMode::Auto);
    let mut images = Vec::with_capacity(req.images.len());
    for p in &req.images {
        let t = synaptix_io::image::png::load_image_rgba(p, Device::Cpu).map_err(|e| format!("{}: {e}", p.display()))?;
        images.push(RgbaImage::from_tensor(&t).map_err(err)?);
    }
    let sizes: Vec<(usize, usize)> = images.iter().map(|i| (i.width, i.height)).collect();
    let resolution = DEFAULT_RESOLUTION;
    let (width, height) = size_or(req, QwenImage21Model::default_size(&sizes, resolution), snap_side);
    let steps = steps_or(req.steps, DEFAULT_STEPS);
    let cfg = req.guidance.unwrap_or(1.0);
    let negative = req.negative.as_deref().filter(|n| !n.trim().is_empty() && cfg > 1.0);
    let cond = m.encode_prompt(&req.prompt, negative, &images, resolution).map_err(err)?;
    let refs = if images.is_empty() { None } else { Some(m.encode_references(&images, resolution).map_err(err)?) };
    let ref_tokens = refs.as_ref().map(|r| r.num_tokens()).unwrap_or(0);
    let txt = cond.embeds.dims()[1].max(cond.negative.as_ref().map(|n| n.0.dims()[1]).unwrap_or(0));
    let t = m.load_transformer(QwenImage21Model::tokens_for(width, height, ref_tokens, txt)).map_err(err)?;
    let p = SampleParams { width, height, steps, cfg, seed: req.seed, kv_cache: true };
    let lat = m.sample(&t, &cond, refs.as_ref(), &p, &mut |i, n| progress(i, n)).map_err(err)?;
    drop(t);
    let image = m.decode(&lat).map_err(err)?;
    let transparent = RgbaImage::from_tensor(&image).map_err(err)?.has_transparency();
    let image = if transparent { image } else { image.narrow(0, 0, 3).map_err(err)?.contiguous().map_err(err)? };
    Ok(ImageOutput { image, width, height, steps, label: format!("Qwen-Image 2.1 {width}×{height}, {steps} шагов, референсов {}", images.len()) })
}
