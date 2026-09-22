//! `synaptix imagine <model_dir> <prompt> -o out.png` — txt2img через нативный
//! SDXL (CLIP×2 + UNet2DConditionModel + AutoencoderKL); по `model_index.json`
//! переключается на FLUX.1, FLUX.2 и Qwen-Image 2.1 (`--image` — референсы).

use std::path::PathBuf;

use synaptix_core::dtype::DType;
use synaptix_image_sdxl::{SdxlPipeline, Txt2ImgParams};

use crate::commands::device;

pub struct ImagineArgs {
    pub model: PathBuf,
    pub prompt: String,
    pub output: PathBuf,
    pub negative: String,
    pub steps: usize,
    pub guidance_scale: f32,
    pub height: usize,
    pub width: usize,
    pub seed: u64,
    pub device: String,
    pub compute_dtype: Option<String>,
    pub quant: Option<String>,
    pub storage_dtype: Option<String>,
    /// Референсы Qwen-Image 2.1.
    pub image: Vec<PathBuf>,
    /// `output_resolution` Qwen-Image 2.1.
    pub resolution: usize,
}

/// `model_index.json` каталога diffusers или `.syn`-бандла.
fn model_index(model: &std::path::Path) -> String {
    if model.is_file() {
        return synaptix_bundle::Bundle::open(model)
            .ok()
            .and_then(|b| b.read_file("model_index.json").ok().map(|c| c.into_owned()))
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .unwrap_or_default();
    }
    std::fs::read_to_string(model.join("model_index.json")).unwrap_or_default()
}

pub fn run(args: ImagineArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.model.exists() {
        return Err(format!("model dir not found: {}", args.model.display()).into());
    }
    let index = model_index(&args.model);
    if index.contains("QwenImage21") {
        return run_qwen21(args);
    }
    // `Flux2Pipeline`/`Flux2KleinPipeline` — другая архитектура, чем FLUX.1.
    if index.contains("Flux2") {
        return run_flux2(args);
    }
    if index.contains("Flux") {
        return run_flux(args);
    }

    let dev = device::resolve(&args.device);
    // Квант весов UNet (как в LLM): --quant nvfp4|mxfp8 (--storage-dtype = алиас).
    // Режет VRAM UNet (attn/GEGLU-линейки), считается в F16. CLIP/VAE — dense.
    let want = args.quant.as_deref().or(args.storage_dtype.as_deref()).unwrap_or("none");
    let quant = match want.to_lowercase().as_str() {
        "none" | "bf16" | "f16" | "f32" => DType::BF16,
        "nvfp4" => DType::NVFP4,
        "mxfp8" | "fp8" => DType::MXFP8,
        other => return Err(format!("неизвестный --quant/--storage-dtype: {other} (none|nvfp4|mxfp8)").into()),
    };
    // Дефолт: BF16 на CUDA (FA-4 tensor-core attention + fused-ядра, ~4× быстрее
    // F32, качество визуально идентично — diffusers тоже гоняет UNet в bf16),
    // F32 на CPU. VAE всегда F32 (F16/BF16 overflow). Явный --compute-dtype важнее.
    // Quant требует F16-активацию → дефолт F16 при quant.
    let dtype = match args.compute_dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(other) => return Err(format!("unknown compute-dtype {other}").into()),
        None if quant.is_quantized() => DType::F16,
        None if dev.is_cuda() => DType::BF16,
        None => DType::F32,
    };

    eprintln!(
        "synaptix imagine: {} | {}×{} | {} steps | cfg {} | seed {} | quant={quant:?} compute={dtype:?} {dev:?}",
        args.model.display(),
        args.width,
        args.height,
        args.steps,
        args.guidance_scale,
        args.seed,
    );

    let t0 = std::time::Instant::now();
    let pipe = SdxlPipeline::from_pretrained_quant(&args.model, dev, dtype, quant)?;
    eprintln!("synaptix imagine: model loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let params = Txt2ImgParams {
        prompt: args.prompt,
        negative_prompt: args.negative,
        height: args.height,
        width: args.width,
        steps: args.steps,
        guidance_scale: args.guidance_scale,
        seed: args.seed,
    };

    let bar = indicatif::ProgressBar::new(params.steps as u64);
    bar.set_style(
        indicatif::ProgressStyle::with_template("  {bar:40} {pos}/{len} steps [{elapsed_precise}]")
            .unwrap(),
    );

    let t1 = std::time::Instant::now();
    let image = pipe.txt2img(&params, |step, _total| bar.set_position(step as u64))?;
    bar.finish_and_clear();
    let dt = t1.elapsed().as_secs_f32();
    eprintln!(
        "synaptix imagine: {} steps in {:.1}s ({:.2}s/step)",
        params.steps,
        dt,
        dt / params.steps.max(1) as f32
    );

    synaptix_io::image::save_image(&image, &args.output)?;
    eprintln!("synaptix imagine: saved {}", args.output.display());
    Ok(())
}

/// FLUX.1-dev txt2img. guidance-distilled (без negative/CFG). Всегда BF16 на
/// CUDA (transformer 23GB; f32 не влезает в VRAM). Компоненты грузятся
/// последовательно (CLIP→T5→transformer→VAE), пик ~23GB.
fn run_flux(args: ImagineArgs) -> Result<(), Box<dyn std::error::Error>> {
    use synaptix_image_flux::{FluxPipeline, Txt2ImgParams as FluxParams};

    let dev = device::resolve(&args.device);
    // Точность как в LLM: --quant (NVFP4/MXFP8) квантует веса трансформера резидентно
    // (VRAM 23GB bf16 → ~6GB nvfp4 / ~12GB mxfp8), --storage-dtype = алиас. --compute-dtype
    // = dtype активаций/энкодеров (quant требует F16-активацию → дефолт F16 при quant,
    // иначе BF16 = Python-качество). Дефолт без --quant = dense bf16 (прежнее поведение).
    let want = args.quant.as_deref().or(args.storage_dtype.as_deref()).unwrap_or("none");
    let quant = match want.to_lowercase().as_str() {
        "none" | "bf16" | "f16" | "f32" => DType::BF16,
        "nvfp4" => DType::NVFP4,
        "mxfp8" | "fp8" => DType::MXFP8,
        other => return Err(format!("неизвестный --quant/--storage-dtype: {other} (none|nvfp4|mxfp8)").into()),
    };
    let dtype = match args.compute_dtype.as_deref() {
        Some("f32") => DType::F32,
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        _ => if quant.is_quantized() { DType::F16 } else { DType::BF16 },
    };
    eprintln!(
        "synaptix imagine [FLUX]: {} | {}×{} | {} steps | guidance {} | seed {} | quant={quant:?} compute={dtype:?} {dev:?}",
        args.model.display(), args.width, args.height, args.steps, args.guidance_scale, args.seed,
    );

    let t0 = std::time::Instant::now();
    let pipe = FluxPipeline::from_pretrained_quant(&args.model, dev, dtype, quant)?;
    eprintln!("synaptix imagine [FLUX]: tokenizers loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let params = FluxParams {
        prompt: args.prompt,
        height: args.height,
        width: args.width,
        steps: args.steps,
        guidance_scale: args.guidance_scale,
        seed: args.seed,
    };

    let bar = indicatif::ProgressBar::new(params.steps as u64);
    bar.set_style(
        indicatif::ProgressStyle::with_template("  {bar:40} {pos}/{len} steps [{elapsed_precise}]")
            .unwrap(),
    );
    let t1 = std::time::Instant::now();
    let image = pipe.txt2img(&params, |step, _total| bar.set_position(step as u64))?;
    bar.finish_and_clear();
    let dt = t1.elapsed().as_secs_f32();
    eprintln!(
        "synaptix imagine [FLUX]: {} steps in {:.1}s ({:.2}s/step)",
        params.steps, dt, dt / params.steps.max(1) as f32
    );

    synaptix_io::image::save_image(&image, &args.output)?;
    eprintln!("synaptix imagine [FLUX]: saved {}", args.output.display());
    Ok(())
}

/// FLUX.2 (dev / klein): стадии `Flux2Model` подряд. Энкодер — BF16, DiT —
/// `--quant nvfp4|mxfp8` или плотный BF16; что не влезло в VRAM, стримится.
fn run_flux2(args: ImagineArgs) -> Result<(), Box<dyn std::error::Error>> {
    use synaptix_image_flux2::{Flux2Model, SampleParams};

    let dev = device::resolve(&args.device);
    let want = args.quant.as_deref().or(args.storage_dtype.as_deref()).unwrap_or("none");
    let quant = match want.to_lowercase().as_str() {
        "none" | "bf16" | "f16" | "f32" => DType::BF16,
        "nvfp4" => DType::NVFP4,
        "mxfp8" | "fp8" => DType::MXFP8,
        other => return Err(format!("неизвестный --quant/--storage-dtype: {other} (none|nvfp4|mxfp8)").into()),
    };
    let t0 = std::time::Instant::now();
    let m = Flux2Model::open(&args.model, dev, DType::BF16, quant)?;
    eprintln!(
        "synaptix imagine [FLUX.2 {:?}]: {} | {}×{} | {} steps | guidance {} | seed {} | quant={quant:?} {dev:?}",
        m.variant(),
        args.model.display(),
        args.width,
        args.height,
        args.steps,
        args.guidance_scale,
        args.seed,
    );
    let cond = m.encode_prompt(&args.prompt, m.variant().uses_cfg(args.guidance_scale))?;
    let t = m.load_transformer(Flux2Model::tokens_for(args.width, args.height, 0))?;
    let p = SampleParams {
        width: args.width,
        height: args.height,
        steps: args.steps,
        guidance: args.guidance_scale,
        seed: args.seed,
        denoise: 1.0,
    };
    let bar = indicatif::ProgressBar::new(p.steps as u64);
    bar.set_style(
        indicatif::ProgressStyle::with_template("  {bar:40} {pos}/{len} steps [{elapsed_precise}]").unwrap(),
    );
    let lat = m.sample(&t, &cond, None, None, &p, &mut |i, _| {
        bar.set_position(i as u64);
        true
    })?;
    bar.finish_and_clear();
    drop(t);
    let image = m.decode(&lat)?;
    synaptix_io::image::save_image(&image, &args.output)?;
    eprintln!(
        "synaptix imagine [FLUX.2]: saved {} ({:.1}s)",
        args.output.display(),
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

/// Qwen-Image 2.1: картинка по тексту, правка по референсам (`--image`, до 10)
/// и RGBA (PNG с альфой, если модель её нарисовала). Энкодер Qwen3-VL — BF16,
/// DiT — `--quant nvfp4|mxfp8` или плотный BF16; `--cfg` > 1 с непустым
/// `--negative` включает true CFG (по умолчанию модель идёт без него).
fn run_qwen21(args: ImagineArgs) -> Result<(), Box<dyn std::error::Error>> {
    use synaptix_core::device::Device;
    use synaptix_image_qwen21::{QwenImage21Model, RgbaImage, SampleParams};

    let dev = device::resolve(&args.device);
    let want = args.quant.as_deref().or(args.storage_dtype.as_deref()).unwrap_or("none");
    let quant = match want.to_lowercase().as_str() {
        "none" | "bf16" | "f16" | "f32" => DType::BF16,
        "nvfp4" => DType::NVFP4,
        "mxfp8" | "fp8" => DType::MXFP8,
        other => return Err(format!("неизвестный --quant/--storage-dtype: {other} (none|nvfp4|mxfp8)").into()),
    };
    let t0 = std::time::Instant::now();
    let m = QwenImage21Model::open(&args.model, dev, DType::BF16, quant)?;
    let mut images = Vec::with_capacity(args.image.len());
    for p in &args.image {
        let t = synaptix_io::image::png::load_image_rgba(p, Device::Cpu)?;
        images.push(RgbaImage::from_tensor(&t)?);
    }
    let sizes: Vec<(usize, usize)> = images.iter().map(|i| (i.width, i.height)).collect();
    let (width, height) = if args.width == 0 || args.height == 0 {
        QwenImage21Model::default_size(&sizes, args.resolution)
    } else {
        (args.width, args.height)
    };
    let steps = if args.steps == 0 { synaptix_image_qwen21::model::DEFAULT_STEPS } else { args.steps };
    let cfg_on = args.guidance_scale > 1.0 && !args.negative.is_empty();
    eprintln!(
        "synaptix imagine [Qwen-Image 2.1]: {} | {}×{} | {} steps | cfg {} | seed {} | референсов {} | quant={quant:?} {dev:?}",
        args.model.display(),
        width,
        height,
        steps,
        if cfg_on { args.guidance_scale.to_string() } else { "выкл".into() },
        args.seed,
        images.len(),
    );
    let negative = cfg_on.then_some(args.negative.as_str());
    let cond = m.encode_prompt(&args.prompt, negative, &images, args.resolution)?;
    let refs = if images.is_empty() { None } else { Some(m.encode_references(&images, args.resolution)?) };
    let ref_tokens = refs.as_ref().map(|r| r.num_tokens()).unwrap_or(0);
    let txt = cond.embeds.dims()[1].max(cond.negative.as_ref().map(|n| n.0.dims()[1]).unwrap_or(0));
    let t = m.load_transformer(QwenImage21Model::tokens_for(width, height, ref_tokens, txt))?;
    let p = SampleParams { width, height, steps, cfg: args.guidance_scale, seed: args.seed, kv_cache: true };
    let bar = indicatif::ProgressBar::new(p.steps as u64);
    bar.set_style(
        indicatif::ProgressStyle::with_template("  {bar:40} {pos}/{len} steps [{elapsed_precise}]").unwrap(),
    );
    let lat = m.sample(&t, &cond, refs.as_ref(), &p, &mut |i, _| {
        bar.set_position(i as u64);
        true
    })?;
    bar.finish_and_clear();
    drop(t);
    let image = m.decode(&lat)?;
    // Без настоящей прозрачности (альфа лишь шумит у 255) — обычный RGB PNG.
    let transparent = RgbaImage::from_tensor(&image)?.has_transparency();
    let out = if transparent { image } else { image.narrow(0, 0, 3)?.contiguous()? };
    synaptix_io::image::save_image(&out, &args.output)?;
    eprintln!(
        "synaptix imagine [Qwen-Image 2.1]: saved {}{} ({:.1}s)",
        args.output.display(),
        if transparent { " (RGBA)" } else { "" },
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
