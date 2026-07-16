//! `synaptix imagine <model_dir> <prompt> -o out.png` — txt2img через нативный
//! SDXL (CLIP×2 + UNet2DConditionModel + AutoencoderKL).

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
}

fn is_flux(model: &std::path::Path) -> bool {
    std::fs::read_to_string(model.join("model_index.json"))
        .map(|s| s.contains("Flux"))
        .unwrap_or(false)
}

pub fn run(args: ImagineArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.model.exists() {
        return Err(format!("model dir not found: {}", args.model.display()).into());
    }
    if is_flux(&args.model) {
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
