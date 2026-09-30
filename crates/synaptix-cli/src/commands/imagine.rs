use std::path::{Path, PathBuf};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::commands::device;

pub struct ImagineArgs {
    pub model: PathBuf,
    pub prompt: String,
    pub output: PathBuf,
    pub negative: Option<String>,
    pub steps: Option<usize>,
    pub guidance_scale: Option<f32>,
    pub height: Option<usize>,
    pub width: Option<usize>,
    pub seed: u64,
    pub device: String,
    pub compute_dtype: Option<String>,
    pub quant: Option<String>,
    pub storage_dtype: Option<String>,
    pub image: Vec<PathBuf>,
    pub resolution: usize,
    pub init_image: Option<PathBuf>,
    pub strength: Option<f32>,
    pub fit: String,
    pub memory: String,
    pub t5_len: Option<usize>,
    pub no_kv_cache: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    Sdxl,
    Flux,
    Flux2,
    QwenImage,
    QwenImage21,
}

type R<T> = Result<T, Box<dyn std::error::Error>>;

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

fn detect(model: &Path) -> R<Family> {
    let index = model_index(model);
    let class = serde_json::from_str::<serde_json::Value>(&index)
        .ok()
        .and_then(|v| v.get("_class_name").and_then(|c| c.as_str()).map(str::to_string))
        .unwrap_or_default();
    let family = if class.contains("QwenImage21") {
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
        return Err(format!(
            "{}: неизвестный пайплайн `{class}` в model_index.json (ожидались SDXL, FLUX.1, FLUX.2, Qwen-Image, Qwen-Image 2.1)",
            model.display()
        )
        .into());
    };
    Ok(family)
}

fn parse_quant(args: &ImagineArgs) -> R<DType> {
    let want = args.quant.as_deref().or(args.storage_dtype.as_deref()).unwrap_or("none");
    match want.to_lowercase().as_str() {
        "none" | "dense" | "bf16" | "f16" | "f32" => Ok(DType::BF16),
        "nvfp4" => Ok(DType::NVFP4),
        "mxfp8" | "fp8" => Ok(DType::MXFP8),
        other => Err(format!("неизвестный --quant/--storage-dtype: {other} (none|nvfp4|mxfp8)").into()),
    }
}

fn parse_compute(name: Option<&str>) -> R<Option<DType>> {
    match name {
        None => Ok(None),
        Some("f32") => Ok(Some(DType::F32)),
        Some("f16") => Ok(Some(DType::F16)),
        Some("bf16") => Ok(Some(DType::BF16)),
        Some(other) => Err(format!("unknown compute-dtype {other} (f32|f16|bf16)").into()),
    }
}

fn parse_fit(fit: &str) -> R<bool> {
    match fit {
        "stretch" => Ok(false),
        "crop" | "center-crop" => Ok(true),
        other => Err(format!("--fit: stretch | crop, а не `{other}`").into()),
    }
}

fn warn_unused(family: Family, flags: &[(&str, bool)]) {
    for (flag, used) in flags {
        if *used {
            eprintln!("synaptix imagine: {flag} не поддерживается у {family:?} — игнорируется");
        }
    }
}

fn load_rgb(path: &Path) -> R<Tensor> {
    Ok(synaptix_io::image::load_image(path, Device::Cpu)?)
}

fn image_size(t: &Tensor) -> (usize, usize) {
    let d = t.dims();
    (d[2], d[1])
}

fn progress_bar(steps: usize) -> indicatif::ProgressBar {
    let bar = indicatif::ProgressBar::new(steps as u64);
    bar.set_style(
        indicatif::ProgressStyle::with_template("  {bar:40} {pos}/{len} steps [{elapsed_precise}]").unwrap(),
    );
    bar
}

fn pick_steps(requested: Option<usize>, model_default: usize) -> usize {
    match requested {
        Some(0) | None => model_default,
        Some(v) => v,
    }
}

fn target_size(
    args: &ImagineArgs,
    fallback: (usize, usize),
    snap: impl Fn(usize) -> usize,
) -> (usize, usize) {
    let w = args.width.filter(|&v| v > 0).unwrap_or(fallback.0);
    let h = args.height.filter(|&v| v > 0).unwrap_or(fallback.1);
    (snap(w), snap(h))
}

fn finish(image: &Tensor, output: &Path, label: &str, started: std::time::Instant) -> R<()> {
    synaptix_io::image::save_image(image, output)?;
    eprintln!(
        "synaptix imagine [{label}]: saved {} ({:.1}s)",
        output.display(),
        started.elapsed().as_secs_f32()
    );
    Ok(())
}

fn memory_label(memory: &str) -> R<&'static str> {
    match memory {
        "auto" => Ok("auto"),
        "resident" => Ok("resident"),
        "stream" | "offload" | "block_offload" => Ok("stream"),
        other => Err(format!("--memory: auto | resident | stream, а не `{other}`").into()),
    }
}

pub fn run(args: ImagineArgs) -> R<()> {
    if !args.model.exists() {
        return Err(format!("model not found: {}", args.model.display()).into());
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let family = detect(&args.model)?;
    memory_label(&args.memory)?;
    if let Some(s) = args.strength {
        if !(0.0..=1.0).contains(&s) {
            return Err(format!("--strength вне [0, 1]: {s}").into());
        }
    }
    match family {
        Family::Sdxl => run_sdxl(args),
        Family::Flux => run_flux(args),
        Family::Flux2 => run_flux2(args),
        Family::QwenImage => run_qwen(args),
        Family::QwenImage21 => run_qwen21(args),
    }
}

fn run_sdxl(args: ImagineArgs) -> R<()> {
    use synaptix_image_sdxl::stages::snap_side;
    use synaptix_image_sdxl::{SdxlCheckpoint, SdxlSampleParams};

    warn_unused(
        Family::Sdxl,
        &[
            ("--image", !args.image.is_empty()),
            ("--compute-dtype", args.compute_dtype.is_some()),
            ("--memory", args.memory != "auto"),
            ("--t5-len", args.t5_len.is_some()),
            ("--no-kv-cache", args.no_kv_cache),
        ],
    );
    let dev = device::resolve(&args.device);
    let quant = parse_quant(&args)?;
    let center_crop = parse_fit(&args.fit)?;
    let started = std::time::Instant::now();
    let ck = SdxlCheckpoint::open(&args.model, dev)?;
    let init_src = args.init_image.as_deref().map(load_rgb).transpose()?;
    let fallback = init_src.as_ref().map(image_size).unwrap_or((1024, 1024));
    let (width, height) = target_size(&args, fallback, snap_side);
    let steps = pick_steps(args.steps, 30);
    let guidance = args.guidance_scale.unwrap_or(5.0);
    let denoise = if init_src.is_some() { args.strength.unwrap_or(0.6) } else { 1.0 };
    eprintln!(
        "synaptix imagine [SDXL]: {} | {width}×{height} | {steps} steps | cfg {guidance} | seed {} | img2img {} | quant={quant:?} {dev:?}",
        args.model.display(),
        args.seed,
        init_src.as_ref().map(|_| format!("strength {denoise}")).unwrap_or_else(|| "нет".into()),
    );
    let cond = ck.encode_prompt(&args.prompt, args.negative.as_deref().unwrap_or(""))?;
    let init = match &init_src {
        Some(img) => {
            let fitted = synaptix_io::image::fit_image(img, width, height, center_crop)?;
            Some(ck.encode_image(&fitted)?)
        }
        None => None,
    };
    let unet = ck.load_unet(quant)?;
    let p = SdxlSampleParams { width, height, steps, guidance, seed: args.seed, denoise };
    let bar = progress_bar(steps);
    let lat = ck.sample(&unet, &cond, init.as_ref(), &p, &mut |i, _| {
        bar.set_position(i as u64);
        true
    })?;
    bar.finish_and_clear();
    drop(unet);
    let image = ck.decode(&lat)?;
    finish(&image, &args.output, "SDXL", started)
}

fn run_flux(args: ImagineArgs) -> R<()> {
    use synaptix_image_flux::model::snap_side;
    use synaptix_image_flux::{FluxModel, OffloadMode, SampleParams};

    warn_unused(
        Family::Flux,
        &[
            ("--negative", args.negative.is_some()),
            ("--image", !args.image.is_empty()),
            ("--no-kv-cache", args.no_kv_cache),
        ],
    );
    let dev = device::resolve(&args.device);
    let quant = parse_quant(&args)?;
    let compute = match parse_compute(args.compute_dtype.as_deref())? {
        Some(d) => d,
        None if !dev.is_cuda() => DType::F32,
        None if quant.is_quantized() => DType::F16,
        None => DType::BF16,
    };
    let quant = if dev.is_cuda() { quant } else { compute };
    let mode = match memory_label(&args.memory)? {
        "resident" => OffloadMode::Resident,
        "stream" => OffloadMode::Stream,
        _ => OffloadMode::Auto,
    };
    let center_crop = parse_fit(&args.fit)?;
    let started = std::time::Instant::now();
    let m = FluxModel::open(&args.model, dev, compute, quant)?.with_offload(mode);
    let distilled = m.guidance_distilled();
    let seq = args.t5_len.unwrap_or(m.default_max_seq_len());
    let init_src = args.init_image.as_deref().map(load_rgb).transpose()?;
    let fallback = init_src.as_ref().map(image_size).unwrap_or((1024, 1024));
    let (width, height) = target_size(&args, fallback, snap_side);
    let steps = pick_steps(args.steps, if distilled { 28 } else { 4 });
    let guidance = args.guidance_scale.unwrap_or(3.5);
    let denoise = if init_src.is_some() { args.strength.unwrap_or(0.75) } else { 1.0 };
    eprintln!(
        "synaptix imagine [FLUX.1 {}]: {} | {width}×{height} | {steps} steps | guidance {guidance} | T5 {seq} | seed {} | img2img {} | quant={quant:?} compute={compute:?} memory={mode:?} {dev:?}",
        if distilled { "dev" } else { "schnell" },
        args.model.display(),
        args.seed,
        init_src.as_ref().map(|_| format!("strength {denoise}")).unwrap_or_else(|| "нет".into()),
    );
    let cond = m.encode_prompt(&args.prompt, seq)?;
    let init = match &init_src {
        Some(img) => {
            let fitted = synaptix_io::image::fit_image(img, width, height, center_crop)?;
            Some(m.encode_image(&fitted)?)
        }
        None => None,
    };
    let t = m.load_transformer(FluxModel::tokens_for(width, height, seq))?;
    let p = SampleParams { width, height, steps, guidance, seed: args.seed, denoise };
    let bar = progress_bar(steps);
    let lat = m.sample(&t, &cond, init.as_ref(), &p, &mut |i, _| {
        bar.set_position(i as u64);
        true
    })?;
    bar.finish_and_clear();
    drop(t);
    let image = m.decode(&lat)?;
    finish(&image, &args.output, "FLUX.1", started)
}

fn run_flux2(args: ImagineArgs) -> R<()> {
    use synaptix_image_flux2::model::{reference_size, snap_side};
    use synaptix_image_flux2::{Flux2Model, Flux2Variant, MemoryMode, SampleParams};

    warn_unused(
        Family::Flux2,
        &[
            ("--compute-dtype", args.compute_dtype.is_some()),
            ("--t5-len", args.t5_len.is_some()),
            ("--no-kv-cache", args.no_kv_cache),
        ],
    );
    if args.image.len() > 10 {
        return Err(format!("FLUX.2 принимает до 10 референсов, передано {}", args.image.len()).into());
    }
    let dev = device::resolve(&args.device);
    let quant = parse_quant(&args)?;
    let mode = match memory_label(&args.memory)? {
        "resident" => MemoryMode::Resident,
        "stream" => MemoryMode::Stream,
        _ => MemoryMode::Auto,
    };
    let center_crop = parse_fit(&args.fit)?;
    let started = std::time::Instant::now();
    let m = Flux2Model::open(&args.model, dev, DType::BF16, quant)?.with_memory(mode);
    let variant = m.variant();
    if args.negative.is_some() {
        eprintln!("synaptix imagine: FLUX.2 не принимает свой негатив (klein base берёт пустой) — --negative игнорируется");
    }
    let mut references = Vec::with_capacity(args.image.len());
    for p in &args.image {
        let img = load_rgb(p)?;
        let (w, h) = image_size(&img);
        let (rw, rh) = reference_size(w, h);
        references.push(synaptix_io::image::fit_image(&img, rw, rh, true)?);
    }
    let init_src = args.init_image.as_deref().map(load_rgb).transpose()?;
    let fallback = init_src
        .as_ref()
        .map(image_size)
        .or(references.first().map(image_size))
        .unwrap_or((1024, 1024));
    let (width, height) = target_size(&args, fallback, snap_side);
    let steps = pick_steps(args.steps, variant.default_steps());
    let guidance = args.guidance_scale.unwrap_or(variant.default_guidance());
    let denoise = if init_src.is_some() { args.strength.unwrap_or(0.75) } else { 1.0 };
    if variant == Flux2Variant::KleinDistilled && args.guidance_scale.is_some() {
        eprintln!("synaptix imagine: дистиллированный klein не использует guidance — --cfg игнорируется");
    }
    eprintln!(
        "synaptix imagine [FLUX.2 {variant:?}]: {} | {width}×{height} | {steps} steps | guidance {guidance} | seed {} | референсов {} | img2img {} | quant={quant:?} memory={mode:?} {dev:?}",
        args.model.display(),
        args.seed,
        references.len(),
        init_src.as_ref().map(|_| format!("strength {denoise}")).unwrap_or_else(|| "нет".into()),
    );
    let cond = m.encode_prompt(&args.prompt, variant.uses_cfg(guidance))?;
    let refs = if references.is_empty() { None } else { Some(m.encode_references(&references)?) };
    let init = match &init_src {
        Some(img) => {
            let fitted = synaptix_io::image::fit_image(img, width, height, center_crop)?;
            Some(m.encode_image(&fitted)?)
        }
        None => None,
    };
    let ref_tokens = refs.as_ref().map(|r| r.num_tokens()).unwrap_or(0);
    let t = m.load_transformer(Flux2Model::tokens_for(width, height, ref_tokens))?;
    let p = SampleParams { width, height, steps, guidance, seed: args.seed, denoise };
    let bar = progress_bar(steps);
    let lat = m.sample(&t, &cond, refs.as_ref(), init.as_ref(), &p, &mut |i, _| {
        bar.set_position(i as u64);
        true
    })?;
    bar.finish_and_clear();
    drop(t);
    let image = m.decode(&lat)?;
    finish(&image, &args.output, "FLUX.2", started)
}

fn run_qwen(args: ImagineArgs) -> R<()> {
    use synaptix_image_qwen::model::snap_side;
    use synaptix_image_qwen::{MemoryMode, QwenImageModel, SampleParams};

    warn_unused(
        Family::QwenImage,
        &[
            ("--init-image", args.init_image.is_some()),
            ("--strength", args.strength.is_some()),
            ("--compute-dtype", args.compute_dtype.is_some()),
            ("--t5-len", args.t5_len.is_some()),
            ("--no-kv-cache", args.no_kv_cache),
        ],
    );
    let dev = device::resolve(&args.device);
    let quant = parse_quant(&args)?;
    let mode = match memory_label(&args.memory)? {
        "resident" => MemoryMode::Resident,
        "stream" => MemoryMode::Stream,
        _ => MemoryMode::Auto,
    };
    let started = std::time::Instant::now();
    let m = QwenImageModel::open(&args.model, dev, DType::BF16, quant)?.with_memory(mode);
    let variant = m.variant();
    if args.image.len() > variant.max_images() {
        return Err(format!(
            "{} принимает картинок: {}, передано {}",
            variant.label(),
            variant.max_images(),
            args.image.len()
        )
        .into());
    }
    if variant.is_edit() && args.image.is_empty() {
        return Err(format!("{} правит картинку — нужен --image", variant.label()).into());
    }
    let images: Vec<Tensor> = args.image.iter().map(|p| load_rgb(p)).collect::<R<_>>()?;
    let sizes: Vec<(usize, usize)> = images.iter().map(image_size).collect();
    let (width, height) = target_size(&args, m.default_size(&sizes), snap_side);
    let steps = pick_steps(args.steps, variant.default_steps());
    let cfg = args.guidance_scale.unwrap_or(4.0);
    let negative = args.negative.clone().unwrap_or_else(|| " ".to_string());
    let negative = (!negative.is_empty() && cfg > 1.0).then_some(negative);
    eprintln!(
        "synaptix imagine [{}]: {} | {width}×{height} | {steps} steps | true cfg {} | seed {} | картинок {} | quant={quant:?} memory={mode:?} {dev:?}",
        variant.label(),
        args.model.display(),
        if negative.is_some() { cfg.to_string() } else { "выкл".into() },
        args.seed,
        images.len(),
    );
    let cond = m.encode_prompt(&args.prompt, negative.as_deref(), &images)?;
    let refs = if images.is_empty() { None } else { Some(m.encode_references(&images)?) };
    let ref_tokens = refs.as_ref().map(|r| r.num_tokens()).unwrap_or(0);
    let txt = cond.embeds.dims()[1].max(cond.negative.as_ref().map(|n| n.dims()[1]).unwrap_or(0));
    let t = m.load_transformer(QwenImageModel::tokens_for(width, height, ref_tokens, txt))?;
    let p = SampleParams { width, height, steps, cfg, seed: args.seed };
    let bar = progress_bar(steps);
    let lat = m.sample(&t, &cond, refs.as_ref(), &p, &mut |i, _| {
        bar.set_position(i as u64);
        true
    })?;
    bar.finish_and_clear();
    drop(t);
    let image = m.decode(&lat)?;
    finish(&image, &args.output, variant.label(), started)
}

fn run_qwen21(args: ImagineArgs) -> R<()> {
    use synaptix_image_qwen21::model::snap_side;
    use synaptix_image_qwen21::{MemoryMode, QwenImage21Model, RgbaImage, SampleParams};

    warn_unused(
        Family::QwenImage21,
        &[
            ("--init-image", args.init_image.is_some()),
            ("--strength", args.strength.is_some()),
            ("--compute-dtype", args.compute_dtype.is_some()),
            ("--t5-len", args.t5_len.is_some()),
        ],
    );
    let dev = device::resolve(&args.device);
    let quant = parse_quant(&args)?;
    let mode = match memory_label(&args.memory)? {
        "resident" => MemoryMode::Resident,
        "stream" => MemoryMode::Stream,
        _ => MemoryMode::Auto,
    };
    let started = std::time::Instant::now();
    let m = QwenImage21Model::open(&args.model, dev, DType::BF16, quant)?.with_memory(mode);
    let mut images = Vec::with_capacity(args.image.len());
    for p in &args.image {
        let t = synaptix_io::image::png::load_image_rgba(p, Device::Cpu)?;
        images.push(RgbaImage::from_tensor(&t)?);
    }
    let sizes: Vec<(usize, usize)> = images.iter().map(|i| (i.width, i.height)).collect();
    let (width, height) =
        target_size(&args, QwenImage21Model::default_size(&sizes, args.resolution), snap_side);
    let steps = pick_steps(args.steps, synaptix_image_qwen21::model::DEFAULT_STEPS);
    let cfg = args.guidance_scale.unwrap_or(1.0);
    let negative = args.negative.as_deref().filter(|n| !n.trim().is_empty() && cfg > 1.0);
    eprintln!(
        "synaptix imagine [Qwen-Image 2.1]: {} | {width}×{height} | {steps} steps | cfg {} | seed {} | референсов {} | kv-cache {} | quant={quant:?} memory={mode:?} {dev:?}",
        args.model.display(),
        if negative.is_some() { cfg.to_string() } else { "выкл".into() },
        args.seed,
        images.len(),
        !args.no_kv_cache,
    );
    let cond = m.encode_prompt(&args.prompt, negative, &images, args.resolution)?;
    let refs = if images.is_empty() { None } else { Some(m.encode_references(&images, args.resolution)?) };
    let ref_tokens = refs.as_ref().map(|r| r.num_tokens()).unwrap_or(0);
    let txt = cond.embeds.dims()[1].max(cond.negative.as_ref().map(|n| n.0.dims()[1]).unwrap_or(0));
    let t = m.load_transformer(QwenImage21Model::tokens_for(width, height, ref_tokens, txt))?;
    let p = SampleParams { width, height, steps, cfg, seed: args.seed, kv_cache: !args.no_kv_cache };
    let bar = progress_bar(steps);
    let lat = m.sample(&t, &cond, refs.as_ref(), &p, &mut |i, _| {
        bar.set_position(i as u64);
        true
    })?;
    bar.finish_and_clear();
    drop(t);
    let image = m.decode(&lat)?;
    let transparent = RgbaImage::from_tensor(&image)?.has_transparency();
    let out = if transparent { image } else { image.narrow(0, 0, 3)?.contiguous()? };
    finish(&out, &args.output, if transparent { "Qwen-Image 2.1, RGBA" } else { "Qwen-Image 2.1" }, started)
}

pub fn run_depth(model: &Path, image: &Path, output: &Path, device_name: &str) -> R<()> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = device::resolve(device_name);
    let started = std::time::Instant::now();
    let net = synaptix_depth_anything::DepthAnything::load(model, dev)?;
    let img = load_rgb(image)?.to_device(dev)?;
    let depth = net.depth_rgb(&img)?.to_device(Device::Cpu)?;
    synaptix_io::image::save_image(&depth, output)?;
    eprintln!(
        "synaptix depth: {} → {} ({:.1}s)",
        image.display(),
        output.display(),
        started.elapsed().as_secs_f32()
    );
    Ok(())
}
