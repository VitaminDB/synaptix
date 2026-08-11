use std::path::PathBuf;
use std::process::Command;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_video_minimax_h3 as h3;

pub struct H3Args {
    pub model_dir: PathBuf,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub output: PathBuf,
    pub encoder: Option<PathBuf>,
    pub first_frame: Option<PathBuf>,
    pub last_frame: Option<PathBuf>,
    pub width: usize,
    pub height: usize,
    pub duration: f64,
    pub frames: Option<usize>,
    pub steps: usize,
    pub cfg_scale: f32,
    pub seed: Option<u64>,
    pub lora: Option<PathBuf>,
    pub lora_strength: f32,
    pub quant_transformer: Option<String>,
    pub quant_encoder: Option<String>,
    pub compute_dtype: Option<String>,
    pub memory_mode: String,
    pub pipeline: Option<String>,
    pub list_pipelines: bool,
    pub variant: Option<String>,
    pub device: usize,
    pub prof: bool,
    pub keep_wav: bool,
}

fn parse_dtype(s: Option<&str>, default: DType) -> Result<DType, String> {
    match s.map(|x| x.to_lowercase()) {
        None => Ok(default),
        Some(q) => match q.as_str() {
            "none" | "bf16" => Ok(DType::BF16),
            "f16" => Ok(DType::F16),
            "f32" => Ok(DType::F32),
            "nvfp4" => Ok(DType::NVFP4),
            "mxfp8" | "fp8" => Ok(DType::MXFP8),
            other => Err(format!("неизвестный dtype: {other} (bf16|f16|mxfp8|nvfp4)")),
        },
    }
}

pub fn run(args: H3Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.list_pipelines {
        for s in h3::spec::registry() {
            let lora = if s.needs_lora { " [нужна Turbo LoRA]" } else { "" };
            println!(
                "{:<14} {:<7} шагов {:<3} cfg {:<4} — {}{}",
                s.name,
                s.variant.dir_name(),
                s.default_steps,
                s.default_cfg,
                s.desc,
                lora
            );
        }
        return Ok(());
    }

    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();

    h3::runtime::set_h3_prof(args.prof);
    let mode = h3::H3MemoryMode::parse(&args.memory_mode)
        .ok_or_else(|| format!("неизвестный memory-mode: {}", args.memory_mode))?;
    mode.install();

    let device = Device::Cuda(args.device);
    let compute = parse_dtype(args.compute_dtype.as_deref(), DType::BF16)?;
    let quant_dit = parse_dtype(args.quant_transformer.as_deref(), DType::NVFP4)?;
    let quant_enc = parse_dtype(args.quant_encoder.as_deref(), DType::MXFP8)?;

    let variant = args
        .variant
        .as_deref()
        .and_then(h3::config::H3Variant::parse)
        .unwrap_or(h3::config::H3Variant::Fl2va);
    let paths = h3::H3Paths::open_variant(&args.model_dir, variant)?;

    let spec = match &args.pipeline {
        Some(name) => Some(
            h3::spec::by_name(name)
                .ok_or_else(|| format!("неизвестный пайплайн: {name}"))?,
        ),
        None => None,
    };
    let steps = if args.steps > 0 {
        args.steps
    } else {
        spec.as_ref().map(|s| s.default_steps).unwrap_or(20)
    };
    let cfg_scale = if args.cfg_scale > 0.0 {
        args.cfg_scale
    } else {
        spec.as_ref().map(|s| s.default_cfg).unwrap_or(5.0)
    };

    let frame_count = args
        .frames
        .unwrap_or_else(|| h3::config::frames_for_duration(args.duration));
    let geometry = h3::pipeline::Geometry::new(args.width, args.height, frame_count);
    eprintln!(
        "[h3] {}x{} {} кадров ({:.2} с), латент {}x{}x{}, аудио {} латент-кадров",
        geometry.width,
        geometry.height,
        geometry.frame_count,
        geometry.frame_count as f64 / h3::config::FPS,
        geometry.latent_t,
        geometry.latent_h,
        geometry.latent_w,
        geometry.audio_t
    );

    let encoder_dir = args.encoder.unwrap_or_else(|| paths.text_encoder_dir());
    eprintln!("[h3] загрузка энкодера Qwen3-VL из {}", encoder_dir.display());
    let encoder = h3::text_encoder::EncoderHandle::load(&encoder_dir, device, quant_enc)?;

    let mut images: Vec<(Tensor, h3::text_encoder::ImageGrid)> = Vec::new();
    let mut keyframe_paths: Vec<(usize, PathBuf)> = Vec::new();
    if let Some(p) = &args.first_frame {
        keyframe_paths.push((0, p.clone()));
    }
    if let Some(p) = &args.last_frame {
        keyframe_paths.push((geometry.frame_count - 1, p.clone()));
    }
    let mut keyframe_rgb: Vec<Tensor> = Vec::new();
    for (_, p) in &keyframe_paths {
        let img = synaptix_io::image::png::load_image(p, Device::Cpu)?;
        let (patches, grid) = encoder.prepare_image(&img)?;
        images.push((patches, grid));
        keyframe_rgb.push(img);
    }

    let merge = encoder.merge_size();
    let grids: Vec<h3::text_encoder::ImageGrid> = images.iter().map(|(_, g)| *g).collect();
    let presentation = if grids.is_empty() {
        h3::text_encoder::presentation_t2va(&args.prompt)
    } else {
        h3::text_encoder::presentation_fl2va(&args.prompt, &grids, merge)
    };

    eprintln!("[h3] кодирование промпта");
    let cond = encoder.encode(&presentation, &images)?;
    let negative = match &args.negative_prompt {
        Some(np) if cfg_scale > 1.0 => {
            let np_pres = h3::text_encoder::presentation_t2va(np);
            Some(encoder.encode(&np_pres, &[])?)
        }
        _ => None,
    };
    drop(encoder);
    h3::memory::trim_pool(device);

    eprintln!("[h3] загрузка DiT ({})", paths.transformer_dir().display());
    let mut ckpt = h3::H3Checkpoint::open(paths.clone(), device, compute)?;
    if let Some(lp) = &args.lora {
        let lw = h3::LoraWeights::open(lp, device, args.lora_strength)?;
        ckpt = ckpt.with_lora(std::sync::Arc::new(lw));
        eprintln!("[h3] LoRA {} (сила {})", lp.display(), args.lora_strength);
    }

    let sched = h3::H3Scheduler::new(
        steps,
        ckpt.config.sigma_shift_video as f64,
        ckpt.config.sigma_shift_audio as f64,
    );

    let conditioning = h3::pipeline::Conditioning {
        context: cond.hidden,
        text_tags: cond.tags,
    };
    let mut req = h3::pipeline::DenoiseRequest::new(geometry, &conditioning);
    let neg_cond;
    if let Some(n) = negative {
        neg_cond = h3::pipeline::Conditioning { context: n.hidden, text_tags: n.tags };
        req.negative = Some(&neg_cond);
        req.guider = h3::guider::GuiderParams::cfg(cfg_scale);
    }
    req.seed = args.seed;
    req.keyframes = keyframe_paths
        .iter()
        .map(|(i, _)| h3::layout::Keyframe { resolved_frame_index: *i })
        .collect();

    let dit = h3::dit::H3Dit::load(&ckpt, device, compute, quant_dit)?;
    let prep = h3::pipeline::prepare(&dit, &req, &sched)?;

    if !keyframe_rgb.is_empty() {
        eprintln!("[h3] кодирование ключевых кадров через VAE");
        let vae_cfg = ckpt.vae_config()?;
        let vw = h3::loader::ComponentLoader::open_file(paths.video_vae_file(), device)?;
        let enc = h3::vae::VaeEncoder::load(&vw, vae_cfg, device, compute)?;
        let mut latents = Vec::with_capacity(keyframe_rgb.len());
        for img in &keyframe_rgb {
            let d = img.dims().to_vec();
            let x = img
                .to_device(device)?
                .reshape(vec![1, d[0], 1, d[1], d[2]])?
                .mul_scalar(2.0)?
                .add_scalar(-1.0)?;
            latents.push(enc.encode(&x)?);
        }
        drop(enc);
        h3::memory::trim_pool(device);
        req.cond_rows.video = h3::pipeline::cond_rows_from_keyframe_latents(
            &latents,
            dit.cfg.patch_size,
            None,
            args.seed.unwrap_or(0),
        )?;
    }

    let plan = h3::memory::plan(
        &ckpt,
        &prep.plan,
        prep.layout.seq_len,
        quant_dit,
        compute,
        compute,
        mode,
        device,
    )?;
    eprintln!("[h3] {}", plan.summary());

    eprintln!("[h3] предвычисление adaLN на {steps} шагов");
    let cache = h3::pipeline::build_adaln_cache(&dit, &ckpt, &prep, compute)?;

    eprintln!("[h3] денойзинг: {steps} шагов, cfg {cfg_scale}");
    let progress = |p: h3::pipeline::DenoiseProgress| {
        eprint!("\r[h3] шаг {}/{} sigma {:.4}   ", p.step, p.total, p.sigma);
    };
    let hooks = h3::pipeline::DenoiseHooks {
        progress: Some(&progress),
        cancel: None,
    };
    let out = h3::pipeline::denoise_av(&dit, &cache, &prep, &req, &sched, &hooks)?;
    eprintln!();

    drop(cache);
    drop(prep);
    drop(dit);
    h3::memory::trim_pool(device);

    eprintln!("[h3] декодирование видео");
    let vae_cfg = ckpt.vae_config()?;
    let vw = h3::loader::ComponentLoader::open_file(paths.video_vae_file(), device)?;
    let vdec = h3::vae::VaeDecoder::load(&vw, vae_cfg, device, compute)?;
    let rgb = vdec.decode(&out.video_latent)?;
    drop(vdec);
    drop(vw);
    h3::memory::trim_pool(device);

    eprintln!("[h3] декодирование звука");
    let acfg = ckpt.audio_vae_config()?;
    let aw = h3::loader::ComponentLoader::open_file(paths.audio_vae_file(), device)?;
    let adec = h3::audio_vae::AudioVae::load_decoder(&aw, acfg, device, compute)?;
    let wave = adec.decode(&out.audio_latent)?;
    let sample_rate = adec.sample_rate();
    drop(adec);
    h3::memory::trim_pool(device);

    write_mp4(&rgb, Some(&wave), sample_rate, h3::config::FPS, &args.output, args.keep_wav)?;
    eprintln!("[h3] записано: {}", args.output.display());
    Ok(())
}

fn write_wav(path: &PathBuf, wave: &Tensor, sample_rate: usize) -> Result<(), Box<dyn std::error::Error>> {
    let pcm = h3::audio_vae::interleave_stereo(wave)?;
    let channels = wave.dims()[1] as u16;
    let bits = 16u16;
    let byte_rate = sample_rate as u32 * channels as u32 * (bits / 8) as u32;
    let block_align = channels * (bits / 8);
    let data_len = (pcm.len() * 2) as u32;
    let mut buf = Vec::with_capacity(44 + pcm.len() * 2);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVEfmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&(sample_rate as u32).to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for s in &pcm {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, buf)?;
    Ok(())
}

fn write_mp4(
    rgb: &Tensor,
    wave: Option<&Tensor>,
    sample_rate: usize,
    fps: f64,
    out: &PathBuf,
    keep_wav: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join("synaptix_h3_frames");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;

    let frames = h3::vae::rgb_to_frames(rgb)?;
    for (i, fr) in frames.iter().enumerate() {
        let (h, w) = (fr.dims()[1], fr.dims()[2]);
        let planar: Vec<f32> = fr
            .to_device(Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape(vec![3 * h * w])?
            .to_vec1::<f32>()?;
        let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
        ppm.reserve(3 * h * w);
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let v = planar[c * h * w + y * w + x];
                    ppm.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
        }
        std::fs::write(dir.join(format!("f{i:05}.ppm")), ppm)?;
    }

    let wav = dir.join("audio.wav");
    if let Some(w) = wave {
        write_wav(&wav, w, sample_rate)?;
    }

    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y")
        .arg("-framerate")
        .arg(format!("{fps}"))
        .arg("-i")
        .arg(dir.join("f%05d.ppm"));
    if wave.is_some() {
        cmd.arg("-i").arg(&wav).arg("-c:a").arg("aac").arg("-b:a").arg("192k").arg("-shortest");
    }
    let status = cmd
        .arg("-c:v")
        .arg("libx264")
        .arg("-crf")
        .arg("17")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(out)
        .status()?;
    if !status.success() {
        return Err("ffmpeg завершился с ошибкой".into());
    }
    if keep_wav {
        if let Some(parent) = out.parent() {
            let dest = parent.join(format!(
                "{}.wav",
                out.file_stem().and_then(|s| s.to_str()).unwrap_or("audio")
            ));
            let _ = std::fs::copy(&wav, dest);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
