//! Видео одним вызовом. MiniMax-H3: видео со звуком по описанию, по первому
//! и/или последнему кадру (fl2va). Порядок загрузки этапов — как у
//! `synaptix h3`: энкодер → (VAE ключевых кадров) → DiT → VAE видео → VAE
//! звука, между этапами пулы ужимаются. Прогресс по шагам денойза, отмена
//! флагом. Результат — mp4 (ffmpeg).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::AtomicBool;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_video_minimax_h3 as h3;

type R<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

#[derive(Debug, Clone, Default)]
pub struct VideoRequest {
    pub prompt: String,
    pub negative: Option<String>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    /// Длительность, секунды.
    pub seconds: Option<f64>,
    pub steps: Option<usize>,
    pub guidance: Option<f32>,
    pub seed: u64,
    /// Первый кадр (картинка).
    pub first_frame: Option<PathBuf>,
    /// Последний кадр (картинка).
    pub last_frame: Option<PathBuf>,
    /// Референсы (ref2va): картинки (до 9), видео (до 3), звук (до 3 — голос
    /// для липсинка, музыка), всего до 12. Не сочетаются с ключевыми кадрами.
    pub refs: Vec<PathBuf>,
    /// Квант трансформера и энкодера: `nvfp4`, `mxfp8`, `bf16` (плотный);
    /// пусто/`auto` — умолчание модели.
    pub quant: Option<String>,
}

/// Квант из настройки; `None` — умолчание пайплайна.
fn quant_setting(q: Option<&str>) -> Option<DType> {
    match q.unwrap_or("").to_ascii_lowercase().as_str() {
        "nvfp4" => Some(DType::NVFP4),
        "mxfp8" | "fp8" => Some(DType::MXFP8),
        "bf16" | "dense" | "none" => Some(DType::BF16),
        _ => None,
    }
}

pub struct VideoOutput {
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps: f64,
    pub label: String,
}

/// H3: видео в `out` (mp4). `progress(шаг, всего)`.
pub fn h3(model: &Path, encoder: Option<&Path>, device: Device, req: &VideoRequest, out: &Path, cancel: &AtomicBool, progress: &(dyn Fn(usize, usize) + Sync)) -> R<VideoOutput> {
    if !device.is_cuda() {
        return Err("MiniMax-H3 работает только на видеокарте".into());
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let mode = h3::H3MemoryMode::parse("auto").ok_or("режим памяти")?;
    mode.install();
    let compute = DType::BF16;
    let q = quant_setting(req.quant.as_deref()).unwrap_or(DType::NVFP4);
    let (quant_dit, quant_enc) = (q, q);
    h3::runtime::set_sigma_shift(None, None);
    h3::runtime::set_vae_tile(None);
    h3::runtime::set_load_reserve_bytes(13 << 29);
    let snap = |v: usize| (v / 16).max(1) * 16;
    let width = snap(req.width.filter(|&v| v > 0).unwrap_or(1344));
    let height = snap(req.height.filter(|&v| v > 0).unwrap_or(768));
    let steps = req.steps.filter(|&s| s > 0).unwrap_or(20);
    let cfg_scale = req.guidance.unwrap_or(5.0);
    let seed = req.seed;

    let ref_sources = req.refs.iter().map(|p| h3::refs::RefSource::from_path(p, true)).collect::<Result<Vec<_>, _>>().map_err(err)?;
    h3::refs::validate(&ref_sources).map_err(err)?;
    if !ref_sources.is_empty() && (req.first_frame.is_some() || req.last_frame.is_some()) {
        return Err("референсы и ключевые кадры — разные варианты H3 (ref2va и fl2va), вместе не идут".into());
    }
    let variant = if ref_sources.is_empty() { h3::config::H3Variant::Fl2va } else { h3::config::H3Variant::Ref2va };
    let source = h3::H3Source::open(model, variant).map_err(err)?;
    let frame_count = h3::config::frames_for_duration(req.seconds.filter(|s| *s > 0.0).unwrap_or(5.0));
    let geometry = h3::pipeline::Geometry::new(width, height, frame_count);

    let encoder_source = match encoder {
        Some(p) => h3::H3EncoderSource::open(p).map_err(err)?,
        None => h3::H3EncoderSource::from_model(&source).ok_or_else(|| format!("в {} нет энкодера Qwen3-VL", source.path().display()))?,
    };
    let encoder = h3::text_encoder::EncoderHandle::load_source(&encoder_source, device, compute, quant_enc).map_err(err)?;

    let mut images: Vec<(Tensor, h3::text_encoder::ImageGrid)> = Vec::new();
    let mut keyframes: Vec<(usize, Tensor)> = Vec::new();
    for (idx, p) in [(0usize, &req.first_frame), (geometry.frame_count - 1, &req.last_frame)] {
        let Some(p) = p else { continue };
        let raw = synaptix_io::image::png::load_image(p, Device::Cpu).map_err(|e| format!("{}: {e}", p.display()))?;
        let img = synaptix_io::image::fit_image(&raw, geometry.width, geometry.height, true).map_err(err)?;
        images.push(encoder.prepare_image(&img).map_err(err)?);
        keyframes.push((idx, img));
    }
    let ref_media = if ref_sources.is_empty() {
        Vec::new()
    } else {
        h3::refs::decode(
            &ref_sources,
            &h3::refs::RefOptions { image_size: h3::refs::RefImageSize::Match, target_width: geometry.width, target_height: geometry.height, frame_count: geometry.frame_count },
        )
        .map_err(err)?
    };
    let merge = encoder.merge_size();
    let grids: Vec<h3::text_encoder::ImageGrid> = images.iter().map(|(_, g)| *g).collect();
    let presentation = if !ref_media.is_empty() {
        let inputs = h3::refs::encoder_inputs(&encoder, &ref_media).map_err(err)?;
        images = inputs.vision;
        h3::text_encoder::presentation_ref2va(&req.prompt, &inputs.items, merge)
    } else if grids.is_empty() {
        h3::text_encoder::presentation_t2va(&req.prompt)
    } else {
        h3::text_encoder::presentation_fl2va(&req.prompt, &grids, merge)
    };
    let cond = encoder.encode(&presentation, &images).map_err(err)?;
    let negative = match req.negative.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
        Some(np) if cfg_scale > 1.0 => Some(encoder.encode(&h3::text_encoder::presentation_t2va(np), &[]).map_err(err)?),
        _ => None,
    };
    drop(encoder);
    h3::memory::trim_pool(device);

    let ckpt = h3::H3Checkpoint::open_source(source.clone(), device, compute).map_err(err)?;
    let sched = h3::H3Scheduler::new(steps, ckpt.config.sigma_shift_video as f64, ckpt.config.sigma_shift_audio as f64);
    let conditioning = h3::pipeline::Conditioning { context: cond.hidden, text_tags: cond.tags };
    let mut dreq = h3::pipeline::DenoiseRequest::new(geometry, &conditioning);
    let neg_cond;
    if let Some(n) = negative {
        neg_cond = h3::pipeline::Conditioning { context: n.hidden, text_tags: n.tags };
        dreq.negative = Some(&neg_cond);
        dreq.guider = h3::guider::GuiderParams { cfg_scale, rescale: 0.0, skip_steps: 0 };
    }
    dreq.seed = Some(seed);
    dreq.keyframes = keyframes.iter().map(|(i, _)| h3::layout::Keyframe { resolved_frame_index: *i }).collect();

    if !ref_media.is_empty() {
        if !ckpt.config.supports_references() {
            return Err(format!("{} — вариант FL2VA, референсы понимает только Ref2VA", source.path().display()));
        }
        let vw = h3::loader::ComponentLoader::open_component(&source, h3::H3Component::VideoVae, device).map_err(err)?;
        let enc = h3::vae::VaeEncoder::load(&vw, ckpt.vae_config().map_err(err)?, device, compute).map_err(err)?;
        let has_audio = ref_media.iter().any(|m| match m {
            h3::refs::RefMedia::Audio(_) => true,
            h3::refs::RefMedia::Video(v) => v.audio.is_some(),
            h3::refs::RefMedia::Image(_) => false,
        });
        let audio_enc = if has_audio {
            let aw = h3::loader::ComponentLoader::open_component(&source, h3::H3Component::AudioVae, device).map_err(err)?;
            Some(h3::audio_vae::AudioVae::load_full(&aw, ckpt.audio_vae_config().map_err(err)?, device, compute).map_err(err)?)
        } else {
            None
        };
        let latents = h3::refs::encode_latents(&ref_media, &enc, audio_enc.as_ref(), ckpt.config.patch_size, device, seed).map_err(err)?;
        drop(enc);
        drop(audio_enc);
        h3::memory::trim_pool(device);
        dreq.refs = latents.blocks;
        dreq.cond_rows = latents.cond_rows;
    }
    drop(ref_media);
    let dit = h3::dit::H3Dit::load(&ckpt, device, compute, quant_dit).map_err(err)?;
    let prep = h3::pipeline::prepare(&dit, &dreq, &sched).map_err(err)?;
    if !keyframes.is_empty() {
        let vw = h3::loader::ComponentLoader::open_component(&source, h3::H3Component::VideoVae, device).map_err(err)?;
        let enc = h3::vae::VaeEncoder::load(&vw, ckpt.vae_config().map_err(err)?, device, compute).map_err(err)?;
        let mut latents = Vec::with_capacity(keyframes.len());
        for (_, img) in &keyframes {
            let d = img.dims().to_vec();
            let x = img.to_device(device).map_err(err)?.reshape(vec![1, d[0], 1, d[1], d[2]]).map_err(err)?.mul_scalar(2.0).map_err(err)?.add_scalar(-1.0).map_err(err)?;
            latents.push(enc.encode(&x).map_err(err)?);
        }
        drop(enc);
        h3::memory::trim_pool(device);
        dreq.cond_rows.video = h3::pipeline::cond_rows_from_keyframe_latents(&latents, dit.cfg.patch_size, None, seed).map_err(err)?;
    }
    let _plan = h3::memory::plan(&ckpt, &prep.plan, prep.layout.seq_len, quant_dit, compute, compute, mode, device).map_err(err)?;
    let cache = h3::pipeline::build_adaln_cache(&dit, &ckpt, &prep, compute).map_err(err)?;
    let on_step = |p: h3::pipeline::DenoiseProgress| progress(p.step, p.total);
    let hooks = h3::pipeline::DenoiseHooks { progress: Some(&on_step), cancel: Some(cancel) };
    let lat = h3::pipeline::denoise_av(&dit, &cache, &prep, &dreq, &sched, &hooks).map_err(err)?;
    drop(cache);
    drop(prep);
    drop(dit);
    h3::memory::trim_pool(device);

    let vw = h3::loader::ComponentLoader::open_component(&source, h3::H3Component::VideoVae, device).map_err(err)?;
    let vdec = h3::vae::VaeDecoder::load(&vw, ckpt.vae_config().map_err(err)?, device, compute).map_err(err)?;
    let rgb = vdec.decode(&lat.video_latent).map_err(err)?;
    drop(vdec);
    drop(vw);
    h3::memory::trim_pool(device);
    let aw = h3::loader::ComponentLoader::open_component(&source, h3::H3Component::AudioVae, device).map_err(err)?;
    let adec = h3::audio_vae::AudioVae::load_decoder(&aw, ckpt.audio_vae_config().map_err(err)?, device, compute).map_err(err)?;
    let wave = adec.decode(&lat.audio_latent).map_err(err)?;
    let sample_rate = adec.sample_rate();
    drop(adec);
    h3::memory::trim_pool(device);

    let frames = h3::vae::rgb_to_frames(&rgb).map_err(err)?;
    let pcm = h3::audio_vae::interleave_stereo(&wave).map_err(err)?;
    let channels = wave.dims()[1] as u16;
    write_mp4(&frames, Some((&pcm, sample_rate as u32, channels)), h3::config::FPS, out)?;
    let label = format!("MiniMax-H3 {width}×{height}, {:.1} с, {steps} шагов{}", geometry.frame_count as f64 / h3::config::FPS, if keyframes.is_empty() { String::new() } else { format!(", ключевых кадров {}", keyframes.len()) }) + &if req.refs.is_empty() { String::new() } else { format!(", референсов {}", req.refs.len()) };
    Ok(VideoOutput { width, height, frames: geometry.frame_count, fps: h3::config::FPS, label })
}

/// Кадры `[3,H,W]` 0..1 и звук (перемежённые отсчёты) → mp4 через ffmpeg.
pub fn write_mp4(frames: &[Tensor], audio: Option<(&[f32], u32, u16)>, fps: f64, out: &Path) -> R<()> {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("synaptix_gen_{}_{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(err)?;
    let result = (|| -> R<()> {
        for (i, fr) in frames.iter().enumerate() {
            let (h, w) = (fr.dims()[1], fr.dims()[2]);
            let planar: Vec<f32> = fr.to_device(Device::Cpu).map_err(err)?.to_dtype(DType::F32).map_err(err)?.reshape(vec![3 * h * w]).map_err(err)?.to_vec1::<f32>().map_err(err)?;
            let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
            ppm.reserve(3 * h * w);
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        ppm.push((planar[c * h * w + y * w + x].clamp(0.0, 1.0) * 255.0).round() as u8);
                    }
                }
            }
            std::fs::write(dir.join(format!("f{i:05}.ppm")), ppm).map_err(err)?;
        }
        let wav = dir.join("audio.wav");
        if let Some((pcm, sr, ch)) = audio {
            std::fs::write(&wav, crate::music::wav_bytes(pcm, sr, ch)).map_err(err)?;
        }
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error", "-y"]).arg("-framerate").arg(format!("{fps}")).arg("-i").arg(dir.join("f%05d.ppm"));
        if audio.is_some() {
            cmd.arg("-i").arg(&wav).args(["-c:a", "aac", "-b:a", "192k", "-shortest"]);
        }
        let status = cmd.args(["-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]).arg(out).status().map_err(|e| format!("ffmpeg: {e}"))?;
        if !status.success() {
            return Err("ffmpeg завершился с ошибкой".into());
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// Компоненты LTX-2.3: чекпойнт (DiT+VAE+вокодер), Gemma-3 (текст), distilled
/// LoRA (нужна dev-чекпойнту для дистиллированных сигм).
pub struct LtxPaths {
    pub model: PathBuf,
    pub gemma: PathBuf,
    pub distilled_lora: Option<PathBuf>,
}

/// LTX-2.3: видео со звуком одной стадией (дистиллированные сигмы, 8 шагов),
/// по описанию или с первого кадра (`first_frame`). Порядок — как у
/// `synaptix video`: Gemma (выгружается) → коннекторы → DiT → VAE → звук.
pub fn ltx(paths: &LtxPaths, device: Device, req: &VideoRequest, out: &Path, cancel: &AtomicBool, progress: &(dyn Fn(usize, usize) + Sync)) -> R<VideoOutput> {
    use synaptix_core::precision::PrecisionConfig;
    use synaptix_llm_gemma3::pipeline::GemmaPipeline;
    use synaptix_video_ltx23::audio_vae::AudioVaeDecoder;
    use synaptix_video_ltx23::dit::AvDit;
    use synaptix_video_ltx23::loader::{LoraWeights, LtxCheckpoint};
    use synaptix_video_ltx23::pipeline as lp;
    use synaptix_video_ltx23::text_encoder::{AudioTextConditioner, VideoTextConditioner};
    use synaptix_video_ltx23::vae::{VaeDecoder, VaeEncoder};
    use synaptix_video_ltx23::vocoder::VocoderWithBwe;
    if !device.is_cuda() {
        return Err("LTX-2.3 работает только на видеокарте".into());
    }
    if !req.refs.is_empty() || req.last_frame.is_some() {
        return Err("LTX-2.3 здесь умеет видео по описанию и с первого кадра; референсы и последний кадр — у MiniMax-H3".into());
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let ord = if let Device::Cuda(o) = device { o } else { 0 };
    let release = || {
        let _ = synaptix_core::device::cuda::synchronize_all(ord);
        let _ = synaptix_core::memory::cuda_pool::hard_trim_cuda_mempool_device(ord);
    };
    let compute = DType::BF16;
    let quant_dit = quant_setting(req.quant.as_deref()).unwrap_or(compute);
    let quant_enc = match quant_dit {
        DType::NVFP4 => DType::NVFP4,
        _ => DType::MXFP8,
    };
    let fps = 24.0;
    let width = req.width.filter(|&v| v > 0).unwrap_or(960);
    let height = req.height.filter(|&v| v > 0).unwrap_or(544);
    let (hp, wp) = lp::latent_grid(width, height);
    let fp = lp::frames_for_duration(req.seconds.filter(|s| *s > 0.0).unwrap_or(5.0), fps);
    let seed = Some(req.seed.max(1));

    let ckpt = LtxCheckpoint::open(&paths.model, Device::Cpu, DType::BF16).map_err(|e| format!("LTX: {e}"))?;
    // Плотный DiT — всегда стриминг блоков с хоста; квантованный — на карту,
    // если влезает с VAE и активациями (оценка как у CLI).
    let offload = quant_dit == compute || {
        let dit_b = synaptix_video_ltx23::dit::dit_resident_bytes(&ckpt, quant_dit, compute);
        let vae_b: usize = ckpt.infos().filter(|(n, _, _)| n.starts_with("vae.")).map(|(_, _, s)| compute.bytes_for_numel(s.iter().product())).sum();
        let act_b = fp * hp * wp * 4096 * 2 * 28;
        let pad_b = (fp * hp * wp + 256) * (16384 * 2 + 16384 * 2 + 12288 * 2);
        let need = dit_b + vae_b + act_b + pad_b + (1usize << 30);
        synaptix_core::device::cuda::mem_info(ord).map(|(free, _)| need > free).unwrap_or(true)
    };
    let pin = synaptix_core::device::cuda::OffloadPinCacheGuard::new_paused(&ckpt.shard_bytes());
    let distilled_name = paths.model.file_name().map(|n| n.to_string_lossy().to_lowercase().contains("distilled")).unwrap_or(false);
    let lora = match (&paths.distilled_lora, distilled_name) {
        (Some(p), false) => Some(LoraWeights::open(p, device, 1.0).map_err(|e| format!("LoRA {}: {e}", p.display()))?),
        _ => None,
    };

    synaptix_core::device::cuda::set_offload_pinned(true);
    let (states, mask, nag_states) = {
        let prec = PrecisionConfig { compute, attn_w: quant_enc, mlp_w: quant_enc, lm_head: DType::BF16, embed: DType::BF16, kv: DType::BF16 };
        let gemma = GemmaPipeline::load_with_precision(&paths.gemma, device, prec, Some(1024)).map_err(|e| format!("Gemma: {e}"))?;
        pin.resume();
        let (s, m) = gemma.encode_for_ltx(&req.prompt, 1024, device).map_err(|e| format!("Gemma: {e}"))?;
        let nag = gemma.encode_for_ltx(lp::DEFAULT_NAG_PROMPT, 1024, device).map_err(|e| format!("Gemma: {e}"))?;
        (s, m, nag)
    };
    release();
    let (v_enc, a_enc, nag_enc) = {
        let ckpt_gpu = ckpt.view_on(device);
        let vtc = VideoTextConditioner::load(&ckpt_gpu, device, compute).map_err(err)?;
        let v = vtc.forward(&states, &mask).map_err(err)?;
        let a = AudioTextConditioner::load(&ckpt_gpu, device, compute).map_err(err)?.forward(&states, &mask).map_err(err)?;
        let n = vtc.forward(&nag_states.0, &nag_states.1).map_err(err)?;
        (v, a, n)
    };
    drop(states);
    drop(nag_states);
    synaptix_core::device::cuda::set_offload_pinned(false);
    let v_nag = Some((&nag_enc, lp::NAG_DEFAULT_SCALE, lp::NAG_DEFAULT_ALPHA, lp::NAG_DEFAULT_TAU));

    let on_step = |p: lp::DenoiseProgress| progress(p.step + 1, p.total);
    let hooks = lp::DenoiseHooks { progress: Some(&on_step), cancel: Some(cancel) };
    let (latent, a_tok) = synaptix_core::grad::no_grad(|| -> R<(Tensor, Tensor)> {
        let lora_ckpt = lora.map(|l| ckpt.view_on(Device::Cpu).with_lora(std::sync::Arc::new(l)));
        let dit = AvDit::load_with(lora_ckpt.as_ref().unwrap_or(&ckpt), device, compute, quant_dit, offload).map_err(|e| format!("DiT: {e}"))?;
        let conds = match &req.first_frame {
            Some(p) => {
                let raw = synaptix_io::image::load_image(p, Device::Cpu).map_err(|e| format!("{}: {e}", p.display()))?;
                let enc = VaeEncoder::load(&ckpt, device).map_err(|e| format!("VAE: {e}"))?;
                let (ph, pw) = (hp * 32, wp * 32);
                let img = synaptix_io::image::fit_image(&raw, pw, ph, true).map_err(err)?.to_device(device).map_err(err)?.affine(2.0, -1.0).map_err(err)?.contiguous().map_err(err)?.reshape(vec![1, 3, 1, ph, pw]).map_err(err)?;
                let lat = enc.encode(&img).map_err(|e| format!("VAE: {e}"))?;
                vec![(0usize, lp::frame_latent_to_tokens(&lat).map_err(err)?, 1.0f32)]
            }
            None => Vec::new(),
        };
        let r = lp::denoise_av(&dit, &v_enc, &a_enc, fp, hp, wp, &lp::DISTILLED_SIGMAS, None, None, fps, device, v_nag, &conds, seed, &hooks).map_err(err)?;
        drop(dit);
        synaptix_core::tensor::ops::conv_filter_cache_gc();
        Ok(r)
    })?;
    release();
    let rgb = synaptix_core::grad::no_grad(|| -> R<Tensor> {
        let vae = VaeDecoder::load(&ckpt, device).map_err(|e| format!("VAE: {e}"))?;
        vae.decode(&latent).map_err(|e| format!("VAE: {e}"))
    })?;
    release();
    let wave = synaptix_core::grad::no_grad(|| -> R<Tensor> {
        let avae = AudioVaeDecoder::load(&ckpt, device).map_err(|e| format!("VAE звука: {e}"))?;
        let voc = VocoderWithBwe::load(&paths.model, device).map_err(|e| format!("вокодер: {e}"))?;
        lp::decode_audio_tokens(&avae, &voc, &a_tok).map_err(err)
    })?;
    release();
    let frames = lp::rgb_to_frames(&rgb).map_err(err)?;
    let (ch, l) = (wave.dims()[1], wave.dims()[2]);
    let planar: Vec<f32> = wave.reshape(vec![ch * l]).map_err(err)?.to_dtype(DType::F32).map_err(err)?.to_vec1::<f32>().map_err(err)?;
    let mut pcm = Vec::with_capacity(ch * l);
    for i in 0..l {
        for c in 0..ch {
            pcm.push(planar[c * l + i]);
        }
    }
    let n = frames.len();
    write_mp4(&frames, Some((&pcm, 48_000, ch as u16)), fps, out)?;
    Ok(VideoOutput { width: wp * 32, height: hp * 32, frames: n, fps, label: format!("LTX-2.3 {}×{}, {:.1} с{}", wp * 32, hp * 32, n as f64 / fps, if req.first_frame.is_some() { ", с первого кадра" } else { "" }) })
}
