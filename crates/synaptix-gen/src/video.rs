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
    let (quant_dit, quant_enc) = (DType::NVFP4, DType::NVFP4);
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
