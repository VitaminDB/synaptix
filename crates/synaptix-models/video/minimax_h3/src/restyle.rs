use std::path::Path;
use std::process::Command;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::audio_vae::AudioVae;
use crate::config::{AUDIO_SAMPLES_PER_LATENT, AUDIO_SAMPLE_RATE, FPS};
use crate::pipeline::Geometry;
use crate::scheduler::H3Scheduler;
use crate::vae::VaeEncoder;
use crate::H3Error;

pub struct RestyleSource {
    pub video: Tensor,
    pub audio: Tensor,
}

fn ffmpeg(path: &Path, args: &[&str]) -> Result<Vec<u8>, H3Error> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args(args)
        .arg("pipe:1")
        .output()
        .map_err(|e| H3Error::Io(format!("ffmpeg: {e}")))?;
    if !out.status.success() {
        return Err(H3Error::Io(format!(
            "ffmpeg {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(out.stdout)
}

fn decode_frames(path: &Path, g: Geometry) -> Result<Vec<u8>, H3Error> {
    let vf = format!("fps={FPS},scale={}:{}", g.width, g.height);
    let frames = g.frame_count.to_string();
    let mut rgb = ffmpeg(path, &["-an", "-vf", &vf, "-frames:v", &frames, "-f", "rawvideo", "-pix_fmt", "rgb24"])?;
    let frame = g.width * g.height * 3;
    let have = rgb.len() / frame;
    if have == 0 {
        return Err(H3Error::Io(format!("{}: видео не дало кадров", path.display())));
    }
    rgb.truncate(have * frame);
    let last = rgb[(have - 1) * frame..].to_vec();
    for _ in have..g.frame_count {
        rgb.extend_from_slice(&last);
    }
    Ok(rgb)
}

fn decode_audio(path: &Path, g: Geometry) -> Result<Tensor, H3Error> {
    let len = g.audio_t * AUDIO_SAMPLES_PER_LATENT;
    let rate = AUDIO_SAMPLE_RATE.to_string();
    let raw = ffmpeg(path, &["-vn", "-ac", "2", "-ar", &rate, "-f", "f32le"]).unwrap_or_default();
    let inter: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let have = (inter.len() / 2).min(len);
    let mut planar = vec![0f32; 2 * len];
    for i in 0..have {
        planar[i] = inter[2 * i];
        planar[len + i] = inter[2 * i + 1];
    }
    Ok(Tensor::from_vec(planar, vec![1, 2, len], Device::Cpu)?)
}

fn rgb_to_clip(rgb: &[u8], len: usize, width: usize, height: usize) -> Result<Tensor, H3Error> {
    let plane = width * height;
    let mut out = vec![0f32; 3 * len * plane];
    for (i, px) in rgb.chunks_exact(3).enumerate() {
        let (t, p) = (i / plane, i % plane);
        for c in 0..3 {
            out[(c * len + t) * plane + p] = px[c] as f32 / 127.5 - 1.0;
        }
    }
    Ok(Tensor::from_vec(out, vec![1, 3, len, height, width], Device::Cpu)?)
}

pub fn encode_source(
    path: &Path,
    geometry: Geometry,
    vae: &VaeEncoder,
    audio_vae: &AudioVae,
    device: Device,
) -> Result<RestyleSource, H3Error> {
    let rgb = decode_frames(path, geometry)?;
    let frame = geometry.width * geometry.height * 3;
    let video = vae.encode_clips(geometry.frame_count, &mut |start, len| {
        rgb_to_clip(&rgb[start * frame..(start + len) * frame], len, geometry.width, geometry.height)
    })?;
    let wave = decode_audio(path, geometry)?.to_device(device)?;
    let audio = audio_vae.encode(&wave)?;
    let at = audio.dims()[3];
    let audio = if at > geometry.audio_t {
        audio.narrow(3, 0, geometry.audio_t)?.contiguous()?
    } else {
        audio
    };
    Ok(RestyleSource { video: video.to_dtype(DType::F32)?, audio: audio.to_dtype(DType::F32)? })
}

fn noised(clean: &Tensor, sigma: f64, rng: &mut synaptix_ops::rng::Philox4x32, dtype: DType) -> Result<Tensor, H3Error> {
    let n = clean.dims().iter().product();
    let mut noise = vec![0f32; n];
    synaptix_ops::rng::fill_normal_f32(rng, &mut noise);
    let noise = Tensor::from_vec(noise, clean.dims().to_vec(), Device::Cpu)?.to_device(clean.device())?;
    Ok(noise
        .mul_scalar(sigma as f32)?
        .add(&clean.to_dtype(DType::F32)?.mul_scalar((1.0 - sigma) as f32)?)?
        .to_dtype(dtype)?)
}

pub fn start(
    sched: &H3Scheduler,
    strength: f32,
    source: &RestyleSource,
    seed: u64,
    device: Device,
    dtype: DType,
) -> Result<(H3Scheduler, Tensor, Tensor), H3Error> {
    let steps = sched.steps();
    if !(0.0..=1.0).contains(&strength) || steps == 0 {
        return Err(H3Error::Config(format!("restyle: сила вне [0, 1]: {strength}")));
    }
    let skip = (((1.0 - strength as f64) * steps as f64).round() as usize).min(steps - 1);
    let mut rng = synaptix_ops::rng::Philox4x32::new(seed);
    let video = noised(&source.video.to_device(device)?, sched.video_sigma(skip), &mut rng, dtype)?;
    let audio = noised(&source.audio.to_device(device)?, sched.audio_sigma(skip), &mut rng, dtype)?;
    let tail = H3Scheduler::from_sigmas(sched.sigmas[skip..].to_vec(), sched.shift_video, sched.shift_audio);
    Ok((tail, video, audio))
}
