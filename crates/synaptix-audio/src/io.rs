use std::path::Path;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

use crate::error::{AudioError, Result};

/// Сэмплы WAV как f32 (interleaved), частота и число каналов.
fn read_wav_interleaved_f32(path: &Path) -> Result<(Vec<f32>, u32, u16)> {
    let mut reader = WavReader::open(path).map_err(AudioError::from)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max).map_err(AudioError::from))
                .collect::<Result<Vec<_>>>()?
        }
        SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map_err(AudioError::from))
            .collect::<Result<Vec<_>>>()?,
    };
    Ok((samples, spec.sample_rate, spec.channels))
}

/// Стерео WAV в планарном виде `[L0..Ln, R0..Rn]` (моно дублируется, у
/// многоканального берутся первые два канала) — вход VAE ACE-Step, которому
/// стерео-панорама помогает отделять стемы.
pub fn read_wav_stereo_f32(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    let (samples, sr, channels) = read_wav_interleaved_f32(path.as_ref())?;
    let ch = channels.max(1) as usize;
    let n = samples.len() / ch;
    let mut planar = vec![0.0f32; 2 * n];
    for i in 0..n {
        planar[i] = samples[i * ch];
        planar[n + i] = samples[i * ch + (ch > 1) as usize];
    }
    Ok((planar, sr))
}

pub fn read_wav_mono_f32(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    let (samples, sr, channels) = read_wav_interleaved_f32(path.as_ref())?;
    if channels == 1 {
        return Ok((samples, sr));
    }
    let ch = channels as usize;
    let n_frames = samples.len() / ch;
    let mut mono = vec![0.0f32; n_frames];
    for i in 0..n_frames {
        let mut acc = 0.0f32;
        for c in 0..ch {
            acc += samples[i * ch + c];
        }
        mono[i] = acc / ch as f32;
    }
    Ok((mono, sr))
}

pub fn write_wav_mono_f32(path: impl AsRef<Path>, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let p = path.as_ref();
    let mut writer = WavWriter::create(p, spec).map_err(AudioError::from)?;
    for &s in samples {
        writer.write_sample(s).map_err(AudioError::from)?;
    }
    writer.finalize().map_err(AudioError::from)?;
    Ok(())
}
