use std::path::Path;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

use crate::error::{AudioError, Result};

pub fn read_wav_mono_f32(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    let p = path.as_ref();
    let mut reader = WavReader::open(p).map_err(AudioError::from)?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
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
    if spec.channels == 1 {
        return Ok((samples, sr));
    }
    let ch = spec.channels as usize;
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
