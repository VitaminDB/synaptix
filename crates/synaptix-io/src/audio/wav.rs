use std::path::Path;
use hound::{WavReader, WavSpec, WavWriter, SampleFormat};
use crate::error::{IoError, Result};
use super::AudioBuffer;

pub fn read_wav(path: impl AsRef<Path>) -> Result<AudioBuffer> {
    let mut reader = WavReader::open(path).map_err(|e| IoError::Audio(e.to_string()))?;
    let spec = reader.spec();
    let channels = spec.channels;
    let sample_rate = spec.sample_rate;
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => {
            reader.samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| IoError::Audio(e.to_string()))?
        }
        (SampleFormat::Int, 16) => {
            reader.samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| IoError::Audio(e.to_string()))?
                .into_iter()
                .map(|s| s as f32 / i16::MAX as f32)
                .collect()
        }
        (SampleFormat::Int, 24) => {
            reader.samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| IoError::Audio(e.to_string()))?
                .into_iter()
                .map(|s| s as f32 / 8_388_607.0_f32)
                .collect()
        }
        (SampleFormat::Int, 32) => {
            reader.samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| IoError::Audio(e.to_string()))?
                .into_iter()
                .map(|s| s as f32 / i32::MAX as f32)
                .collect()
        }
        (fmt, bits) => {
            return Err(IoError::Audio(format!("unsupported WAV format {fmt:?} {bits}-bit")));
        }
    };
    Ok(AudioBuffer::new(samples, sample_rate, channels))
}

pub fn write_wav(buf: &AudioBuffer, path: impl AsRef<Path>) -> Result<()> {
    let spec = WavSpec {
        channels: buf.channels,
        sample_rate: buf.sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec).map_err(|e| IoError::Audio(e.to_string()))?;
    for &s in &buf.samples {
        writer.write_sample(s).map_err(|e| IoError::Audio(e.to_string()))?;
    }
    writer.finalize().map_err(|e| IoError::Audio(e.to_string()))?;
    Ok(())
}
