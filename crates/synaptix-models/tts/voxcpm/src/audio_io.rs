use synaptix_audio::io::read_wav_mono_f32;
use synaptix_audio::resample::resample_linear;

use crate::VoxError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    Left,
    Right,
}

pub fn load_resampled(path: &str, target_sr: usize) -> Result<Vec<f32>, VoxError> {
    let (samples, sr) = read_wav_mono_f32(path).map_err(|e| VoxError::Audio(e.to_string()))?;
    if sr as usize == target_sr {
        Ok(samples)
    } else {
        resample_linear(&samples, sr, target_sr as u32).map_err(|e| VoxError::Audio(e.to_string()))
    }
}

pub fn pad_to_multiple(mut samples: Vec<f32>, multiple: usize, mode: PadMode) -> Vec<f32> {
    let len = samples.len();
    let rem = len % multiple;
    if rem == 0 {
        return samples;
    }
    let pad = multiple - rem;
    match mode {
        PadMode::Right => {
            samples.extend(std::iter::repeat(0.0).take(pad));
            samples
        }
        PadMode::Left => {
            let mut out = vec![0.0f32; pad];
            out.extend_from_slice(&samples);
            out
        }
    }
}
