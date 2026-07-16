use std::f32::consts::PI;

use crate::error::Result;
use crate::mel::{log_mel_spectrogram, MelConfig};
use crate::stft::StftConfig;

#[derive(Debug, Clone, Copy)]
pub struct MfccConfig {
    pub n_mfcc: usize,
    pub use_log_mel: bool,
}

impl Default for MfccConfig {
    fn default() -> Self {
        Self { n_mfcc: 13, use_log_mel: true }
    }
}

pub fn mfcc(
    audio: &[f32],
    stft_cfg: &StftConfig,
    mel_cfg: &MelConfig,
    mfcc_cfg: &MfccConfig,
) -> Result<Vec<Vec<f32>>> {
    let log_mel = log_mel_spectrogram(audio, stft_cfg, mel_cfg)?;
    Ok(dct_ii_per_frame(&log_mel, mfcc_cfg.n_mfcc))
}

pub fn dct_ii_per_frame(input: &[Vec<f32>], n_out: usize) -> Vec<Vec<f32>> {
    input.iter().map(|frame| dct_ii(frame, n_out)).collect()
}

fn dct_ii(x: &[f32], n_out: usize) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return vec![0.0; n_out];
    }
    let mut out = vec![0.0f32; n_out];
    let scale_first = (1.0f32 / n as f32).sqrt();
    let scale_rest = (2.0f32 / n as f32).sqrt();
    for k in 0..n_out {
        let mut acc = 0.0f32;
        for i in 0..n {
            acc += x[i] * (PI / n as f32 * (i as f32 + 0.5) * k as f32).cos();
        }
        let scale = if k == 0 { scale_first } else { scale_rest };
        out[k] = acc * scale;
    }
    out
}

pub fn delta(frames: &[Vec<f32>], window: usize) -> Vec<Vec<f32>> {
    let n = frames.len();
    if n == 0 {
        return Vec::new();
    }
    let dim = frames[0].len();
    let mut out = vec![vec![0.0f32; dim]; n];
    let mut denom = 0.0f32;
    for j in 1..=window {
        denom += (j * j) as f32;
    }
    denom *= 2.0;
    for t in 0..n {
        for d in 0..dim {
            let mut acc = 0.0f32;
            for j in 1..=window {
                let plus = (t + j).min(n - 1);
                let minus = t.saturating_sub(j);
                acc += j as f32 * (frames[plus][d] - frames[minus][d]);
            }
            out[t][d] = acc / denom.max(1e-12);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct_ii_constant_signal_concentrates_in_k0() {
        let x = vec![1.0f32; 8];
        let d = dct_ii(&x, 4);
        let energy: f32 = d.iter().skip(1).map(|v| v.abs()).sum();
        assert!(d[0].abs() > 1.0);
        assert!(energy < 1e-4);
    }

    #[test]
    fn delta_zero_for_constant_frames() {
        let frames = vec![vec![1.0, 2.0]; 5];
        let d = delta(&frames, 2);
        for f in d {
            for v in f {
                assert!(v.abs() < 1e-6);
            }
        }
    }

    #[test]
    fn mfcc_runs_on_whisper_default() {
        let audio: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
        let stft_cfg = StftConfig::whisper_default();
        let mel_cfg = MelConfig::whisper_default();
        let cfg = MfccConfig::default();
        let m = mfcc(&audio, &stft_cfg, &mel_cfg, &cfg).unwrap();
        assert!(!m.is_empty());
        assert_eq!(m[0].len(), 13);
    }
}
