use crate::error::Result;
use crate::stft::{power_spectrogram, stft, StftConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelScale {
    Htk,
    Slaney,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelNorm {
    None,
    Slaney,
}

#[derive(Debug, Clone, Copy)]
pub struct MelConfig {
    pub n_mels: usize,
    pub f_min: f32,
    pub f_max: f32,
    pub n_fft: usize,
    pub sample_rate: u32,
    pub mel_scale: MelScale,
    pub norm: MelNorm,
}

impl MelConfig {
    pub fn whisper_default() -> Self {
        Self {
            n_mels: 80,
            f_min: 0.0,
            f_max: 8000.0,
            n_fft: 400,
            sample_rate: 16000,
            mel_scale: MelScale::Slaney,
            norm: MelNorm::Slaney,
        }
    }
}

pub fn mel_filterbank(cfg: &MelConfig) -> Vec<Vec<f32>> {
    let n_freqs = cfg.n_fft / 2 + 1;
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * cfg.sample_rate as f32 / cfg.n_fft as f32)
        .collect();

    let mel_min = hz_to_mel(cfg.f_min, cfg.mel_scale);
    let mel_max = hz_to_mel(cfg.f_max, cfg.mel_scale);
    let mel_points: Vec<f32> = (0..cfg.n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (cfg.n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m, cfg.mel_scale)).collect();

    let mut fb = vec![vec![0.0f32; n_freqs]; cfg.n_mels];
    for m in 0..cfg.n_mels {
        let left = hz_points[m];
        let center = hz_points[m + 1];
        let right = hz_points[m + 2];
        for (k, &f) in fft_freqs.iter().enumerate() {
            let v = if f < left || f > right {
                0.0
            } else if f <= center {
                (f - left) / (center - left).max(1e-12)
            } else {
                (right - f) / (right - center).max(1e-12)
            };
            fb[m][k] = v.max(0.0);
        }
        if matches!(cfg.norm, MelNorm::Slaney) {
            let enorm = 2.0 / (right - left).max(1e-12);
            for v in &mut fb[m] {
                *v *= enorm;
            }
        }
    }
    fb
}

pub fn apply_mel_filterbank(power: &[Vec<f32>], fb: &[Vec<f32>]) -> Vec<Vec<f32>> {
    power
        .iter()
        .map(|frame| {
            fb.iter()
                .map(|filter| {
                    let mut acc = 0.0f32;
                    for (p, w) in frame.iter().zip(filter) {
                        acc += p * w;
                    }
                    acc
                })
                .collect()
        })
        .collect()
}

pub fn log_mel_spectrogram(
    audio: &[f32],
    stft_cfg: &StftConfig,
    mel_cfg: &MelConfig,
) -> Result<Vec<Vec<f32>>> {
    let spec = stft(audio, stft_cfg)?;
    let power = power_spectrogram(&spec);
    let fb = mel_filterbank(mel_cfg);
    let mel = apply_mel_filterbank(&power, &fb);
    Ok(power_to_log(&mel, 1e-10))
}

pub fn power_to_log(power: &[Vec<f32>], floor: f32) -> Vec<Vec<f32>> {
    power
        .iter()
        .map(|frame| frame.iter().map(|&v| (v.max(floor)).ln()).collect())
        .collect()
}

pub fn power_to_db(power: &[Vec<f32>], top_db: Option<f32>) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = power
        .iter()
        .map(|frame| frame.iter().map(|&v| 10.0 * v.max(1e-10).log10()).collect())
        .collect();
    if let Some(top) = top_db {
        let max = out
            .iter()
            .flat_map(|f| f.iter().copied())
            .fold(f32::NEG_INFINITY, f32::max);
        if max.is_finite() {
            let floor = max - top;
            for frame in &mut out {
                for v in frame {
                    if *v < floor {
                        *v = floor;
                    }
                }
            }
        }
    }
    out
}

fn hz_to_mel(hz: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
        MelScale::Slaney => {
            let min_log_hz = 1000.0f32;
            let min_log_mel = (min_log_hz - 0.0) / (200.0 / 3.0);
            let logstep = (6.4_f32.ln()) / 27.0;
            if hz < min_log_hz {
                hz / (200.0 / 3.0)
            } else {
                min_log_mel + (hz / min_log_hz).ln() / logstep
            }
        }
    }
}

fn mel_to_hz(mel: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney => {
            let min_log_hz = 1000.0f32;
            let min_log_mel = min_log_hz / (200.0 / 3.0);
            let logstep = (6.4_f32.ln()) / 27.0;
            if mel < min_log_mel {
                mel * (200.0 / 3.0)
            } else {
                min_log_hz * (logstep * (mel - min_log_mel)).exp()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_filterbank_shape() {
        let cfg = MelConfig::whisper_default();
        let fb = mel_filterbank(&cfg);
        assert_eq!(fb.len(), 80);
        assert_eq!(fb[0].len(), 201);
    }

    #[test]
    fn mel_filterbank_rows_sum_positive() {
        let cfg = MelConfig::whisper_default();
        let fb = mel_filterbank(&cfg);
        for (i, row) in fb.iter().enumerate() {
            let s: f32 = row.iter().sum();
            assert!(s > 0.0, "row {i} sum should be positive, got {s}");
        }
    }

    #[test]
    fn hz_mel_round_trip_slaney() {
        for &f in &[100.0f32, 500.0, 1000.0, 4000.0, 8000.0] {
            let m = hz_to_mel(f, MelScale::Slaney);
            let f_back = mel_to_hz(m, MelScale::Slaney);
            assert!((f_back - f).abs() / f < 1e-4);
        }
    }

    #[test]
    fn log_mel_dims_correct() {
        let audio: Vec<f32> = vec![0.0; 16000];
        let stft_cfg = StftConfig::whisper_default();
        let mel_cfg = MelConfig::whisper_default();
        let lm = log_mel_spectrogram(&audio, &stft_cfg, &mel_cfg).unwrap();
        assert!(!lm.is_empty());
        assert_eq!(lm[0].len(), 80);
    }
}
