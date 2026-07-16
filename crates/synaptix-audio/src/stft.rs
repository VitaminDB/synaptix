use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::error::{AudioError, Result};
use crate::window::{build, WindowKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    Zero,
    Reflect,
}

#[derive(Debug, Clone, Copy)]
pub struct StftConfig {
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub window: WindowKind,
    pub center: bool,
    pub pad_mode: PadMode,
}

impl StftConfig {
    pub fn whisper_default() -> Self {
        Self {
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            window: WindowKind::Hann,
            center: true,
            pad_mode: PadMode::Reflect,
        }
    }

    pub fn num_freqs(&self) -> usize {
        self.n_fft / 2 + 1
    }

    fn validate(&self) -> Result<()> {
        if self.n_fft == 0 || self.hop_length == 0 || self.win_length == 0 {
            return Err(AudioError::invalid_arg("n_fft/hop_length/win_length must be > 0"));
        }
        if self.win_length > self.n_fft {
            return Err(AudioError::invalid_arg("win_length must be <= n_fft"));
        }
        Ok(())
    }
}

pub fn stft(audio: &[f32], cfg: &StftConfig) -> Result<Vec<Vec<Complex32>>> {
    cfg.validate()?;
    let window = pad_window(&build(cfg.window, cfg.win_length), cfg.n_fft);
    let signal = if cfg.center {
        pad_signal(audio, cfg.n_fft / 2, cfg.pad_mode)
    } else {
        audio.to_vec()
    };
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(cfg.n_fft);
    let n_freqs = cfg.num_freqs();
    let mut frames: Vec<Vec<Complex32>> = Vec::new();
    let mut start = 0usize;
    while start + cfg.n_fft <= signal.len() {
        let frame = stft_frame(&signal[start..start + cfg.n_fft], &window, &fft, n_freqs);
        frames.push(frame);
        start += cfg.hop_length;
    }
    Ok(frames)
}

/// STFT с явным окном (`win`, длиной `win_length` ≤ `n_fft`; центрируется в кадре
/// `n_fft` нулями по краям — как `torch.stft` с `win_length < n_fft`). Нужен когда
/// окно не выражается через [`WindowKind`] (напр. NeMo сохраняет точный
/// симметричный Hann-вектор в чекпойнте).
pub fn stft_with_window(
    audio: &[f32],
    n_fft: usize,
    hop_length: usize,
    win: &[f32],
    center: bool,
    pad_mode: PadMode,
) -> Result<Vec<Vec<Complex32>>> {
    if n_fft == 0 || hop_length == 0 || win.is_empty() {
        return Err(AudioError::invalid_arg("n_fft/hop_length/win must be > 0"));
    }
    if win.len() > n_fft {
        return Err(AudioError::invalid_arg("win length must be <= n_fft"));
    }
    let window = pad_window(win, n_fft);
    let signal = if center {
        pad_signal(audio, n_fft / 2, pad_mode)
    } else {
        audio.to_vec()
    };
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let n_freqs = n_fft / 2 + 1;
    let mut frames: Vec<Vec<Complex32>> = Vec::new();
    let mut start = 0usize;
    while start + n_fft <= signal.len() {
        frames.push(stft_frame(&signal[start..start + n_fft], &window, &fft, n_freqs));
        start += hop_length;
    }
    Ok(frames)
}

pub fn istft(spectrogram: &[Vec<Complex32>], cfg: &StftConfig) -> Result<Vec<f32>> {
    cfg.validate()?;
    let window = pad_window(&build(cfg.window, cfg.win_length), cfg.n_fft);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_inverse(cfg.n_fft);

    let n_freqs = cfg.num_freqs();
    let n_frames = spectrogram.len();
    if n_frames == 0 {
        return Ok(Vec::new());
    }
    for frame in spectrogram {
        if frame.len() != n_freqs {
            return Err(AudioError::invalid_arg(format!(
                "spectrogram frame len {} != expected {}",
                frame.len(),
                n_freqs
            )));
        }
    }

    let out_len = (n_frames - 1) * cfg.hop_length + cfg.n_fft;
    let mut out = vec![0.0f32; out_len];
    let mut weight = vec![0.0f32; out_len];

    let inv_scale = 1.0f32 / cfg.n_fft as f32;
    for (i, frame) in spectrogram.iter().enumerate() {
        let mut buf = mirror_full_spectrum(frame, cfg.n_fft);
        fft.process(&mut buf);
        let start = i * cfg.hop_length;
        for n in 0..cfg.n_fft {
            let v = buf[n].re * inv_scale * window[n];
            out[start + n] += v;
            weight[start + n] += window[n] * window[n];
        }
    }
    for n in 0..out_len {
        if weight[n] > 1e-12 {
            out[n] /= weight[n];
        }
    }
    if cfg.center {
        let trim = cfg.n_fft / 2;
        if out.len() > 2 * trim {
            out = out[trim..out.len() - trim].to_vec();
        }
    }
    Ok(out)
}

pub fn power_spectrogram(spectrogram: &[Vec<Complex32>]) -> Vec<Vec<f32>> {
    spectrogram
        .iter()
        .map(|frame| frame.iter().map(|c| c.re * c.re + c.im * c.im).collect())
        .collect()
}

pub fn magnitude_spectrogram(spectrogram: &[Vec<Complex32>]) -> Vec<Vec<f32>> {
    spectrogram
        .iter()
        .map(|frame| frame.iter().map(|c| (c.re * c.re + c.im * c.im).sqrt()).collect())
        .collect()
}

fn stft_frame(
    samples: &[f32],
    window: &[f32],
    fft: &Arc<dyn Fft<f32>>,
    n_freqs: usize,
) -> Vec<Complex32> {
    let mut buf: Vec<Complex32> = samples
        .iter()
        .zip(window)
        .map(|(&s, &w)| Complex32::new(s * w, 0.0))
        .collect();
    fft.process(&mut buf);
    buf.truncate(n_freqs);
    buf
}

fn mirror_full_spectrum(half: &[Complex32], n_fft: usize) -> Vec<Complex32> {
    let mut buf = vec![Complex32::new(0.0, 0.0); n_fft];
    for (i, &c) in half.iter().enumerate() {
        buf[i] = c;
    }
    for k in 1..(n_fft / 2) {
        let idx = n_fft - k;
        buf[idx] = Complex32::new(half[k].re, -half[k].im);
    }
    buf
}

fn pad_window(win: &[f32], n_fft: usize) -> Vec<f32> {
    if win.len() == n_fft {
        return win.to_vec();
    }
    let pad = n_fft - win.len();
    let left = pad / 2;
    let right = pad - left;
    let mut out = Vec::with_capacity(n_fft);
    out.extend(std::iter::repeat(0.0).take(left));
    out.extend_from_slice(win);
    out.extend(std::iter::repeat(0.0).take(right));
    out
}

fn pad_signal(audio: &[f32], pad: usize, mode: PadMode) -> Vec<f32> {
    if pad == 0 || audio.is_empty() {
        return audio.to_vec();
    }
    let mut out = Vec::with_capacity(audio.len() + 2 * pad);
    match mode {
        PadMode::Zero => {
            out.extend(std::iter::repeat(0.0).take(pad));
            out.extend_from_slice(audio);
            out.extend(std::iter::repeat(0.0).take(pad));
        }
        PadMode::Reflect => {
            let n = audio.len();
            for k in 0..pad {
                let src = (pad - k).min(n - 1);
                out.push(audio[src]);
            }
            out.extend_from_slice(audio);
            for k in 0..pad {
                let src = n.saturating_sub(2 + k);
                out.push(audio[src]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_returns_correct_shape() {
        let audio: Vec<f32> = vec![0.0; 16000];
        let cfg = StftConfig::whisper_default();
        let spec = stft(&audio, &cfg).unwrap();
        assert_eq!(spec[0].len(), cfg.num_freqs());
        assert!(!spec.is_empty());
    }

    #[test]
    fn power_spec_non_negative() {
        let audio: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
        let cfg = StftConfig {
            n_fft: 128,
            hop_length: 32,
            win_length: 128,
            window: WindowKind::Hann,
            center: false,
            pad_mode: PadMode::Zero,
        };
        let spec = stft(&audio, &cfg).unwrap();
        let pwr = power_spectrogram(&spec);
        for frame in &pwr {
            for &v in frame {
                assert!(v >= 0.0);
            }
        }
    }
}
