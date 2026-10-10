//! Признаки как в Kaldi: `compute-mfcc-feats` (окно Povey, preemphasis,
//! mel-банки по шкале 1127·ln(1+f/700), DCT, лифтеринг, `snip-edges=false`),
//! `apply-cmvn` (среднее по высказыванию), `splice-feats`, `transform-feats`.

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    pub sample_frequency: f32,
    /// мс.
    pub frame_shift: f32,
    pub frame_length: f32,
    pub low_frequency: f32,
    pub high_frequency: f32,
    pub num_mel_bins: usize,
    pub num_coefficients: usize,
    pub cepstral_lifter: f32,
    pub preemphasis_coefficient: f32,
    pub snip_edges: bool,
    pub use_energy: bool,
    pub splice_left_context: usize,
    pub splice_right_context: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            sample_frequency: 16000.0,
            frame_shift: 10.0,
            frame_length: 25.0,
            low_frequency: 20.0,
            high_frequency: 7800.0,
            num_mel_bins: 23,
            num_coefficients: 13,
            cepstral_lifter: 22.0,
            preemphasis_coefficient: 0.97,
            snip_edges: false,
            use_energy: false,
            splice_left_context: 3,
            splice_right_context: 3,
        }
    }
}

fn mel(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

/// Вычислитель MFCC с предрасчитанными окном, банками и DCT.
pub struct Mfcc {
    cfg: FeatureConfig,
    win_len: usize,
    shift: usize,
    padded: usize,
    window: Vec<f32>,
    /// Для каждого мел-банка: (первый бин FFT, веса).
    banks: Vec<(usize, Vec<f32>)>,
    /// `[num_coefficients × num_mel_bins]`.
    dct: Vec<f32>,
    lifter: Vec<f32>,
}

impl Mfcc {
    pub fn new(cfg: FeatureConfig) -> Self {
        let win_len = (cfg.sample_frequency * cfg.frame_length / 1000.0) as usize;
        let shift = (cfg.sample_frequency * cfg.frame_shift / 1000.0) as usize;
        let padded = win_len.next_power_of_two();
        let window: Vec<f32> = (0..win_len)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI / (win_len - 1) as f64;
                (0.5 - 0.5 * (a * i as f64).cos()).powf(0.85) as f32
            })
            .collect();
        let nb = cfg.num_mel_bins;
        let num_fft_bins = padded / 2;
        let bin_width = cfg.sample_frequency / padded as f32;
        let (ml, mh) = (mel(cfg.low_frequency), mel(cfg.high_frequency));
        let delta = (mh - ml) / (nb + 1) as f32;
        let mut banks = Vec::with_capacity(nb);
        for b in 0..nb {
            let left = ml + b as f32 * delta;
            let center = ml + (b + 1) as f32 * delta;
            let right = ml + (b + 2) as f32 * delta;
            let mut first = usize::MAX;
            let mut w = Vec::new();
            for i in 0..num_fft_bins {
                let m = mel(bin_width * i as f32);
                if m > left && m < right {
                    let v = if m <= center { (m - left) / (center - left) } else { (right - m) / (right - center) };
                    if first == usize::MAX {
                        first = i;
                    }
                    w.resize(i - first, 0.0);
                    w.push(v);
                }
            }
            banks.push((if first == usize::MAX { 0 } else { first }, w));
        }
        let nc = cfg.num_coefficients;
        let mut dct = vec![0f32; nc * nb];
        for k in 0..nc {
            let norm = if k == 0 { (1.0 / nb as f64).sqrt() } else { (2.0 / nb as f64).sqrt() };
            for n in 0..nb {
                dct[k * nb + n] = (norm * (std::f64::consts::PI / nb as f64 * (n as f64 + 0.5) * k as f64).cos()) as f32;
            }
        }
        let q = cfg.cepstral_lifter;
        let lifter = (0..nc).map(|i| 1.0 + 0.5 * q * (std::f32::consts::PI * i as f32 / q).sin()).collect();
        Self { cfg, win_len, shift, padded, window, banks, dct, lifter }
    }

    pub fn config(&self) -> &FeatureConfig {
        &self.cfg
    }

    pub fn num_frames(&self, n: usize) -> usize {
        if self.cfg.snip_edges {
            if n < self.win_len { 0 } else { 1 + (n - self.win_len) / self.shift }
        } else {
            (n + self.shift / 2) / self.shift
        }
    }

    /// MFCC `[frames × num_coefficients]`. Сэмплы — в шкале int16 (как Kaldi
    /// читает WAV), т. е. f32 из [-1, 1] нужно умножить на 32768.
    pub fn compute(&self, wav: &[f32]) -> Vec<f32> {
        let n = wav.len();
        let frames = self.num_frames(n);
        let nc = self.cfg.num_coefficients;
        let nb = self.cfg.num_mel_bins;
        let mut out = vec![0f32; frames * nc];
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.padded);
        let mut buf = vec![Complex32::new(0.0, 0.0); self.padded];
        let mut frame = vec![0f32; self.win_len];
        let mut melv = vec![0f32; nb];
        for f in 0..frames {
            let start: isize = if self.cfg.snip_edges {
                (f * self.shift) as isize
            } else {
                (f * self.shift + self.shift / 2) as isize - (self.win_len / 2) as isize
            };
            for (j, v) in frame.iter_mut().enumerate() {
                let mut s = start + j as isize;
                while s < 0 || s >= n as isize {
                    s = if s < 0 { -s - 1 } else { 2 * n as isize - 1 - s };
                }
                *v = wav[s as usize];
            }
            let mean = frame.iter().map(|&x| x as f64).sum::<f64>() / self.win_len as f64;
            for v in frame.iter_mut() {
                *v -= mean as f32;
            }
            let c = self.cfg.preemphasis_coefficient;
            for j in (1..self.win_len).rev() {
                frame[j] -= c * frame[j - 1];
            }
            frame[0] -= c * frame[0];
            for (b, (x, w)) in buf.iter_mut().zip(frame.iter().zip(&self.window)) {
                *b = Complex32::new(x * w, 0.0);
            }
            for b in buf[self.win_len..].iter_mut() {
                *b = Complex32::new(0.0, 0.0);
            }
            fft.process(&mut buf);
            for (m, (first, w)) in melv.iter_mut().zip(&self.banks) {
                let mut e = 0f32;
                for (k, &wk) in w.iter().enumerate() {
                    e += wk * buf[first + k].norm_sqr();
                }
                *m = e.max(f32::EPSILON).ln();
            }
            for k in 0..nc {
                let mut acc = 0f32;
                for (j, &mv) in melv.iter().enumerate() {
                    acc += self.dct[k * nb + j] * mv;
                }
                out[f * nc + k] = acc * self.lifter[k];
            }
        }
        out
    }
}

/// Вычесть среднее по кадрам (CMVN без нормировки дисперсии).
pub fn cmvn(feats: &mut [f32], dim: usize) {
    let frames = feats.len() / dim;
    if frames == 0 {
        return;
    }
    let mut mean = vec![0f64; dim];
    for r in feats.chunks(dim) {
        for (m, &v) in mean.iter_mut().zip(r) {
            *m += v as f64;
        }
    }
    for r in feats.chunks_mut(dim) {
        for (v, m) in r.iter_mut().zip(&mean) {
            *v -= (*m / frames as f64) as f32;
        }
    }
}

/// `splice-feats`: склейка ±контекста, края — повтор крайнего кадра.
pub fn splice(feats: &[f32], dim: usize, left: usize, right: usize) -> Vec<f32> {
    let frames = feats.len() / dim;
    let width = left + right + 1;
    let mut out = vec![0f32; frames * dim * width];
    for t in 0..frames {
        for c in 0..width {
            let src = (t as isize + c as isize - left as isize).clamp(0, frames as isize - 1) as usize;
            out[(t * width + c) * dim..(t * width + c + 1) * dim].copy_from_slice(&feats[src * dim..(src + 1) * dim]);
        }
    }
    out
}

/// `transform-feats`: `y = M·x` (матрица `[out × in]` или `[out × (in+1)]` со
/// смещением в последнем столбце).
pub fn transform(feats: &[f32], dim_in: usize, m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let frames = feats.len() / dim_in;
    let affine = cols == dim_in + 1;
    let mut out = vec![0f32; frames * rows];
    for t in 0..frames {
        let x = &feats[t * dim_in..(t + 1) * dim_in];
        for r in 0..rows {
            let row = &m[r * cols..(r + 1) * cols];
            let mut acc: f32 = row[..dim_in].iter().zip(x).map(|(a, b)| a * b).sum();
            if affine {
                acc += row[dim_in];
            }
            out[t * rows + r] = acc;
        }
    }
    out
}
