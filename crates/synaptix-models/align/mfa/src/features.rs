//! Признаки как в Kaldi: `compute-mfcc-feats` (окно Povey, preemphasis,
//! mel-банки по шкале 1127·ln(1+f/700), DCT, лифтеринг, `snip-edges=false`),
//! `apply-cmvn` (среднее по высказыванию), `splice-feats`, `transform-feats`.

use serde::{Deserialize, Serialize};
use synaptix_audio::kaldi::{Fbank, FbankConfig};

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

/// Вычислитель MFCC: лог-мел-энергии Kaldi ([`synaptix_audio::kaldi::Fbank`]) → DCT → лифтеринг.
pub struct Mfcc {
    cfg: FeatureConfig,
    fbank: Fbank,
    /// `[num_coefficients × num_mel_bins]`.
    dct: Vec<f32>,
    lifter: Vec<f32>,
}

impl Mfcc {
    pub fn new(cfg: FeatureConfig) -> Self {
        let fbank = Fbank::new(FbankConfig {
            sample_frequency: cfg.sample_frequency,
            frame_shift: cfg.frame_shift,
            frame_length: cfg.frame_length,
            low_frequency: cfg.low_frequency,
            high_frequency: cfg.high_frequency,
            num_mel_bins: cfg.num_mel_bins,
            preemphasis_coefficient: cfg.preemphasis_coefficient,
            snip_edges: cfg.snip_edges,
        });
        let nb = cfg.num_mel_bins;
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
        Self { cfg, fbank, dct, lifter }
    }

    pub fn config(&self) -> &FeatureConfig {
        &self.cfg
    }

    pub fn num_frames(&self, n: usize) -> usize {
        self.fbank.num_frames(n)
    }

    /// MFCC `[frames × num_coefficients]`. Сэмплы — в шкале int16 (как Kaldi
    /// читает WAV), т. е. f32 из [-1, 1] нужно умножить на 32768.
    pub fn compute(&self, wav: &[f32]) -> Vec<f32> {
        let nc = self.cfg.num_coefficients;
        let nb = self.cfg.num_mel_bins;
        let mel = self.fbank.log_mel(wav);
        let frames = mel.len() / nb.max(1);
        let mut out = vec![0f32; frames * nc];
        for f in 0..frames {
            let melv = &mel[f * nb..(f + 1) * nb];
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
