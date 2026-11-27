//! Признаки как в Kaldi (`compute-fbank-feats` / `compute-mfcc-feats` до DCT,
//! `torchaudio.compliance.kaldi.fbank`): кадры с окном Povey, вычитание
//! среднего кадра, preemphasis, спектр мощности (FFT, дополненный до степени
//! двойки), треугольные мел-банки по шкале `1127·ln(1 + f/700)`, логарифм.
//!
//! Масштаб сэмплов задаёт вызывающий: Kaldi читает WAV в шкале int16
//! (f32 из [-1, 1] × 32768), torchaudio — как подали.

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

#[derive(Debug, Clone, Copy)]
pub struct FbankConfig {
    pub sample_frequency: f32,
    /// мс.
    pub frame_shift: f32,
    /// мс.
    pub frame_length: f32,
    pub low_frequency: f32,
    /// Верхняя граница банков, Гц; `<= 0` — смещение от частоты Найквиста
    /// (0 — сама Найквиста), как `high_freq` в Kaldi.
    pub high_frequency: f32,
    pub num_mel_bins: usize,
    pub preemphasis_coefficient: f32,
    /// `true` — только кадры, целиком лежащие в сигнале; `false` — кадры по
    /// центрам с зеркальным продолжением краёв.
    pub snip_edges: bool,
}

impl FbankConfig {
    /// Умолчания `torchaudio.compliance.kaldi.fbank` при `num_mel_bins` банках.
    pub fn torchaudio(num_mel_bins: usize, sample_frequency: f32) -> Self {
        Self {
            sample_frequency,
            frame_shift: 10.0,
            frame_length: 25.0,
            low_frequency: 20.0,
            high_frequency: 0.0,
            num_mel_bins,
            preemphasis_coefficient: 0.97,
            snip_edges: true,
        }
    }
}

fn mel(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

/// Вычислитель лог-мел-энергий с предрасчитанными окном и банками.
pub struct Fbank {
    cfg: FbankConfig,
    win_len: usize,
    shift: usize,
    padded: usize,
    window: Vec<f32>,
    /// Для каждого мел-банка: (первый бин FFT, веса).
    banks: Vec<(usize, Vec<f32>)>,
}

impl Fbank {
    pub fn new(cfg: FbankConfig) -> Self {
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
        let high = if cfg.high_frequency > 0.0 { cfg.high_frequency } else { 0.5 * cfg.sample_frequency + cfg.high_frequency };
        let (ml, mh) = (mel(cfg.low_frequency), mel(high));
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
        Self { cfg, win_len, shift, padded, window, banks }
    }

    pub fn config(&self) -> &FbankConfig {
        &self.cfg
    }

    pub fn num_frames(&self, n: usize) -> usize {
        if self.cfg.snip_edges {
            if n < self.win_len { 0 } else { 1 + (n - self.win_len) / self.shift }
        } else {
            (n + self.shift / 2) / self.shift
        }
    }

    /// Лог-мел-энергии `[frames × num_mel_bins]` (натуральный логарифм, пол — `f32::EPSILON`).
    pub fn log_mel(&self, wav: &[f32]) -> Vec<f32> {
        let n = wav.len();
        let frames = self.num_frames(n);
        let nb = self.cfg.num_mel_bins;
        let mut out = vec![0f32; frames * nb];
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.padded);
        let mut buf = vec![Complex32::new(0.0, 0.0); self.padded];
        let mut frame = vec![0f32; self.win_len];
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
            for (m, (first, w)) in out[f * nb..(f + 1) * nb].iter_mut().zip(&self.banks) {
                let mut e = 0f32;
                for (k, &wk) in w.iter().enumerate() {
                    e += wk * buf[first + k].norm_sqr();
                }
                *m = e.max(f32::EPSILON).ln();
            }
        }
        out
    }
}

/// Вычесть среднее по кадрам (CMN по высказыванию), `feats` — `[frames × dim]`.
pub fn subtract_mean(feats: &mut [f32], dim: usize) {
    let frames = feats.len() / dim.max(1);
    if frames == 0 {
        return;
    }
    for d in 0..dim {
        let mean = (0..frames).map(|f| feats[f * dim + d] as f64).sum::<f64>() / frames as f64;
        for f in 0..frames {
            feats[f * dim + d] -= mean as f32;
        }
    }
}
