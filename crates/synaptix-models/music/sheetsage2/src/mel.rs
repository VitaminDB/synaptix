//! Лог-мел фронтенд MERT2 — всегда в F32, как в релизе (там автокаст на нём
//! выключен явно).
//!
//! `torchaudio.Spectrogram(n_fft 2048, hop 240, center, reflect)` → мощность
//! → мел-банк из весов (`feature_extractor.mel_scale.fb`) → `10·log10(max(x,
//! 1e-10))` → последний кадр отбрасывается → нормировка средним и СКО по
//! полосам из весов. На окне 300 с это 30 000 кадров по 128 полос.
//!
//! БПФ считается на CPU (кадры тишины — нули без БПФ: окно почти всегда
//! дополнено тишиной до 300 с), проекция на мел-банк и логарифм — на
//! устройстве модели.

use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::config::Mert2Config;
use crate::SheetError;

pub struct MelFrontend {
    /// Окно БПФ (периодический Ханн из весов), длина `win_length`.
    window: Vec<f32>,
    /// `[n_fft/2 + 1, n_mels]`, F32, на устройстве.
    filterbank: Tensor,
    /// `[1, 1, n_mels]`, F32.
    mean: Tensor,
    /// `[1, 1, n_mels]`, F32, уже `max(std, 1e-5)`.
    std: Tensor,
    n_fft: usize,
    hop: usize,
    device: Device,
}

impl MelFrontend {
    pub fn new(
        config: &Mert2Config,
        window: Vec<f32>,
        filterbank: Tensor,
        mean: Tensor,
        std: Tensor,
        device: Device,
    ) -> Result<Self, SheetError> {
        if window.len() != config.win_length || config.win_length != config.n_fft {
            return Err(SheetError::Config("окно БПФ MERT2 должно совпадать с n_fft".into()));
        }
        let n_mels = config.num_mel_bins;
        let mean = mean.to_device(device)?.to_dtype(DType::F32)?.reshape(vec![1usize, 1, n_mels])?;
        let std = std
            .to_device(device)?
            .to_dtype(DType::F32)?
            .clamp(1e-5, f32::MAX)?
            .reshape(vec![1usize, 1, n_mels])?;
        Ok(Self {
            window,
            filterbank: filterbank.to_device(device)?.to_dtype(DType::F32)?,
            mean,
            std,
            n_fft: config.n_fft,
            hop: config.hop_length,
            device,
        })
    }

    /// Спектр мощности `[кадры, n_fft/2+1]` (кадр `k` — центр в `k·hop`).
    pub fn power_spectrogram(&self, audio: &[f32]) -> Vec<f32> {
        let n_fft = self.n_fft;
        let half = n_fft / 2;
        let bins = half + 1;
        let frames = 1 + audio.len() / self.hop;
        // reflect-паддинг на n_fft/2 с каждой стороны
        let padded_len = audio.len() + 2 * half;
        let sample = |i: usize| -> f32 {
            let j = i as isize - half as isize;
            let n = audio.len() as isize;
            let j = if j < 0 {
                -j
            } else if j >= n {
                2 * (n - 1) - j
            } else {
                j
            };
            audio[j as usize]
        };
        debug_assert!(padded_len >= n_fft);
        let mut out = vec![0f32; frames * bins];
        let fft = FftPlanner::<f32>::new().plan_fft_forward(n_fft);
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
        let chunk = frames.div_ceil(threads);
        let window = &self.window;
        std::thread::scope(|scope| {
            for (t, rows) in out.chunks_mut(chunk * bins).enumerate() {
                let fft = Arc::clone(&fft);
                let sample = &sample;
                scope.spawn(move || {
                    let mut buf = vec![Complex32::new(0.0, 0.0); n_fft];
                    let mut scratch = vec![Complex32::new(0.0, 0.0); fft.get_inplace_scratch_len()];
                    let first = t * chunk;
                    for (r, row) in rows.chunks_mut(bins).enumerate() {
                        let start = (first + r) * self.hop;
                        let mut silent = true;
                        for (k, b) in buf.iter_mut().enumerate() {
                            let x = sample(start + k);
                            silent &= x == 0.0;
                            *b = Complex32::new(x * window[k], 0.0);
                        }
                        if silent {
                            row.fill(0.0);
                            continue;
                        }
                        fft.process_with_scratch(&mut buf, &mut scratch);
                        for (dst, c) in row.iter_mut().zip(&buf[..bins]) {
                            let mag = c.re.hypot(c.im);
                            *dst = mag * mag;
                        }
                    }
                });
            }
        });
        out
    }

    /// Нормированный лог-мел `[1, кадры − 1, n_mels]` (F32, на устройстве).
    pub fn forward(&self, audio: &[f32]) -> Result<Tensor, SheetError> {
        let bins = self.n_fft / 2 + 1;
        let power = self.power_spectrogram(audio);
        let frames = power.len() / bins;
        let spec = Tensor::from_vec(power, vec![frames, bins], Device::Cpu)?.to_device(self.device)?;
        let mel = spec.matmul(&self.filterbank)?; // [кадры, n_mels]
        // 10·log10(max(x, 1e-10)) = ln(x) · 10/ln 10
        let db = mel.clamp(1e-10, f32::MAX)?.log()?.affine(10.0 / std::f32::consts::LN_10, 0.0)?;
        let n_mels = db.dims()[1];
        let db = db.narrow(0, 0, frames - 1)?.contiguous()?.reshape(vec![1usize, frames - 1, n_mels])?;
        Ok(db.broadcast_sub(&self.mean)?.broadcast_div(&self.std)?)
    }
}
