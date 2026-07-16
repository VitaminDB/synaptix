//! NeMo `FilterbankFeatures` mel-фронтенд (как `AudioToMelSpectrogramPreprocessor`).
//!
//! Источник истины: NeMo `nemo/collections/asr/parts/preprocessing/features.py`.
//! Точная цепочка (v2.1, normalize="NA", dither=0 для детерминизма):
//!   preemph(0.97) → STFT(n_fft=512, hop=160, win=400 **симметричный** Hann, center
//!   reflect-pad n_fft/2) → power=|X|² (re²+im²) → mel = fb @ power → log(x + 2⁻²⁴).
//! Окно (`preprocessor.featurizer.window`) и фильтрбанк (`preprocessor.featurizer.fb`,
//! librosa-mel) берутся ИЗ чекпойнта (не пересчитываются) — гарантирует bit-мэтч.
//! Кадры обрезаются до `floor((L + n_fft/2·2 − n_fft) / hop)` = NeMo `get_seq_len`.

use synaptix_audio::stft::{stft_with_window, PadMode};

pub struct MelFrontend {
    window: Vec<f32>, // win_length (400), симметричный Hann из чекпойнта
    fb: Vec<f32>,     // n_mels * n_freqs row-major (librosa mel из чекпойнта)
    n_mels: usize,
    n_freqs: usize,
    n_fft: usize,
    hop: usize,
    preemph: f32,
    log_guard: f32,
}

impl MelFrontend {
    pub fn nemo_v21(window: Vec<f32>, fb: Vec<f32>, n_mels: usize, n_freqs: usize) -> Self {
        Self {
            window,
            fb,
            n_mels,
            n_freqs,
            n_fft: 512,
            hop: 160,
            preemph: 0.97,
            log_guard: 2f32.powi(-24),
        }
    }

    /// `audio` (16 кГц моно) → (flat `[n_mels, T]` row-major, `T`).
    pub fn forward(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        // preemph: y[0]=x[0]; y[t]=x[t] − 0.97·x[t−1].
        let mut x = vec![0.0f32; audio.len()];
        if !audio.is_empty() {
            x[0] = audio[0];
        }
        for t in 1..audio.len() {
            x[t] = audio[t] - self.preemph * audio[t - 1];
        }

        let spec = stft_with_window(&x, self.n_fft, self.hop, &self.window, true, PadMode::Reflect)
            .expect("sortformer mel stft");
        let n_frames_full = spec.len();

        // NeMo get_seq_len: floor((L + (n_fft/2)*2 − n_fft) / hop).
        let pad_amount = (self.n_fft / 2) * 2;
        let seq_len = (audio.len() + pad_amount - self.n_fft) / self.hop;
        let t_out = seq_len.min(n_frames_full);

        // mel = fb @ power, затем log(x + guard). out row-major [n_mels, t_out].
        let mut out = vec![0.0f32; self.n_mels * t_out];
        for t in 0..t_out {
            let frame = &spec[t];
            for m in 0..self.n_mels {
                let base = m * self.n_freqs;
                let mut acc = 0.0f32;
                for f in 0..self.n_freqs {
                    let p = frame[f].re * frame[f].re + frame[f].im * frame[f].im;
                    acc += self.fb[base + f] * p;
                }
                out[m * t_out + t] = (acc + self.log_guard).ln();
            }
        }
        (out, t_out)
    }
}
