//! Мел-спектрограмма эталона (`modules/audio.py: mel_spectrogram`): 22,05 кГц,
//! n_fft = win = 1024 (Hann, периодическое), hop 256, 80 полос, 0…Найквист,
//! зеркальное дополнение по 384 сэмпла, без центрирования; амплитуда
//! `sqrt(re² + im² + 1e-9)`, банки librosa (Slaney + норма Slaney), `ln(max(x, 1e-5))`.

use synaptix_audio::mel::{mel_filterbank, MelConfig, MelNorm, MelScale};
use synaptix_audio::stft::{stft, PadMode, StftConfig};
use synaptix_audio::window::WindowKind;

use crate::{SeedVcError, HOP, N_MELS, R, SAMPLE_RATE};

const N_FFT: usize = 1024;

/// Волна 22,05 кГц → мел `[N_MELS × T]` построчно по полосам (как тензор `[80, T]`), `T = ⌊N / 256⌋`.
pub fn mel_spectrogram(wave: &[f32]) -> R<(Vec<f32>, usize)> {
    let pad = (N_FFT - HOP) / 2;
    if wave.len() <= pad {
        return Err(SeedVcError::Input("запись слишком короткая".into()));
    }
    let n = wave.len();
    let mut y = Vec::with_capacity(n + 2 * pad);
    y.extend((1..=pad).rev().map(|i| wave[i]));
    y.extend_from_slice(wave);
    y.extend((0..pad).map(|i| wave[n - 2 - i]));
    let cfg = StftConfig { n_fft: N_FFT, hop_length: HOP, win_length: N_FFT, window: WindowKind::Hann, center: false, pad_mode: PadMode::Reflect };
    let spec = stft(&y, &cfg).map_err(|e| SeedVcError::Other(format!("stft: {e}")))?;
    let fb = mel_filterbank(&MelConfig { n_mels: N_MELS, f_min: 0.0, f_max: SAMPLE_RATE as f32 / 2.0, n_fft: N_FFT, sample_rate: SAMPLE_RATE, mel_scale: MelScale::Slaney, norm: MelNorm::Slaney });
    let t = spec.len();
    let mut out = vec![0f32; N_MELS * t];
    let mut mag = vec![0f32; N_FFT / 2 + 1];
    for (i, frame) in spec.iter().enumerate() {
        for (m, c) in mag.iter_mut().zip(frame) {
            *m = (c.re * c.re + c.im * c.im + 1e-9).sqrt();
        }
        for (b, filt) in fb.iter().enumerate() {
            let acc: f32 = mag.iter().zip(filt).map(|(p, w)| p * w).sum();
            out[b * t + i] = acc.max(1e-5).ln();
        }
    }
    Ok((out, t))
}
