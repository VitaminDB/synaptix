//! Log-mel фронтенд GigaAM (torchaudio `MelSpectrogram` + natural-log).
//!
//! torchaudio-конфиг (см. `preprocess.py`): `n_fft=320`, `win_length=320`,
//! `hop_length=160`, периодическое Hann-окно, `center=false`, `power=2.0`,
//! mel-фильтрбанк htk (`mel_scale="htk"`), без Slaney-нормализации
//! (`mel_norm=null`), `f_min=0`, `f_max=sr/2`. Затем `log(clamp(x, 1e-9, 1e9))`
//! (натуральный логарифм, НЕ log10/дБ как у Whisper).

use synaptix_audio::mel::{apply_mel_filterbank, mel_filterbank, MelConfig, MelNorm, MelScale};
use synaptix_audio::stft::{power_spectrogram, stft, PadMode, StftConfig};
use synaptix_audio::window::WindowKind;

use crate::config::PreprocessorConfig;

/// `audio` (16 кГц моно) → flat row-major `[n_mels, n_frames]` + (n_mels, n_frames).
pub fn log_mel(audio: &[f32], pre: &PreprocessorConfig) -> (Vec<f32>, usize, usize) {
    let stft_cfg = StftConfig {
        n_fft: pre.n_fft,
        hop_length: pre.hop_length,
        win_length: pre.win_length,
        window: WindowKind::Hann,
        center: pre.center,
        pad_mode: PadMode::Reflect,
    };
    let mel_cfg = MelConfig {
        n_mels: pre.features,
        f_min: 0.0,
        f_max: pre.sample_rate as f32 / 2.0,
        n_fft: pre.n_fft,
        sample_rate: pre.sample_rate,
        mel_scale: MelScale::Htk,
        norm: MelNorm::None,
    };

    let spec = stft(audio, &stft_cfg).expect("stft");
    let power = power_spectrogram(&spec); // [time][n_freqs]
    let fb = mel_filterbank(&mel_cfg);
    let mel = apply_mel_filterbank(&power, &fb); // [time][n_mels]
    let n_frames = mel.len();
    let n_mels = pre.features;

    // log(clamp(x, 1e-9, 1e9)) → [n_mels, n_frames] row-major.
    let mut out = vec![0.0f32; n_mels * n_frames];
    for (t, frame) in mel.iter().enumerate() {
        for (m, &v) in frame.iter().enumerate() {
            let clamped = v.clamp(1e-9, 1e9);
            out[m * n_frames + t] = clamped.ln();
        }
    }
    (out, n_mels, n_frames)
}
