//! Log-mel фронтенд Whisper (точная HF-нормализация поверх synaptix-audio STFT).
//!
//! HF `WhisperFeatureExtractor`: STFT(n_fft=400, hop=160, periodic-hann, center,
//! reflect) → |·|² → mel(Slaney, n_mels, fmin0 fmax8000, Slaney-norm) → log10 →
//! `max(x, x.max()-8)` → `(x+4)/4`. HF также отбрасывает последний STFT-фрейм
//! (`stft[..., :-1]`), поэтому 480000 сэмплов → 3000 фреймов.

use synaptix_audio::mel::{apply_mel_filterbank, mel_filterbank, MelConfig, MelNorm, MelScale};
use synaptix_audio::stft::{power_spectrogram, stft, StftConfig};

use crate::WhisperError;

pub fn whisper_mel_config(n_mels: usize) -> MelConfig {
    MelConfig {
        n_mels,
        f_min: 0.0,
        f_max: 8000.0,
        n_fft: 400,
        sample_rate: 16000,
        mel_scale: MelScale::Slaney,
        norm: MelNorm::Slaney,
    }
}

/// `audio` (16 kHz моно, дополнить до 30 с заранее) → flat row-major
/// `[n_mels, n_frames]` + `(n_mels, n_frames)`. `target_frames` = ожидаемое
/// число фреймов (HF: `max_source_positions * 2`, обычно 3000) — лишний хвост
/// STFT отбрасывается.
pub fn whisper_log_mel(
    audio: &[f32],
    n_mels: usize,
    target_frames: usize,
) -> Result<(Vec<f32>, usize, usize), WhisperError> {
    let stft_cfg = StftConfig::whisper_default();
    let mel_cfg = whisper_mel_config(n_mels);

    let spec = stft(audio, &stft_cfg).map_err(|e| WhisperError::Audio(e.to_string()))?;
    let power = power_spectrogram(&spec); // [time][n_freqs]
    let fb = mel_filterbank(&mel_cfg);
    let mut mel = apply_mel_filterbank(&power, &fb); // [time][n_mels]

    // HF отбрасывает последний фрейм (stft[..., :-1]).
    if mel.len() > target_frames {
        mel.truncate(target_frames);
    }
    let n_frames = mel.len();

    // log10 + глобальный максимум.
    let mut global_max = f32::NEG_INFINITY;
    for frame in &mut mel {
        for v in frame.iter_mut() {
            let l = v.max(1e-10).log10();
            *v = l;
            if l > global_max {
                global_max = l;
            }
        }
    }
    let floor = global_max - 8.0;

    // clamp + rescale, транспонируем в [n_mels, n_frames].
    let mut out = vec![0.0f32; n_mels * n_frames];
    for (t, frame) in mel.iter().enumerate() {
        for (m, &v) in frame.iter().enumerate() {
            let clamped = v.max(floor);
            out[m * n_frames + t] = (clamped + 4.0) / 4.0;
        }
    }
    Ok((out, n_mels, n_frames))
}
