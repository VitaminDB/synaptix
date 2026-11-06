//! F5-TTS (SWivid/F5-TTS, v1 Base: DiT 22×1024 + условный flow matching) с вокодером Vocos
//! (charactr/vocos-mel-24khz). Версии для русского — Misha24-10/F5-TTS_RUSSIAN. Голос задаёт образец
//! (запись + её текст): модель «продолжает» мел образца новым текстом.

pub mod dit;
pub mod pipeline;
pub mod text;
pub mod vocos;

pub use pipeline::{F5Pipeline, Options, VoicePrompt};

#[derive(Debug, thiserror::Error)]
pub enum F5Error {
    #[error("load: {0}")]
    Load(String),
    #[error("tensor: {0}")]
    Tensor(#[from] synaptix_core::error::SynaptixError),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, F5Error>;

pub const SAMPLE_RATE: u32 = 24_000;
pub const HOP: usize = 256;
pub const N_MELS: usize = 100;

const GOLD: u64 = 0x9E37_79B9_7F4A_7C15;

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Начальный шум: splitmix64 по счётчику, Бокс — Мюллер (как `reference/f5_ref.py`).
pub fn noise(seed: u64, n: usize) -> Vec<f32> {
    let u = |i: u64| ((mix(seed.wrapping_add(i.wrapping_mul(GOLD))) >> 40) as f64 + 0.5) / (1u64 << 24) as f64;
    (0..n as u64)
        .map(|k| {
            let (u1, u2) = (u(2 * k + 1), u(2 * k + 2));
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        })
        .collect()
}

/// Лог-мел как у F5/Vocos: STFT n_fft 1024, hop 256, окно Ханна, центр с отражением, |X|,
/// 100 полос HTK без нормировки (0…12 кГц), ln(max(x, 1e-5)). Выход `[кадр][полоса]`.
pub fn log_mel(audio: &[f32]) -> Vec<Vec<f32>> {
    use synaptix_audio::mel::{apply_mel_filterbank, mel_filterbank, MelConfig, MelNorm, MelScale};
    use synaptix_audio::stft::{magnitude_spectrogram, stft, PadMode, StftConfig};
    use synaptix_audio::window::WindowKind;
    let st = StftConfig { n_fft: 1024, hop_length: HOP, win_length: 1024, window: WindowKind::Hann, center: true, pad_mode: PadMode::Reflect };
    let spec = stft(audio, &st).unwrap_or_default();
    let mag = magnitude_spectrogram(&spec);
    let fb = mel_filterbank(&MelConfig { n_mels: N_MELS, f_min: 0.0, f_max: SAMPLE_RATE as f32 / 2.0, n_fft: 1024, sample_rate: SAMPLE_RATE, mel_scale: MelScale::Htk, norm: MelNorm::None });
    apply_mel_filterbank(&mag, &fb).into_iter().map(|r| r.into_iter().map(|v| v.max(1e-5).ln()).collect()).collect()
}
