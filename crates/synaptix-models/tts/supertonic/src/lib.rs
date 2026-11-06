//! Supertonic 3 (HF `Supertone/supertonic-3`) — нативный порт. Оригинал опубликован
//! только графами ONNX; архитектура восстановлена по ним (`SPEC.md`), веса — из
//! инициализаторов (`synaptix_io::onnx`).
//!
//! Конвейер: предсказатель длительности (секунды) → текстовый энкодер с подсказкой
//! стилем → N шагов Эйлера векторного поля flow matching (CFG 4·v_c − 3·v_u) из
//! гауссова шума → вокодер (латент 24×6 на кадр → 3072 семпла, 44.1 кГц).
//! Голоса — готовые стили (`voices.json`: style_ttl [50, 256], style_dp [8, 16]).

pub mod model;
pub mod pipeline;
pub mod text;

pub use pipeline::{Options, SupertonicPipeline, Voice};

#[derive(Debug, thiserror::Error)]
pub enum StError {
    #[error("load: {0}")]
    Load(String),
    #[error("tensor: {0}")]
    Tensor(#[from] synaptix_core::error::SynaptixError),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, StError>;

pub const SAMPLE_RATE: u32 = 44_100;

const GOLD: u64 = 0x9E37_79B9_7F4A_7C15;

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Начальный шум: splitmix64 по счётчику, Бокс — Мюллер (пары u1, u2 в f64),
/// как в эталоне `reference/supertonic_ref.py`.
pub fn noise(seed: u64, n: usize) -> Vec<f32> {
    let u = |i: u64| ((mix(seed.wrapping_add(i.wrapping_mul(GOLD))) >> 40) as f64 + 0.5) / (1u64 << 24) as f64;
    (0..n as u64)
        .map(|k| {
            let (u1, u2) = (u(2 * k + 1), u(2 * k + 2));
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        })
        .collect()
}
