//! SheetSage2 — запись → партитура: мелодия (вокал и инструмент), аккорды,
//! доли, тональность и структура песни, а из них — ABC.
//!
//! Модель — энкодер-декодер. Энкодер — MERT-v2-FullSong (конформер 24×1024
//! поверх лог-мел спектра, 25 кадров в секунду) с влитыми при упаковке
//! LoRA-адаптерами внимания; его слои смешиваются обученными весами и
//! проецируются в 512. Декодер — BART на 6 слоёв, который жадно пишет
//! событийные токены под грамматикой: сдвиг по сетке долей → метка времени,
//! размер и позиция, секция, тональность, аккорд, ноты с длительностями.
//!
//! Раскладка по модулям:
//!
//! - [`vocab`] — словарь событий и разбор последовательности;
//! - [`grammar`] — какие токены разрешены на каждом шаге генерации;
//! - [`mel`] — лог-мел фронтенд (F32, как в релизе);
//! - [`encoder`] — MERT2 + смешивание слоёв + проекция;
//! - [`decoder`] — BART-декодер с KV-кэшем;
//! - [`generate`] — жадная генерация под грамматикой;
//! - [`stitch`] — окна по 300 с, карта времени, склейка, префикс перекрытия;
//! - [`export`] — события → доли, интервалы и ноты для нотации;
//! - [`notation`] — сборка и проверка двухголосного ABC;
//! - [`pipeline`] — транскрипция целиком;
//! - [`pack`] — упаковка релиза (слить LoRA, записать компонент в бандл).
//!
//! Партитуру без аккордов (`melody_only`) YuE2 принимает как план кавера с
//! `cot = melody`.

pub mod config;
pub mod decoder;
pub mod encoder;
pub mod export;
pub mod generate;
pub mod grammar;
pub mod loader;
pub mod mel;
pub mod nn;
pub mod notation;
pub mod pack;
pub mod pipeline;
pub mod stitch;
pub mod vocab;

use synaptix_core::error::SynaptixError;

#[derive(Debug, thiserror::Error)]
pub enum SheetError {
    #[error("конфигурация: {0}")]
    Config(String),
    #[error("загрузка: {0}")]
    Load(String),
    #[error("тензор: {0}")]
    Tensor(#[from] SynaptixError),
    #[error("последовательность: {0}")]
    Sequence(String),
    #[error("нотация: {0}")]
    Notation(String),
    #[error("прервано: {0}")]
    Cancelled(&'static str),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, SheetError>;

/// Имя компонента в бандле: `tensors:sheetsage2` и файлы под `sheetsage2/`.
pub const COMPONENT: &str = "sheetsage2";

pub use config::{Mert2Config, SheetSage2Config};
pub use export::VoiceSelect;
pub use pipeline::{SheetSage2, TranscribeOptions, Transcription};

pub use vocab::Vocab;
