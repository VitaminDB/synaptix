//! Leokani-tts 7M (HF `milliwattlabs/leokani-tts-7M`) — нативный порт.

pub mod g2p;
pub mod model;
pub mod pipeline;

pub use pipeline::LeokaniPipeline;

#[derive(Debug, thiserror::Error)]
pub enum LkError {
    #[error("load: {0}")]
    Load(String),
    #[error("tensor: {0}")]
    Tensor(#[from] synaptix_core::error::SynaptixError),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, LkError>;

/// Символы с присоединёнными комбинирующими знаками (как `clusters` upstream) — для словаря фонем.
pub fn model_clusters(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in s.chars() {
        let mark = unicode_normalization::char::is_combining_mark(c) || c == '\u{200d}';
        match out.last_mut() {
            Some(last) if mark => last.push(c),
            _ => out.push(c.to_string()),
        }
    }
    out
}
