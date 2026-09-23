
pub mod ar;
pub mod cond_encoder;
pub mod config;
pub mod dcw;
pub mod detokenizer;
pub mod dit;
pub mod encoder;
pub mod fsq;
pub mod lm;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod scheduler;
pub mod text_encoder;
pub mod tokenizer;
pub mod vae;

use synaptix_core::error::SynaptixError;

#[derive(Debug, thiserror::Error)]
pub enum AceError {
    #[error("config: {0}")]
    Config(String),
    #[error("load: {0}")]
    Load(String),
    #[error("tensor: {0}")]
    Tensor(#[from] SynaptixError),
    #[error("{0}")]
    Other(String),
    /// Генерация остановлена флагом из [`with_cancel`].
    #[error("генерация остановлена")]
    Cancelled,
}

thread_local! {
    static CANCEL: std::cell::RefCell<Option<std::sync::Arc<std::sync::atomic::AtomicBool>>> =
        const { std::cell::RefCell::new(None) };
}

/// Выполнить `f` с флагом отмены: шаги диффузии, авторегрессия кодов и
/// переходы между стадиями `generate_music` опрашивают его и выходят с
/// [`AceError::Cancelled`]. Потоковый контекст, а не поле опций — чтобы не
/// менять `SamplerOptions`/`CodesGenOptions` у всех вызывающих.
pub fn with_cancel<R>(
    flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    f: impl FnOnce() -> R,
) -> R {
    let prev = CANCEL.with(|c| c.replace(flag));
    let out = f();
    CANCEL.with(|c| *c.borrow_mut() = prev);
    out
}

/// `Err(Cancelled)`, если флаг из [`with_cancel`] взведён.
pub(crate) fn check_cancel() -> Result<()> {
    let set = CANCEL.with(|c| {
        c.borrow()
            .as_ref()
            .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed))
    });
    if set {
        Err(AceError::Cancelled)
    } else {
        Ok(())
    }
}

pub type Result<T> = std::result::Result<T, AceError>;

pub use config::{DitConfig, LmConfig, VaeConfig};
pub use lm::AceStepLm;
pub use vae::AceStepVae;

#[cfg(test)]
mod cancel_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn cancel_flag_is_scoped_to_with_cancel() {
        assert!(check_cancel().is_ok());
        let flag = Arc::new(AtomicBool::new(false));
        with_cancel(Some(flag.clone()), || {
            assert!(check_cancel().is_ok());
            flag.store(true, Ordering::Relaxed);
            assert!(matches!(check_cancel(), Err(AceError::Cancelled)));
        });
        assert!(check_cancel().is_ok(), "вне with_cancel флаг не действует");
    }
}
