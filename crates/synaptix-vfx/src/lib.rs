//! synaptix-vfx — видеофильтры на GPU в реальном времени для плеера: повышение разрешения и
//! интерполяция кадров. Свои ядра CUDA (`vfx.cu`: свёртка на тензорных ядрах со слитым эпилогом),
//! веса — `.syn` или safetensors с `arch` в метаданных.
//!
//! Поток: кадры RGBA8 по порядку → [`Upscaler::push`] → увеличенные кадры через `emit` (сразу или с
//! задержкой окна у видеомоделей). Перемотка — [`Upscaler::reset`].

pub mod gpu;
pub mod nanovsr;
pub mod span;
pub mod weights;

use std::path::Path;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("CUDA: {0}")]
    Cuda(String),
    #[error("веса: {0}")]
    Weights(String),
    #[error("{0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Кадр на выходе фильтра (живёт до следующего вызова).
pub struct Frame<'a> {
    pub rgba: &'a [u8],
    pub width: u32,
    pub height: u32,
}

/// Повышение разрешения потока кадров.
pub trait Upscaler: Send {
    fn name(&self) -> &str;
    /// Во сколько раз больше кадр на выходе.
    fn scale(&self) -> u32;
    /// На сколько кадров выход отстаёт от входа (0 — покадровая модель).
    fn latency(&self) -> usize {
        0
    }
    /// Следующий кадр RGBA8 `width×height`; готовые кадры — через `emit` по порядку.
    fn push(&mut self, rgba: &[u8], width: u32, height: u32, emit: &mut dyn FnMut(Frame)) -> Result<()>;
    /// Конец потока: отдать придержанные кадры.
    fn flush(&mut self, _emit: &mut dyn FnMut(Frame)) -> Result<()> {
        Ok(())
    }
    /// Перемотка: забыть придержанные кадры и состояние.
    fn reset(&mut self);
}

/// Открыть модель повышения разрешения по `arch` из метаданных весов.
pub fn open_upscaler(path: &Path, ordinal: usize) -> Result<Box<dyn Upscaler>> {
    let w = weights::Weights::open(path)?;
    let g = gpu::Gpu::new(ordinal)?;
    open_with(g, &w)
}

pub fn open_with(g: Arc<gpu::Gpu>, w: &weights::Weights) -> Result<Box<dyn Upscaler>> {
    match w.arch() {
        "nanovsr" => Ok(Box::new(nanovsr::NanoVsr::load(g, w)?)),
        "span" => Ok(Box::new(span::Span::load(g, w)?)),
        a => Err(Error::Unsupported(format!("неизвестная модель «{a}»"))),
    }
}
