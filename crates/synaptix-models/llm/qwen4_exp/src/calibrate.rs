//! Калибровка квантования экспертов MoE: вторые моменты входов каждой
//! матрицы каждого эксперта и матрицы связей входов слоёв на своём тексте →
//! файл для `synaptix quantize --calib` (важность столбцов и компенсация
//! ошибки, см. `synaptix_core::quant::{calib, gptq}`). Общая обвязка —
//! [`calib::calibrate_text`].

use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::quant::calib;

pub use calib::{TextCalibOpts as CalibOpts, TextCalibReport as CalibReport};

use crate::Qwen4ExpPipeline;

/// Прогнать `text` через модель `bundle` и записать калибровку в `out`.
/// Ядра устройства должны быть зарегистрированы вызывающим.
pub fn calibrate_experts(bundle: &Path, text: &str, out: &Path, device: Device, opts: &CalibOpts, progress: &dyn Fn(usize, usize)) -> Result<CalibReport, Box<dyn std::error::Error>> {
    let pipeline = Qwen4ExpPipeline::load_with_precision(bundle, device, PrecisionConfig::nvfp4(), Some(opts.window + 64))?;
    let ids = pipeline.encode(text)?;
    let mut cache = None;
    calib::calibrate_text(
        &ids,
        out,
        opts,
        |part, window| -> Result<(), Box<dyn std::error::Error>> {
            if let Some(len) = window {
                cache = Some(pipeline.make_cache(len + 8)?);
            }
            pipeline.model.forward(part, cache.as_mut().expect("окно открыто"))?;
            Ok(())
        },
        progress,
    )
}
