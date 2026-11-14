//! Калибровка квантования гибрида Qwen3.5/3.6 (GatedDeltaNet + full-attn,
//! эксперты MoE): вторые моменты входов матриц каждого эксперта и матрицы
//! связей входов слоёв MoE (запись — в общем слое `common::moe`) → файл для
//! `synaptix quantize --calib`: выбор масштабов NVFP4 по важности столбцов
//! и компенсация ошибки (GPTQ). Общая обвязка — [`calib::calibrate_text`].
//!
//! Модель грузится с квантованием весов при загрузке в `precision`
//! (по умолчанию MXFP8 — активации ближе к оригиналу, чем у NVFP4).

use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::quant::calib::{self, TextCalibOpts, TextCalibReport};
use synaptix_core::tensor::Tensor;

use crate::pipeline::HybridPipeline;

pub fn calibrate(
    bundle: &Path,
    text: &str,
    out: &Path,
    device: Device,
    precision: PrecisionConfig,
    opts: &TextCalibOpts,
    progress: &dyn Fn(usize, usize),
) -> Result<TextCalibReport, Box<dyn std::error::Error>> {
    let pipeline = HybridPipeline::load_with_precision(bundle, device, precision, Some(opts.window + 64))?;
    if pipeline.config.moe.is_none() {
        return Err("калибровка гибрида: у модели нет экспертов MoE (плотный MLP пока не пишет статистику)".into());
    }
    let ids = pipeline.encode(text)?;
    let mut kv = None;
    calib::calibrate_text(
        &ids,
        out,
        opts,
        |part, window| -> Result<(), Box<dyn std::error::Error>> {
            if let Some(len) = window {
                kv = Some(pipeline.model.make_kv_cache(1, len + 8)?);
            }
            let t = Tensor::from_vec(part.to_vec(), vec![1usize, part.len()], device)?;
            synaptix_core::grad::no_grad(|| pipeline.model.forward_trunk(&t, kv.as_mut().expect("окно открыто")))?;
            Ok(())
        },
        progress,
    )
}
