//! Реестр мапперов по `general.architecture`. Каждый маппер даёт
//! [`ConversionPlan`]: имена тензоров HF ← ggml (с перестановками и
//! преобразованиями), синтезированные `config.json`, токенизатор и шаблон.
//! Тот же план используют конвертер `.gguf → .syn` и рантайм-источник
//! [`crate::source::GgufSource`].

pub mod common;
pub mod dense;
pub mod gemma3;
pub mod gemma4;
pub mod qwen35;

use crate::error::{GgufError, Result};
use crate::plan::ConversionPlan;
use crate::reader::GgufFile;

/// Архитектуры, для которых есть маппер. `qwen3moe` и `qwen3next` читаются и
/// конвертируются, но движок их пока не исполняет (нет MoE-ветки Qwen).
pub const SUPPORTED: &[&str] = &["llama", "qwen2", "qwen3", "qwen3moe", "qwen35", "gemma3", "gemma4"];

pub fn is_supported(arch: &str) -> bool {
    SUPPORTED.contains(&arch)
}

pub fn build_plan(model: &GgufFile, mmproj: Option<&GgufFile>, bundle_id: &str) -> Result<ConversionPlan> {
    let arch = model.architecture()?.to_string();
    match arch.as_str() {
        "qwen35" => qwen35::build_plan(model, mmproj, bundle_id),
        "llama" | "qwen2" | "qwen3" | "qwen3moe" => dense::build_plan(model, bundle_id),
        "gemma3" => gemma3::build_plan(model, bundle_id),
        "gemma4" => gemma4::build_plan(model, bundle_id),
        other => Err(GgufError::UnsupportedArch(format!(
            "{other} (поддержаны: {})",
            SUPPORTED.join(", ")
        ))),
    }
}
