//! Единая детекция архитектуры моделей (заменяет дубли в CLI/адаптерах).
//! Первичный источник — `config.json` `model_type` (HF-каталог или `.syn`-бандл);
//! fallback — `BundleMeta.arch` из бандла, если config.json отсутствует.

use std::path::Path;

use synaptix_bundle::Bundle;

/// Читает файл из HF-каталога (директория) или из `.syn`-бандла (файл).
pub fn read_model_file(model: &Path, name: &str) -> Option<Vec<u8>> {
    if model.is_dir() {
        std::fs::read(model.join(name)).ok()
    } else {
        let bundle = Bundle::open(model).ok()?;
        bundle.read_file(name).ok().map(|c| c.into_owned())
    }
}

/// `model_type` из config.json. None, если файла/поля нет.
pub fn model_type(path: &Path) -> Option<String> {
    let bytes = read_model_file(path, "config.json")?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    v.get("model_type")
        .and_then(|x| x.as_str())
        .map(str::to_string)
}

/// `arch` из метаданных `.syn`-бандла (fallback). None для HF-каталога/пустого.
fn bundle_arch(path: &Path) -> Option<String> {
    if path.is_dir() {
        return None;
    }
    let b = Bundle::open(path).ok()?;
    let a = b.meta().arch.clone();
    if a.is_empty() {
        None
    } else {
        Some(a)
    }
}

/// Универсальный ключ архитектуры: config.json `model_type`, иначе
/// `BundleMeta.arch`. Используется детекторами всех подсистем.
pub fn arch_key(path: &Path) -> Option<String> {
    model_type(path)
        .filter(|s| !s.is_empty())
        .or_else(|| bundle_arch(path))
}

pub fn config_max_seq(path: &Path) -> Option<usize> {
    let bytes = read_model_file(path, "config.json")?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let field = |v: &serde_json::Value| {
        v.get("max_position_embeddings")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
    };
    v.get("text_config").and_then(field).or_else(|| field(&v))
}

/// LLM-архитектура — определяет, какой pipeline грузить.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmArch {
    Qwen3,
    Hybrid,
    Llama,
    Gemma3,
}

/// `qwen3_next`/`qwen3_5`/`qwen3_6` → гибрид (GatedDeltaNet + full-attn);
/// `llama` → Llama; `gemma`/`gemma3` → Gemma3; остальное → Qwen3 (dense/MoE).
pub fn detect_llm_arch(path: &Path) -> Result<LlmArch, String> {
    let key = arch_key(path)
        .ok_or_else(|| format!("config.json/arch не найдены в {}", path.display()))?;
    Ok(match key.as_str() {
        "qwen3_next" | "qwen3_5" | "qwen3_6" => LlmArch::Hybrid,
        "llama" => LlmArch::Llama,
        "gemma" | "gemma3" | "gemma3_text" => LlmArch::Gemma3,
        _ => LlmArch::Qwen3,
    })
}
