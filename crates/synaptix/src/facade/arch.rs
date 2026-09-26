//! Единая детекция архитектуры моделей (заменяет дубли в CLI/адаптерах).
//! Первичный источник — `config.json` `model_type` (HF-каталог или `.syn`-бандл);
//! fallback — `BundleMeta.arch` из бандла, если config.json отсутствует.

use std::path::Path;

use synaptix_bundle::Bundle;
use synaptix_io::WeightLoader as _;


/// Читает файл из HF-каталога, `.syn`-бандла или `.gguf` (у GGUF файлы
/// `config.json`/`tokenizer.json`/… синтезирует маппер).
pub fn read_model_file(model: &Path, name: &str) -> Option<Vec<u8>> {
    synaptix_io::weights::read_model_file(model, name)
}

/// `model_type` из config.json. None, если файла/поля нет.
pub fn model_type(path: &Path) -> Option<String> {
    let bytes = read_model_file(path, "config.json")?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    v.get("model_type")
        .and_then(|x| x.as_str())
        .map(str::to_string)
}

/// Квант-форматы весов файла модели: `(формат, число тензоров)` по
/// убыванию числа; пусто у плотного бандла и HF-каталога. Для панели
/// моделей и предупреждения о двойном кванте при перекодировке.
pub fn bundle_quant_formats(path: &Path) -> Vec<(String, usize)> {
    if path.is_dir() || !synaptix_io::weights::is_model_file(path) {
        return Vec::new();
    }
    let Ok(l) = synaptix_io::SynBundleLoader::open(path) else { return Vec::new() };
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for name in l.names() {
        if name.ends_with(".qpacked") || name.ends_with(".qscales") {
            continue;
        }
        if let Some(kind) = l.quant_kind(name) {
            *counts.entry(synaptix_bundle::quant_layout::format_key(kind)).or_default() += 1;
        }
    }
    let mut v: Vec<(String, usize)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

/// `arch` из метаданных `.syn`-бандла (fallback). None для HF-каталога/пустого.
fn bundle_arch(path: &Path) -> Option<String> {
    if path.is_dir() || synaptix_io::weights::is_gguf_model(path) {
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
    let v = v.get("text_config").filter(|t| field(t).is_some()).unwrap_or(&v);
    let max = field(v)?;
    // Тип `rope_scaling`, который движок не применяет, — ёмкость по исходному
    // контексту (см. `synaptix_llm_common::rope_scaling::resolve`).
    let cap = v
        .get("rope_scaling")
        .filter(|rs| {
            let kind = rs
                .get("rope_type")
                .or_else(|| rs.get("type"))
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            !matches!(kind.as_str(), "" | "default" | "linear" | "llama3" | "yarn")
        })
        .and_then(|rs| rs.get("original_max_position_embeddings"))
        .and_then(|x| x.as_u64())
        .map(|x| x as usize);
    Some(cap.map_or(max, |c| c.min(max)))
}

/// LLM-архитектура — определяет, какой pipeline грузить.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmArch {
    Qwen3,
    Hybrid,
    Llama,
    Gemma3,
    Gemma4,
    MuseGlimmer,
    Qwen4Exp,
}

/// `qwen3_next`/`qwen3_5`/`qwen3_6` → гибрид (GatedDeltaNet + full-attn);
/// `qwen4_exp` → Qwen4Exp (GatedDeltaNet + QSA + MoE + PLE); `llama` → Llama;
/// `gemma`/`gemma3` → Gemma3; `gemma4` → Gemma4 (MoE + широкие global-слои);
/// остальное → Qwen3 (dense/MoE).
pub fn detect_llm_arch(path: &Path) -> Result<LlmArch, String> {
    let key = arch_key(path)
        .ok_or_else(|| format!("config.json/arch не найдены в {}", path.display()))?;
    Ok(match key.as_str() {
        "qwen3_next" | "qwen3_5" | "qwen3_6" => LlmArch::Hybrid,
        "llama" => LlmArch::Llama,
        "gemma" | "gemma3" | "gemma3_text" => LlmArch::Gemma3,
        "gemma4" | "gemma4_text" => LlmArch::Gemma4,
        "muse_glimmer" | "muse_glimmer_text" => LlmArch::MuseGlimmer,
        "qwen4_exp" | "qwen4_exp_text" => LlmArch::Qwen4Exp,
        _ => LlmArch::Qwen3,
    })
}
