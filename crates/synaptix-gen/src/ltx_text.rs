//! Текстовый кодер LTX: Gemma 3 12B (LTX-2.3) или Gemma 4 12B unified
//! (LTX-2.5) — по `model_type` в `config.json` бандла/каталога. Оба отдают
//! все скрытые состояния и маску под коннекторы DiT.

use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_llm_gemma3::pipeline::GemmaPipeline;
use synaptix_llm_gemma4::pipeline::Gemma4Pipeline;

pub enum LtxTextEncoder {
    Gemma3(GemmaPipeline),
    Gemma4(Gemma4Pipeline),
}

/// Gemma 4 ли кодер: `model_type` (корень или `text_config`) начинается с `gemma4`.
pub fn is_gemma4(path: &Path) -> bool {
    let raw = if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("syn")) {
        synaptix_bundle::Bundle::open(path).ok().and_then(|b| b.read_file("config.json").ok().map(|c| c.to_vec()))
    } else {
        std::fs::read(path.join("config.json")).ok()
    };
    let Some(v) = raw.and_then(|r| serde_json::from_slice::<serde_json::Value>(&r).ok()) else { return false };
    let mt = |x: &serde_json::Value| x.get("model_type").and_then(|m| m.as_str()).unwrap_or("").to_string();
    mt(&v).starts_with("gemma4") || v.get("text_config").is_some_and(|t| mt(t).starts_with("gemma4"))
}

impl LtxTextEncoder {
    pub fn load(path: &Path, device: Device, prec: PrecisionConfig, max_seq: Option<usize>) -> Result<Self, String> {
        if is_gemma4(path) {
            Gemma4Pipeline::load_with_precision(path, device, prec, max_seq).map(Self::Gemma4).map_err(|e| format!("Gemma 4: {e}"))
        } else {
            GemmaPipeline::load_with_precision(path, device, prec, max_seq).map(Self::Gemma3).map_err(|e| format!("Gemma: {e}"))
        }
    }

    pub fn encode_for_ltx(&self, prompt: &str, max_seq: usize, device: Device) -> Result<(Vec<Tensor>, Vec<u32>), String> {
        match self {
            Self::Gemma3(g) => g.encode_for_ltx(prompt, max_seq, device).map_err(|e| e.to_string()),
            Self::Gemma4(g) => g.encode_for_ltx(prompt, max_seq, device).map_err(|e| e.to_string()),
        }
    }
}
