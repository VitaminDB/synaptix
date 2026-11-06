//! `config.json` бандла (из `reference/silero_export.py`): символы, голоса, атрибуты модулей.

use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModuleAttrs {
    pub class: Option<String>,
    /// Свёртка — `[p]`, ISTFT — `"same"`/`"center"`.
    pub padding: Option<serde_json::Value>,
    pub num_heads: Option<usize>,
    pub pitch_pred: Option<bool>,
    pub shorten_factor: Option<usize>,
    pub n_fft: Option<usize>,
    pub hop_length: Option<usize>,
    pub win_length: Option<usize>,
    pub taps: Option<usize>,
    pub utt_type_emb: Option<bool>,
    pub focus_emb: Option<bool>,
    pub emphasis_emb: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub component: String,
    #[serde(default)]
    pub mean_std_coef: Vec<f64>,
    pub pitch_strength: Option<f64>,
    pub pitch_emb_type: Option<String>,
    pub n_mel_channels: Option<usize>,
    #[serde(default)]
    pub modules: HashMap<String, ModuleAttrs>,
}

impl ModuleAttrs {
    pub fn conv_pad(&self) -> Option<usize> {
        self.padding.as_ref()?.as_array()?.first()?.as_u64().map(|v| v as usize)
    }
}

impl ModelConfig {
    pub fn module(&self, p: &str) -> Option<&ModuleAttrs> {
        self.modules.get(p)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PackageConfig {
    pub symbols: String,
    pub sos_token: String,
    pub eos_token: String,
    pub alphabet: Vec<String>,
    pub speaker_to_ids: Vec<HashMap<String, usize>>,
    pub speaker_to_model: HashMap<String, usize>,
    pub speakers: Vec<String>,
    pub symbol_to_id: HashMap<String, i64>,
    pub ext_alph: Option<HashMap<String, HashMap<String, String>>>,
    #[serde(default)]
    pub phons: bool,
    pub emph_pitch: Option<bool>,
    pub models: Vec<ModelConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BundleConfig {
    pub source: String,
    pub packages: Vec<PackageConfig>,
    /// Добавляется при упаковке: вариант фронтенда, лицензия, описание.
    #[serde(default)]
    pub meta: serde_json::Value,
}
