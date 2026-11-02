//! Конфиг Whisper из `config.json` (архитектура) и `generation_config.json`
//! (decode-параметры) внутри `.syn`-бандла.

use serde::Deserialize;
use synaptix_bundle::Bundle;

use crate::WhisperError;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WhisperConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub max_target_positions: usize,
    pub vocab_size: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub decoder_start_token_id: u32,
    pub activation_function: String,
    pub layer_norm_eps: f32,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            d_model: 0,
            encoder_layers: 0,
            decoder_layers: 0,
            encoder_attention_heads: 0,
            decoder_attention_heads: 0,
            encoder_ffn_dim: 0,
            decoder_ffn_dim: 0,
            num_mel_bins: 0,
            max_source_positions: 1500,
            max_target_positions: 448,
            vocab_size: 0,
            bos_token_id: 50257,
            eos_token_id: 50257,
            pad_token_id: 50257,
            decoder_start_token_id: 50258,
            activation_function: "gelu".to_string(),
            // В config.json Whisper нет поля layer_norm_eps — HF использует 1e-5.
            layer_norm_eps: 1e-5,
        }
    }
}

impl WhisperConfig {
    pub fn encoder_head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }

    pub fn decoder_head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, WhisperError> {
        let cfg: WhisperConfig =
            serde_json::from_slice(bytes).map_err(|e| WhisperError::Config(e.to_string()))?;
        // gelu_tanh ≠ точный erf-gelu; модель строится под erf-gelu.
        if cfg.activation_function != "gelu" {
            return Err(WhisperError::Config(format!(
                "unsupported activation_function={:?} (expected exact \"gelu\")",
                cfg.activation_function
            )));
        }
        if cfg.d_model == 0 || cfg.encoder_layers == 0 || cfg.vocab_size == 0 {
            return Err(WhisperError::Config(
                "config.json missing d_model/encoder_layers/vocab_size".to_string(),
            ));
        }
        Ok(cfg)
    }

    pub fn from_bundle(bundle: &Bundle) -> Result<Self, WhisperError> {
        let bytes = bundle
            .read_file("config.json")
            .map_err(|e| WhisperError::Bundle(e.to_string()))?;
        Self::from_json_bytes(&bytes)
    }
}

/// Decode-time параметры из `generation_config.json` (suppress-токены).
/// Языковые/задачные токены резолвятся через токенизатор в pipeline.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct GenerationConfig {
    pub suppress_tokens: Vec<u32>,
    pub begin_suppress_tokens: Vec<u32>,
    pub is_multilingual: bool,
    /// Головы cross-attn `(слой, голова)` для выравнивания слов по времени.
    pub alignment_heads: Vec<(usize, usize)>,
}

impl GenerationConfig {
    pub fn from_bundle(bundle: &Bundle) -> Result<Self, WhisperError> {
        let bytes = bundle
            .read_file("generation_config.json")
            .map_err(|e| WhisperError::Bundle(e.to_string()))?;
        serde_json::from_slice(&bytes).map_err(|e| WhisperError::Config(e.to_string()))
    }
}

/// Головы выравнивания: из `generation_config.json`, иначе по известным моделям
/// (как `_ALIGNMENT_HEADS` openai/whisper), иначе — все головы второй половины
/// слоёв декодера (запасной вариант openai).
pub fn alignment_heads(cfg: &WhisperConfig, gen: &GenerationConfig) -> Vec<(usize, usize)> {
    let ok = |v: &[(usize, usize)]| {
        !v.is_empty() && v.iter().all(|&(l, h)| l < cfg.decoder_layers && h < cfg.decoder_attention_heads)
    };
    if ok(&gen.alignment_heads) {
        return gen.alignment_heads.clone();
    }
    let known: &[(usize, usize)] = match (cfg.d_model, cfg.decoder_layers, cfg.vocab_size) {
        // tiny (многоязычная)
        (384, 4, 51865) => &[(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)],
        // large-v3-turbo
        (1280, 4, 51866) => &[(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)],
        // large-v3
        (1280, 32, 51866) => &[(7, 0), (10, 17), (12, 18), (13, 12), (16, 1), (17, 14), (19, 11), (21, 4), (24, 1), (25, 6)],
        _ => &[],
    };
    if ok(known) {
        return known.to_vec();
    }
    let n = cfg.decoder_layers;
    (n / 2..n).flat_map(|l| (0..cfg.decoder_attention_heads).map(move |h| (l, h))).collect()
}
