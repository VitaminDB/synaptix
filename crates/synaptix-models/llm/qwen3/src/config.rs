use std::path::Path;

use serde::Deserialize;
use synaptix_llm_common::{Activation, DecoderConfig, LayerKind, NormGain, RopeSpec};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub tie_word_embeddings: bool,
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 40960,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            attention_bias: false,
            bos_token_id: Some(151643),
            eos_token_id: Some(151645),
            tie_word_embeddings: true,
            use_sliding_window: false,
            sliding_window: None,
        }
    }
}

impl Qwen3Config {
    pub fn from_hf_json(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| ConfigError::Io(format!("read {}: {e}", path.display())))?;
        let cfg: Self = serde_json::from_slice(&bytes)
            .map_err(|e| ConfigError::Parse(format!("parse {}: {e}", path.display())))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.hidden_size == 0 {
            return Err(ConfigError::Invalid("hidden_size == 0".into()));
        }
        if self.num_attention_heads == 0 {
            return Err(ConfigError::Invalid("num_attention_heads == 0".into()));
        }
        if self.num_key_value_heads == 0 || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return Err(ConfigError::Invalid(format!(
                "GQA constraint: heads={} must be divisible by kv_heads={}",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.head_dim == 0 {
            return Err(ConfigError::Invalid("head_dim == 0".into()));
        }
        if self.hidden_act != "silu" && self.hidden_act != "swish" {
            return Err(ConfigError::Invalid(format!(
                "hidden_act unsupported: {} (only silu/swish)",
                self.hidden_act
            )));
        }
        Ok(())
    }

    pub fn to_decoder_config(&self) -> DecoderConfig {
        DecoderConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            norm_gain: NormGain::Plain,
            activation: Activation::Silu,
            sandwich_norms: false,
            post_norm_eps: None,
            qk_norm: true,
            attn_output_gate: false,
            attn_scale: 1.0 / (self.head_dim as f32).sqrt(),
            embed_scale: None,
            embed_rms_norm: false,
            logit_scale: None,
            logit_softcap: None,
            rope_global: RopeSpec {
                theta: self.rope_theta,
                rotary_dim: self.head_dim,
                scaled_freqs: None,
            },
            rope_local: None,
            sliding_window: None,
            sliding_window_pattern: 0,
            layer_kinds: vec![LayerKind::Full; self.num_hidden_layers],
            linear: None,
            tie_word_embeddings: self.tie_word_embeddings,
            bos_token_id: self.bos_token_id,
            eos_token_ids: self.eos_token_id.into_iter().collect(),
        }
    }

    pub fn group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn q_total_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_total_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("config io: {0}")]
    Io(String),
    #[error("config parse: {0}")]
    Parse(String),
    #[error("config invalid: {0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_qwen3_1p7b() {
        let cfg = Qwen3Config::default();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.q_total_dim(), 2048);
        assert_eq!(cfg.kv_total_dim(), 1024);
        assert_eq!(cfg.group_size(), 2);
        assert!(cfg.tie_word_embeddings);
        cfg.validate().unwrap();
    }

    #[test]
    fn parse_from_json_string() {
        let json = r#"{
            "vocab_size": 151936,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000,
            "hidden_act": "silu",
            "attention_bias": false,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "tie_word_embeddings": true,
            "use_sliding_window": false,
            "sliding_window": null,
            "model_type": "qwen3"
        }"#;
        let cfg: Qwen3Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.intermediate_size, 6144);
        assert_eq!(cfg.eos_token_id, Some(151645));
    }

    #[test]
    fn rejects_bad_gqa() {
        let mut cfg = Qwen3Config::default();
        cfg.num_key_value_heads = 5;
        assert!(cfg.validate().is_err());
    }
}
