use std::path::Path;

use serde::Deserialize;
use synaptix_llm_common::{Activation, DecoderConfig, LayerKind, NormGain, RopeSpec};

pub use synaptix_llm_common::rope_scaling::RopeScaling;
use synaptix_llm_common::rope_scaling::{self, ScaledRope};

/// MLX/HF `quantization` секция. Наличие → веса проекций хранятся как affine
/// int`bits` (group `group_size`, scale+bias на группу) и дектвантятся при
/// загрузке (см. `loader`).
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct QuantConfig {
    pub group_size: usize,
    pub bits: usize,
}

/// `eos_token_id` может быть скаляром (Llama-2) либо массивом (Llama-3: несколько
/// служебных токенов конца). Генерация останавливается на любом из них.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl EosTokenId {
    pub fn ids(&self) -> Vec<u32> {
        match self {
            EosTokenId::Single(v) => vec![*v],
            EosTokenId::Multiple(v) => v.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Явный `head_dim` (Llama-3.2 задаёт его независимо от `hidden/heads`).
    /// 0 в JSON → вычисляется как `hidden_size / num_attention_heads`.
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<EosTokenId>,
    pub tie_word_embeddings: bool,
    pub quantization: Option<QuantConfig>,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        // Дефолты под Llama-3.2-1B-Instruct.
        Self {
            vocab_size: 128256,
            hidden_size: 2048,
            intermediate_size: 8192,
            num_hidden_layers: 16,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 64,
            max_position_embeddings: 131072,
            rms_norm_eps: 1.0e-5,
            rope_theta: 500_000.0,
            rope_scaling: None,
            hidden_act: "silu".into(),
            attention_bias: false,
            bos_token_id: Some(128000),
            eos_token_id: Some(EosTokenId::Single(128009)),
            tie_word_embeddings: true,
            quantization: None,
        }
    }
}

impl LlamaConfig {
    pub fn from_hf_json(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| ConfigError::Io(format!("read {}: {e}", path.display())))?;
        Self::from_hf_json_slice(&bytes)
            .map_err(|e| ConfigError::Parse(format!("{}: {e}", path.display())))
    }

    pub fn from_hf_json_slice(bytes: &[u8]) -> Result<Self, ConfigError> {
        let mut cfg: Self = serde_json::from_slice(bytes)
            .map_err(|e| ConfigError::Parse(format!("parse config.json: {e}")))?;
        if cfg.head_dim == 0 {
            cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads.max(1);
        }
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
        if self.head_dim == 0 || self.head_dim % 2 != 0 {
            return Err(ConfigError::Invalid(format!(
                "head_dim must be > 0 and even (RoPE), got {}",
                self.head_dim
            )));
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
        let rope = self.scaled_rope();
        DecoderConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            max_position_embeddings: rope.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            norm_gain: NormGain::Plain,
            activation: Activation::Silu,
            sandwich_norms: false,
            post_norm_eps: None,
            qk_norm: false,
            attn_output_gate: false,
            attn_scale: rope.attn_scale_mul() / (self.head_dim as f32).sqrt(),
            embed_scale: None,
            embed_rms_norm: false,
            logit_scale: None,
            logit_softcap: None,
            rope_global: RopeSpec {
                theta: self.rope_theta,
                rotary_dim: self.head_dim,
                scaled_freqs: rope.freqs,
            },
            rope_local: None,
            sliding_window: None,
            sliding_window_pattern: 0,
            layer_kinds: vec![LayerKind::Full; self.num_hidden_layers],
            linear: None,
            tie_word_embeddings: self.tie_word_embeddings,
            bos_token_id: self.bos_token_id,
            eos_token_ids: self.eos_ids(),
            ext: None,
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

    pub fn eos_ids(&self) -> Vec<u32> {
        self.eos_token_id.as_ref().map(|e| e.ids()).unwrap_or_default()
    }

    /// RoPE с учётом `rope_scaling` (`llama3`/`linear`/`yarn`; прочие типы —
    /// обычный RoPE с ёмкостью `original_max_position_embeddings`).
    pub fn scaled_rope(&self) -> ScaledRope {
        rope_scaling::resolve(
            self.rope_scaling.as_ref(),
            self.rope_theta,
            self.head_dim,
            self.max_position_embeddings,
        )
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
    fn default_matches_llama32_1b() {
        let cfg = LlamaConfig::default();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.q_total_dim(), 2048);
        assert_eq!(cfg.kv_total_dim(), 512);
        assert_eq!(cfg.group_size(), 4);
        assert!(cfg.tie_word_embeddings);
        cfg.validate().unwrap();
    }

    #[test]
    fn parse_mlx_config_with_array_eos_and_quant() {
        let json = r#"{
            "vocab_size": 128256,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-05,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            },
            "hidden_act": "silu",
            "attention_bias": false,
            "bos_token_id": 128000,
            "eos_token_id": [128001, 128008, 128009],
            "tie_word_embeddings": true,
            "quantization": { "group_size": 64, "bits": 4 },
            "model_type": "llama"
        }"#;
        let cfg: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.eos_ids(), vec![128001, 128008, 128009]);
        assert_eq!(cfg.quantization.unwrap().bits, 4);
        assert_eq!(cfg.quantization.unwrap().group_size, 64);
        let freqs = cfg.scaled_rope().freqs.expect("llama3 scaling");
        assert_eq!(freqs.len(), 32);
        // Высокочастотная компонента (i=0, inv_freq=1.0) не масштабируется.
        assert!((freqs[0] - 1.0).abs() < 1e-6);
        // Низкочастотная (i=half-1) делится на factor (длинная волна).
        assert!(freqs[31] < 1.0);
    }

    #[test]
    fn scalar_eos_parses() {
        let json = r#"{ "eos_token_id": 2, "model_type": "llama" }"#;
        let cfg: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.eos_ids(), vec![2]);
    }

    #[test]
    fn rejects_bad_gqa() {
        let mut cfg = LlamaConfig::default();
        cfg.num_key_value_heads = 7;
        assert!(cfg.validate().is_err());
    }
}
