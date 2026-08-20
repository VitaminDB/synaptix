use std::path::Path;

use serde::Deserialize;
use synaptix_llm_common::{Activation, DecoderConfig, LayerKind, NormGain, RopeSpec};

/// `rope_scaling` global-слоёв. Gemma-3 использует `rope_type=linear` (позиции
/// делятся на `factor`, эквивалентно `inv_freq/factor`). Прочие типы → без скейла.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    #[serde(default)]
    pub factor: f32,
}

/// `eos_token_id` Gemma — массив (`[1, 106]` = `<eos>` + `<end_of_turn>`).
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
pub struct Gemma3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    /// База RoPE global-слоёв (1e6).
    pub rope_theta: f32,
    /// База RoPE local (sliding) слоёв (1e4).
    pub rope_local_base_freq: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub sliding_window: usize,
    /// Каждый `sliding_window_pattern`-й слой — global (full attention); остальные
    /// — sliding (local). Global когда `(idx+1) % pattern == 0`.
    pub sliding_window_pattern: usize,
    /// Делитель attention-scale: `scale = query_pre_attn_scalar^-0.5`. `None` → head_dim.
    pub query_pre_attn_scalar: Option<f32>,
    pub hidden_activation: String,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<EosTokenId>,
    pub tie_word_embeddings: bool,
}

impl Default for Gemma3Config {
    fn default() -> Self {
        // Дефолты под gemma-3-12b text_config.
        Self {
            vocab_size: 262208,
            hidden_size: 3840,
            intermediate_size: 15360,
            num_hidden_layers: 48,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 256,
            max_position_embeddings: 131072,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            rope_scaling: None,
            sliding_window: 1024,
            sliding_window_pattern: 6,
            query_pre_attn_scalar: Some(256.0),
            hidden_activation: "gelu_pytorch_tanh".into(),
            bos_token_id: Some(2),
            eos_token_id: Some(EosTokenId::Multiple(vec![1, 106])),
            tie_word_embeddings: true,
        }
    }
}

impl Gemma3Config {
    /// Парсит HF-конфиг. У мультимодального чекпойнта арх-параметры лежат во
    /// вложенном `text_config`, а bos/eos — на верхнем уровне; мерджим (text_config
    /// поверх root, токены root сохраняются). У text-only чекпойнта всё на root.
    pub fn from_hf_json(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| ConfigError::Io(format!("read {}: {e}", path.display())))?;
        Self::from_hf_json_slice(&bytes, &path.display().to_string())
    }

    /// То же, но из уже прочитанных байт: `config.json` внутри `.syn`-бандла
    /// живёт чанком, а не файлом на диске. `origin` — только для сообщений.
    pub fn from_hf_json_slice(bytes: &[u8], origin: &str) -> Result<Self, ConfigError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| ConfigError::Parse(format!("parse {origin}: {e}")))?;
        let mut merged = match root.as_object() {
            Some(m) => m.clone(),
            None => return Err(ConfigError::Parse("config root not an object".into())),
        };
        if let Some(tc) = root.get("text_config").and_then(|v| v.as_object()) {
            for (k, v) in tc {
                merged.insert(k.clone(), v.clone());
            }
        }
        let mut cfg: Self = serde_json::from_value(serde_json::Value::Object(merged))
            .map_err(|e| ConfigError::Parse(format!("deserialize: {e}")))?;
        if cfg.head_dim == 0 {
            cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads.max(1);
        }
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.hidden_size == 0 || self.num_attention_heads == 0 {
            return Err(ConfigError::Invalid("hidden_size/heads == 0".into()));
        }
        if self.num_key_value_heads == 0 || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return Err(ConfigError::Invalid(format!(
                "GQA: heads={} % kv_heads={} != 0",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.head_dim == 0 || self.head_dim % 2 != 0 {
            return Err(ConfigError::Invalid(format!("head_dim must be even, got {}", self.head_dim)));
        }
        let act = self.hidden_activation.as_str();
        if act != "gelu_pytorch_tanh" && act != "gelu_tanh" && act != "gelu" {
            return Err(ConfigError::Invalid(format!(
                "hidden_activation unsupported: {act} (gelu_pytorch_tanh)"
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
            norm_gain: NormGain::OnePlus,
            activation: Activation::GeluTanh,
            sandwich_norms: true,
            post_norm_eps: None,
            qk_norm: true,
            attn_output_gate: false,
            attn_scale: self.attn_scale(),
            embed_scale: Some((self.hidden_size as f32).sqrt()),
            embed_rms_norm: false,
            logit_scale: None,
            logit_softcap: None,
            rope_global: RopeSpec {
                theta: self.rope_theta,
                rotary_dim: self.head_dim,
                scaled_freqs: Some(self.global_rope_freqs()),
            },
            rope_local: Some(RopeSpec {
                theta: self.rope_local_base_freq,
                rotary_dim: self.head_dim,
                scaled_freqs: None,
            }),
            sliding_window: Some(self.sliding_window),
            sliding_window_pattern: self.sliding_window_pattern,
            layer_kinds: vec![LayerKind::Full; self.num_hidden_layers],
            linear: None,
            tie_word_embeddings: self.tie_word_embeddings,
            bos_token_id: self.bos_token_id,
            eos_token_ids: self.eos_ids(),
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

    /// Global (full-attention) слой, если `(idx+1) % pattern == 0`. `pattern==0`
    /// (или 1) → все слои global.
    pub fn is_global_layer(&self, idx: usize) -> bool {
        let p = self.sliding_window_pattern;
        p <= 1 || (idx + 1) % p == 0
    }

    /// `scale = query_pre_attn_scalar^-0.5` (Gemma масштабирует на фикс. скаляр,
    /// не на head_dim). Fallback — head_dim.
    pub fn attn_scale(&self) -> f32 {
        let s = self.query_pre_attn_scalar.unwrap_or(self.head_dim as f32);
        1.0 / s.sqrt()
    }

    /// Частоты RoPE global-слоёв: `1/theta^(2i/d)`, делёные на linear-factor если задан.
    pub fn global_rope_freqs(&self) -> Vec<f32> {
        let factor = match &self.rope_scaling {
            Some(rs) if rs.rope_type.to_ascii_lowercase() == "linear" && rs.factor > 0.0 => {
                rs.factor as f64
            }
            _ => 1.0,
        };
        let theta = self.rope_theta as f64;
        let hd = self.head_dim;
        (0..hd / 2)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / hd as f64) / factor) as f32)
            .collect()
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
    fn defaults_gemma3_12b() {
        let c = Gemma3Config::default();
        assert_eq!(c.head_dim, 256);
        assert_eq!(c.group_size(), 2);
        assert_eq!(c.eos_ids(), vec![1, 106]);
        // attn scale = 1/sqrt(256) = 1/16
        assert!((c.attn_scale() - 1.0 / 16.0).abs() < 1e-6);
        // pattern=6 → global на idx 5,11,...; sliding на 0..4
        assert!(!c.is_global_layer(0));
        assert!(!c.is_global_layer(4));
        assert!(c.is_global_layer(5));
        assert!(c.is_global_layer(11));
        assert_eq!(c.global_rope_freqs().len(), 128);
    }

    #[test]
    fn parses_nested_text_config_and_root_eos() {
        let json = r#"{
            "architectures": ["Gemma3ForConditionalGeneration"],
            "bos_token_id": 2,
            "eos_token_id": [1, 106],
            "text_config": {
                "head_dim": 256,
                "hidden_size": 3840,
                "intermediate_size": 15360,
                "num_attention_heads": 16,
                "num_hidden_layers": 48,
                "num_key_value_heads": 8,
                "query_pre_attn_scalar": 256,
                "rms_norm_eps": 1e-06,
                "rope_local_base_freq": 10000,
                "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
                "rope_theta": 1000000,
                "sliding_window": 1024,
                "sliding_window_pattern": 6,
                "vocab_size": 262208,
                "hidden_activation": "gelu_pytorch_tanh"
            },
            "vision_config": {"hidden_size": 1152}
        }"#;
        let tmp = std::env::temp_dir().join("gemma3_test_config.json");
        std::fs::write(&tmp, json).unwrap();
        let c = Gemma3Config::from_hf_json(&tmp).unwrap();
        assert_eq!(c.hidden_size, 3840);
        assert_eq!(c.num_hidden_layers, 48);
        assert_eq!(c.eos_ids(), vec![1, 106]);
        assert_eq!(c.bos_token_id, Some(2));
        // linear factor 8 → global freqs делятся на 8
        let g = c.global_rope_freqs();
        assert!((g[0] - 1.0 / 8.0).abs() < 1e-6, "global f0 {}", g[0]);
        let _ = std::fs::remove_file(&tmp);
    }
}
