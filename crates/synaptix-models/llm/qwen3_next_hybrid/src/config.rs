use serde::Deserialize;
use synaptix_llm_common as common;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Linear,
    Full,
}

#[derive(Debug, Clone, Deserialize)]
struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    /// M-RoPE: сколько частот rotary-части отдано осям (время, строка,
    /// столбец) — у Qwen3.5 `[11, 11, 10]` на 32 частоты.
    #[serde(default)]
    mrope_section: Option<Vec<usize>>,
    /// Интерливинг осей по частотам (`T H W T H W …`), а не подряд блоками.
    #[serde(default)]
    mrope_interleaved: bool,
}

/// Раскладка M-RoPE (см. `crate::mrope`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MropeSpec {
    pub section: Vec<usize>,
    pub interleaved: bool,
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct TextConfigRaw {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    partial_rotary_factor: f32,
    attn_output_gate: bool,
    hidden_act: String,
    full_attention_interval: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_conv_kernel_dim: usize,
    layer_types: Vec<String>,
    tie_word_embeddings: bool,
    bos_token_id: Option<u32>,
    eos_token_id: EosField,
    mtp_num_hidden_layers: usize,
    rope_parameters: RopeParameters,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(untagged)]
enum EosField {
    #[default]
    Absent,
    One(u32),
    Many(Vec<u32>),
}

impl EosField {
    fn ids(&self) -> Vec<u32> {
        match self {
            EosField::Absent => Vec::new(),
            EosField::One(v) => vec![*v],
            EosField::Many(v) => v.clone(),
        }
    }
    fn first(&self) -> Option<u32> {
        self.ids().first().copied()
    }
}

impl Default for TextConfigRaw {
    fn default() -> Self {
        Self {
            vocab_size: 248320,
            hidden_size: 5120,
            intermediate_size: 17408,
            num_hidden_layers: 64,
            num_attention_heads: 24,
            num_key_value_heads: 4,
            head_dim: 256,
            max_position_embeddings: 262144,
            rms_norm_eps: 1.0e-6,
            partial_rotary_factor: 0.25,
            attn_output_gate: true,
            hidden_act: "silu".into(),
            full_attention_interval: 4,
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            layer_types: Vec::new(),
            tie_word_embeddings: false,
            bos_token_id: Some(248044),
            eos_token_id: EosField::One(248044),
            mtp_num_hidden_layers: 0,
            rope_parameters: RopeParameters {
                rope_theta: default_rope_theta(),
                mrope_section: None,
                mrope_interleaved: false,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct HybridConfig {
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
    pub partial_rotary_factor: f32,
    pub attn_output_gate: bool,
    pub hidden_act: String,
    pub full_attention_interval: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub layer_kinds: Vec<LayerKind>,
    pub tie_word_embeddings: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    pub mtp_num_hidden_layers: usize,
    pub image_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
    pub vision_start_token_id: Option<u32>,
    pub vision_end_token_id: Option<u32>,
    /// M-RoPE из `rope_parameters.mrope_section`; `None` — обычный 1D RoPE
    /// (у text-only сборок и старых конфигов).
    pub mrope: Option<MropeSpec>,
}

impl HybridConfig {
    /// Обратные частоты rotary-части — ровно те, что кладёт в таблицы
    /// `RopeCache::new` (та же f32-арифметика), чтобы M-RoPE-таблицы,
    /// собранные на host, для текста совпадали с обычным путём.
    pub fn rope_inv_freqs(&self) -> Vec<f32> {
        let rd = self.rotary_dim();
        (0..rd / 2)
            .map(|i| self.rope_theta.powf(-(2.0 * i as f32) / (rd as f32)))
            .collect()
    }

    pub fn from_hf_bytes(bytes: &[u8]) -> Result<Self, ConfigError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| ConfigError::Parse(format!("config json: {e}")))?;
        let tc = root
            .get("text_config")
            .cloned()
            .unwrap_or_else(|| root.clone());
        let raw: TextConfigRaw = serde_json::from_value(tc)
            .map_err(|e| ConfigError::Parse(format!("text_config: {e}")))?;
        let mrope = raw.rope_parameters.mrope_section.clone().map(|section| MropeSpec {
            section,
            interleaved: raw.rope_parameters.mrope_interleaved,
        });
        let mut cfg = Self::from_raw(raw)?;
        let id = |k: &str| root.get(k).and_then(|v| v.as_u64()).map(|v| v as u32);
        cfg.image_token_id = id("image_token_id");
        cfg.video_token_id = id("video_token_id");
        cfg.vision_start_token_id = id("vision_start_token_id");
        cfg.vision_end_token_id = id("vision_end_token_id");
        cfg.mrope = mrope;
        Ok(cfg)
    }

    fn from_raw(raw: TextConfigRaw) -> Result<Self, ConfigError> {
        let layer_kinds = if raw.layer_types.is_empty() {
            (0..raw.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % raw.full_attention_interval == 0 {
                        LayerKind::Full
                    } else {
                        LayerKind::Linear
                    }
                })
                .collect()
        } else {
            raw.layer_types
                .iter()
                .map(|s| match s.as_str() {
                    "full_attention" => Ok(LayerKind::Full),
                    "linear_attention" => Ok(LayerKind::Linear),
                    other => Err(ConfigError::Invalid(format!("layer_type: {other}"))),
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        if layer_kinds.len() != raw.num_hidden_layers {
            return Err(ConfigError::Invalid(format!(
                "layer_types len {} != num_hidden_layers {}",
                layer_kinds.len(),
                raw.num_hidden_layers
            )));
        }
        let cfg = Self {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            max_position_embeddings: raw.max_position_embeddings,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta: raw.rope_parameters.rope_theta,
            partial_rotary_factor: raw.partial_rotary_factor,
            attn_output_gate: raw.attn_output_gate,
            hidden_act: raw.hidden_act,
            full_attention_interval: raw.full_attention_interval,
            linear_num_key_heads: raw.linear_num_key_heads,
            linear_num_value_heads: raw.linear_num_value_heads,
            linear_key_head_dim: raw.linear_key_head_dim,
            linear_value_head_dim: raw.linear_value_head_dim,
            linear_conv_kernel_dim: raw.linear_conv_kernel_dim,
            layer_kinds,
            tie_word_embeddings: raw.tie_word_embeddings,
            bos_token_id: raw.bos_token_id,
            eos_token_id: raw.eos_token_id.first(),
            eos_token_ids: raw.eos_token_id.ids(),
            mtp_num_hidden_layers: raw.mtp_num_hidden_layers,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
            mrope: None,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.hidden_size == 0 || self.num_attention_heads == 0 {
            return Err(ConfigError::Invalid("hidden_size/heads == 0".into()));
        }
        if self.num_key_value_heads == 0
            || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return Err(ConfigError::Invalid(format!(
                "GQA: heads={} % kv_heads={} != 0",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.linear_num_key_heads == 0
            || self.linear_num_value_heads % self.linear_num_key_heads != 0
        {
            return Err(ConfigError::Invalid(format!(
                "linear GQA: v_heads={} % k_heads={} != 0",
                self.linear_num_value_heads, self.linear_num_key_heads
            )));
        }
        let rd = self.rotary_dim();
        if rd == 0 || rd % 2 != 0 {
            return Err(ConfigError::Invalid(format!("rotary_dim must be even > 0, got {rd}")));
        }
        if self.hidden_act != "silu" && self.hidden_act != "swish" {
            return Err(ConfigError::Invalid(format!("hidden_act: {}", self.hidden_act)));
        }
        Ok(())
    }

    pub fn to_decoder_config(&self) -> common::DecoderConfig {
        let layer_kinds = self
            .layer_kinds
            .iter()
            .map(|k| match k {
                LayerKind::Linear => common::LayerKind::Linear,
                LayerKind::Full => common::LayerKind::Full,
            })
            .collect();
        common::DecoderConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            norm_gain: common::NormGain::OnePlus,
            activation: common::Activation::Silu,
            sandwich_norms: false,
            post_norm_eps: None,
            qk_norm: true,
            attn_output_gate: self.attn_output_gate,
            attn_scale: 1.0 / (self.head_dim as f32).sqrt(),
            embed_scale: None,
            embed_rms_norm: false,
            logit_scale: None,
            logit_softcap: None,
            rope_global: common::RopeSpec {
                theta: self.rope_theta,
                rotary_dim: self.rotary_dim(),
                scaled_freqs: None,
            },
            rope_local: None,
            sliding_window: None,
            sliding_window_pattern: 0,
            layer_kinds,
            linear: Some(common::LinearAttnConfig {
                num_key_heads: self.linear_num_key_heads,
                num_value_heads: self.linear_num_value_heads,
                key_head_dim: self.linear_key_head_dim,
                value_head_dim: self.linear_value_head_dim,
                conv_kernel: self.linear_conv_kernel_dim,
            }),
            tie_word_embeddings: self.tie_word_embeddings,
            bos_token_id: self.bos_token_id,
            eos_token_ids: self.eos_ids(),
        }
    }

    pub fn rotary_dim(&self) -> usize {
        let rd = (self.head_dim as f32 * self.partial_rotary_factor).round() as usize;
        rd - (rd % 2)
    }

    pub fn group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn linear_group_size(&self) -> usize {
        self.linear_num_value_heads / self.linear_num_key_heads
    }

    pub fn q_total_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_total_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn conv_dim(&self) -> usize {
        self.linear_key_dim() * 2 + self.linear_value_dim()
    }

    pub fn layer_kind(&self, idx: usize) -> LayerKind {
        self.layer_kinds[idx]
    }

    pub fn eos_ids(&self) -> Vec<u32> {
        if self.eos_token_ids.is_empty() {
            self.eos_token_id.into_iter().collect()
        } else {
            self.eos_token_ids.clone()
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("config parse: {0}")]
    Parse(String),
    #[error("config invalid: {0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "model_type": "qwen3_5",
        "text_config": {
            "attn_output_gate": true,
            "bos_token_id": 248044,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "num_attention_heads": 24,
            "num_hidden_layers": 4,
            "num_key_value_heads": 4,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {"rope_theta": 10000000},
            "tie_word_embeddings": false,
            "vocab_size": 248320
        }
    }"#;

    #[test]
    fn parses_text_config() {
        let cfg = HybridConfig::from_hf_bytes(SAMPLE.as_bytes()).unwrap();
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_hidden_layers, 4);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.rotary_dim(), 64);
        assert_eq!(cfg.rope_theta, 10_000_000.0);
        assert_eq!(cfg.group_size(), 6);
        assert_eq!(cfg.linear_group_size(), 3);
        assert_eq!(cfg.linear_key_dim(), 2048);
        assert_eq!(cfg.linear_value_dim(), 6144);
        assert_eq!(cfg.conv_dim(), 10240);
        assert_eq!(cfg.layer_kind(0), LayerKind::Linear);
        assert_eq!(cfg.layer_kind(3), LayerKind::Full);
        assert!(!cfg.tie_word_embeddings);
        assert!(cfg.attn_output_gate);
    }

    #[test]
    fn reads_mrope_spec() {
        let json = r#"{"model_type": "qwen3_5", "rope_parameters": {"mrope_interleaved": true,
            "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000,
            "partial_rotary_factor": 0.25}, "head_dim": 256, "partial_rotary_factor": 0.25}"#;
        let cfg = HybridConfig::from_hf_bytes(json.as_bytes()).unwrap();
        assert_eq!(cfg.mrope, Some(MropeSpec { section: vec![11, 11, 10], interleaved: true }));
        assert_eq!(cfg.rotary_dim(), 64);
        assert_eq!(cfg.rope_inv_freqs().len(), 32);
        let plain = HybridConfig::from_hf_bytes(br#"{"model_type": "qwen3_5"}"#).unwrap();
        assert_eq!(plain.mrope, None);
    }

    #[test]
    fn reads_vision_token_ids() {
        let with = SAMPLE.replace(
            "\"model_type\": \"qwen3_5\",",
            "\"model_type\": \"qwen3_5\", \"image_token_id\": 248056, \"vision_start_token_id\": 248053,",
        );
        let cfg = HybridConfig::from_hf_bytes(with.as_bytes()).unwrap();
        assert_eq!(cfg.image_token_id, Some(248056));
        assert_eq!(cfg.vision_start_token_id, Some(248053));
        let cfg = HybridConfig::from_hf_bytes(SAMPLE.as_bytes()).unwrap();
        assert_eq!(cfg.image_token_id, None);
    }

    #[test]
    fn reads_mtp_layer_count() {
        let cfg = HybridConfig::from_hf_bytes(SAMPLE.as_bytes()).unwrap();
        assert_eq!(cfg.mtp_num_hidden_layers, 0);
        let with = SAMPLE.replace(r#""vocab_size": 248320"#, r#""vocab_size": 248320, "mtp_num_hidden_layers": 1"#);
        let cfg = HybridConfig::from_hf_bytes(with.as_bytes()).unwrap();
        assert_eq!(cfg.mtp_num_hidden_layers, 1);
    }

    #[test]
    fn eos_token_id_accepts_scalar_and_list() {
        let one = SAMPLE.replace(r#""eos_token_id": 248044"#, r#""eos_token_id": 248046"#);
        let cfg = HybridConfig::from_hf_bytes(one.as_bytes()).unwrap();
        assert_eq!(cfg.eos_ids(), vec![248046]);

        let many = SAMPLE.replace(r#""eos_token_id": 248044"#, r#""eos_token_id": [248046, 248044]"#);
        let cfg = HybridConfig::from_hf_bytes(many.as_bytes()).unwrap();
        assert_eq!(cfg.eos_ids(), vec![248046, 248044]);
        assert_eq!(cfg.eos_token_id, Some(248046));
    }
}
