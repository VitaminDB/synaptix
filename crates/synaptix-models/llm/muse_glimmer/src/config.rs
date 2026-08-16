use serde::Deserialize;
use synaptix_llm_common as common;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
}

fn default_rope_theta() -> f32 {
    500_000.0
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
    post_norm_eps: f32,
    qk_scale_factor: f32,
    output_multiplier: f32,
    final_logit_softcapping: f32,
    hidden_activation: String,
    sliding_window: usize,
    layer_types: Vec<String>,
    layer_rope_theta: Vec<f32>,
    tie_word_embeddings: bool,
    bos_token_id: Option<u32>,
    eos_token_id: EosField,
    rope_parameters: RopeParameters,
}

impl Default for TextConfigRaw {
    fn default() -> Self {
        Self {
            vocab_size: 202_048,
            hidden_size: 6656,
            intermediate_size: 19_968,
            num_hidden_layers: 52,
            num_attention_heads: 32,
            num_key_value_heads: 2,
            head_dim: 128,
            max_position_embeddings: 131_072,
            rms_norm_eps: 1.0e-5,
            post_norm_eps: 1.0e-8,
            qk_scale_factor: 3.87,
            output_multiplier: 0.196_116_14,
            final_logit_softcapping: 20.0,
            hidden_activation: "silu".into(),
            sliding_window: 2048,
            layer_types: Vec::new(),
            layer_rope_theta: Vec::new(),
            tie_word_embeddings: false,
            bos_token_id: Some(200_000),
            eos_token_id: EosField::One(200_001),
            rope_parameters: RopeParameters { rope_theta: default_rope_theta() },
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VisionConfigRaw {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub patch_size: usize,
    pub patch_temporal: usize,
    pub merge_size: usize,
    pub pos_emb_height: usize,
    pub pos_emb_width: usize,
    pub layer_norm_eps: f32,
    pub layer_types: Vec<String>,
    pub rope_parameters: RopeParameters,
}

impl Default for VisionConfigRaw {
    fn default() -> Self {
        Self {
            hidden_size: 1536,
            num_hidden_layers: 50,
            intermediate_size: 8960,
            num_attention_heads: 16,
            patch_size: 14,
            patch_temporal: 2,
            merge_size: 2,
            pos_emb_height: 32,
            pos_emb_width: 32,
            layer_norm_eps: 1.0e-5,
            layer_types: Vec::new(),
            rope_parameters: RopeParameters { rope_theta: 10_000.0 },
        }
    }
}

#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub patch_size: usize,
    pub patch_temporal: usize,
    pub merge_size: usize,
    pub pos_emb_side: usize,
    pub layer_norm_eps: f32,
    pub full_layers: Vec<bool>,
    pub rope_theta: f32,
    pub out_hidden_size: usize,
    pub projector_hidden_size: usize,
    pub max_image_tokens: usize,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn window_patches(&self) -> usize {
        self.pos_emb_side
    }

    pub fn patch_features(&self) -> usize {
        self.patch_temporal * 3 * self.patch_size * self.patch_size
    }

    pub fn merge_unit(&self) -> usize {
        self.merge_size * self.merge_size
    }

    pub fn merged_dim(&self) -> usize {
        self.hidden_size * self.merge_unit()
    }
}

#[derive(Debug, Clone)]
pub struct MuseConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub post_norm_eps: f32,
    pub qk_scale_factor: f32,
    pub output_multiplier: f32,
    pub final_logit_softcapping: f32,
    pub sliding_window: usize,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    pub image_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
    pub vision: Option<VisionConfig>,
}

pub const FULL_ATTENTION_INTERVAL: usize = 4;

impl MuseConfig {
    pub fn from_hf_bytes(bytes: &[u8]) -> Result<Self, ConfigError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| ConfigError::Parse(format!("config json: {e}")))?;
        let tc = root
            .get("text_config")
            .cloned()
            .unwrap_or_else(|| root.clone());
        let raw: TextConfigRaw = serde_json::from_value(tc)
            .map_err(|e| ConfigError::Parse(format!("text_config: {e}")))?;

        if !raw.layer_types.is_empty() {
            if raw.layer_types.len() != raw.num_hidden_layers {
                return Err(ConfigError::Invalid(format!(
                    "layer_types len {} != num_hidden_layers {}",
                    raw.layer_types.len(),
                    raw.num_hidden_layers
                )));
            }
            for (i, t) in raw.layer_types.iter().enumerate() {
                let expect_full = (i + 1) % FULL_ATTENTION_INTERVAL == 0;
                let ok = match t.as_str() {
                    "full_attention" => expect_full,
                    "sliding_attention" => !expect_full,
                    other => return Err(ConfigError::Invalid(format!("layer_type: {other}"))),
                };
                if !ok {
                    return Err(ConfigError::Invalid(format!(
                        "layer_types[{i}] = {t} не соответствует паттерну [S,S,S,F]"
                    )));
                }
            }
        }
        if !raw.layer_rope_theta.is_empty() {
            if raw.layer_rope_theta.len() != raw.num_hidden_layers {
                return Err(ConfigError::Invalid("layer_rope_theta длины != num_hidden_layers".into()));
            }
            for (i, th) in raw.layer_rope_theta.iter().enumerate() {
                let expect_nope = (i + 1) % FULL_ATTENTION_INTERVAL == 0;
                if expect_nope != (*th == 0.0) {
                    return Err(ConfigError::Invalid(format!(
                        "layer_rope_theta[{i}] = {th}: NoPE ожидается только на full-слоях"
                    )));
                }
                if *th != 0.0 && *th != raw.rope_parameters.rope_theta {
                    return Err(ConfigError::Invalid(format!(
                        "layer_rope_theta[{i}] = {th} != rope_theta {}",
                        raw.rope_parameters.rope_theta
                    )));
                }
            }
        }

        let id = |k: &str| root.get(k).and_then(|v| v.as_u64()).map(|v| v as u32);
        let vision = root.get("vision_config").map(|vc| -> Result<VisionConfig, ConfigError> {
            let vraw: VisionConfigRaw = serde_json::from_value(vc.clone())
                .map_err(|e| ConfigError::Parse(format!("vision_config: {e}")))?;
            let n = vraw.num_hidden_layers;
            let full_layers = if vraw.layer_types.is_empty() {
                (0..n).map(|i| (i + 1) % 4 == 0 || i == n - 1).collect()
            } else {
                vraw.layer_types
                    .iter()
                    .map(|s| match s.as_str() {
                        "full_attention" => Ok(true),
                        "window_attention" => Ok(false),
                        other => Err(ConfigError::Invalid(format!("vision layer_type: {other}"))),
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };
            if vraw.pos_emb_height != vraw.pos_emb_width {
                return Err(ConfigError::Invalid("pos_emb_height != pos_emb_width".into()));
            }
            Ok(VisionConfig {
                hidden_size: vraw.hidden_size,
                num_hidden_layers: n,
                intermediate_size: vraw.intermediate_size,
                num_attention_heads: vraw.num_attention_heads,
                patch_size: vraw.patch_size,
                patch_temporal: vraw.patch_temporal,
                merge_size: vraw.merge_size,
                pos_emb_side: vraw.pos_emb_height,
                layer_norm_eps: vraw.layer_norm_eps,
                full_layers,
                rope_theta: vraw.rope_parameters.rope_theta,
                out_hidden_size: root
                    .get("out_hidden_size")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(6144),
                projector_hidden_size: root
                    .get("projector_hidden_size")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(4096),
                max_image_tokens: 4096,
            })
        });
        let vision = match vision {
            Some(v) => Some(v?),
            None => None,
        };

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
            post_norm_eps: raw.post_norm_eps,
            qk_scale_factor: raw.qk_scale_factor,
            output_multiplier: raw.output_multiplier,
            final_logit_softcapping: raw.final_logit_softcapping,
            sliding_window: raw.sliding_window,
            rope_theta: raw.rope_parameters.rope_theta,
            tie_word_embeddings: raw.tie_word_embeddings,
            bos_token_id: raw.bos_token_id,
            eos_token_ids: raw.eos_token_id.ids(),
            image_token_id: id("image_token_id"),
            video_token_id: id("video_token_id"),
            vision,
        };
        cfg.validate(&raw.hidden_activation)?;
        Ok(cfg)
    }

    fn validate(&self, activation: &str) -> Result<(), ConfigError> {
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
        if activation != "silu" && activation != "swish" {
            return Err(ConfigError::Invalid(format!("hidden_activation: {activation}")));
        }
        if self.sliding_window == 0 {
            return Err(ConfigError::Invalid("sliding_window == 0".into()));
        }
        Ok(())
    }

    pub fn merge_generation_config(&mut self, bytes: &[u8]) {
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(bytes) else { return };
        let Some(eos) = v.get("eos_token_id") else { return };
        let ids: Vec<u32> = match eos {
            serde_json::Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            serde_json::Value::Array(a) => a
                .iter()
                .filter_map(|x| x.as_u64().map(|v| v as u32))
                .collect(),
            _ => Vec::new(),
        };
        for i in ids {
            if !self.eos_token_ids.contains(&i) {
                self.eos_token_ids.push(i);
            }
        }
    }

    pub fn to_decoder_config(&self) -> common::DecoderConfig {
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
            sandwich_norms: true,
            post_norm_eps: Some(self.post_norm_eps),
            qk_norm: true,
            attn_output_gate: true,
            attn_scale: self.qk_scale_factor / (self.head_dim as f32).sqrt(),
            embed_scale: None,
            embed_rms_norm: true,
            logit_scale: Some(self.output_multiplier),
            logit_softcap: Some(self.final_logit_softcapping),
            rope_global: common::RopeSpec { theta: 0.0, rotary_dim: 0, scaled_freqs: None },
            rope_local: Some(common::RopeSpec {
                theta: self.rope_theta,
                rotary_dim: self.head_dim,
                scaled_freqs: None,
            }),
            sliding_window: Some(self.sliding_window),
            sliding_window_pattern: FULL_ATTENTION_INTERVAL,
            layer_kinds: vec![common::LayerKind::Full; self.num_hidden_layers],
            linear: None,
            tie_word_embeddings: self.tie_word_embeddings,
            bos_token_id: self.bos_token_id,
            eos_token_ids: self.eos_token_ids.clone(),
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
        "model_type": "muse_glimmer",
        "image_token_id": 200092,
        "video_token_id": 200091,
        "out_hidden_size": 6144,
        "projector_hidden_size": 4096,
        "text_config": {
            "hidden_size": 6656,
            "intermediate_size": 19968,
            "num_hidden_layers": 8,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 202048,
            "rms_norm_eps": 1e-05,
            "post_norm_eps": 1e-08,
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.19611613513818404,
            "final_logit_softcapping": 20.0,
            "hidden_activation": "silu",
            "sliding_window": 2048,
            "max_position_embeddings": 131072,
            "layer_types": ["sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
            "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0, 500000.0, 500000.0, 500000.0, 0],
            "tie_word_embeddings": false,
            "bos_token_id": 200000,
            "eos_token_id": 200001,
            "rope_parameters": {"rope_theta": 500000.0}
        },
        "vision_config": {
            "hidden_size": 1536,
            "num_hidden_layers": 50,
            "intermediate_size": 8960,
            "num_attention_heads": 16,
            "patch_size": 14,
            "patch_temporal": 2,
            "merge_size": 2,
            "pos_emb_height": 32,
            "pos_emb_width": 32,
            "layer_norm_eps": 1e-05,
            "rope_parameters": {"rope_theta": 10000.0}
        }
    }"#;

    #[test]
    fn parses_text_and_vision() {
        let cfg = MuseConfig::from_hf_bytes(SAMPLE.as_bytes()).unwrap();
        assert_eq!(cfg.hidden_size, 6656);
        assert_eq!(cfg.num_hidden_layers, 8);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.sliding_window, 2048);
        assert_eq!(cfg.image_token_id, Some(200092));
        let v = cfg.vision.as_ref().unwrap();
        assert_eq!(v.hidden_size, 1536);
        assert_eq!(v.head_dim(), 96);
        assert_eq!(v.patch_features(), 1176);
        assert!(v.full_layers[3]);
        assert!(!v.full_layers[0]);
        assert!(v.full_layers[49]);
        assert_eq!(v.out_hidden_size, 6144);
    }

    #[test]
    fn decoder_config_shape() {
        let cfg = MuseConfig::from_hf_bytes(SAMPLE.as_bytes()).unwrap();
        let d = cfg.to_decoder_config();
        assert!(d.sandwich_norms);
        assert!(d.embed_rms_norm);
        assert_eq!(d.post_norm_eps, Some(1e-8));
        assert_eq!(d.logit_softcap, Some(20.0));
        assert_eq!(d.rope_global.rotary_dim, 0);
        assert_eq!(d.rope_local.as_ref().unwrap().rotary_dim, 128);
        assert_eq!(d.sliding_window_pattern, 4);
        assert!((d.attn_scale - 3.87 / (128.0f32).sqrt()).abs() < 1e-6);
        assert!(!d.is_global_layer(0));
        assert!(d.is_global_layer(3));
        assert!(d.is_global_layer(7));
        assert_eq!(d.window_for(0), Some(2048));
        assert_eq!(d.window_for(3), None);
        assert!(d.graph_decode_ok());
        assert!(!d.graph_prefill_ok());
    }

    #[test]
    fn rejects_wrong_pattern() {
        let bad = SAMPLE.replace(
            r#""layer_rope_theta": [500000.0, 500000.0, 500000.0, 0, 500000.0, 500000.0, 500000.0, 0],"#,
            r#""layer_rope_theta": [0, 500000.0, 500000.0, 500000.0, 500000.0, 500000.0, 500000.0, 0],"#,
        );
        assert!(MuseConfig::from_hf_bytes(bad.as_bytes()).is_err());
    }

    #[test]
    fn merges_generation_config_eos() {
        let mut cfg = MuseConfig::from_hf_bytes(SAMPLE.as_bytes()).unwrap();
        assert_eq!(cfg.eos_token_ids, vec![200001]);
        cfg.merge_generation_config(br#"{"eos_token_id": [200001, 200008]}"#);
        assert_eq!(cfg.eos_token_ids, vec![200001, 200008]);
    }
}
