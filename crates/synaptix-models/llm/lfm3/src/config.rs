//! `config.json` LFM3 → [`DecoderConfig`] общего декодера.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use synaptix_llm_common::config::{Activation, DecoderConfig, DecoderExt, LayerKind, NormGain, RopeSpec};

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Ids {
    One(u32),
    Many(Vec<u32>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct Lfm3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub norm_eps: f32,
    pub layer_types: Vec<String>,
    pub rope_parameters: HashMap<String, RopeParams>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default = "default_conv")]
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub conv_bias: bool,
    #[serde(default)]
    pub use_sandwich_norm: bool,
    #[serde(default = "yes")]
    pub tie_word_embeddings: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<Ids>,
}

fn default_conv() -> usize {
    3
}

fn yes() -> bool {
    true
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("io: {0}")]
    Io(String),
    #[error("parse: {0}")]
    Parse(String),
    #[error("invalid: {0}")]
    Invalid(String),
}

impl Lfm3Config {
    pub fn from_hf_json(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let p = path.as_ref();
        let b = std::fs::read(p).map_err(|e| ConfigError::Io(format!("read {}: {e}", p.display())))?;
        Self::from_hf_json_slice(&b)
    }

    pub fn from_hf_json_slice(bytes: &[u8]) -> Result<Self, ConfigError> {
        let mut c: Self = serde_json::from_slice(bytes).map_err(|e| ConfigError::Parse(format!("config.json: {e}")))?;
        if c.head_dim == 0 {
            c.head_dim = c.hidden_size / c.num_attention_heads.max(1);
        }
        if c.layer_types.len() != c.num_hidden_layers {
            return Err(ConfigError::Invalid(format!("layer_types: {} вместо {}", c.layer_types.len(), c.num_hidden_layers)));
        }
        if c.conv_bias {
            return Err(ConfigError::Invalid("conv_bias=true пока не поддержан".into()));
        }
        Ok(c)
    }

    pub fn eos_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            Some(Ids::One(i)) => vec![*i],
            Some(Ids::Many(v)) => v.clone(),
            None => Vec::new(),
        }
    }

    /// RoPE типа слоя: частичный — «нулевой хвост» (частоты первых `factor·hd/2` пар — по прогрессии
    /// полной головы, остальные нули), т.е. поворот по всей голове с нулевыми частотами хвоста.
    fn rope(&self, kind: &str) -> RopeSpec {
        let p = self.rope_parameters.get(kind).or_else(|| self.rope_parameters.values().next());
        let theta = p.map_or(10_000.0, |p| p.rope_theta);
        let factor = p.and_then(|p| p.partial_rotary_factor).unwrap_or(1.0);
        let hd = self.head_dim;
        let angles = ((factor * hd as f32) as usize) / 2;
        let freqs: Vec<f32> = (0..hd / 2).map(|i| if i < angles { 1.0 / theta.powf((2 * i) as f32 / hd as f32) } else { 0.0 }).collect();
        RopeSpec { theta, rotary_dim: hd, scaled_freqs: Some(freqs), mscale: 1.0 }
    }

    pub fn to_decoder_config(&self) -> DecoderConfig {
        let kinds: Vec<LayerKind> = self.layer_types.iter().map(|t| if t == "conv" { LayerKind::ShortConv } else { LayerKind::Full }).collect();
        let sliding: Vec<bool> = self.layer_types.iter().map(|t| t == "sliding_attention").collect();
        let has_sliding = sliding.iter().any(|&b| b);
        DecoderConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.norm_eps,
            norm_gain: NormGain::Plain,
            activation: Activation::Silu,
            sandwich_norms: self.use_sandwich_norm,
            post_norm_eps: None,
            qk_norm: true,
            attn_output_gate: false,
            attn_scale: 1.0 / (self.head_dim as f32).sqrt(),
            embed_scale: None,
            embed_rms_norm: false,
            logit_scale: None,
            logit_softcap: None,
            rope_global: self.rope("full_attention"),
            rope_local: has_sliding.then(|| self.rope("sliding_attention")),
            sliding_window: if has_sliding { self.sliding_window } else { None },
            sliding_window_pattern: 0,
            layer_kinds: kinds,
            linear: None,
            tie_word_embeddings: self.tie_word_embeddings,
            bos_token_id: self.bos_token_id,
            eos_token_ids: self.eos_ids(),
            ext: Some(DecoderExt { head_gate: true, sliding_layers: sliding, short_conv_kernel: self.conv_l_cache, ..Default::default() }),
        }
    }
}
