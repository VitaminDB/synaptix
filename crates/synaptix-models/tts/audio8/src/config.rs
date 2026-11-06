use serde::Deserialize;

use crate::{Audio8Error, Result};

fn d_true() -> bool {
    true
}

/// `config.json` модели (ArkttsConfig). Нужные поля; остальное игнорируется.
#[derive(Debug, Clone, Deserialize)]
pub struct Audio8Config {
    pub dim: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_local_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    #[serde(default = "d_true")]
    pub attention_qkv_bias: bool,
    #[serde(default)]
    pub attention_o_bias: bool,
    #[serde(default)]
    pub attention_qk_norm: bool,
    pub fast_dim: usize,
    pub n_fast_layer: usize,
    pub fast_n_head: usize,
    pub fast_n_local_heads: usize,
    pub fast_head_dim: usize,
    pub fast_intermediate_size: usize,
    #[serde(default)]
    pub fast_attention_qkv_bias: bool,
    #[serde(default)]
    pub fast_attention_o_bias: bool,
    #[serde(default)]
    pub fast_attention_qk_norm: bool,
    pub norm_eps: f32,
    pub rope_base: f32,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub codebook_size: usize,
    pub num_codebooks: usize,
    pub semantic_begin_id: u32,
    pub semantic_end_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    #[serde(default = "d_true")]
    pub norm_fastlayer_input: bool,
    #[serde(default)]
    pub scale_codebook_embeddings: bool,
    pub ras_temperature: f32,
    pub ras_top_p: f32,
    pub ras_window_size: usize,
    #[serde(default = "d_post_layers")]
    pub codec_post_n_layer: usize,
    #[serde(default = "d_post_heads")]
    pub codec_post_n_head: usize,
    #[serde(default = "d_post_local")]
    pub codec_post_n_local_heads: usize,
    #[serde(default = "d_post_ffn")]
    pub codec_post_intermediate_size: usize,
}

fn d_post_layers() -> usize {
    8
}
fn d_post_heads() -> usize {
    16
}
fn d_post_local() -> usize {
    8
}
fn d_post_ffn() -> usize {
    1216
}

impl Audio8Config {
    pub fn from_json_bytes(b: &[u8]) -> Result<Self> {
        let c: Self = serde_json::from_slice(b).map_err(|e| Audio8Error::Config(e.to_string()))?;
        if c.attention_qk_norm || c.fast_attention_qk_norm {
            return Err(Audio8Error::Config("qk_norm не поддержан (у Audio8 его нет)".into()));
        }
        if c.scale_codebook_embeddings {
            return Err(Audio8Error::Config("scale_codebook_embeddings не поддержан".into()));
        }
        if c.dim != c.fast_dim {
            return Err(Audio8Error::Config("fast_dim ≠ dim (fast_project_in) не поддержан".into()));
        }
        Ok(c)
    }

    /// Число семантических токенов (= размер семантического кодбука).
    pub fn semantic_count(&self) -> usize {
        (self.semantic_end_id - self.semantic_begin_id + 1) as usize
    }
}

/// `generation_config.json`: умолчания сэмплинга модели.
#[derive(Debug, Clone, Deserialize)]
pub struct GenDefaults {
    #[serde(default = "d_temp")]
    pub temperature: f32,
    #[serde(default = "d_top_p")]
    pub top_p: f32,
    #[serde(default = "d_top_k")]
    pub top_k: usize,
    #[serde(default = "d_max_new")]
    pub max_new_tokens: usize,
}

fn d_temp() -> f32 {
    0.7
}
fn d_top_p() -> f32 {
    0.9
}
fn d_top_k() -> usize {
    50
}
fn d_max_new() -> usize {
    512
}

impl Default for GenDefaults {
    fn default() -> Self {
        Self { temperature: d_temp(), top_p: d_top_p(), top_k: d_top_k(), max_new_tokens: d_max_new() }
    }
}
