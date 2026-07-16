use serde::{Deserialize, Serialize};

fn default_vocab_size() -> usize { 2048 }
fn default_max_position_embeddings() -> usize { 2048 }
fn default_num_hidden_layers() -> usize { 48 }
fn default_ffn_dim() -> usize { 16384 }
fn default_num_attention_heads() -> usize { 16 }
fn default_d_model() -> usize { 4096 }
fn default_layerdrop() -> f64 { 0.0 }
fn default_audio_channels() -> usize { 1 }
fn default_num_codebooks() -> usize { 4 }
fn default_pad_token_id() -> u32 { 2048 }
fn default_bos_token_id() -> u32 { 2048 }
fn default_eos_token_id() -> u32 { 2048 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicgenConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_ffn_dim")]
    pub ffn_dim: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_layerdrop")]
    pub layerdrop: f64,
    #[serde(default = "default_audio_channels")]
    pub audio_channels: usize,
    #[serde(default = "default_num_codebooks")]
    pub num_codebooks: usize,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: u32,
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,
}

impl Default for MusicgenConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            max_position_embeddings: default_max_position_embeddings(),
            num_hidden_layers: default_num_hidden_layers(),
            ffn_dim: default_ffn_dim(),
            num_attention_heads: default_num_attention_heads(),
            d_model: default_d_model(),
            layerdrop: default_layerdrop(),
            audio_channels: default_audio_channels(),
            num_codebooks: default_num_codebooks(),
            pad_token_id: default_pad_token_id(),
            bos_token_id: default_bos_token_id(),
            eos_token_id: default_eos_token_id(),
        }
    }
}
