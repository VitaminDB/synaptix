use serde::{Deserialize, Serialize};

fn default_vocab_size() -> usize { 51865 }
fn default_num_mel_bins() -> usize { 80 }
fn default_encoder_layers() -> usize { 24 }
fn default_encoder_attention_heads() -> usize { 16 }
fn default_decoder_layers() -> usize { 24 }
fn default_decoder_attention_heads() -> usize { 16 }
fn default_decoder_ffn_dim() -> usize { 4096 }
fn default_encoder_ffn_dim() -> usize { 4096 }
fn default_d_model() -> usize { 1024 }
fn default_dropout() -> f64 { 0.0 }
fn default_max_source_positions() -> usize { 1500 }
fn default_max_target_positions() -> usize { 448 }
fn default_pad_token_id() -> u32 { 50256 }
fn default_bos_token_id() -> u32 { 50257 }
fn default_eos_token_id() -> u32 { 50256 }
fn default_suppress_tokens() -> Vec<u32> { vec![] }
fn default_begin_suppress_tokens() -> Vec<u32> { vec![220, 50256] }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: usize,
    #[serde(default = "default_encoder_layers")]
    pub encoder_layers: usize,
    #[serde(default = "default_encoder_attention_heads")]
    pub encoder_attention_heads: usize,
    #[serde(default = "default_decoder_layers")]
    pub decoder_layers: usize,
    #[serde(default = "default_decoder_attention_heads")]
    pub decoder_attention_heads: usize,
    #[serde(default = "default_decoder_ffn_dim")]
    pub decoder_ffn_dim: usize,
    #[serde(default = "default_encoder_ffn_dim")]
    pub encoder_ffn_dim: usize,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default = "default_max_source_positions")]
    pub max_source_positions: usize,
    #[serde(default = "default_max_target_positions")]
    pub max_target_positions: usize,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: u32,
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,
    #[serde(default = "default_suppress_tokens")]
    pub suppress_tokens: Vec<u32>,
    #[serde(default = "default_begin_suppress_tokens")]
    pub begin_suppress_tokens: Vec<u32>,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            num_mel_bins: default_num_mel_bins(),
            encoder_layers: default_encoder_layers(),
            encoder_attention_heads: default_encoder_attention_heads(),
            decoder_layers: default_decoder_layers(),
            decoder_attention_heads: default_decoder_attention_heads(),
            decoder_ffn_dim: default_decoder_ffn_dim(),
            encoder_ffn_dim: default_encoder_ffn_dim(),
            d_model: default_d_model(),
            dropout: default_dropout(),
            max_source_positions: default_max_source_positions(),
            max_target_positions: default_max_target_positions(),
            pad_token_id: default_pad_token_id(),
            bos_token_id: default_bos_token_id(),
            eos_token_id: default_eos_token_id(),
            suppress_tokens: default_suppress_tokens(),
            begin_suppress_tokens: default_begin_suppress_tokens(),
        }
    }
}
