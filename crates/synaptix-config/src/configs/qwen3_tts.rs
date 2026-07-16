use serde::{Deserialize, Serialize};

fn default_vocab_size() -> usize { 8192 }
fn default_d_model() -> usize { 1024 }
fn default_num_heads() -> usize { 16 }
fn default_num_layers() -> usize { 16 }
fn default_ffn_dim() -> usize { 4096 }
fn default_num_mel_bins() -> usize { 80 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3TtsConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    #[serde(default = "default_ffn_dim")]
    pub ffn_dim: usize,
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: usize,
}

impl Default for Qwen3TtsConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            d_model: default_d_model(),
            num_heads: default_num_heads(),
            num_layers: default_num_layers(),
            ffn_dim: default_ffn_dim(),
            num_mel_bins: default_num_mel_bins(),
        }
    }
}
