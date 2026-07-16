use serde::{Deserialize, Serialize};

fn default_vocab_size() -> usize { 34 }
fn default_d_model() -> usize { 1024 }
fn default_n_heads() -> usize { 8 }
fn default_n_layers() -> usize { 24 }
fn default_ffn_dim() -> usize { 4096 }
fn default_conv_channels() -> usize { 256 }
fn default_conv_kernel_size() -> usize { 31 }
fn default_subsampling_factor() -> usize { 8 }
fn default_max_seq_len() -> usize { 1024 }
fn default_dropout() -> f64 { 0.1 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GigaAmConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_n_heads")]
    pub n_heads: usize,
    #[serde(default = "default_n_layers")]
    pub n_layers: usize,
    #[serde(default = "default_ffn_dim")]
    pub ffn_dim: usize,
    #[serde(default = "default_conv_channels")]
    pub conv_channels: usize,
    #[serde(default = "default_conv_kernel_size")]
    pub conv_kernel_size: usize,
    #[serde(default = "default_subsampling_factor")]
    pub subsampling_factor: usize,
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
}

impl Default for GigaAmConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            d_model: default_d_model(),
            n_heads: default_n_heads(),
            n_layers: default_n_layers(),
            ffn_dim: default_ffn_dim(),
            conv_channels: default_conv_channels(),
            conv_kernel_size: default_conv_kernel_size(),
            subsampling_factor: default_subsampling_factor(),
            max_seq_len: default_max_seq_len(),
            dropout: default_dropout(),
        }
    }
}
