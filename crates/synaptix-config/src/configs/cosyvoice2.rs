use serde::{Deserialize, Serialize};

fn default_vocab_size() -> usize { 8192 }
fn default_d_model() -> usize { 1024 }
fn default_num_heads() -> usize { 16 }
fn default_num_layers() -> usize { 14 }
fn default_ffn_dim() -> usize { 4096 }
fn default_flow_num_layers() -> usize { 22 }
fn default_flow_d_model() -> usize { 512 }
fn default_flow_num_heads() -> usize { 8 }
fn default_num_mel_bins() -> usize { 80 }
fn default_sampling_rate() -> usize { 22050 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosyVoice2Config {
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
    #[serde(default = "default_flow_num_layers")]
    pub flow_num_layers: usize,
    #[serde(default = "default_flow_d_model")]
    pub flow_d_model: usize,
    #[serde(default = "default_flow_num_heads")]
    pub flow_num_heads: usize,
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: usize,
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: usize,
}

impl Default for CosyVoice2Config {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            d_model: default_d_model(),
            num_heads: default_num_heads(),
            num_layers: default_num_layers(),
            ffn_dim: default_ffn_dim(),
            flow_num_layers: default_flow_num_layers(),
            flow_d_model: default_flow_d_model(),
            flow_num_heads: default_flow_num_heads(),
            num_mel_bins: default_num_mel_bins(),
            sampling_rate: default_sampling_rate(),
        }
    }
}
