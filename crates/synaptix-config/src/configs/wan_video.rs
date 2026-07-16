use serde::{Deserialize, Serialize};

fn default_model_type() -> String { "wan_video".into() }
fn default_num_attention_heads() -> usize { 40 }
fn default_attention_head_dim() -> usize { 128 }
fn default_in_channels() -> usize { 16 }
fn default_out_channels() -> usize { 16 }
fn default_num_layers() -> usize { 40 }
fn default_cross_attention_dim() -> usize { 5120 }
fn default_ffn_dim_multiplier() -> f64 { 4.0 / 3.0 }
fn default_patch_size() -> Vec<usize> { vec![1, 2, 2] }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WanVideoConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_attention_head_dim")]
    pub attention_head_dim: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    #[serde(default = "default_cross_attention_dim")]
    pub cross_attention_dim: usize,
    #[serde(default = "default_ffn_dim_multiplier")]
    pub ffn_dim_multiplier: f64,
    #[serde(default = "default_patch_size")]
    pub patch_size: Vec<usize>,
}

impl Default for WanVideoConfig {
    fn default() -> Self {
        Self {
            model_type: default_model_type(),
            num_attention_heads: default_num_attention_heads(),
            attention_head_dim: default_attention_head_dim(),
            in_channels: default_in_channels(),
            out_channels: default_out_channels(),
            num_layers: default_num_layers(),
            cross_attention_dim: default_cross_attention_dim(),
            ffn_dim_multiplier: default_ffn_dim_multiplier(),
            patch_size: default_patch_size(),
        }
    }
}
