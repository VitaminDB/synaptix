use serde::{Deserialize, Serialize};

fn default_num_attention_heads() -> usize { 48 }
fn default_attention_head_dim() -> usize { 64 }
fn default_in_channels() -> usize { 16 }
fn default_out_channels() -> usize { 16 }
fn default_num_layers() -> usize { 42 }
fn default_dropout() -> f64 { 0.0 }
fn default_cross_attention_dim() -> usize { 4096 }
fn default_attention_bias() -> bool { true }
fn default_sample_size() -> usize { 60 }
fn default_sample_size_t() -> usize { 7 }
fn default_patch_size() -> usize { 2 }
fn default_patch_size_t() -> usize { 2 }
fn default_time_embed_dim() -> usize { 512 }
fn default_timestep_activation_fn() -> String { "silu".into() }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CogVideoConfig {
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
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default = "default_cross_attention_dim")]
    pub cross_attention_dim: usize,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_sample_size")]
    pub sample_size: usize,
    #[serde(default = "default_sample_size_t")]
    pub sample_size_t: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_patch_size_t")]
    pub patch_size_t: usize,
    #[serde(default = "default_time_embed_dim")]
    pub time_embed_dim: usize,
    #[serde(default = "default_timestep_activation_fn")]
    pub timestep_activation_fn: String,
}

impl Default for CogVideoConfig {
    fn default() -> Self {
        Self {
            num_attention_heads: default_num_attention_heads(),
            attention_head_dim: default_attention_head_dim(),
            in_channels: default_in_channels(),
            out_channels: default_out_channels(),
            num_layers: default_num_layers(),
            dropout: default_dropout(),
            cross_attention_dim: default_cross_attention_dim(),
            attention_bias: default_attention_bias(),
            sample_size: default_sample_size(),
            sample_size_t: default_sample_size_t(),
            patch_size: default_patch_size(),
            patch_size_t: default_patch_size_t(),
            time_embed_dim: default_time_embed_dim(),
            timestep_activation_fn: default_timestep_activation_fn(),
        }
    }
}
