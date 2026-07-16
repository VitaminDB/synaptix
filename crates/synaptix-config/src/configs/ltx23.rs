use serde::{Deserialize, Serialize};

fn default_num_attention_heads() -> usize { 32 }
fn default_attention_head_dim() -> usize { 64 }
fn default_in_channels() -> usize { 128 }
fn default_out_channels() -> usize { 128 }
fn default_num_layers() -> usize { 48 }
fn default_cross_attention_dim() -> usize { 2048 }
fn default_caption_projection_dim() -> usize { 2048 }
fn default_qk_norm() -> String { "rms_norm_across_heads".into() }
fn default_positional_embedding_type() -> String { "rope".into() }
fn default_positional_embedding_theta() -> f64 { 10000.0 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ltx23Config {
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
    #[serde(default = "default_caption_projection_dim")]
    pub caption_projection_dim: usize,
    #[serde(default = "default_qk_norm")]
    pub qk_norm: String,
    #[serde(default = "default_positional_embedding_type")]
    pub positional_embedding_type: String,
    #[serde(default = "default_positional_embedding_theta")]
    pub positional_embedding_theta: f64,
}

impl Default for Ltx23Config {
    fn default() -> Self {
        Self {
            num_attention_heads: default_num_attention_heads(),
            attention_head_dim: default_attention_head_dim(),
            in_channels: default_in_channels(),
            out_channels: default_out_channels(),
            num_layers: default_num_layers(),
            cross_attention_dim: default_cross_attention_dim(),
            caption_projection_dim: default_caption_projection_dim(),
            qk_norm: default_qk_norm(),
            positional_embedding_type: default_positional_embedding_type(),
            positional_embedding_theta: default_positional_embedding_theta(),
        }
    }
}
