use serde::{Deserialize, Serialize};

fn default_in_channels() -> usize { 16 }
fn default_out_channels() -> usize { 16 }
fn default_patch_size() -> usize { 2 }
fn default_num_layers() -> usize { 24 }
fn default_hidden_size() -> usize { 4096 }
fn default_num_heads() -> usize { 16 }
fn default_mlp_ratio() -> f64 { 4.0 }
fn default_caption_projection_dim() -> usize { 1536 }
fn default_pooled_projection_dim() -> usize { 2048 }
fn default_pos_embed_max_size() -> usize { 96 }
fn default_dual_attention_layers() -> Vec<usize> { vec![] }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sd3Config {
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,
    #[serde(default = "default_caption_projection_dim")]
    pub caption_projection_dim: usize,
    #[serde(default = "default_pooled_projection_dim")]
    pub pooled_projection_dim: usize,
    #[serde(default = "default_pos_embed_max_size")]
    pub pos_embed_max_size: usize,
    #[serde(default = "default_dual_attention_layers")]
    pub dual_attention_layers: Vec<usize>,
    #[serde(default)]
    pub qk_norm: Option<String>,
}

impl Default for Sd3Config {
    fn default() -> Self {
        Self {
            in_channels: default_in_channels(),
            out_channels: default_out_channels(),
            patch_size: default_patch_size(),
            num_layers: default_num_layers(),
            hidden_size: default_hidden_size(),
            num_heads: default_num_heads(),
            mlp_ratio: default_mlp_ratio(),
            caption_projection_dim: default_caption_projection_dim(),
            pooled_projection_dim: default_pooled_projection_dim(),
            pos_embed_max_size: default_pos_embed_max_size(),
            dual_attention_layers: default_dual_attention_layers(),
            qk_norm: None,
        }
    }
}
