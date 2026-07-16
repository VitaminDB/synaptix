use serde::{Deserialize, Serialize};

fn default_in_channels() -> usize { 64 }
fn default_out_channels() -> usize { 64 }
fn default_vec_in_dim() -> usize { 768 }
fn default_context_in_dim() -> usize { 4096 }
fn default_hidden_size() -> usize { 3072 }
fn default_mlp_ratio() -> f64 { 4.0 }
fn default_num_heads() -> usize { 24 }
fn default_depth() -> usize { 19 }
fn default_depth_single_blocks() -> usize { 38 }
fn default_axes_dim() -> Vec<usize> { vec![16, 56, 56] }
fn default_theta() -> f64 { 10000.0 }
fn default_qkv_bias() -> bool { true }
fn default_guidance_embed() -> bool { false }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxConfig {
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,
    #[serde(default = "default_vec_in_dim")]
    pub vec_in_dim: usize,
    #[serde(default = "default_context_in_dim")]
    pub context_in_dim: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default = "default_depth_single_blocks")]
    pub depth_single_blocks: usize,
    #[serde(default = "default_axes_dim")]
    pub axes_dim: Vec<usize>,
    #[serde(default = "default_theta")]
    pub theta: f64,
    #[serde(default = "default_qkv_bias")]
    pub qkv_bias: bool,
    #[serde(default = "default_guidance_embed")]
    pub guidance_embed: bool,
}

impl Default for FluxConfig {
    fn default() -> Self {
        Self {
            in_channels: default_in_channels(),
            out_channels: default_out_channels(),
            vec_in_dim: default_vec_in_dim(),
            context_in_dim: default_context_in_dim(),
            hidden_size: default_hidden_size(),
            mlp_ratio: default_mlp_ratio(),
            num_heads: default_num_heads(),
            depth: default_depth(),
            depth_single_blocks: default_depth_single_blocks(),
            axes_dim: default_axes_dim(),
            theta: default_theta(),
            qkv_bias: default_qkv_bias(),
            guidance_embed: default_guidance_embed(),
        }
    }
}
