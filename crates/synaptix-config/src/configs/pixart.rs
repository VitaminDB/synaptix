use serde::{Deserialize, Serialize};

fn default_num_attention_heads() -> usize { 16 }
fn default_attention_head_dim() -> usize { 72 }
fn default_in_channels() -> usize { 4 }
fn default_out_channels() -> usize { 8 }
fn default_num_layers() -> usize { 28 }
fn default_dropout() -> f64 { 0.0 }
fn default_cross_attention_dim() -> usize { 1152 }
fn default_attention_bias() -> bool { true }
fn default_sample_size() -> usize { 128 }
fn default_activation_fn() -> String { "gelu-approximate".into() }
fn default_use_linear_projection() -> bool { false }
fn default_leakyrelu_slope() -> f64 { 0.01 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PixartConfig {
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
    #[serde(default)]
    pub num_vector_embeds: Option<usize>,
    #[serde(default = "default_activation_fn")]
    pub activation_fn: String,
    #[serde(default)]
    pub num_embeds_ada_norm: Option<usize>,
    #[serde(default = "default_use_linear_projection")]
    pub use_linear_projection: bool,
    #[serde(default = "default_leakyrelu_slope")]
    pub leakyrelu_slope: f64,
    #[serde(default)]
    pub caption_channels: Option<usize>,
}

impl Default for PixartConfig {
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
            num_vector_embeds: None,
            activation_fn: default_activation_fn(),
            num_embeds_ada_norm: None,
            use_linear_projection: default_use_linear_projection(),
            leakyrelu_slope: default_leakyrelu_slope(),
            caption_channels: None,
        }
    }
}
