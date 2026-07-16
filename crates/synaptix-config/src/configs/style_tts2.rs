use serde::{Deserialize, Serialize};

fn default_d_model() -> usize { 512 }
fn default_num_heads() -> usize { 8 }
fn default_num_layers() -> usize { 6 }
fn default_style_dim() -> usize { 128 }
fn default_max_duration() -> f64 { 50.0 }
fn default_multispeaker() -> bool { true }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleTts2Config {
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    #[serde(default = "default_style_dim")]
    pub style_dim: usize,
    #[serde(default = "default_max_duration")]
    pub max_duration: f64,
    #[serde(default = "default_multispeaker")]
    pub multispeaker: bool,
}

impl Default for StyleTts2Config {
    fn default() -> Self {
        Self {
            d_model: default_d_model(),
            num_heads: default_num_heads(),
            num_layers: default_num_layers(),
            style_dim: default_style_dim(),
            max_duration: default_max_duration(),
            multispeaker: default_multispeaker(),
        }
    }
}
