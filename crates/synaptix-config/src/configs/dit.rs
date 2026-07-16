use serde::{Deserialize, Serialize};

fn default_in_channels() -> usize { 4 }
fn default_out_channels() -> usize { 8 }
fn default_patch_size() -> usize { 2 }
fn default_hidden_size() -> usize { 1152 }
fn default_depth() -> usize { 28 }
fn default_num_heads() -> usize { 16 }
fn default_mlp_ratio() -> f64 { 4.0 }
fn default_class_dropout_prob() -> f64 { 0.1 }
fn default_num_classes() -> usize { 1000 }
fn default_learn_sigma() -> bool { true }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DitConfig {
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,
    #[serde(default = "default_class_dropout_prob")]
    pub class_dropout_prob: f64,
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
    #[serde(default = "default_learn_sigma")]
    pub learn_sigma: bool,
}

impl Default for DitConfig {
    fn default() -> Self {
        Self {
            in_channels: default_in_channels(),
            out_channels: default_out_channels(),
            patch_size: default_patch_size(),
            hidden_size: default_hidden_size(),
            depth: default_depth(),
            num_heads: default_num_heads(),
            mlp_ratio: default_mlp_ratio(),
            class_dropout_prob: default_class_dropout_prob(),
            num_classes: default_num_classes(),
            learn_sigma: default_learn_sigma(),
        }
    }
}
