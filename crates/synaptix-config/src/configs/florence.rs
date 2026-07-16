use serde::{Deserialize, Serialize};

fn default_vision_encoder() -> String { "DaViT-Large".into() }
fn default_language_model() -> String { "florence-language-model".into() }
fn default_projection_dim() -> usize { 1024 }
fn default_image_size() -> usize { 768 }
fn default_patch_size() -> usize { 32 }
fn default_max_seq_len() -> usize { 1024 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlorenceConfig {
    #[serde(default = "default_vision_encoder")]
    pub vision_encoder: String,
    #[serde(default = "default_language_model")]
    pub language_model: String,
    #[serde(default = "default_projection_dim")]
    pub projection_dim: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
}

impl Default for FlorenceConfig {
    fn default() -> Self {
        Self {
            vision_encoder: default_vision_encoder(),
            language_model: default_language_model(),
            projection_dim: default_projection_dim(),
            image_size: default_image_size(),
            patch_size: default_patch_size(),
            max_seq_len: default_max_seq_len(),
        }
    }
}
