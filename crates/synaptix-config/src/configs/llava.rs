use serde::{Deserialize, Serialize};

fn default_mm_projector_type() -> String { "mlp2x_gelu".into() }
fn default_image_grid_pinpoints() -> Vec<Vec<usize>> { vec![] }
fn default_mm_vision_tower() -> String { "openai/clip-vit-large-patch14-336".into() }
fn default_mm_hidden_size() -> usize { 1024 }
fn default_image_aspect_ratio() -> String { "pad".into() }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlavaConfig {
    #[serde(default = "default_mm_projector_type")]
    pub mm_projector_type: String,
    #[serde(default = "default_image_grid_pinpoints")]
    pub image_grid_pinpoints: Vec<Vec<usize>>,
    #[serde(default = "default_mm_vision_tower")]
    pub mm_vision_tower: String,
    #[serde(default = "default_mm_hidden_size")]
    pub mm_hidden_size: usize,
    #[serde(default = "default_image_aspect_ratio")]
    pub image_aspect_ratio: String,
}

impl Default for LlavaConfig {
    fn default() -> Self {
        Self {
            mm_projector_type: default_mm_projector_type(),
            image_grid_pinpoints: default_image_grid_pinpoints(),
            mm_vision_tower: default_mm_vision_tower(),
            mm_hidden_size: default_mm_hidden_size(),
            image_aspect_ratio: default_image_aspect_ratio(),
        }
    }
}
