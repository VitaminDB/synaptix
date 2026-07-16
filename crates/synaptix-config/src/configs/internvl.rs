use serde::{Deserialize, Serialize};

fn default_vision_model() -> String { "InternViT-6B".into() }
fn default_llm_model() -> String { "InternLM2-20B".into() }
fn default_select_layer() -> i64 { -1 }
fn default_downsample_ratio() -> f64 { 0.5 }
fn default_ps_version() -> String { "v2".into() }
fn default_max_dynamic_patch() -> usize { 12 }
fn default_use_thumbnail() -> bool { true }
fn default_img_size() -> usize { 448 }
fn default_patch_size() -> usize { 14 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternVlConfig {
    #[serde(default = "default_vision_model")]
    pub vision_model: String,
    #[serde(default = "default_llm_model")]
    pub llm_model: String,
    #[serde(default = "default_select_layer")]
    pub select_layer: i64,
    #[serde(default = "default_downsample_ratio")]
    pub downsample_ratio: f64,
    #[serde(default = "default_ps_version")]
    pub ps_version: String,
    #[serde(default = "default_max_dynamic_patch")]
    pub max_dynamic_patch: usize,
    #[serde(default = "default_use_thumbnail")]
    pub use_thumbnail: bool,
    #[serde(default = "default_img_size")]
    pub img_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
}

impl Default for InternVlConfig {
    fn default() -> Self {
        Self {
            vision_model: default_vision_model(),
            llm_model: default_llm_model(),
            select_layer: default_select_layer(),
            downsample_ratio: default_downsample_ratio(),
            ps_version: default_ps_version(),
            max_dynamic_patch: default_max_dynamic_patch(),
            use_thumbnail: default_use_thumbnail(),
            img_size: default_img_size(),
            patch_size: default_patch_size(),
        }
    }
}
