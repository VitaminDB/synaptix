use serde::{Deserialize, Serialize};

fn default_vocab_size() -> usize { 151936 }
fn default_hidden_size() -> usize { 8192 }
fn default_intermediate_size() -> usize { 29568 }
fn default_num_hidden_layers() -> usize { 80 }
fn default_num_attention_heads() -> usize { 64 }
fn default_num_key_value_heads() -> usize { 8 }
fn default_max_position_embeddings() -> usize { 128000 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_rope_theta() -> f64 { 1000000.0 }
fn default_vision_start_token_id() -> u32 { 151652 }
fn default_vision_end_token_id() -> u32 { 151653 }
fn default_image_token_id() -> u32 { 151655 }
fn default_video_token_id() -> u32 { 151656 }

fn default_vision_depth() -> usize { 32 }
fn default_vision_embed_dim() -> usize { 1280 }
fn default_vision_num_heads() -> usize { 16 }
fn default_vision_in_channels() -> usize { 3 }
fn default_vision_patch_size() -> usize { 14 }
fn default_vision_temporal_patch_size() -> usize { 2 }
fn default_vision_spatial_merge_size() -> usize { 2 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_vision_depth")]
    pub depth: usize,
    #[serde(default = "default_vision_embed_dim")]
    pub embed_dim: usize,
    #[serde(default = "default_vision_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_vision_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_vision_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_vision_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_vision_spatial_merge_size")]
    pub spatial_merge_size: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            depth: default_vision_depth(),
            embed_dim: default_vision_embed_dim(),
            num_heads: default_vision_num_heads(),
            in_channels: default_vision_in_channels(),
            patch_size: default_vision_patch_size(),
            temporal_patch_size: default_vision_temporal_patch_size(),
            spatial_merge_size: default_vision_spatial_merge_size(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen2VlConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_vision_start_token_id")]
    pub vision_start_token_id: u32,
    #[serde(default = "default_vision_end_token_id")]
    pub vision_end_token_id: u32,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
    #[serde(default = "default_video_token_id")]
    pub video_token_id: u32,
    #[serde(default)]
    pub vision_config: VisionConfig,
}

impl Default for Qwen2VlConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: default_num_key_value_heads(),
            max_position_embeddings: default_max_position_embeddings(),
            rms_norm_eps: default_rms_norm_eps(),
            rope_theta: default_rope_theta(),
            vision_start_token_id: default_vision_start_token_id(),
            vision_end_token_id: default_vision_end_token_id(),
            image_token_id: default_image_token_id(),
            video_token_id: default_video_token_id(),
            vision_config: VisionConfig::default(),
        }
    }
}
