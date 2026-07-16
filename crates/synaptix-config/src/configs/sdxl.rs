use serde::{Deserialize, Serialize};

fn default_cross_attention_dim() -> usize { 2048 }
fn default_attention_head_dim() -> Vec<usize> { vec![5, 10, 20] }
fn default_transformer_layers_per_block() -> Vec<usize> { vec![1, 2, 10] }
fn default_block_out_channels() -> Vec<usize> { vec![320, 640, 1280] }
fn default_layers_per_block() -> usize { 2 }
fn default_addition_embed_type() -> String { "text_time".into() }
fn default_addition_time_embed_dim() -> usize { 256 }
fn default_projection_class_embeddings_input_dim() -> usize { 2816 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdxlConfig {
    #[serde(default = "default_cross_attention_dim")]
    pub cross_attention_dim: usize,
    #[serde(default = "default_attention_head_dim")]
    pub attention_head_dim: Vec<usize>,
    #[serde(default = "default_transformer_layers_per_block")]
    pub transformer_layers_per_block: Vec<usize>,
    #[serde(default = "default_block_out_channels")]
    pub block_out_channels: Vec<usize>,
    #[serde(default = "default_layers_per_block")]
    pub layers_per_block: usize,
    #[serde(default = "default_addition_embed_type")]
    pub addition_embed_type: String,
    #[serde(default = "default_addition_time_embed_dim")]
    pub addition_time_embed_dim: usize,
    #[serde(default = "default_projection_class_embeddings_input_dim")]
    pub projection_class_embeddings_input_dim: usize,
}

impl Default for SdxlConfig {
    fn default() -> Self {
        Self {
            cross_attention_dim: default_cross_attention_dim(),
            attention_head_dim: default_attention_head_dim(),
            transformer_layers_per_block: default_transformer_layers_per_block(),
            block_out_channels: default_block_out_channels(),
            layers_per_block: default_layers_per_block(),
            addition_embed_type: default_addition_embed_type(),
            addition_time_embed_dim: default_addition_time_embed_dim(),
            projection_class_embeddings_input_dim: default_projection_class_embeddings_input_dim(),
        }
    }
}
