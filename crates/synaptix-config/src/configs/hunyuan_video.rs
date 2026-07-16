use serde::{Deserialize, Serialize};

fn default_in_channels() -> usize { 16 }
fn default_out_channels() -> usize { 16 }
fn default_patch_size() -> Vec<usize> { vec![1, 2, 2] }
fn default_hidden_size() -> usize { 3072 }
fn default_heads_num() -> usize { 24 }
fn default_mm_double_blocks_depth() -> usize { 20 }
fn default_mm_single_blocks_depth() -> usize { 40 }
fn default_rope_dim_list() -> Vec<usize> { vec![16, 56, 56] }
fn default_guidance_embed() -> bool { true }
fn default_text_states_dim() -> usize { 4096 }
fn default_text_states_dim_2() -> usize { 768 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HunyuanVideoConfig {
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: Vec<usize>,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_heads_num")]
    pub heads_num: usize,
    #[serde(default = "default_mm_double_blocks_depth")]
    pub mm_double_blocks_depth: usize,
    #[serde(default = "default_mm_single_blocks_depth")]
    pub mm_single_blocks_depth: usize,
    #[serde(default = "default_rope_dim_list")]
    pub rope_dim_list: Vec<usize>,
    #[serde(default = "default_guidance_embed")]
    pub guidance_embed: bool,
    #[serde(default = "default_text_states_dim")]
    pub text_states_dim: usize,
    #[serde(default = "default_text_states_dim_2")]
    pub text_states_dim_2: usize,
}

impl Default for HunyuanVideoConfig {
    fn default() -> Self {
        Self {
            in_channels: default_in_channels(),
            out_channels: default_out_channels(),
            patch_size: default_patch_size(),
            hidden_size: default_hidden_size(),
            heads_num: default_heads_num(),
            mm_double_blocks_depth: default_mm_double_blocks_depth(),
            mm_single_blocks_depth: default_mm_single_blocks_depth(),
            rope_dim_list: default_rope_dim_list(),
            guidance_embed: default_guidance_embed(),
            text_states_dim: default_text_states_dim(),
            text_states_dim_2: default_text_states_dim_2(),
        }
    }
}
