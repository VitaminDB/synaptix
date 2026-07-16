use serde::{Deserialize, Serialize};

fn default_d_model() -> usize { 4096 }
fn default_num_heads() -> usize { 32 }
fn default_num_layers() -> usize { 40 }
fn default_vocab_size() -> usize { 122753 }
fn default_max_audio_len() -> usize { 3000 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoxCpmConfig {
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_max_audio_len")]
    pub max_audio_len: usize,
}

impl Default for VoxCpmConfig {
    fn default() -> Self {
        Self {
            d_model: default_d_model(),
            num_heads: default_num_heads(),
            num_layers: default_num_layers(),
            vocab_size: default_vocab_size(),
            max_audio_len: default_max_audio_len(),
        }
    }
}
