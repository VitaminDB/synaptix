use serde::{Deserialize, Serialize};

fn default_vocab_size() -> usize { 32000 }
fn default_d_model() -> usize { 4096 }
fn default_num_heads() -> usize { 32 }
fn default_num_layers() -> usize { 32 }
fn default_ffn_dim() -> usize { 16384 }
fn default_num_codebooks() -> usize { 8 }
fn default_frame_rate() -> usize { 12 }
fn default_audio_channels() -> usize { 1 }
fn default_streaming_chunk_size() -> usize { 64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoshiConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    #[serde(default = "default_ffn_dim")]
    pub ffn_dim: usize,
    #[serde(default = "default_num_codebooks")]
    pub num_codebooks: usize,
    #[serde(default = "default_frame_rate")]
    pub frame_rate: usize,
    #[serde(default = "default_audio_channels")]
    pub audio_channels: usize,
    #[serde(default = "default_streaming_chunk_size")]
    pub streaming_chunk_size: usize,
}

impl Default for MoshiConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            d_model: default_d_model(),
            num_heads: default_num_heads(),
            num_layers: default_num_layers(),
            ffn_dim: default_ffn_dim(),
            num_codebooks: default_num_codebooks(),
            frame_rate: default_frame_rate(),
            audio_channels: default_audio_channels(),
            streaming_chunk_size: default_streaming_chunk_size(),
        }
    }
}
