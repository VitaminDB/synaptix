use serde::{Deserialize, Serialize};

fn default_in_channels() -> usize { 64 }
fn default_out_channels() -> usize { 64 }
fn default_num_layers() -> usize { 24 }
fn default_d_model() -> usize { 1536 }
fn default_num_heads() -> usize { 24 }
fn default_audio_channels() -> usize { 2 }
fn default_sample_rate() -> usize { 44100 }
fn default_latent_dim() -> usize { 64 }
fn default_downsampling_ratio() -> usize { 512 }
fn default_min_input_length() -> usize { 65536 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StableAudioConfig {
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_audio_channels")]
    pub audio_channels: usize,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: usize,
    #[serde(default = "default_latent_dim")]
    pub latent_dim: usize,
    #[serde(default = "default_downsampling_ratio")]
    pub downsampling_ratio: usize,
    #[serde(default = "default_min_input_length")]
    pub min_input_length: usize,
}

impl Default for StableAudioConfig {
    fn default() -> Self {
        Self {
            in_channels: default_in_channels(),
            out_channels: default_out_channels(),
            num_layers: default_num_layers(),
            d_model: default_d_model(),
            num_heads: default_num_heads(),
            audio_channels: default_audio_channels(),
            sample_rate: default_sample_rate(),
            latent_dim: default_latent_dim(),
            downsampling_ratio: default_downsampling_ratio(),
            min_input_length: default_min_input_length(),
        }
    }
}
