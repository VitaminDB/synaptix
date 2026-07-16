use serde::{Deserialize, Serialize};

fn default_vocab_size() -> usize { 512 }
fn default_d_model() -> usize { 1024 }
fn default_num_heads() -> usize { 8 }
fn default_num_dit_layers() -> usize { 8 }
fn default_num_ar_layers() -> usize { 32 }
fn default_ar_hidden_size() -> usize { 2048 }
fn default_num_ar_heads() -> usize { 16 }
fn default_latent_channels() -> usize { 64 }
fn default_sample_rate() -> usize { 44100 }
fn default_hop_length() -> usize { 512 }
fn default_num_audio_tokens_per_frame() -> usize { 1 }
fn default_text_encoder() -> String { "qwen".into() }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AceStepConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_num_dit_layers")]
    pub num_dit_layers: usize,
    #[serde(default = "default_num_ar_layers")]
    pub num_ar_layers: usize,
    #[serde(default = "default_ar_hidden_size")]
    pub ar_hidden_size: usize,
    #[serde(default = "default_num_ar_heads")]
    pub num_ar_heads: usize,
    #[serde(default = "default_latent_channels")]
    pub latent_channels: usize,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: usize,
    #[serde(default = "default_hop_length")]
    pub hop_length: usize,
    #[serde(default = "default_num_audio_tokens_per_frame")]
    pub num_audio_tokens_per_frame: usize,
    #[serde(default = "default_text_encoder")]
    pub text_encoder: String,
}

impl Default for AceStepConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            d_model: default_d_model(),
            num_heads: default_num_heads(),
            num_dit_layers: default_num_dit_layers(),
            num_ar_layers: default_num_ar_layers(),
            ar_hidden_size: default_ar_hidden_size(),
            num_ar_heads: default_num_ar_heads(),
            latent_channels: default_latent_channels(),
            sample_rate: default_sample_rate(),
            hop_length: default_hop_length(),
            num_audio_tokens_per_frame: default_num_audio_tokens_per_frame(),
            text_encoder: default_text_encoder(),
        }
    }
}
