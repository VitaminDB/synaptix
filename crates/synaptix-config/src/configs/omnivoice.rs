use serde::{Deserialize, Serialize};

fn default_vocab_size() -> usize { 32000 }
fn default_d_model() -> usize { 4096 }
fn default_num_heads() -> usize { 32 }
fn default_num_layers() -> usize { 32 }
fn default_codec_vocab_size() -> usize { 4096 }
fn default_num_codec_tokens_per_frame() -> usize { 8 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniVoiceConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    #[serde(default = "default_codec_vocab_size")]
    pub codec_vocab_size: usize,
    #[serde(default = "default_num_codec_tokens_per_frame")]
    pub num_codec_tokens_per_frame: usize,
}

impl Default for OmniVoiceConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            d_model: default_d_model(),
            num_heads: default_num_heads(),
            num_layers: default_num_layers(),
            codec_vocab_size: default_codec_vocab_size(),
            num_codec_tokens_per_frame: default_num_codec_tokens_per_frame(),
        }
    }
}
