use serde::{Deserialize, Serialize};

fn default_vocab_size() -> usize { 1025 }
fn default_d_model() -> usize { 1024 }
fn default_n_heads() -> usize { 8 }
fn default_n_encoder_layers() -> usize { 24 }
fn default_n_decoder_layers() -> usize { 24 }
fn default_ffn_dim() -> usize { 4096 }
fn default_conv_channels() -> usize { 256 }
fn default_conv_kernel_size() -> usize { 31 }
fn default_subsampling_factor() -> usize { 4 }
fn default_max_audio_len() -> usize { 3000 }
fn default_max_text_len() -> usize { 512 }
fn default_dropout() -> f64 { 0.1 }
fn default_rnnt_joint_dim() -> usize { 1024 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParakeetConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_n_heads")]
    pub n_heads: usize,
    #[serde(default = "default_n_encoder_layers")]
    pub n_encoder_layers: usize,
    #[serde(default = "default_n_decoder_layers")]
    pub n_decoder_layers: usize,
    #[serde(default = "default_ffn_dim")]
    pub ffn_dim: usize,
    #[serde(default = "default_conv_channels")]
    pub conv_channels: usize,
    #[serde(default = "default_conv_kernel_size")]
    pub conv_kernel_size: usize,
    #[serde(default = "default_subsampling_factor")]
    pub subsampling_factor: usize,
    #[serde(default = "default_max_audio_len")]
    pub max_audio_len: usize,
    #[serde(default = "default_max_text_len")]
    pub max_text_len: usize,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default = "default_rnnt_joint_dim")]
    pub rnnt_joint_dim: usize,
}

impl Default for ParakeetConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            d_model: default_d_model(),
            n_heads: default_n_heads(),
            n_encoder_layers: default_n_encoder_layers(),
            n_decoder_layers: default_n_decoder_layers(),
            ffn_dim: default_ffn_dim(),
            conv_channels: default_conv_channels(),
            conv_kernel_size: default_conv_kernel_size(),
            subsampling_factor: default_subsampling_factor(),
            max_audio_len: default_max_audio_len(),
            max_text_len: default_max_text_len(),
            dropout: default_dropout(),
            rnnt_joint_dim: default_rnnt_joint_dim(),
        }
    }
}
