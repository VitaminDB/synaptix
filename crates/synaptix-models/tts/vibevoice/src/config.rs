use serde::Deserialize;

use crate::{Result, VibeVoiceError};

#[derive(Debug, Clone, Deserialize)]
pub struct AcousticTokenizerConfig {
    #[serde(default = "one")]
    pub channels: usize,
    #[serde(default = "yes")]
    pub causal: bool,
    #[serde(default = "vae64")]
    pub vae_dim: usize,
    #[serde(default = "half")]
    pub fix_std: f32,
    #[serde(default = "gaussian")]
    pub std_dist_type: String,
    #[serde(default = "depthwise")]
    pub mixer_layer: String,
    #[serde(default = "none_str")]
    pub conv_norm: String,
    #[serde(default = "constant")]
    pub pad_mode: String,
    #[serde(default = "yes")]
    pub disable_last_norm: bool,
    #[serde(default = "rmsnorm")]
    pub layernorm: String,
    #[serde(default = "eps5")]
    pub layernorm_eps: f32,
    #[serde(default = "yes")]
    pub layernorm_elementwise_affine: bool,
    #[serde(default = "yes")]
    pub conv_bias: bool,
    #[serde(default = "scale6")]
    pub layer_scale_init_value: f32,
    #[serde(default = "filters32")]
    pub encoder_n_filters: usize,
    #[serde(default = "default_ratios")]
    pub encoder_ratios: Vec<usize>,
    #[serde(default = "default_depths")]
    pub encoder_depths: String,
    #[serde(default = "filters32")]
    pub decoder_n_filters: usize,
    #[serde(default)]
    pub decoder_ratios: Option<Vec<usize>>,
    #[serde(default)]
    pub decoder_depths: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SemanticTokenizerConfig {
    #[serde(default = "one")]
    pub channels: usize,
    #[serde(default = "yes")]
    pub causal: bool,
    #[serde(default = "vae128")]
    pub vae_dim: usize,
    #[serde(default = "depthwise")]
    pub mixer_layer: String,
    #[serde(default = "none_str")]
    pub conv_norm: String,
    #[serde(default = "constant")]
    pub pad_mode: String,
    #[serde(default = "yes")]
    pub disable_last_norm: bool,
    #[serde(default = "rmsnorm")]
    pub layernorm: String,
    #[serde(default = "eps5")]
    pub layernorm_eps: f32,
    #[serde(default = "yes")]
    pub layernorm_elementwise_affine: bool,
    #[serde(default = "yes")]
    pub conv_bias: bool,
    #[serde(default = "scale6")]
    pub layer_scale_init_value: f32,
    #[serde(default = "filters32")]
    pub encoder_n_filters: usize,
    #[serde(default = "default_ratios")]
    pub encoder_ratios: Vec<usize>,
    #[serde(default = "default_depths")]
    pub encoder_depths: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "vocab_default")]
    pub vocab_size: usize,
    #[serde(default = "eps6")]
    pub rms_norm_eps: f32,
    #[serde(default = "theta_default")]
    pub rope_theta: f64,
    #[serde(default = "max_pos_default")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl DecoderConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionHeadConfig {
    pub hidden_size: usize,
    #[serde(default = "four")]
    pub head_layers: usize,
    #[serde(default = "ffn3")]
    pub head_ffn_ratio: f32,
    #[serde(default = "eps5")]
    pub rms_norm_eps: f32,
    #[serde(default = "vae64")]
    pub latent_size: usize,
    #[serde(default = "v_pred")]
    pub prediction_type: String,
    #[serde(default = "ddpm_steps")]
    pub ddpm_num_steps: usize,
    #[serde(default = "ddpm_infer")]
    pub ddpm_num_inference_steps: usize,
    #[serde(default = "cosine")]
    pub ddpm_beta_schedule: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VibeVoiceConfig {
    pub acoustic_tokenizer_config: AcousticTokenizerConfig,
    pub semantic_tokenizer_config: SemanticTokenizerConfig,
    pub decoder_config: DecoderConfig,
    pub diffusion_head_config: DiffusionHeadConfig,
}

impl VibeVoiceConfig {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| VibeVoiceError::Config(format!("config.json: {e}")))
    }

    pub fn acoustic_vae_dim(&self) -> usize {
        self.acoustic_tokenizer_config.vae_dim
    }

    pub fn semantic_vae_dim(&self) -> usize {
        self.semantic_tokenizer_config.vae_dim
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioProcessorConfig {
    #[serde(default = "sr24k")]
    pub sampling_rate: u32,
    #[serde(default = "yes")]
    pub normalize_audio: bool,
    #[serde(default = "db25", rename = "target_dB_FS")]
    pub target_db_fs: f32,
    #[serde(default = "eps_audio")]
    pub eps: f32,
}

impl Default for AudioProcessorConfig {
    fn default() -> Self {
        Self {
            sampling_rate: 24_000,
            normalize_audio: true,
            target_db_fs: -25.0,
            eps: 1e-6,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessorConfig {
    #[serde(default = "compress3200")]
    pub speech_tok_compress_ratio: usize,
    #[serde(default = "yes")]
    pub db_normalize: bool,
    #[serde(default)]
    pub audio_processor: AudioProcessorConfig,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            speech_tok_compress_ratio: 3200,
            db_normalize: true,
            audio_processor: AudioProcessorConfig::default(),
        }
    }
}

impl PreprocessorConfig {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| VibeVoiceError::Config(format!("preprocessor_config.json: {e}")))
    }
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub cfg_scale: f32,
    pub ddpm_inference_steps: usize,
    pub max_length_times: f32,
    pub max_new_tokens: Option<usize>,
    pub seed: u64,
    pub zero_noise: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            cfg_scale: 1.3,
            ddpm_inference_steps: 20,
            max_length_times: 2.0,
            max_new_tokens: None,
            seed: 0,
            zero_noise: false,
        }
    }
}

pub fn parse_depths(spec: &str) -> Result<Vec<usize>> {
    spec.split('-')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| VibeVoiceError::Config(format!("depths '{spec}': {e}")))
        })
        .collect()
}

fn one() -> usize {
    1
}
fn four() -> usize {
    4
}
fn yes() -> bool {
    true
}
fn vae64() -> usize {
    64
}
fn vae128() -> usize {
    128
}
fn half() -> f32 {
    0.5
}
fn gaussian() -> String {
    "gaussian".into()
}
fn depthwise() -> String {
    "depthwise_conv".into()
}
fn none_str() -> String {
    "none".into()
}
fn constant() -> String {
    "constant".into()
}
fn rmsnorm() -> String {
    "RMSNorm".into()
}
fn eps5() -> f32 {
    1e-5
}
fn eps6() -> f32 {
    1e-6
}
fn scale6() -> f32 {
    1e-6
}
fn filters32() -> usize {
    32
}
fn default_ratios() -> Vec<usize> {
    vec![8, 5, 5, 4, 2, 2]
}
fn default_depths() -> String {
    "3-3-3-3-3-3-8".into()
}
fn vocab_default() -> usize {
    151_936
}
fn theta_default() -> f64 {
    1_000_000.0
}
fn max_pos_default() -> usize {
    32_768
}
fn ffn3() -> f32 {
    3.0
}
fn v_pred() -> String {
    "v_prediction".into()
}
fn ddpm_steps() -> usize {
    1000
}
fn ddpm_infer() -> usize {
    20
}
fn cosine() -> String {
    "cosine".into()
}
fn sr24k() -> u32 {
    24_000
}
fn db25() -> f32 {
    -25.0
}
fn eps_audio() -> f32 {
    1e-6
}
fn compress3200() -> usize {
    3200
}
