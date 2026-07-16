use serde::Deserialize;

use crate::VoxError;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(rename = "type")]
    pub kind: String,
    pub long_factor: Vec<f32>,
    pub short_factor: Vec<f32>,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LmConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub kv_channels: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub use_mup: bool,
    #[serde(default = "default_scale_emb")]
    pub scale_emb: f32,
    #[serde(default = "default_dim_model_base")]
    pub dim_model_base: usize,
    #[serde(default = "default_scale_depth")]
    pub scale_depth: f32,
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default)]
    pub eos_token_id: u32,
}

fn default_scale_emb() -> f32 { 12.0 }
fn default_dim_model_base() -> usize { 256 }
fn default_scale_depth() -> f32 { 1.4 }

#[derive(Debug, Clone, Deserialize)]
pub struct SubTransformerConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub kv_channels: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CfmConfig {
    #[serde(default = "default_sigma_min")]
    pub sigma_min: f32,
    #[serde(default = "default_solver")]
    pub solver: String,
    #[serde(default = "default_t_scheduler")]
    pub t_scheduler: String,
    #[serde(default = "default_cfg_rate")]
    pub inference_cfg_rate: f32,
}

fn default_sigma_min() -> f32 { 1e-6 }
fn default_solver() -> String { "euler".to_string() }
fn default_t_scheduler() -> String { "log-norm".to_string() }
fn default_cfg_rate() -> f32 { 2.0 }

#[derive(Debug, Clone, Deserialize)]
pub struct DitConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub kv_channels: usize,
    #[serde(default)]
    pub mean_mode: bool,
    pub cfm_config: CfmConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioVaeConfig {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub sr_bin_boundaries: Vec<i64>,
    pub sample_rate: usize,
    pub out_sample_rate: usize,
}

impl AudioVaeConfig {
    pub fn hop_length(&self) -> usize {
        self.encoder_rates.iter().product()
    }
    pub fn decode_chunk_size(&self) -> usize {
        self.decoder_rates.iter().product()
    }
    pub fn sr_bucket(&self, sr: usize) -> usize {
        self.sr_bin_boundaries
            .iter()
            .filter(|&&b| (sr as i64) >= b)
            .count()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxConfig {
    pub architecture: String,
    pub lm_config: LmConfig,
    pub patch_size: usize,
    pub feat_dim: usize,
    pub scalar_quantization_latent_dim: usize,
    pub scalar_quantization_scale: f32,
    pub residual_lm_num_layers: usize,
    pub residual_lm_no_rope: bool,
    pub encoder_config: SubTransformerConfig,
    pub dit_config: DitConfig,
    pub audio_vae_config: AudioVaeConfig,
    pub max_length: usize,
}

impl VoxConfig {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, VoxError> {
        let cfg: VoxConfig = serde_json::from_slice(bytes)
            .map_err(|e| VoxError::Config(format!("parse config.json: {e}")))?;
        if cfg.architecture != "voxcpm2" {
            return Err(VoxError::Config(format!(
                "expected architecture=voxcpm2, got '{}'",
                cfg.architecture
            )));
        }
        Ok(cfg)
    }
}
