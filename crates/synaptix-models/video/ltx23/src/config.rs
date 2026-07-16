//! Конфиг LTX-2.3, разбираемый из `__metadata__["config"]` чекпойнта
//! (`AVTransformer3DModel` + `CausalVideoAutoencoder` + RectifiedFlow + audio).
//!
//! Источник истины — JSON внутри весов, а не захардкоженные числа: [`Ltx23Config`]
//! десериализуется напрямую из метаданных safetensors. Секции audio_vae/vocoder
//! пока хранятся сырым [`serde_json::Value`] (типизируются на аудио-фазах).

use std::collections::HashMap;

use serde::Deserialize;

use crate::LtxError;

/// Полный конфиг чекпойнта LTX-2.3.
#[derive(Debug, Clone)]
pub struct Ltx23Config {
    pub model_version: String,
    pub transformer: TransformerConfig,
    pub vae: VaeConfig,
    pub scheduler: SchedulerConfig,
    /// Весь JSON `config` (для секций audio_vae/vocoder до их типизации).
    pub raw: serde_json::Value,
}

impl Ltx23Config {
    /// Разобрать из `SafetensorsLoader::metadata()` (ключи `config` + `model_version`).
    pub fn from_metadata(meta: &HashMap<String, String>) -> Result<Self, LtxError> {
        let cfg_str = meta
            .get("config")
            .ok_or_else(|| LtxError::Config("checkpoint __metadata__ has no 'config'".into()))?;
        let raw: serde_json::Value = serde_json::from_str(cfg_str)
            .map_err(|e| LtxError::Config(format!("config json: {e}")))?;
        let transformer: TransformerConfig = serde_json::from_value(raw["transformer"].clone())
            .map_err(|e| LtxError::Config(format!("transformer config: {e}")))?;
        let vae: VaeConfig = serde_json::from_value(raw["vae"].clone())
            .map_err(|e| LtxError::Config(format!("vae config: {e}")))?;
        let scheduler: SchedulerConfig = serde_json::from_value(raw["scheduler"].clone())
            .map_err(|e| LtxError::Config(format!("scheduler config: {e}")))?;
        Ok(Self {
            model_version: meta.get("model_version").cloned().unwrap_or_default(),
            transformer,
            vae,
            scheduler,
            raw,
        })
    }
}

/// `AVTransformer3DModel` (denoiser, prefix `model.diffusion_model`).
///
/// Незаявленные поля JSON игнорируются (нет `deny_unknown_fields`): объявляем
/// только то, что используем.
#[derive(Debug, Clone, Deserialize)]
pub struct TransformerConfig {
    pub activation_fn: String,
    pub attention_bias: bool,
    pub attention_head_dim: usize,
    pub caption_channels: usize,
    pub cross_attention_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub norm_elementwise_affine: bool,
    pub norm_eps: f64,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub num_embeds_ada_norm: usize,
    pub qk_norm: String,
    pub standardization_norm: String,
    pub positional_embedding_type: String,
    pub positional_embedding_theta: f64,
    pub positional_embedding_max_pos: Vec<usize>,
    pub timestep_scale_multiplier: f64,
    pub av_ca_timestep_scale_multiplier: f64,
    pub causal_temporal_positioning: bool,
    pub rope_type: String,
    pub frequencies_precision: String,
    pub text_encoder_norm_type: String,
    pub apply_gated_attention: bool,
    pub cross_attention_adaln: bool,

    // --- audio stream ---
    pub audio_num_attention_heads: usize,
    pub audio_attention_head_dim: usize,
    pub audio_out_channels: usize,
    pub audio_cross_attention_dim: usize,
    pub audio_positional_embedding_max_pos: Vec<usize>,
    pub use_audio_video_cross_attention: bool,
    pub av_cross_ada_norm: bool,
    pub share_ff: bool,

    // --- text embeddings connector (perceiver-resampler) ---
    pub use_embeddings_connector: bool,
    pub connector_attention_head_dim: usize,
    pub connector_num_attention_heads: usize,
    pub connector_num_layers: usize,
    pub connector_positional_embedding_max_pos: Vec<usize>,
    pub connector_num_learnable_registers: usize,
    pub connector_norm_output: bool,
    pub connector_apply_gated_attention: bool,
    pub connector_learnable_registers_std: f64,
    pub caption_proj_before_connector: bool,
    pub use_middle_indices_grid: bool,
    pub audio_connector_attention_head_dim: usize,
    pub audio_connector_num_attention_heads: usize,
}

impl TransformerConfig {
    /// Скрытая размерность видео-потока (`num_attention_heads * attention_head_dim`).
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
    /// Скрытая размерность аудио-потока.
    pub fn audio_inner_dim(&self) -> usize {
        self.audio_num_attention_heads * self.audio_attention_head_dim
    }
    /// Внутренняя размерность FF видео-потока (mult=4).
    pub fn ff_inner_dim(&self) -> usize {
        4 * self.inner_dim()
    }
    /// Внутренняя размерность FF аудио-потока (mult=4).
    pub fn audio_ff_inner_dim(&self) -> usize {
        4 * self.audio_inner_dim()
    }
    /// Размер входа `text_embedding_projection` (`caption_channels * num_hidden_states`,
    /// где hidden-states = `num_hidden_layers + 1` энкодера Gemma).
    pub fn text_aggregate_in(&self, num_text_hidden_states: usize) -> usize {
        self.caption_channels * num_text_hidden_states
    }
}

/// `CausalVideoAutoencoder` (prefix `vae`).
#[derive(Debug, Clone, Deserialize)]
pub struct VaeConfig {
    pub dims: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub encoder_blocks: Vec<VaeBlock>,
    pub decoder_blocks: Vec<VaeBlock>,
    pub scaling_factor: f64,
    pub norm_layer: String,
    pub patch_size: usize,
    pub latent_log_var: String,
    pub use_quant_conv: bool,
    pub causal_decoder: bool,
    pub timestep_conditioning: bool,
    pub normalize_latent_channels: bool,
    pub encoder_base_channels: usize,
    pub decoder_base_channels: usize,
    pub spatial_padding_mode: String,
}

/// Запись блока VAE: `["res_x", {"num_layers": 4}]` / `["compress_all", {"multiplier": 2}]`.
#[derive(Debug, Clone, Deserialize)]
pub struct VaeBlock(pub String, pub VaeBlockParams);

#[derive(Debug, Clone, Deserialize)]
pub struct VaeBlockParams {
    #[serde(default)]
    pub num_layers: Option<usize>,
    #[serde(default)]
    pub multiplier: Option<f64>,
}

/// `RectifiedFlowScheduler` (секция `scheduler`).
#[derive(Debug, Clone, Deserialize)]
pub struct SchedulerConfig {
    pub num_train_timesteps: usize,
    pub sampler: String,
    #[serde(default)]
    pub shifting: Option<String>,
    #[serde(default)]
    pub base_resolution: Option<f64>,
}
