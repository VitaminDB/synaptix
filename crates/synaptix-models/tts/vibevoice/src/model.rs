use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::config::VibeVoiceConfig;
use crate::head::DiffusionHead;
use crate::loader::{VibeVoiceCheckpoint, WeightSource};
use crate::qwen2::Qwen2Model;
use crate::schedule::DpmSolverMultistep;
use crate::vae::{AcousticTokenizer, SemanticTokenizer, SpeechConnector};
use crate::{err, Result};

pub struct VibeVoiceModel {
    pub lm: Qwen2Model,
    pub acoustic: AcousticTokenizer,
    pub semantic: SemanticTokenizer,
    pub acoustic_connector: SpeechConnector,
    pub semantic_connector: SpeechConnector,
    pub head: DiffusionHead,
    pub speech_scaling_factor: f32,
    pub speech_bias_factor: f32,
    pub config: VibeVoiceConfig,
    pub device: Device,
    pub dtype: DType,
}

impl VibeVoiceModel {
    pub fn load(ckpt: &VibeVoiceCheckpoint, rope_capacity: usize) -> Result<Self> {
        let cfg = ckpt.config.clone();
        let src = ckpt.source();
        let lm = Qwen2Model::load(
            src,
            &cfg.decoder_config,
            "model.language_model",
            "lm_head.weight",
            rope_capacity,
        )?;
        let acoustic = AcousticTokenizer::load(
            src,
            &cfg.acoustic_tokenizer_config,
            "model.acoustic_tokenizer",
        )?;
        let semantic = SemanticTokenizer::load(
            src,
            &cfg.semantic_tokenizer_config,
            "model.semantic_tokenizer",
        )?;
        let acoustic_connector = SpeechConnector::load(src, "model.acoustic_connector")?;
        let semantic_connector = SpeechConnector::load(src, "model.semantic_connector")?;
        let head = DiffusionHead::load(src, &cfg.diffusion_head_config, "model.prediction_head")?;
        let scaling = read_scalar(src, "model.speech_scaling_factor")?;
        let bias = read_scalar(src, "model.speech_bias_factor")?;
        Ok(Self {
            lm,
            acoustic,
            semantic,
            acoustic_connector,
            semantic_connector,
            head,
            speech_scaling_factor: scaling,
            speech_bias_factor: bias,
            config: cfg,
            device: ckpt.device,
            dtype: ckpt.dtype,
        })
    }

    pub fn new_scheduler(&self) -> Result<DpmSolverMultistep> {
        DpmSolverMultistep::new(
            self.config.diffusion_head_config.ddpm_num_steps,
            &self.config.diffusion_head_config.ddpm_beta_schedule,
        )
    }

    pub fn scale_latents(&self, latents: &Tensor) -> Result<Tensor> {
        latents
            .affine(1.0, self.speech_bias_factor)
            .and_then(|t| t.affine(self.speech_scaling_factor, 0.0))
            .map_err(err)
    }

    pub fn unscale_latents(&self, latents: &Tensor) -> Result<Tensor> {
        latents
            .affine(1.0 / self.speech_scaling_factor, 0.0)
            .and_then(|t| t.affine(1.0, -self.speech_bias_factor))
            .map_err(err)
    }
}

fn read_scalar(src: &dyn WeightSource, name: &str) -> Result<f32> {
    let t = src.get(name)?;
    let v = t
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(err)?;
    Ok(v[0])
}
