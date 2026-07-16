//! Конфиг GigaAM-v3-e2e-CTC из `config.json` (упакованный HF-снапшот).

use serde::Deserialize;

use crate::GigaAmError;

#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessorConfig {
    pub sample_rate: u32,
    pub features: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub mel_scale: String,
    #[serde(default)]
    pub mel_norm: Option<String>,
    pub center: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub feat_in: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub subsampling: String,
    pub subs_kernel_size: usize,
    pub subsampling_factor: usize,
    pub ff_expansion_factor: usize,
    pub self_attention_model: String,
    pub pos_emb_max_len: usize,
    pub n_heads: usize,
    pub conv_kernel_size: usize,
    pub conv_norm_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HeadConfig {
    pub feat_in: usize,
    pub num_classes: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GigaAmConfig {
    pub model_name: String,
    pub model_class: String,
    pub sample_rate: u32,
    pub preprocessor: PreprocessorConfig,
    pub encoder: EncoderConfig,
    pub head: HeadConfig,
}

impl GigaAmConfig {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, GigaAmError> {
        let cfg: GigaAmConfig =
            serde_json::from_slice(bytes).map_err(|e| GigaAmError::Config(e.to_string()))?;
        if cfg.encoder.self_attention_model != "rotary" {
            return Err(GigaAmError::Config(format!(
                "unsupported self_attention_model={:?} (expected \"rotary\")",
                cfg.encoder.self_attention_model
            )));
        }
        if cfg.encoder.subsampling != "conv1d" {
            return Err(GigaAmError::Config(format!(
                "unsupported subsampling={:?} (expected \"conv1d\")",
                cfg.encoder.subsampling
            )));
        }
        if cfg.encoder.conv_norm_type != "layer_norm" {
            return Err(GigaAmError::Config(format!(
                "unsupported conv_norm_type={:?} (expected \"layer_norm\")",
                cfg.encoder.conv_norm_type
            )));
        }
        Ok(cfg)
    }

    pub fn head_dim(&self) -> usize {
        self.encoder.d_model / self.encoder.n_heads
    }

    /// blank-id для CTC = размер словаря токенизатора (= num_classes - 1).
    pub fn blank_id(&self) -> usize {
        self.head.num_classes - 1
    }
}
