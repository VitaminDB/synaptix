
use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_llm_common::{
    Activation, DecoderConfig, DecoderModel, KvCache, LayerKind, ModelError, NormGain, RopeSpec,
    WeightSource,
};

use crate::config::LmConfig;
use crate::loader::CompLoader;
use crate::AceError;

fn strip(key: &str) -> &str {
    key.strip_prefix("model.").unwrap_or(key)
}

pub struct BundleWeightSource {
    loader: CompLoader,
}

impl BundleWeightSource {
    pub fn new(loader: CompLoader) -> Self {
        Self { loader }
    }
}

impl WeightSource for BundleWeightSource {
    fn tensor(&self, key: &str, _device: Device, dtype: DType) -> Result<Tensor, ModelError> {
        self.loader
            .get(strip(key), dtype)
            .map_err(|e| ModelError::Load(e.to_string()))
    }
    fn contains(&self, key: &str) -> bool {
        self.loader.has(strip(key))
    }
}

pub fn to_decoder_config(cfg: &LmConfig) -> DecoderConfig {
    DecoderConfig {
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        max_position_embeddings: cfg.max_position_embeddings,
        rms_norm_eps: cfg.rms_norm_eps,
        norm_gain: NormGain::Plain,
        activation: Activation::Silu,
        sandwich_norms: false,
        qk_norm: true,
        attn_output_gate: false,
        attn_scale: 1.0 / (cfg.head_dim as f32).sqrt(),
        embed_scale: None,
        rope_global: RopeSpec {
            theta: cfg.rope_theta,
            rotary_dim: cfg.head_dim,
            scaled_freqs: None,
        },
        rope_local: None,
        sliding_window: None,
        sliding_window_pattern: 0,
        layer_kinds: vec![LayerKind::Full; cfg.num_hidden_layers],
        linear: None,
        tie_word_embeddings: true,
        bos_token_id: Some(cfg.bos_token_id),
        eos_token_ids: vec![cfg.eos_token_id],
    }
}

pub struct AceStepLm {
    pub model: DecoderModel,
    pub config: LmConfig,
    pub device: Device,
    pub compute: DType,
}

impl AceStepLm {
    pub fn open(
        path: impl AsRef<Path>,
        device: Device,
        compute: DType,
        quant_w: DType,
        rope_capacity: usize,
    ) -> Result<Self, AceError> {
        let loader = CompLoader::open(path, None, device)?;
        let src = BundleWeightSource { loader };
        let config = LmConfig::lm_1_7b();
        let dcfg = to_decoder_config(&config);
        let model = DecoderModel::build(
            &dcfg, &src, device, compute, quant_w, quant_w, compute, compute, rope_capacity,
        )
        .map_err(|e| AceError::Load(e.to_string()))?;
        Ok(Self { model, config, device, compute })
    }

    pub fn make_kv(&self, batch: usize, max_seq: usize) -> Result<KvCache, AceError> {
        self.model
            .make_kv_cache(batch, max_seq)
            .map_err(|e| AceError::Other(e.to_string()))
    }

    pub fn forward(&self, ids: &Tensor, kv: &mut KvCache) -> Result<Tensor, AceError> {
        self.model.forward(ids, kv).map_err(|e| AceError::Other(e.to_string()))
    }
}
