
use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_llm_common::{
    Activation, DecoderConfig, DecoderModel, KvCache, LayerKind, ModelError, NormGain, RopeSpec,
    WeightSource,
};

use crate::config::LmConfig;
use crate::loader::{read_bundle_file, CompLoader};
use crate::AceError;

fn strip(key: &str) -> &str {
    key.strip_prefix("model.").unwrap_or(key)
}

/// `WeightSource` поверх .syn-бандла. `DecoderModel` просит ключи в HF-раскладке
/// (`model.embed_tokens.weight`, `model.layers.N.…`); в бандле они лежат либо
/// так же (`acestep_5hz_lm_4b.syn` — упакован из Qwen3ForCausalLM), либо без
/// префикса `model.` (`acestep_5hz_lm_1.7b.syn`, `qwen3-embedding-0.6b.syn` —
/// Qwen3Model). Берём то имя, которое есть; раньше префикс срезался всегда, и
/// 4b-бандл падал на `embed_tokens.weight: tensor not found`.
pub struct BundleWeightSource {
    loader: CompLoader,
}

impl BundleWeightSource {
    pub fn new(loader: CompLoader) -> Self {
        Self { loader }
    }

    fn resolve<'a>(&self, key: &'a str) -> &'a str {
        if self.loader.has(key) {
            key
        } else {
            strip(key)
        }
    }
}

impl WeightSource for BundleWeightSource {
    fn tensor(&self, key: &str, _device: Device, dtype: DType) -> Result<Tensor, ModelError> {
        self.loader
            .get(self.resolve(key), dtype)
            .map_err(|e| ModelError::Load(e.to_string()))
    }
    fn contains(&self, key: &str) -> bool {
        self.loader.has(key) || self.loader.has(strip(key))
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
        post_norm_eps: None,
        qk_norm: true,
        attn_output_gate: false,
        attn_scale: 1.0 / (cfg.head_dim as f32).sqrt(),
        embed_scale: None,
        embed_rms_norm: false,
        logit_scale: None,
        logit_softcap: None,
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
        let path = path.as_ref();
        let loader = CompLoader::open(path, None, device)?;
        let src = BundleWeightSource { loader };
        // Архитектура — из config.json бандла (1.7b и 4b различаются числом
        // слоёв/голов/hidden); без него — прежний захардкоженный 1.7b.
        let config = match read_bundle_file(path, "config.json") {
            Ok(bytes) => LmConfig::from_hf_json(&bytes)?,
            Err(_) => LmConfig::lm_1_7b(),
        };
        eprintln!(
            "[acestep-lm] {}: layers={} hidden={} heads={}/{} inter={}",
            path.display(),
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.intermediate_size
        );
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
