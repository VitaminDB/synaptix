//! Веса LFM3: HF-каталог (safetensors) или `.syn`-бандл с исходными именами. Общий декодер просит
//! веса в именах Llama/Gemma — здесь они переводятся в имена LFM3.

use std::path::{Path, PathBuf};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_llm_common::ModelError;

use crate::config::Lfm3Config;

enum Source {
    Files(SafetensorsLoader),
    Model(SynBundleLoader),
}

pub struct Lfm3Weights {
    pub config: Lfm3Config,
    source: Source,
}

/// Имя общего декодера → имя LFM3.
pub fn lfm_name(key: &str) -> String {
    if key == "model.norm.weight" {
        return "model.embedding_norm.weight".into();
    }
    if key == "lm_head.weight" {
        return "model.embed_tokens.weight".into();
    }
    const MAP: &[(&str, &str)] = &[
        (".input_layernorm.", ".operator_norm."),
        (".post_attention_layernorm.", ".post_operator_norm."),
        (".pre_feedforward_layernorm.", ".ffn_norm."),
        (".post_feedforward_layernorm.", ".post_ffn_norm."),
        (".mlp.gate_proj.", ".feed_forward.w1."),
        (".mlp.up_proj.", ".feed_forward.w3."),
        (".mlp.down_proj.", ".feed_forward.w2."),
        (".self_attn.head_gate.", ".self_attn.gate_proj."),
        (".short_conv.conv.", ".conv.conv."),
        (".short_conv.in_proj.", ".conv.in_proj."),
        (".short_conv.out_proj.", ".conv.out_proj."),
    ];
    for (a, b) in MAP {
        if key.contains(a) {
            return key.replacen(a, b, 1);
        }
    }
    key.to_string()
}

impl Lfm3Weights {
    pub fn load(path: impl AsRef<Path>, device: Device) -> Result<Self, ModelError> {
        let path = path.as_ref();
        if synaptix_io::weights::is_model_file(path) {
            let l = SynBundleLoader::open(path).map_err(|e| ModelError::Load(e.to_string()))?.with_device(device);
            let cfg = l.read_file("config.json").ok_or_else(|| ModelError::Load("config.json: нет файла".into()))?;
            let config = Lfm3Config::from_hf_json_slice(&cfg).map_err(|e| ModelError::Load(e.to_string()))?;
            return Ok(Self { config, source: Source::Model(l) });
        }
        let config = Lfm3Config::from_hf_json(path.join("config.json")).map_err(|e| ModelError::Load(e.to_string()))?;
        let shards = shards(path)?;
        let l = SafetensorsLoader::open_sharded(&shards).map_err(|e| ModelError::Load(e.to_string()))?.with_device(device);
        Ok(Self { config, source: Source::Files(l) })
    }

    fn loader(&self) -> &dyn WeightLoader {
        match &self.source {
            Source::Files(l) => l,
            Source::Model(l) => l,
        }
    }
}

fn shards(dir: &Path) -> Result<Vec<PathBuf>, ModelError> {
    let one = dir.join("model.safetensors");
    if one.exists() {
        return Ok(vec![one]);
    }
    let idx = dir.join("model.safetensors.index.json");
    let b = std::fs::read(&idx).map_err(|e| ModelError::Load(format!("{}: {e}", idx.display())))?;
    let v: serde_json::Value = serde_json::from_slice(&b).map_err(|e| ModelError::Load(e.to_string()))?;
    let mut s: Vec<PathBuf> = v["weight_map"].as_object().map(|m| m.values().filter_map(|x| x.as_str()).map(|x| dir.join(x)).collect()).unwrap_or_default();
    s.sort();
    s.dedup();
    Ok(s)
}

impl synaptix_llm_common::WeightSource for Lfm3Weights {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, ModelError> {
        let n = lfm_name(key);
        self.loader().load_to(&n, device, dtype).map_err(|e| ModelError::Load(format!("load '{n}': {e}")))
    }

    fn contains(&self, key: &str) -> bool {
        let n = lfm_name(key);
        match &self.source {
            Source::Files(l) => l.contains(&n),
            Source::Model(l) => l.contains(&n) || l.quant_dims(&n).is_some(),
        }
    }

    fn quant(&self, key: &str, device: Device) -> Option<Result<synaptix_core::tensor::quant::QuantWeight, ModelError>> {
        let Source::Model(m) = &self.source else { return None };
        m.load_quant(&lfm_name(key), device).map(|r| r.map_err(|e| ModelError::Load(e.to_string())))
    }
}
