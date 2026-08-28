use std::path::Path;

use synaptix_bundle::Bundle;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;

use crate::config::HybridConfig;

pub const LM_PREFIX: &str = "model.language_model.";

pub struct HybridWeights {
    pub config: HybridConfig,
    loader: SynBundleLoader,
    pub tokenizer_json: Vec<u8>,
    pub device: Device,
    pub dtype: DType,
}

impl HybridWeights {
    pub fn load(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let bundle = Bundle::open(path).map_err(|e| LoadError::Io(e.to_string()))?;
        let config_bytes = bundle
            .read_file("config.json")
            .map_err(|e| LoadError::Io(format!("read config.json: {e}")))?;
        let config =
            HybridConfig::from_hf_bytes(&config_bytes).map_err(|e| LoadError::Config(e.to_string()))?;
        let tokenizer_json = bundle
            .read_file("tokenizer.json")
            .map_err(|e| LoadError::Io(format!("read tokenizer.json: {e}")))?
            .into_owned();
        drop(bundle);

        let loader = SynBundleLoader::open(path)
            .map_err(|e| LoadError::Io(e.to_string()))?
            .with_device(device);
        Ok(Self { config, loader, tokenizer_json, device, dtype })
    }

    pub fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, LoadError> {
        self.loader
            .load_to(key, device, dtype)
            .map_err(|e| LoadError::Io(format!("load '{key}': {e}")))
    }

    pub fn lm_tensor(&self, suffix: &str, device: Device, dtype: DType) -> Result<Tensor, LoadError> {
        let key = format!("{LM_PREFIX}{suffix}");
        self.tensor(&key, device, dtype)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.loader.names().iter().any(|n| *n == key)
    }

    /// Имя тензора внутри бандла: у гибрида языковая часть лежит под
    /// `model.language_model.`, а снаружи адресуется как `model.`.
    fn bundle_key(key: &str) -> String {
        if key == "lm_head.weight" {
            return key.to_string();
        }
        match key.strip_prefix("model.") {
            Some(rest) => format!("{LM_PREFIX}{rest}"),
            None => key.to_string(),
        }
    }
}

impl synaptix_llm_common::WeightSource for HybridWeights {
    fn tensor(
        &self,
        key: &str,
        device: Device,
        dtype: DType,
    ) -> Result<Tensor, synaptix_llm_common::ModelError> {
        let r = if key == "lm_head.weight" {
            HybridWeights::tensor(self, key, device, dtype)
        } else if let Some(rest) = key.strip_prefix("model.") {
            self.lm_tensor(rest, device, dtype)
        } else {
            HybridWeights::tensor(self, key, device, dtype)
        };
        r.map_err(|e| synaptix_llm_common::ModelError::Load(e.to_string()))
    }

    fn contains(&self, key: &str) -> bool {
        HybridWeights::contains(self, &Self::bundle_key(key))
    }

    /// Готовый квант-вес из бандла, собранного с `syn-quant-v1`.
    fn quant(
        &self,
        key: &str,
        device: Device,
    ) -> Option<Result<synaptix_core::tensor::quant::QuantWeight, synaptix_llm_common::ModelError>>
    {
        let r = self.loader.load_quant(&Self::bundle_key(key), device)?;
        Some(r.map_err(|e| synaptix_llm_common::ModelError::Load(e.to_string())))
    }

    /// Стопка экспертов MoE, упакованная одним блобом.
    fn quant_stack(
        &self,
        key: &str,
        device: Device,
    ) -> Option<
        Result<Vec<synaptix_core::tensor::quant::QuantWeight>, synaptix_llm_common::ModelError>,
    > {
        let r = self.loader.load_quant_stack(&Self::bundle_key(key), device)?;
        Some(r.map_err(|e| synaptix_llm_common::ModelError::Load(e.to_string())))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(String),
    #[error("config: {0}")]
    Config(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn bundle_path() -> Option<PathBuf> {
        let p = PathBuf::from("models/qwen3.6 27B.syn");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    #[test]
    fn probe_shapes() {
        if std::env::var("SYN_QWEN_NEXT_PROBE").is_err() {
            return;
        }
        let Some(path) = bundle_path() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let w = HybridWeights::load(&path, Device::Cpu, DType::F32).expect("open");
        let c = &w.config;
        eprintln!(
            "cfg: H={} layers={} heads={}/{} hd={} rot={} lin(k={},v={},dk={},dv={},conv_k={})",
            c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads,
            c.head_dim, c.rotary_dim(), c.linear_num_key_heads, c.linear_num_value_heads,
            c.linear_key_head_dim, c.linear_value_head_dim, c.linear_conv_kernel_dim,
        );
        let names = [
            "embed_tokens.weight",
            "norm.weight",
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.0.linear_attn.in_proj_qkv.weight",
            "layers.0.linear_attn.in_proj_a.weight",
            "layers.0.linear_attn.in_proj_b.weight",
            "layers.0.linear_attn.in_proj_z.weight",
            "layers.0.linear_attn.conv1d.weight",
            "layers.0.linear_attn.A_log",
            "layers.0.linear_attn.dt_bias",
            "layers.0.linear_attn.norm.weight",
            "layers.0.linear_attn.out_proj.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "layers.3.self_attn.q_proj.weight",
            "layers.3.self_attn.k_proj.weight",
            "layers.3.self_attn.v_proj.weight",
            "layers.3.self_attn.o_proj.weight",
            "layers.3.self_attn.q_norm.weight",
            "layers.3.self_attn.k_norm.weight",
        ];
        for n in names {
            match w.lm_tensor(n, Device::Cpu, DType::F32) {
                Ok(t) => eprintln!("  {n}: {:?}", t.dims()),
                Err(e) => eprintln!("  {n}: ERR {e}"),
            }
        }
        match w.tensor("lm_head.weight", Device::Cpu, DType::F32) {
            Ok(t) => eprintln!("  lm_head.weight: {:?}", t.dims()),
            Err(e) => eprintln!("  lm_head.weight: ERR {e}"),
        }
    }

    #[test]
    fn opens_and_reads_config_and_one_tensor() {
        if std::env::var("SYN_QWEN_NEXT_LOAD").is_err() {
            return;
        }
        let Some(path) = bundle_path() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let w = HybridWeights::load(&path, Device::Cpu, DType::BF16).expect("open");
        assert_eq!(w.config.num_hidden_layers, 64);
        assert_eq!(w.config.hidden_size, 5120);
        assert!(!w.tokenizer_json.is_empty());
        let emb = w
            .lm_tensor("embed_tokens.weight", Device::Cpu, DType::BF16)
            .expect("embed");
        assert_eq!(emb.dims(), &[w.config.vocab_size, w.config.hidden_size]);
        let lm = w.tensor("lm_head.weight", Device::Cpu, DType::BF16).expect("lm_head");
        assert_eq!(lm.dims(), &[w.config.vocab_size, w.config.hidden_size]);
    }
}
