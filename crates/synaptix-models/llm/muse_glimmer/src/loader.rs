use std::path::Path;

use synaptix_bundle::Bundle;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;

use crate::config::MuseConfig;

pub const LM_PREFIX: &str = "model.language_model.";

pub struct MuseWeights {
    pub config: MuseConfig,
    loader: SynBundleLoader,
    pub tokenizer_json: Vec<u8>,
    pub device: Device,
    pub dtype: DType,
}

impl MuseWeights {
    pub fn load(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let bundle = Bundle::open(path).map_err(|e| LoadError::Io(e.to_string()))?;
        let config_bytes = bundle
            .read_file("config.json")
            .map_err(|e| LoadError::Io(format!("read config.json: {e}")))?;
        let mut config =
            MuseConfig::from_hf_bytes(&config_bytes).map_err(|e| LoadError::Config(e.to_string()))?;
        if let Ok(gen_bytes) = bundle.read_file("generation_config.json") {
            config.merge_generation_config(&gen_bytes);
        }
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

    fn fused_q_gate(
        &self,
        layer_suffix: &str,
        device: Device,
        dtype: DType,
    ) -> Result<Tensor, LoadError> {
        let q = self.lm_tensor(&format!("{layer_suffix}self_attn.q_proj.weight"), device, dtype)?;
        let g = self.lm_tensor(&format!("{layer_suffix}self_attn.gate_proj.weight"), device, dtype)?;
        let c = &self.config;
        let (nh, hd) = (c.num_attention_heads, c.head_dim);
        let hidden = c.hidden_size;
        let err = |e: synaptix_core::error::SynaptixError| LoadError::Io(format!("fuse q|gate: {e}"));
        let qh = q.reshape(vec![nh, hd, hidden]).map_err(err)?;
        let gh = g.reshape(vec![nh, hd, hidden]).map_err(err)?;
        Tensor::cat(&[&qh, &gh], 1)
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![nh * 2 * hd, hidden]))
            .map_err(err)
    }
}

fn layer_suffix<'a>(rest: &'a str, tail: &str) -> Option<&'a str> {
    rest.strip_suffix(tail)
        .filter(|p| p.starts_with("layers."))
}

impl synaptix_llm_common::WeightSource for MuseWeights {
    fn tensor(
        &self,
        key: &str,
        device: Device,
        dtype: DType,
    ) -> Result<Tensor, synaptix_llm_common::ModelError> {
        let map_err = |e: LoadError| synaptix_llm_common::ModelError::Load(e.to_string());
        if key == "lm_head.weight" {
            return MuseWeights::tensor(self, key, device, dtype).map_err(map_err);
        }
        let Some(rest) = key.strip_prefix("model.") else {
            return MuseWeights::tensor(self, key, device, dtype).map_err(map_err);
        };
        if rest == "norm.weight" {
            let w = self.lm_tensor(rest, device, dtype).map_err(map_err)?;
            return w
                .add_scalar(-1.0)
                .map_err(|e| synaptix_llm_common::ModelError::Load(format!("norm-1: {e}")));
        }
        if layer_suffix(rest, "self_attn.q_norm.weight").is_some()
            || layer_suffix(rest, "self_attn.k_norm.weight").is_some()
        {
            return Tensor::zeros(vec![self.config.head_dim], dtype, device)
                .map_err(|e| synaptix_llm_common::ModelError::Load(format!("qk_norm zeros: {e}")));
        }
        if let Some(prefix) = layer_suffix(rest, "self_attn.q_proj.weight") {
            return self.fused_q_gate(prefix, device, dtype).map_err(map_err);
        }
        self.lm_tensor(rest, device, dtype).map_err(map_err)
    }

    fn contains(&self, key: &str) -> bool {
        if key == "lm_head.weight" {
            return MuseWeights::contains(self, key);
        }
        let Some(rest) = key.strip_prefix("model.") else {
            return MuseWeights::contains(self, key);
        };
        if layer_suffix(rest, "self_attn.q_norm.weight").is_some()
            || layer_suffix(rest, "self_attn.k_norm.weight").is_some()
        {
            return true;
        }
        MuseWeights::contains(self, &format!("{LM_PREFIX}{rest}"))
    }

    /// Готовый квант-вес из бандла, собранного с `syn-quant-v1`.
    ///
    /// Синтетические и склеиваемые на лету веса сюда не попадают: `q_proj`
    /// собирается из двух тензоров ([`MuseWeights::fused_q_gate`]), а
    /// `q_norm`/`k_norm` в бандле вообще нет — для них квант-блоба быть не
    /// может, и спрашивать его бессмысленно.
    fn quant(
        &self,
        key: &str,
        device: Device,
    ) -> Option<Result<synaptix_core::tensor::quant::QuantWeight, synaptix_llm_common::ModelError>>
    {
        let bundle_key = if key == "lm_head.weight" {
            key.to_string()
        } else {
            let rest = key.strip_prefix("model.")?;
            if layer_suffix(rest, "self_attn.q_proj.weight").is_some()
                || layer_suffix(rest, "self_attn.q_norm.weight").is_some()
                || layer_suffix(rest, "self_attn.k_norm.weight").is_some()
                || rest == "norm.weight"
            {
                return None;
            }
            format!("{LM_PREFIX}{rest}")
        };
        let r = self.loader.load_quant(&bundle_key, device)?;
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

    fn bundle_path() -> Option<std::path::PathBuf> {
        std::env::var("SYN_MUSE_BUNDLE").ok().map(std::path::PathBuf::from)
    }

    #[test]
    fn probe_config_and_remaps() {
        let Some(path) = bundle_path() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let w = MuseWeights::load(&path, Device::Cpu, DType::BF16).expect("open");
        let c = &w.config;
        assert_eq!(c.num_hidden_layers, 52);
        assert_eq!(c.hidden_size, 6656);
        assert_eq!(c.num_key_value_heads, 2);
        assert_eq!(c.eos_token_ids, vec![200_001, 200_008]);
        assert!(c.vision.is_some());
        assert!(!w.tokenizer_json.is_empty());

        use synaptix_llm_common::WeightSource;
        let emb = WeightSource::tensor(&w, "model.embed_tokens.weight", Device::Cpu, DType::BF16)
            .expect("embed");
        assert_eq!(emb.dims(), &[c.vocab_size, c.hidden_size]);
        let fused = WeightSource::tensor(
            &w,
            "model.layers.0.self_attn.q_proj.weight",
            Device::Cpu,
            DType::BF16,
        )
        .expect("fused q|gate");
        assert_eq!(fused.dims(), &[2 * c.num_attention_heads * c.head_dim, c.hidden_size]);
        let qn = WeightSource::tensor(
            &w,
            "model.layers.0.self_attn.q_norm.weight",
            Device::Cpu,
            DType::F32,
        )
        .expect("q_norm zeros");
        assert_eq!(qn.dims(), &[c.head_dim]);
        let v = qn.flatten_all().and_then(|t| t.to_vec1::<f32>()).unwrap();
        assert!(v.iter().all(|x| *x == 0.0));
        assert!(WeightSource::contains(&w, "lm_head.weight"));
        assert!(WeightSource::contains(&w, "model.layers.51.mlp.down_proj.weight"));
        assert!(!WeightSource::contains(&w, "mtp.fc.weight"));

        let q = w
            .lm_tensor("layers.0.self_attn.q_proj.weight", Device::Cpu, DType::F32)
            .unwrap()
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .unwrap();
        let g = w
            .lm_tensor("layers.0.self_attn.gate_proj.weight", Device::Cpu, DType::F32)
            .unwrap()
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .unwrap();
        let f = WeightSource::tensor(
            &w,
            "model.layers.0.self_attn.q_proj.weight",
            Device::Cpu,
            DType::F32,
        )
        .unwrap()
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap();
        let hd = w.config.head_dim;
        let hidden = w.config.hidden_size;
        let row = |h: usize, d: usize| (h * hd + d) * hidden;
        let frow = |h: usize, d: usize| (h * 2 * hd + d) * hidden;
        for (h, d) in [(0usize, 0usize), (0, hd - 1), (5, 17), (31, 100)] {
            assert_eq!(f[frow(h, d)], q[row(h, d)], "q row ({h},{d})");
            assert_eq!(f[frow(h, hd + d)], g[row(h, d)], "gate row ({h},{d})");
        }
    }
}
