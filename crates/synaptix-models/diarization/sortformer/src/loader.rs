//! Загрузка весов Sortformer из `.syn`-бандла через [`SynBundleLoader`].
//!
//! Веса под HF-именами (`encoder.layers.{i}.self_attn.linear_q.weight`, `head.layers.{i}.*`,
//! `encoder.pre_encode.conv.{0,2,3,5,6}.*` и т.п.) в чанке `tensors:main`. Маппинг имён — в
//! `SPEC.md`. Вспомогательные файлы (config.json) читаются через [`Bundle::read_file`].

use std::collections::HashSet;
use std::path::Path;

use safetensors::SafeTensors;
use synaptix_bundle::Bundle;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;

use crate::config::SortformerConfig;
use crate::SortformerError;

pub struct SortformerWeights {
    bundle: Bundle,
    loader: SynBundleLoader,
    names: HashSet<String>,
    pub config: SortformerConfig,
    pub device: Device,
    pub dtype: DType,
}

impl SortformerWeights {
    pub fn open(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, SortformerError> {
        let path = path.as_ref();
        let bundle = Bundle::open(path).map_err(|e| SortformerError::Bundle(e.to_string()))?;
        let config = SortformerConfig::from_bundle(&bundle)?;
        let loader = SynBundleLoader::open(path)
            .map_err(|e| SortformerError::Load(e.to_string()))?
            .with_device(device);

        // SynBundleLoader::names() отдаёт dangling-ссылки (st дропается) — собираем
        // владеющий набор имён сами из заголовка safetensors один раз при открытии.
        let slice = bundle
            .tensors_slice()
            .map_err(|e| SortformerError::Bundle(e.to_string()))?;
        let st = SafeTensors::deserialize(slice)
            .map_err(|e| SortformerError::Load(format!("safetensors header: {e}")))?;
        let names: HashSet<String> = st.names().into_iter().map(|s| s.to_string()).collect();

        Ok(Self { bundle, loader, names, config, device, dtype })
    }

    /// Тензор по HF-имени, приведённый к compute-dtype на целевом устройстве.
    pub fn get(&self, name: &str) -> Result<Tensor, SortformerError> {
        self.loader
            .load_to(name, self.device, self.dtype)
            .map_err(|e| SortformerError::Load(format!("'{name}': {e}")))
    }

    /// Тензор в явном dtype (напр. F32 для running_mean/var BatchNorm, pos_bias).
    pub fn get_dtype(&self, name: &str, dtype: DType) -> Result<Tensor, SortformerError> {
        self.loader
            .load_to(name, self.device, dtype)
            .map_err(|e| SortformerError::Load(format!("'{name}': {e}")))
    }

    pub fn get_opt(&self, name: &str) -> Option<Tensor> {
        if self.contains(name) {
            self.get(name).ok()
        } else {
            None
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    pub fn read_aux(&self, name: &str) -> Result<Vec<u8>, SortformerError> {
        self.bundle
            .read_file(name)
            .map(|c| c.into_owned())
            .map_err(|e| SortformerError::Bundle(format!("'{name}': {e}")))
    }
}

/// `encoder.layers.{i}.{suffix}` — слой FastConformer-энкодера.
pub fn enc_layer(i: usize, suffix: &str) -> String {
    format!("encoder.layers.{i}.{suffix}")
}

/// `head.layers.{i}.{suffix}` — слой Sortformer-head'а.
pub fn head_layer(i: usize, suffix: &str) -> String {
    format!("head.layers.{i}.{suffix}")
}
