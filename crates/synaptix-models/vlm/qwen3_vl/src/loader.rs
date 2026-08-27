use std::path::Path;

use synaptix_bundle::Bundle;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;

use crate::config::VisionConfig;
use crate::model::{VisionError, VisionTower, VisionWeights};

/// Отдельный компонент башни — так её кладёт GGUF-репак (`--mmproj`).
pub const VISION_COMPONENT: &str = "vision";

/// Тензор, по которому узнаём башню среди весов основного компонента:
/// HF-упаковка (`syn-pack` по safetensors) кладёт `model.visual.*` в
/// `tensors:main` вместе с языковой моделью.
fn probe_key() -> String {
    format!("{}.patch_embed.proj.weight", crate::model::VIS)
}

pub struct BundleVisionWeights {
    loader: SynBundleLoader,
}

impl BundleVisionWeights {
    /// Открывает веса башни: из компонента `vision`, если он есть в бандле,
    /// иначе — из основного компонента (HF-имена `model.visual.*`).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, VisionError> {
        let path = path.as_ref();
        let has_component = Bundle::open(path)
            .map(|b| b.meta().components.contains_key(VISION_COMPONENT))
            .map_err(|e| VisionError::Load(e.to_string()))?;
        let mut loader =
            SynBundleLoader::open(path).map_err(|e| VisionError::Load(e.to_string()))?;
        if has_component {
            loader = loader.with_component(VISION_COMPONENT);
        }
        Ok(Self { loader })
    }

    pub fn has(&self, key: &str) -> bool {
        self.loader.names().iter().any(|n| *n == key)
    }
}

impl VisionWeights for BundleVisionWeights {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, VisionError> {
        self.loader
            .load_to(key, device, dtype)
            .map_err(|e| VisionError::Load(format!("{key}: {e}")))
    }
}

/// Есть ли в бандле vision-башня: отдельным компонентом `vision` или
/// тензорами `model.visual.*` в основном компоненте. Без загрузки весов —
/// читается только заголовок safetensors (mmap).
pub fn bundle_has_vision(path: impl AsRef<Path>) -> bool {
    let path = path.as_ref();
    let Ok(b) = Bundle::open(path) else {
        return false;
    };
    if b.meta().components.contains_key(VISION_COMPONENT) {
        return true;
    }
    drop(b);
    BundleVisionWeights::open(path)
        .map(|w| w.has(&probe_key()))
        .unwrap_or(false)
}

pub fn load_from_bundle(
    path: impl AsRef<Path>,
    device: Device,
    dtype: DType,
) -> Result<VisionTower, VisionError> {
    let path = path.as_ref();
    let bundle = Bundle::open(path).map_err(|e| VisionError::Load(e.to_string()))?;
    let cfg_bytes = bundle
        .read_file("config.json")
        .map_err(|e| VisionError::Load(format!("config.json: {e}")))?
        .into_owned();
    drop(bundle);
    let config =
        VisionConfig::from_hf_bytes(&cfg_bytes).map_err(|e| VisionError::Load(e.to_string()))?;
    let weights = BundleVisionWeights::open(path)?;
    VisionTower::build(config, &weights, device, dtype)
}
