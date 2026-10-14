//! Веса башни из `.syn`-бандла: отдельный компонент `vision`, если он есть,
//! иначе тензоры `model.vision_tower.*` / `model.merger.*` в основном
//! компоненте (так кладёт `synaptix convert` по HF-каталогу).

use std::path::Path;

use synaptix_bundle::Bundle;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;

use crate::config::MiniCpmConfig;
use crate::model::{MiniCpmVision, VisionError, VisionWeights, TOWER};

/// Компонент бандла с башней (если выделен отдельно).
pub const VISION_COMPONENT: &str = "vision";

/// Тензор, по которому узнаём башню среди весов основного компонента.
fn probe_key() -> String {
    format!("{TOWER}.embeddings.patch_embedding.weight")
}

pub struct BundleVisionWeights {
    loader: SynBundleLoader,
}

impl BundleVisionWeights {
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

/// Есть ли в бандле башня MiniCPM-V: компонент `vision` или тензор
/// патч-эмбеддинга в основном компоненте. Веса не читаются.
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
) -> Result<MiniCpmVision, VisionError> {
    let path = path.as_ref();
    let bundle = Bundle::open(path).map_err(|e| VisionError::Load(e.to_string()))?;
    let cfg_bytes = bundle
        .read_file("config.json")
        .map_err(|e| VisionError::Load(format!("config.json: {e}")))?
        .into_owned();
    drop(bundle);
    let config =
        MiniCpmConfig::from_hf_bytes(&cfg_bytes).map_err(|e| VisionError::Load(e.to_string()))?;
    let weights = BundleVisionWeights::open(path)?;
    MiniCpmVision::build(config, &weights, device, dtype)
}
