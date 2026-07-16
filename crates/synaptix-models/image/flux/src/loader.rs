//! Загрузка одного компонента FLUX из HF-директории.
//!
//! FLUX (diffusers-раскладка) хранит каждую подмодель в своём подкаталоге
//! (`text_encoder/`, `text_encoder_2/`, `transformer/`, `vae/`). Крупные —
//! шардированы (`transformer/` 3 файла, `text_encoder_2/` 2). [`ComponentWeights`]
//! авто-детектит шарды через [`scan_shards`] и отдаёт `get`-замыкание формата,
//! который ждут `*::load` из [`synaptix_nn`]
//! (`Fn(&str) -> Result<Tensor, SynaptixError>`, каст к compute-dtype).

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_io::weights::safetensors::{scan_shards, SafetensorsLoader};
use synaptix_io::weights::WeightLoader;

use crate::FluxError;

pub struct ComponentWeights {
    loader: SafetensorsLoader,
    device: Device,
    dtype: DType,
}

impl ComponentWeights {
    /// Открыть подкаталог компонента (`<model>/transformer`, `<model>/vae` …).
    /// Все `*.safetensors` внутри объединяются в единый индекс (поддержка шардов).
    pub fn open_dir(
        dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, FluxError> {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            return Err(FluxError::Load(format!("not a dir: {}", dir.display())));
        }
        let shards = scan_shards(dir).map_err(|e| FluxError::Load(e.to_string()))?;
        if shards.is_empty() {
            return Err(FluxError::Load(format!("no .safetensors in {}", dir.display())));
        }
        let loader = SafetensorsLoader::open_sharded(&shards)
            .map_err(|e| FluxError::Load(e.to_string()))?
            .with_device(device);
        Ok(Self { loader, device, dtype })
    }

    /// Открыть один safetensors-файл (для нешардированных весов вроде `ae.safetensors`).
    pub fn open_file(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, FluxError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(FluxError::Load(format!("not found: {}", path.display())));
        }
        let loader = SafetensorsLoader::open(path)
            .map_err(|e| FluxError::Load(e.to_string()))?
            .with_device(device);
        Ok(Self { loader, device, dtype })
    }

    /// Тензор по HF-имени, приведённый к compute-dtype на целевом устройстве.
    pub fn get(&self, name: &str) -> Result<Tensor, SynaptixError> {
        self.loader
            .load_to(name, self.device, self.dtype)
            .map_err(|e| SynaptixError::Other(format!("load '{name}': {e}")))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.loader.names().iter().any(|n| *n == name)
    }
}
