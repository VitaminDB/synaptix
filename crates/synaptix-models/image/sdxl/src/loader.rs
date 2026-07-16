//! Загрузка одного компонента SDXL из safetensors-файла HF-директории.
//!
//! SDXL хранит каждую подмодель в своём подкаталоге (`text_encoder/`,
//! `text_encoder_2/`, `unet/`, `vae/`) отдельным safetensors-файлом в
//! HF-раскладке имён. [`ComponentWeights`] оборачивает [`SafetensorsLoader`]
//! и отдаёт `get`-замыкание, которое ждут `*::load` из [`synaptix_nn`]
//! (`Fn(&str) -> Result<Tensor, SynaptixError>`, приведение к compute-dtype).

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;

use crate::SdxlError;

pub struct ComponentWeights {
    loader: SafetensorsLoader,
    device: Device,
    dtype: DType,
}

impl ComponentWeights {
    pub fn open(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, SdxlError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(SdxlError::Load(format!("not found: {}", path.display())));
        }
        let loader = SafetensorsLoader::open(path)
            .map_err(|e| SdxlError::Load(e.to_string()))?
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
        self.loader.contains(name)
    }
}
