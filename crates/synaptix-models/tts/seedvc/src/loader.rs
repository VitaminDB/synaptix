//! Веса и файлы из `.syn`-бандла по компонентам (`ar`, `cfm`, `astral_wide`,
//! `astral_narrow`, `hubert`, `campplus`, `bigvgan`).

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;

use crate::{SeedVcError, R};

pub struct Comp {
    inner: SynBundleLoader,
    device: Device,
}

impl Comp {
    pub fn open(path: &Path, component: &str, device: Device) -> R<Self> {
        let l = SynBundleLoader::open(path).map_err(|e| SeedVcError::Load(format!("{}: {e}", path.display())))?.with_device(device).with_component(component);
        Ok(Self { inner: l, device })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn get(&self, name: &str, dtype: DType) -> R<Tensor> {
        self.inner.load_to(name, self.device, dtype).map_err(|e| SeedVcError::Load(format!("тензор `{name}`: {e}")))
    }

    /// Тензор на хосте, плоским `Vec<f32>`, с формой.
    pub fn host(&self, name: &str) -> R<(Vec<f32>, Vec<usize>)> {
        let t = self.inner.load_to(name, Device::Cpu, DType::F32).map_err(|e| SeedVcError::Load(format!("тензор `{name}`: {e}")))?;
        let dims = t.dims().to_vec();
        Ok((t.contiguous()?.flatten_all()?.to_vec1::<f32>()?, dims))
    }

    pub fn has(&self, name: &str) -> bool {
        self.inner.contains(name)
    }
}
