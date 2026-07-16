use std::path::Path;
use std::sync::Arc;

use synaptix_bundle::Bundle;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_nn::module::Module;
use synaptix_nn::quant_linear::QuantLinear;

use crate::config::VoxConfig;
use crate::VoxError;

const BASE_COMPONENT: &str = "base";
const VAE_COMPONENT: &str = "audiovae";

pub struct VoxCheckpoint {
    bundle: Arc<Bundle>,
    base: SynBundleLoader,
    vae: SynBundleLoader,
    pub config: VoxConfig,
    pub device: Device,
    pub compute: DType,
    pub quant: DType,
}

impl VoxCheckpoint {
    pub fn open(path: impl AsRef<Path>, device: Device, compute: DType) -> Result<Self, VoxError> {
        Self::open_quant(path, device, compute, compute)
    }

    pub fn open_quant(
        path: impl AsRef<Path>,
        device: Device,
        compute: DType,
        quant: DType,
    ) -> Result<Self, VoxError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(VoxError::Load(format!("not found: {}", path.display())));
        }
        let bundle = Bundle::open(path).map_err(|e| VoxError::Load(e.to_string()))?;
        let cfg_bytes = bundle
            .read_file("config.json")
            .map_err(|e| VoxError::Load(format!("config.json: {e}")))?;
        let config = VoxConfig::from_json_bytes(&cfg_bytes)?;

        let base = SynBundleLoader::open(path)
            .map_err(|e| VoxError::Load(e.to_string()))?
            .with_component(BASE_COMPONENT)
            .with_device(device);
        let vae = SynBundleLoader::open(path)
            .map_err(|e| VoxError::Load(e.to_string()))?
            .with_component(VAE_COMPONENT)
            .with_device(device);

        Ok(Self {
            bundle: Arc::new(bundle),
            base,
            vae,
            config,
            device,
            compute,
            quant,
        })
    }

    pub fn read_file(&self, name: &str) -> Result<Vec<u8>, VoxError> {
        self.bundle
            .read_file(name)
            .map(|c| c.into_owned())
            .map_err(|e| VoxError::Load(format!("read {name}: {e}")))
    }

    pub fn has_file(&self, name: &str) -> bool {
        self.bundle.read_file(name).is_ok()
    }

    pub fn get(&self, name: &str) -> Result<Tensor, VoxError> {
        self.base
            .load_to(name, self.device, self.compute)
            .map_err(|e| VoxError::Load(format!("get '{name}': {e}")))
    }

    pub fn get_raw(&self, name: &str) -> Result<Tensor, VoxError> {
        self.base
            .load(name)
            .map_err(|e| VoxError::Load(format!("get_raw '{name}': {e}")))
    }

    pub fn vae(&self, name: &str) -> Result<Tensor, VoxError> {
        self.vae
            .load_to(name, self.device, DType::F32)
            .map_err(|e| VoxError::Load(format!("vae '{name}': {e}")))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.base.names().iter().any(|n| *n == name)
    }
}

pub struct Lin(QuantLinear);

impl Lin {
    pub fn load(ck: &VoxCheckpoint, prefix: &str, key: &str, bias: bool) -> Result<Self, VoxError> {
        Self::load_direct(ck, &format!("{prefix}.{key}"), bias)
    }

    pub fn load_direct(ck: &VoxCheckpoint, base: &str, bias: bool) -> Result<Self, VoxError> {
        let w = ck.get_raw(&format!("{base}.weight"))?;
        let b = if bias {
            Some(ck.get_raw(&format!("{base}.bias"))?)
        } else {
            None
        };
        let q = QuantLinear::build(w, b, ck.quant, ck.compute)?;
        Ok(Self(q))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, VoxError> {
        Ok(self.0.forward(x)?)
    }

    pub fn forward_add(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor, VoxError> {
        Ok(self.0.forward_add(x, residual)?)
    }
}
