pub mod pinned_loader;
pub mod safetensors;
pub mod streaming_loader;
pub mod syn_bundle;
pub mod transcode;
pub use syn_bundle::{is_gguf_model, is_model_file, read_model_file};

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use crate::error::Result;

pub trait WeightLoader: Send + Sync {
    fn load(&self, name: &str) -> Result<Tensor>;
    fn load_to(&self, name: &str, device: Device, dtype: DType) -> Result<Tensor>;
    fn names(&self) -> Vec<&str>;
    fn contains(&self, name: &str) -> bool {
        self.names().iter().any(|n| *n == name)
    }
    /// Готовый квант-вес (квантованный бандл); `None` — вес плотный или
    /// загрузчик про кванты не знает. NVFP4 на карте уже перемешан.
    fn prequant(&self, _name: &str, _device: Device) -> Option<Result<synaptix_core::tensor::quant::QuantWeight>> {
        None
    }
}
