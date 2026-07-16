pub mod pinned_loader;
pub mod safetensors;
pub mod streaming_loader;
pub mod syn_bundle;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use crate::error::Result;

pub trait WeightLoader: Send + Sync {
    fn load(&self, name: &str) -> Result<Tensor>;
    fn load_to(&self, name: &str, device: Device, dtype: DType) -> Result<Tensor>;
    fn names(&self) -> Vec<&str>;
    fn contains(&self, name: &str) -> bool {
        self.names().iter().any(|n| *n == name)
    }
}
