use std::collections::HashMap;
use std::sync::Arc;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use crate::op::{OpKey, OpImpl, OpAttrs};
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use crate::error::Result;

type OpTable = HashMap<OpKey, Arc<dyn OpImpl>>;
static REGISTRY: Lazy<RwLock<OpTable>> = Lazy::new(|| RwLock::new(HashMap::new()));

pub fn register(key: OpKey, imp: Arc<dyn OpImpl>) {
    REGISTRY.write().insert(key, imp);
}

pub fn dispatch(name: &str, device: Device, dtype: DType, inputs: &[&Tensor], attrs: &OpAttrs) -> Result<Vec<Tensor>> {
    let key = OpKey::new(name, device, dtype);
    let guard = REGISTRY.read();
    if let Some(imp) = guard.get(&key) {
        imp.call(inputs, attrs)
    } else {
        crate::fallback::cpu_fallback(name, inputs, attrs)
    }
}

pub fn is_registered(name: &str, device: Device, dtype: DType) -> bool {
    let key = OpKey::new(name, device, dtype);
    REGISTRY.read().contains_key(&key)
}
