use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use crate::error::Result;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct OpKey {
    pub name: String,
    pub device: String,
    pub dtype: String,
}

impl OpKey {
    pub fn new(name: &str, device: Device, dtype: DType) -> Self {
        Self {
            name: name.into(),
            device: format!("{device:?}"),
            dtype: format!("{dtype:?}"),
        }
    }
}

pub trait OpImpl: Send + Sync {
    fn call(&self, inputs: &[&Tensor], attrs: &OpAttrs) -> Result<Vec<Tensor>>;
}

#[derive(Debug, Clone, Default)]
pub struct OpAttrs {
    pub ints: std::collections::HashMap<String, i64>,
    pub floats: std::collections::HashMap<String, f64>,
    pub strings: std::collections::HashMap<String, String>,
    pub bools: std::collections::HashMap<String, bool>,
}

impl OpAttrs {
    pub fn new() -> Self { Self::default() }
    pub fn int(mut self, k: &str, v: i64) -> Self { self.ints.insert(k.into(), v); self }
    pub fn float(mut self, k: &str, v: f64) -> Self { self.floats.insert(k.into(), v); self }
    pub fn string(mut self, k: &str, v: &str) -> Self { self.strings.insert(k.into(), v.into()); self }
    pub fn bool(mut self, k: &str, v: bool) -> Self { self.bools.insert(k.into(), v); self }
    pub fn get_int(&self, k: &str) -> Option<i64> { self.ints.get(k).copied() }
    pub fn get_float(&self, k: &str) -> Option<f64> { self.floats.get(k).copied() }
    pub fn get_str(&self, k: &str) -> Option<&str> { self.strings.get(k).map(|s| s.as_str()) }
    pub fn get_bool(&self, k: &str) -> Option<bool> { self.bools.get(k).copied() }
}
