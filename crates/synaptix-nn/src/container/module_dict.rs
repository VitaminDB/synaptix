use std::collections::BTreeMap;

use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::module::{Module, join_path};
use crate::parameter::Parameter;

pub struct ModuleDict {
    modules: BTreeMap<String, Box<dyn Module>>,
}

impl ModuleDict {
    pub fn new() -> Self { Self { modules: BTreeMap::new() } }

    pub fn insert<M: Module + 'static>(&mut self, key: impl Into<String>, module: M) {
        self.modules.insert(key.into(), Box::new(module));
    }

    pub fn get(&self, key: &str) -> Option<&dyn Module> {
        self.modules.get(key).map(|m| m.as_ref())
    }

    pub fn contains(&self, key: &str) -> bool { self.modules.contains_key(key) }
    pub fn len(&self) -> usize { self.modules.len() }
    pub fn is_empty(&self) -> bool { self.modules.is_empty() }
    pub fn keys(&self) -> impl Iterator<Item = &str> { self.modules.keys().map(|s| s.as_str()) }
}

impl Default for ModuleDict {
    fn default() -> Self { Self::new() }
}

impl Module for ModuleDict {
    fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        Err(SynaptixError::Unsupported(
            "ModuleDict has no forward — access by key",
        ))
    }

    fn parameters(&self) -> Vec<&Parameter> {
        let mut out = Vec::new();
        for m in self.modules.values() {
            out.extend(m.parameters());
        }
        out
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Parameter)> {
        let mut out = Vec::new();
        for (k, m) in &self.modules {
            let child_prefix = join_path(prefix, k);
            out.extend(m.named_parameters(&child_prefix));
        }
        out
    }

    fn set_training(&self, training: bool) {
        for m in self.modules.values() {
            m.set_training(training);
        }
    }
}
