use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::module::{Module, join_path};
use crate::parameter::Parameter;

pub struct ModuleList {
    modules: Vec<Box<dyn Module>>,
}

impl ModuleList {
    pub fn new() -> Self { Self { modules: Vec::new() } }

    pub fn with(modules: Vec<Box<dyn Module>>) -> Self { Self { modules } }

    pub fn push<M: Module + 'static>(&mut self, module: M) {
        self.modules.push(Box::new(module));
    }

    pub fn len(&self) -> usize { self.modules.len() }
    pub fn is_empty(&self) -> bool { self.modules.is_empty() }

    pub fn iter(&self) -> impl Iterator<Item = &dyn Module> {
        self.modules.iter().map(|m| m.as_ref())
    }

    pub fn get(&self, idx: usize) -> Option<&dyn Module> {
        self.modules.get(idx).map(|m| m.as_ref())
    }
}

impl Default for ModuleList {
    fn default() -> Self { Self::new() }
}

impl Module for ModuleList {
    fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        Err(SynaptixError::Unsupported(
            "ModuleList has no forward — iterate manually",
        ))
    }

    fn parameters(&self) -> Vec<&Parameter> {
        let mut out = Vec::new();
        for m in &self.modules {
            out.extend(m.parameters());
        }
        out
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Parameter)> {
        let mut out = Vec::new();
        for (i, m) in self.modules.iter().enumerate() {
            let child_prefix = join_path(prefix, &i.to_string());
            out.extend(m.named_parameters(&child_prefix));
        }
        out
    }

    fn set_training(&self, training: bool) {
        for m in &self.modules {
            m.set_training(training);
        }
    }
}
