use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::module::{Module, join_path};
use crate::parameter::Parameter;

pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self { Self { modules: Vec::new() } }

    pub fn with(modules: Vec<Box<dyn Module>>) -> Self { Self { modules } }

    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(module));
        self
    }

    pub fn push<M: Module + 'static>(&mut self, module: M) {
        self.modules.push(Box::new(module));
    }

    pub fn len(&self) -> usize { self.modules.len() }
    pub fn is_empty(&self) -> bool { self.modules.is_empty() }
}

impl Default for Sequential {
    fn default() -> Self { Self::new() }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for m in &self.modules {
            out = m.forward(&out)?;
        }
        Ok(out)
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
