use std::collections::BTreeMap;

use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::parameter::Parameter;

pub trait Module: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;

    fn parameters(&self) -> Vec<&Parameter> { Vec::new() }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Parameter)> {
        let _ = prefix;
        Vec::new()
    }

    fn state_dict(&self) -> BTreeMap<String, Tensor> {
        let mut dict = BTreeMap::new();
        for (name, param) in self.named_parameters("") {
            dict.insert(name, param.tensor());
        }
        dict
    }

    fn load_state_dict(&self, dict: &BTreeMap<String, Tensor>) -> Result<()> {
        for (name, param) in self.named_parameters("") {
            let value = dict.get(&name).ok_or_else(|| {
                SynaptixError::Other(format!("load_state_dict: missing key '{name}'"))
            })?;
            param.set(value.clone())?;
        }
        Ok(())
    }

    fn set_training(&self, _training: bool) {}
}

pub trait ModuleExt: Module {
    fn forward_with(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.len() != 1 {
            return Err(SynaptixError::Unsupported(
                "ModuleExt::forward_with: default expects 1 input",
            ));
        }
        self.forward(inputs[0])
    }
}

impl<M: Module> ModuleExt for M {}

pub fn join_path(prefix: &str, child: &str) -> String {
    if prefix.is_empty() {
        child.to_string()
    } else if child.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}.{child}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_core::device::Device;
    use synaptix_core::dtype::DType;

    struct Identity;
    impl Module for Identity {
        fn forward(&self, x: &Tensor) -> Result<Tensor> { Ok(x.clone()) }
    }

    #[test]
    fn default_parameters_empty() {
        let m = Identity;
        assert!(m.parameters().is_empty());
        assert!(m.named_parameters("").is_empty());
        assert!(m.state_dict().is_empty());
    }

    #[test]
    fn forward_passes_through() {
        synaptix_kernels_cpu::ensure_registered();
        let t = Tensor::zeros((2usize, 3), DType::F32, Device::Cpu).unwrap();
        let m = Identity;
        let out = m.forward(&t).unwrap();
        assert_eq!(out.dims(), &[2, 3]);
    }
}
