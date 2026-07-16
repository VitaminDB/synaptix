use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::parameter::Parameter;

pub struct PrefixTuning {
    pub prefix_keys: Parameter,
    pub prefix_values: Parameter,
    pub prefix_len: usize,
    pub num_layers: usize,
}

impl PrefixTuning {
    pub fn new(prefix_len: usize, num_layers: usize, hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        let pk = crate::init::init_tensor(&[num_layers, prefix_len, hidden_size], InitMethod::Normal { mean: 0.0, std: 0.02 }, dtype, 0, device)?;
        let pv = crate::init::init_tensor(&[num_layers, prefix_len, hidden_size], InitMethod::Normal { mean: 0.0, std: 0.02 }, dtype, 1, device)?;
        Ok(Self {
            prefix_keys: Parameter::new(pk),
            prefix_values: Parameter::new(pv),
            prefix_len,
            num_layers,
        })
    }

    pub fn get_prefix(&self, layer: usize) -> Result<(Tensor, Tensor)> {
        let keys = self.prefix_keys.tensor();
        let values = self.prefix_values.tensor();
        let k = keys.narrow(0, layer, 1)?;
        let v = values.narrow(0, layer, 1)?;
        Ok((k, v))
    }
}
