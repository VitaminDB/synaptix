use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use synaptix_ops::norm::layer_norm;

use crate::dit::dit_block::modulate;
use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

pub struct FinalLayer {
    pub norm_final: Parameter,
    pub linear: Linear,
    pub adaln_modulation: Linear,
    pub hidden_size: usize,
}

impl FinalLayer {
    pub fn new(hidden_size: usize, patch_size: usize, out_channels: usize, cond_dim: usize, device: Device, dtype: DType) -> Result<Self> {
        let nf = crate::init::init_tensor(&[hidden_size], InitMethod::Ones, dtype, 0, device)?;
        Ok(Self {
            norm_final: Parameter::new(nf),
            linear: Linear::from_init(hidden_size, patch_size * patch_size * out_channels, true, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 0)?,
            adaln_modulation: Linear::from_init(cond_dim, 2 * hidden_size, true, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 1)?,
            hidden_size,
        })
    }

    pub fn from_weights(
        linear_w: Tensor, linear_b: Option<Tensor>,
        adaln_w: Tensor, adaln_b: Option<Tensor>,
    ) -> Result<Self> {
        let hidden_size = linear_w.dims()[1];
        let device = linear_w.device();
        let dtype = linear_w.dtype();
        let nf = crate::init::init_tensor(&[hidden_size], InitMethod::Ones, dtype, 0, device)?;
        Ok(Self {
            norm_final: Parameter::new(nf),
            linear: Linear::new(linear_w, linear_b)?,
            adaln_modulation: Linear::new(adaln_w, adaln_b)?,
            hidden_size,
        })
    }

    pub fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let cond_silu = cond.silu()?;
        let mod_out = self.adaln_modulation.forward(&cond_silu)?;
        let shift = mod_out.narrow(1, 0, self.hidden_size)?.contiguous()?;
        let scale = mod_out.narrow(1, self.hidden_size, self.hidden_size)?.contiguous()?;
        let h = layer_norm(x, None, None, 1e-6)?;
        let h = modulate(&h, &shift, &scale)?;
        self.linear.forward(&h)
    }
}
