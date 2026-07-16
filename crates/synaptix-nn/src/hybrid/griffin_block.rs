use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::norm::layer_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// Griffin minimal block (pre-LN gated GLU + residual).
///
/// Полный Griffin (Google DeepMind) комбинирует RG-LRU и local attention; здесь
/// реализован semantic-плоский inference-stub: pre-LN + gated FFN с
/// SwiGLU-разбиением `fc_in [hidden → 2·inner]` → `silu(a) ⊙ b` → `fc_out`.
///
/// `forward(x: [B, T, hidden])` → `x + fc_out(silu(a) ⊙ b)`, `[a, b] = split(fc_in(LN(x)))`.
pub struct GriffinBlock {
    pub norm_w: Parameter,
    pub norm_b: Parameter,
    pub fc_in: Linear,
    pub fc_out: Linear,
    pub hidden_size: usize,
    pub inner_size: usize,
    pub eps: f32,
}

impl GriffinBlock {
    pub fn new(hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        let inner_size = hidden_size;
        Ok(Self {
            norm_w: Parameter::new(Tensor::ones(vec![hidden_size], dtype, device)?),
            norm_b: Parameter::new(Tensor::zeros(vec![hidden_size], dtype, device)?),
            fc_in: Linear::from_init(
                hidden_size, inner_size * 2, true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            fc_out: Linear::from_init(
                inner_size, hidden_size, true,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 1,
            )?,
            hidden_size, inner_size, eps: 1e-5,
        })
    }

    pub fn from_weights(
        norm_w: Tensor, norm_b: Tensor,
        fc_in_w: Tensor, fc_in_b: Option<Tensor>,
        fc_out_w: Tensor, fc_out_b: Option<Tensor>,
        eps: f32,
    ) -> Result<Self> {
        let fc_in = Linear::new(fc_in_w, fc_in_b)?;
        let fc_out = Linear::new(fc_out_w, fc_out_b)?;
        let hidden_size = fc_in.in_features();
        let inner_size = fc_out.in_features();
        if fc_in.out_features() != inner_size * 2 {
            return Err(SynaptixError::Unsupported("GriffinBlock: fc_in.out_features must be 2 · fc_out.in_features"));
        }
        Ok(Self {
            norm_w: Parameter::new(norm_w),
            norm_b: Parameter::new(norm_b),
            fc_in, fc_out, hidden_size, inner_size, eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 || x.dims()[2] != self.hidden_size {
            return Err(SynaptixError::Unsupported("GriffinBlock: expects x [B, T, hidden]"));
        }
        let h = layer_norm(x, Some(&self.norm_w.tensor()), Some(&self.norm_b.tensor()), self.eps)?;
        let ab = self.fc_in.forward(&h)?;
        let a = ab.narrow(2, 0, self.inner_size)?.contiguous()?;
        let b = ab.narrow(2, self.inner_size, self.inner_size)?.contiguous()?;
        let gated = a.silu()?.mul(&b)?;
        let out = self.fc_out.forward(&gated)?;
        x.add(&out)
    }
}
