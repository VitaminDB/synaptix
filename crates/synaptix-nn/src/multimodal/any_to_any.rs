use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;
use synaptix_ops::norm::rms_norm::rms_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// Any-to-any projector — единый Linear с опциональным RMSNorm на выходе.
/// Используется в multimodal LMs (Chameleon/AnyGPT) для проекции из
/// shared latent space в любую модальность.
pub struct AnyToAnyProjector {
    pub proj: Linear,
    pub norm: Option<Parameter>,
    pub norm_eps: f32,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl AnyToAnyProjector {
    pub fn new(
        in_dim: usize, out_dim: usize, with_norm: bool,
        device: Device, dtype: DType,
    ) -> Result<Self> {
        let norm = if with_norm {
            let n = crate::init::init_tensor(&[out_dim], InitMethod::Ones, dtype, 0, device)?;
            Some(Parameter::new(n))
        } else {
            None
        };
        Ok(Self {
            proj: Linear::from_init(
                in_dim, out_dim, true,
                InitMethod::XavierUniform { fan_in: in_dim, fan_out: out_dim },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            norm,
            norm_eps: 1e-6,
            in_dim,
            out_dim,
        })
    }

    pub fn from_weights(
        weight: Tensor, bias: Option<Tensor>, norm: Option<Tensor>, norm_eps: f32,
    ) -> Result<Self> {
        let proj = Linear::new(weight, bias)?;
        let in_dim = proj.in_features();
        let out_dim = proj.out_features();
        Ok(Self {
            proj,
            norm: norm.map(Parameter::new),
            norm_eps,
            in_dim,
            out_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let projected = self.proj.forward(x)?;
        match self.norm.as_ref() {
            Some(n) => rms_norm(&projected, &n.tensor(), self.norm_eps),
            None => Ok(projected),
        }
    }
}
