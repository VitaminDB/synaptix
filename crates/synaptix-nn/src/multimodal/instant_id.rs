use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

/// InstantID projector (Wang et al. 2024) — проекция face identity embedding
/// (ArcFace 512-d) в hidden space SD UNet (cross-attn context_dim).
/// Минимально — `Linear(id_dim → hidden_size)`. Полная InstantID имеет
/// IdentityNet (ControlNet-side) + adapter-style residual в cross-attention —
/// откладывается до Phase O.
pub struct InstantIdProjector {
    pub proj: Linear,
    pub id_dim: usize,
    pub hidden_size: usize,
}

impl InstantIdProjector {
    pub fn new(id_dim: usize, hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_init(
                id_dim, hidden_size, true,
                InitMethod::XavierUniform { fan_in: id_dim, fan_out: hidden_size },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            id_dim,
            hidden_size,
        })
    }

    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let proj = Linear::new(weight, bias)?;
        let id_dim = proj.in_features();
        let hidden_size = proj.out_features();
        Ok(Self { proj, id_dim, hidden_size })
    }

    pub fn forward(&self, id_embedding: &Tensor) -> Result<Tensor> {
        self.proj.forward(id_embedding)
    }
}
