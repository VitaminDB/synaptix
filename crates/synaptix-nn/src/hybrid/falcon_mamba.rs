use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::norm::layer_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// Falcon-Mamba minimal block (pre-LN linear-SiLU-linear с residual).
///
/// Полная Falcon-Mamba (TII) использует selective-state-space-кочегарку Mamba;
/// здесь реализован semantic-плоский inference-stub с тем же API, что позволяет
/// собрать модель и заменить тело forward'а реальным SSM-ядром позже.
///
/// `forward(x: [B, T, hidden])` → `x + fc2(SiLU(fc1(LN(x))))`.
pub struct FalconMamba {
    pub norm_w: Parameter,
    pub norm_b: Parameter,
    pub fc1: Linear,
    pub fc2: Linear,
    pub hidden_size: usize,
    pub eps: f32,
}

impl FalconMamba {
    pub fn new(hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        let n_w = Tensor::ones(vec![hidden_size], dtype, device)?;
        let n_b = Tensor::zeros(vec![hidden_size], dtype, device)?;
        Ok(Self {
            norm_w: Parameter::new(n_w),
            norm_b: Parameter::new(n_b),
            fc1: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            fc2: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 1,
            )?,
            hidden_size,
            eps: 1e-5,
        })
    }

    pub fn from_weights(
        norm_w: Tensor, norm_b: Tensor,
        fc1_w: Tensor, fc1_b: Option<Tensor>,
        fc2_w: Tensor, fc2_b: Option<Tensor>,
        eps: f32,
    ) -> Result<Self> {
        let fc1 = Linear::new(fc1_w, fc1_b)?;
        let fc2 = Linear::new(fc2_w, fc2_b)?;
        let hidden_size = fc1.in_features();
        Ok(Self {
            norm_w: Parameter::new(norm_w),
            norm_b: Parameter::new(norm_b),
            fc1, fc2, hidden_size, eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 || x.dims()[2] != self.hidden_size {
            return Err(SynaptixError::Unsupported("FalconMamba: expects x [B, T, hidden]"));
        }
        let h = layer_norm(x, Some(&self.norm_w.tensor()), Some(&self.norm_b.tensor()), self.eps)?;
        let h = self.fc1.forward(&h)?;
        let h = h.silu()?;
        let h = self.fc2.forward(&h)?;
        x.add(&h)
    }
}
