use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

/// VLM Projector — 2-layer MLP с GELU-exact (как LLaVA/Qwen-VL).
///
/// `fc1 (in→hidden) → GELU(exact) → fc2 (hidden→out)`. Проекция
/// image-features из vision encoder в hidden space LLM.
pub struct MlpProjector {
    pub fc1: Linear,
    pub fc2: Linear,
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub out_dim: usize,
}

impl MlpProjector {
    pub fn new(
        in_dim: usize, hidden_dim: usize, out_dim: usize,
        device: Device, dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            fc1: Linear::from_init(
                in_dim, hidden_dim, true,
                InitMethod::KaimingUniform { fan_in: in_dim, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            fc2: Linear::from_init(
                hidden_dim, out_dim, true,
                InitMethod::KaimingUniform { fan_in: hidden_dim, a: 0.0 },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            in_dim,
            hidden_dim,
            out_dim,
        })
    }

    pub fn from_weights(
        fc1_w: Tensor, fc1_b: Option<Tensor>,
        fc2_w: Tensor, fc2_b: Option<Tensor>,
    ) -> Result<Self> {
        let fc1 = Linear::new(fc1_w, fc1_b)?;
        let fc2 = Linear::new(fc2_w, fc2_b)?;
        let in_dim = fc1.in_features();
        let hidden_dim = fc1.out_features();
        let out_dim = fc2.out_features();
        Ok(Self { fc1, fc2, in_dim, hidden_dim, out_dim })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?.gelu_exact()?;
        self.fc2.forward(&h)
    }
}
