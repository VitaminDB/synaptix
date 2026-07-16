use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::norm::layer_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// Conformer macaron-style FeedForward Module.
///
/// Pre-LN → fc1 → swish (SiLU) → fc2, applied as residual with **half-step**
/// scale `0.5`. То есть `output = x + 0.5 · fc2(silu(fc1(LN(x))))`.
///
/// Совпадает с torchaudio.models.Conformer._FeedForwardModule и ESPnet
/// `PositionwiseFeedForward` обёрнутый в `MacaronStyle`.
pub struct FeedForwardModule {
    pub norm_w: Parameter,
    pub norm_b: Parameter,
    pub fc1: Linear,
    pub fc2: Linear,
    pub hidden_size: usize,
    pub ffn_dim: usize,
    pub eps: f32,
}

impl FeedForwardModule {
    pub fn new(hidden_size: usize, ffn_dim: usize, device: Device, dtype: DType) -> Result<Self> {
        let norm_w = Tensor::ones(vec![hidden_size], dtype, device)?;
        let norm_b = Tensor::zeros(vec![hidden_size], dtype, device)?;
        Ok(Self {
            norm_w: Parameter::new(norm_w).with_name("norm.weight"),
            norm_b: Parameter::new(norm_b).with_name("norm.bias"),
            fc1: Linear::from_init(
                hidden_size, ffn_dim, true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            fc2: Linear::from_init(
                ffn_dim, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: ffn_dim, a: 0.0 },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            hidden_size,
            ffn_dim,
            eps: 1e-5,
        })
    }

    /// `norm_w`/`norm_b`: `[hidden_size]`. `fc1_w`: `[ffn_dim, hidden_size]`,
    /// `fc1_b`: `[ffn_dim]`. `fc2_w`: `[hidden_size, ffn_dim]`, `fc2_b`: `[hidden_size]`.
    pub fn from_weights(
        norm_w: Tensor, norm_b: Tensor,
        fc1_w: Tensor, fc1_b: Option<Tensor>,
        fc2_w: Tensor, fc2_b: Option<Tensor>,
        eps: f32,
    ) -> Result<Self> {
        let fc1 = Linear::new(fc1_w, fc1_b)?;
        let fc2 = Linear::new(fc2_w, fc2_b)?;
        let hidden_size = fc1.in_features();
        let ffn_dim = fc1.out_features();
        if fc2.in_features() != ffn_dim || fc2.out_features() != hidden_size {
            return Err(SynaptixError::Unsupported(
                "FeedForwardModule::from_weights: fc2 must be [hidden, ffn]",
            ));
        }
        if norm_w.rank() != 1 || norm_w.dims()[0] != hidden_size
            || norm_b.rank() != 1 || norm_b.dims()[0] != hidden_size
        {
            return Err(SynaptixError::Unsupported(
                "FeedForwardModule::from_weights: norm_w/norm_b must be [hidden_size]",
            ));
        }
        Ok(Self {
            norm_w: Parameter::new(norm_w).with_name("norm.weight"),
            norm_b: Parameter::new(norm_b).with_name("norm.bias"),
            fc1, fc2, hidden_size, ffn_dim, eps,
        })
    }

    /// `x: [..., hidden_size]` → same shape. Residual прибавляется внутри.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = layer_norm(x, Some(&self.norm_w.tensor()), Some(&self.norm_b.tensor()), self.eps)?;
        let h = self.fc1.forward(&h)?.silu()?;
        let h = self.fc2.forward(&h)?;
        x.add(&h.affine(0.5, 0.0)?)
    }
}
