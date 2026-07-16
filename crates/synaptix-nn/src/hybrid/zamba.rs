use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::norm::layer_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// Zamba / Zamba2 minimal block (pre-LN + Mamba-stub + shared-attn projection
/// + residual).
///
/// Полный Zamba (Zyphra) использует «глобальный» shared attention поверх
/// каскада Mamba-блоков. Здесь stub: 2 параллельных Linear-ветви
/// (`mamba_proj`, `shared_attn_proj`), их сумма проходит через `out_proj`.
///
/// `forward(x: [B, T, hidden])` →
/// `x + out_proj(silu(mamba_proj(LN(x))) + shared_attn_proj(LN(x)))`.
pub struct Zamba {
    pub norm_w: Parameter,
    pub norm_b: Parameter,
    pub mamba_proj: Linear,
    pub shared_attn_proj: Linear,
    pub out_proj: Linear,
    pub hidden_size: usize,
    pub eps: f32,
}

impl Zamba {
    pub fn new(hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            norm_w: Parameter::new(Tensor::ones(vec![hidden_size], dtype, device)?),
            norm_b: Parameter::new(Tensor::zeros(vec![hidden_size], dtype, device)?),
            mamba_proj: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            shared_attn_proj: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            out_proj: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 2,
            )?,
            hidden_size, eps: 1e-5,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        norm_w: Tensor, norm_b: Tensor,
        mamba_w: Tensor, mamba_b: Option<Tensor>,
        shared_attn_w: Tensor, shared_attn_b: Option<Tensor>,
        out_w: Tensor, out_b: Option<Tensor>,
        eps: f32,
    ) -> Result<Self> {
        let mamba_proj = Linear::new(mamba_w, mamba_b)?;
        let shared_attn_proj = Linear::new(shared_attn_w, shared_attn_b)?;
        let out_proj = Linear::new(out_w, out_b)?;
        let hidden_size = mamba_proj.in_features();
        Ok(Self {
            norm_w: Parameter::new(norm_w),
            norm_b: Parameter::new(norm_b),
            mamba_proj, shared_attn_proj, out_proj, hidden_size, eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 || x.dims()[2] != self.hidden_size {
            return Err(SynaptixError::Unsupported("Zamba: expects x [B, T, hidden]"));
        }
        let h = layer_norm(x, Some(&self.norm_w.tensor()), Some(&self.norm_b.tensor()), self.eps)?;
        let m = self.mamba_proj.forward(&h)?.silu()?;
        let s = self.shared_attn_proj.forward(&h)?;
        let sum = m.add(&s)?;
        let out = self.out_proj.forward(&sum)?;
        x.add(&out)
    }
}
