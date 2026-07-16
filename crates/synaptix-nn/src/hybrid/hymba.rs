use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::norm::layer_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// Hymba minimal block (pre-LN parallel ветки → fuse + residual).
///
/// Полная Hymba (NVIDIA) параллельно прогоняет attention и SSM-ветви на одном
/// входе, конкатенирует и проецирует обратно. Здесь stub: две независимые
/// `attn_proj`/`ssm_proj` Linear-ветки, конкат вдоль hidden, потом fuse.
///
/// `forward(x: [B, T, hidden])` →
/// `x + fuse(cat([SiLU(attn_proj(LN(x))), SiLU(ssm_proj(LN(x)))], dim=-1))`.
pub struct Hymba {
    pub norm_w: Parameter,
    pub norm_b: Parameter,
    pub attn_proj: Linear,
    pub ssm_proj: Linear,
    pub fuse: Linear,
    pub hidden_size: usize,
    pub eps: f32,
}

impl Hymba {
    pub fn new(hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            norm_w: Parameter::new(Tensor::ones(vec![hidden_size], dtype, device)?),
            norm_b: Parameter::new(Tensor::zeros(vec![hidden_size], dtype, device)?),
            attn_proj: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            ssm_proj: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            fuse: Linear::from_init(
                hidden_size * 2, hidden_size, true,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 2,
            )?,
            hidden_size, eps: 1e-5,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        norm_w: Tensor, norm_b: Tensor,
        attn_proj_w: Tensor, attn_proj_b: Option<Tensor>,
        ssm_proj_w: Tensor, ssm_proj_b: Option<Tensor>,
        fuse_w: Tensor, fuse_b: Option<Tensor>,
        eps: f32,
    ) -> Result<Self> {
        let attn_proj = Linear::new(attn_proj_w, attn_proj_b)?;
        let ssm_proj = Linear::new(ssm_proj_w, ssm_proj_b)?;
        let fuse = Linear::new(fuse_w, fuse_b)?;
        let hidden_size = attn_proj.in_features();
        Ok(Self {
            norm_w: Parameter::new(norm_w),
            norm_b: Parameter::new(norm_b),
            attn_proj, ssm_proj, fuse, hidden_size, eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 || x.dims()[2] != self.hidden_size {
            return Err(SynaptixError::Unsupported("Hymba: expects x [B, T, hidden]"));
        }
        let h = layer_norm(x, Some(&self.norm_w.tensor()), Some(&self.norm_b.tensor()), self.eps)?;
        let a = self.attn_proj.forward(&h)?.silu()?;
        let s = self.ssm_proj.forward(&h)?.silu()?;
        let cat = Tensor::cat(&[&a.contiguous()?, &s.contiguous()?], 2)?;
        let out = self.fuse.forward(&cat)?;
        x.add(&out)
    }
}
