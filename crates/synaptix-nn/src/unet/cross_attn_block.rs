use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::layer_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// UNet cross-attention блок: Pre-LN MHA (Q из x, K/V из context) + out_proj + residual.
///
/// `x: [B, T, hidden_size]`, `context: [B, S_ctx, context_dim]` → output `[B, T, hidden_size]`.
pub struct UNetCrossAttnBlock {
    pub norm_w: Parameter,
    pub norm_b: Parameter,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub context_dim: usize,
    pub eps: f32,
}

impl UNetCrossAttnBlock {
    pub fn new(
        hidden_size: usize,
        context_dim: usize,
        num_heads: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        if hidden_size % num_heads != 0 {
            return Err(SynaptixError::Unsupported(
                "UNetCrossAttnBlock: hidden_size must be divisible by num_heads",
            ));
        }
        let head_dim = hidden_size / num_heads;
        Ok(Self {
            norm_w: Parameter::new(Tensor::ones(vec![hidden_size], dtype, device)?),
            norm_b: Parameter::new(Tensor::zeros(vec![hidden_size], dtype, device)?),
            q_proj: Linear::from_init(
                hidden_size, hidden_size, false,
                InitMethod::XavierUniform { fan_in: hidden_size, fan_out: hidden_size },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            k_proj: Linear::from_init(
                context_dim, hidden_size, false,
                InitMethod::XavierUniform { fan_in: context_dim, fan_out: hidden_size },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            v_proj: Linear::from_init(
                context_dim, hidden_size, false,
                InitMethod::XavierUniform { fan_in: context_dim, fan_out: hidden_size },
                InitMethod::Zeros, device, dtype, 2,
            )?,
            out_proj: Linear::from_init(
                hidden_size, hidden_size, false,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 3,
            )?,
            num_heads, head_dim, hidden_size, context_dim, eps: 1e-5,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        norm_w: Tensor, norm_b: Tensor,
        q_w: Tensor, k_w: Tensor, v_w: Tensor, o_w: Tensor,
        num_heads: usize, eps: f32,
    ) -> Result<Self> {
        let q_proj = Linear::new(q_w, None)?;
        let k_proj = Linear::new(k_w, None)?;
        let v_proj = Linear::new(v_w, None)?;
        let out_proj = Linear::new(o_w, None)?;
        let hidden_size = q_proj.out_features();
        let context_dim = k_proj.in_features();
        if hidden_size % num_heads != 0 {
            return Err(SynaptixError::Unsupported(
                "UNetCrossAttnBlock::from_weights: hidden_size must be divisible by num_heads",
            ));
        }
        Ok(Self {
            norm_w: Parameter::new(norm_w),
            norm_b: Parameter::new(norm_b),
            q_proj, k_proj, v_proj, out_proj,
            num_heads,
            head_dim: hidden_size / num_heads,
            hidden_size, context_dim, eps,
        })
    }

    pub fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 || x.dims()[2] != self.hidden_size {
            return Err(SynaptixError::Unsupported("UNetCrossAttnBlock: expects x [B, T, hidden]"));
        }
        if context.rank() != 3 || context.dims()[2] != self.context_dim {
            return Err(SynaptixError::Unsupported("UNetCrossAttnBlock: expects context [B, S_ctx, context_dim]"));
        }
        let b = x.dims()[0];
        let s = x.dims()[1];
        let s_ctx = context.dims()[1];

        let h = layer_norm(x, Some(&self.norm_w.tensor()), Some(&self.norm_b.tensor()), self.eps)?;
        let q = self.q_proj.forward(&h)?
            .reshape(vec![b, s, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?.contiguous()?;
        let k = self.k_proj.forward(context)?
            .reshape(vec![b, s_ctx, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?.contiguous()?;
        let v = self.v_proj.forward(context)?
            .reshape(vec![b, s_ctx, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn = scaled_dot_attention(&q, &k, &v, scale, None)?;
        let merged = attn.permute(vec![0, 2, 1, 3])?.contiguous()?
            .reshape(vec![b, s, self.hidden_size])?;
        let out = self.out_proj.forward(&merged)?;
        x.add(&out)
    }
}
