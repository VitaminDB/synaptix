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

/// VLM block (Flamingo / LLaVA cross-attention style):
/// `pre_norm → cross-attn(Q=x, K/V=context fused KV) → +skip`.
///
/// `cross_attn_kv` — это **fused** Linear `context_dim → 2*hidden_size` (K и
/// V в одном matmul, как в HF `LlamaAttention` fused QKV). Pre-norm на x.
pub struct VlmBlock {
    pub norm_w: Parameter,
    pub norm_b: Parameter,
    pub cross_attn_q: Linear,
    pub cross_attn_kv: Linear,
    pub cross_attn_out: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub context_dim: usize,
    pub eps: f32,
}

impl VlmBlock {
    pub fn new(
        hidden_size: usize, context_dim: usize, num_heads: usize,
        device: Device, dtype: DType,
    ) -> Result<Self> {
        if hidden_size % num_heads != 0 {
            return Err(SynaptixError::Unsupported(
                "VlmBlock: hidden_size must be divisible by num_heads",
            ));
        }
        Ok(Self {
            norm_w: Parameter::new(Tensor::ones(vec![hidden_size], dtype, device)?),
            norm_b: Parameter::new(Tensor::zeros(vec![hidden_size], dtype, device)?),
            cross_attn_q: Linear::from_init(
                hidden_size, hidden_size, false,
                InitMethod::XavierUniform { fan_in: hidden_size, fan_out: hidden_size },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            cross_attn_kv: Linear::from_init(
                context_dim, hidden_size * 2, false,
                InitMethod::XavierUniform { fan_in: context_dim, fan_out: hidden_size * 2 },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            cross_attn_out: Linear::from_init(
                hidden_size, hidden_size, false,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 2,
            )?,
            num_heads,
            head_dim: hidden_size / num_heads,
            hidden_size,
            context_dim,
            eps: 1e-5,
        })
    }

    pub fn from_weights(
        norm_w: Tensor, norm_b: Tensor,
        q_w: Tensor, kv_w: Tensor, o_w: Tensor,
        num_heads: usize, eps: f32,
    ) -> Result<Self> {
        let cross_attn_q = Linear::new(q_w, None)?;
        let cross_attn_kv = Linear::new(kv_w, None)?;
        let cross_attn_out = Linear::new(o_w, None)?;
        let hidden_size = cross_attn_q.out_features();
        let context_dim = cross_attn_kv.in_features();
        if hidden_size % num_heads != 0 {
            return Err(SynaptixError::Unsupported(
                "VlmBlock::from_weights: hidden_size must be divisible by num_heads",
            ));
        }
        Ok(Self {
            norm_w: Parameter::new(norm_w),
            norm_b: Parameter::new(norm_b),
            cross_attn_q,
            cross_attn_kv,
            cross_attn_out,
            num_heads,
            head_dim: hidden_size / num_heads,
            hidden_size,
            context_dim,
            eps,
        })
    }

    /// `x: [B, Sq, hidden_size]`, `context: [B, Sk, context_dim]` →
    /// `[B, Sq, hidden_size]`.
    pub fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 || context.rank() != 3 {
            return Err(SynaptixError::Unsupported(
                "VlmBlock: x [B,Sq,H], context [B,Sk,Dc]",
            ));
        }
        let (b, sq) = (x.dims()[0], x.dims()[1]);
        let sk = context.dims()[1];

        let normed = layer_norm(
            x,
            Some(&self.norm_w.tensor()),
            Some(&self.norm_b.tensor()),
            self.eps,
        )?;

        let q = self.cross_attn_q.forward(&normed)?
            .reshape(vec![b, sq, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?.contiguous()?;
        let kv = self.cross_attn_kv.forward(context)?;
        let k = kv.narrow(2, 0, self.hidden_size)?.contiguous()?
            .reshape(vec![b, sk, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?.contiguous()?;
        let v = kv.narrow(2, self.hidden_size, self.hidden_size)?.contiguous()?
            .reshape(vec![b, sk, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn = scaled_dot_attention(&q, &k, &v, scale, None)?;
        let merged = attn.permute(vec![0, 2, 1, 3])?.contiguous()?
            .reshape(vec![b, sq, self.hidden_size])?;
        let out = self.cross_attn_out.forward(&merged)?;
        x.add(&out)
    }
}
