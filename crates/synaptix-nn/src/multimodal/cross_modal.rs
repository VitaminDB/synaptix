use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::softmax::scaled_dot_attention;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

/// Cross-modal multi-head attention: `Q ← x` (query modality),
/// `K, V ← context` (context modality). K/V projections принимают
/// `context_dim` и приводят к `query_dim` (стандартная схема для CLIP-LLM
/// и Flamingo cross-attention).
pub struct CrossModalAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub query_dim: usize,
    pub context_dim: usize,
}

impl CrossModalAttention {
    pub fn new(
        query_dim: usize, context_dim: usize, num_heads: usize,
        device: Device, dtype: DType,
    ) -> Result<Self> {
        if query_dim % num_heads != 0 {
            return Err(SynaptixError::Unsupported(
                "CrossModalAttention: query_dim must be divisible by num_heads",
            ));
        }
        let head_dim = query_dim / num_heads;
        Ok(Self {
            q_proj: Linear::from_init(
                query_dim, query_dim, false,
                InitMethod::XavierUniform { fan_in: query_dim, fan_out: query_dim },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            k_proj: Linear::from_init(
                context_dim, query_dim, false,
                InitMethod::XavierUniform { fan_in: context_dim, fan_out: query_dim },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            v_proj: Linear::from_init(
                context_dim, query_dim, false,
                InitMethod::XavierUniform { fan_in: context_dim, fan_out: query_dim },
                InitMethod::Zeros, device, dtype, 2,
            )?,
            out_proj: Linear::from_init(
                query_dim, query_dim, false,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 3,
            )?,
            num_heads,
            head_dim,
            query_dim,
            context_dim,
        })
    }

    pub fn from_weights(
        q_w: Tensor, k_w: Tensor, v_w: Tensor, o_w: Tensor, num_heads: usize,
    ) -> Result<Self> {
        let q_proj = Linear::new(q_w, None)?;
        let k_proj = Linear::new(k_w, None)?;
        let v_proj = Linear::new(v_w, None)?;
        let out_proj = Linear::new(o_w, None)?;
        let query_dim = q_proj.out_features();
        let context_dim = k_proj.in_features();
        if query_dim % num_heads != 0 {
            return Err(SynaptixError::Unsupported(
                "CrossModalAttention::from_weights: query_dim must be divisible by num_heads",
            ));
        }
        let head_dim = query_dim / num_heads;
        Ok(Self {
            q_proj, k_proj, v_proj, out_proj,
            num_heads, head_dim, query_dim, context_dim,
        })
    }

    /// `x: [B, Sq, query_dim]`, `context: [B, Sk, context_dim]` →
    /// `[B, Sq, query_dim]`. `mask` (опц.) совместима с
    /// `scaled_dot_attention`.
    pub fn forward(&self, x: &Tensor, context: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        if x.rank() != 3 || context.rank() != 3 {
            return Err(SynaptixError::Unsupported(
                "CrossModalAttention: x [B,Sq,Dq], context [B,Sk,Dc]",
            ));
        }
        let (b, sq) = (x.dims()[0], x.dims()[1]);
        let sk = context.dims()[1];

        let q = self.q_proj.forward(x)?
            .reshape(vec![b, sq, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?.contiguous()?;
        let k = self.k_proj.forward(context)?
            .reshape(vec![b, sk, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?.contiguous()?;
        let v = self.v_proj.forward(context)?
            .reshape(vec![b, sk, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn = scaled_dot_attention(&q, &k, &v, scale, mask)?;
        let merged = attn.permute(vec![0, 2, 1, 3])?.contiguous()?
            .reshape(vec![b, sq, self.query_dim])?;
        self.out_proj.forward(&merged)
    }
}
