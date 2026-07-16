use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use synaptix_ops::attention::softmax::gqa_attention;
use synaptix_ops::norm::layer_norm;
use synaptix_ops::activation::gelu_tanh;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

pub struct TransformerBlock {
    pub norm1_weight: Parameter,
    pub norm1_bias: Parameter,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub norm2_weight: Parameter,
    pub norm2_bias: Parameter,
    pub fc1: Linear,
    pub fc2: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl TransformerBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        ffn_dim: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let w1 = crate::init::init_tensor(&[hidden_size], InitMethod::Ones, dtype, 0, device)?;
        let b1 = crate::init::init_tensor(&[hidden_size], InitMethod::Zeros, dtype, 1, device)?;
        let w2 = crate::init::init_tensor(&[hidden_size], InitMethod::Ones, dtype, 2, device)?;
        let b2 = crate::init::init_tensor(&[hidden_size], InitMethod::Zeros, dtype, 3, device)?;
        Ok(Self {
            norm1_weight: Parameter::new(w1),
            norm1_bias: Parameter::new(b1),
            q_proj: Linear::from_init(hidden_size, hidden_size, false, InitMethod::XavierUniform { fan_in: hidden_size, fan_out: hidden_size }, InitMethod::Zeros, device, dtype, 10)?,
            k_proj: Linear::from_init(hidden_size, hidden_size, false, InitMethod::XavierUniform { fan_in: hidden_size, fan_out: hidden_size }, InitMethod::Zeros, device, dtype, 11)?,
            v_proj: Linear::from_init(hidden_size, hidden_size, false, InitMethod::XavierUniform { fan_in: hidden_size, fan_out: hidden_size }, InitMethod::Zeros, device, dtype, 12)?,
            out_proj: Linear::from_init(hidden_size, hidden_size, false, InitMethod::XavierUniform { fan_in: hidden_size, fan_out: hidden_size }, InitMethod::Zeros, device, dtype, 13)?,
            norm2_weight: Parameter::new(w2),
            norm2_bias: Parameter::new(b2),
            fc1: Linear::from_init(hidden_size, ffn_dim, true, InitMethod::XavierUniform { fan_in: hidden_size, fan_out: ffn_dim }, InitMethod::Zeros, device, dtype, 20)?,
            fc2: Linear::from_init(ffn_dim, hidden_size, true, InitMethod::XavierUniform { fan_in: ffn_dim, fan_out: hidden_size }, InitMethod::Zeros, device, dtype, 21)?,
            num_heads,
            head_dim,
        })
    }

    pub fn from_weights(
        n1_w: Tensor, n1_b: Tensor,
        q_w: Tensor, k_w: Tensor, v_w: Tensor, o_w: Tensor,
        n2_w: Tensor, n2_b: Tensor,
        fc1_w: Tensor, fc1_b: Option<Tensor>,
        fc2_w: Tensor, fc2_b: Option<Tensor>,
        num_heads: usize,
    ) -> Result<Self> {
        let hidden_size = q_w.dims()[0];
        let head_dim = hidden_size / num_heads;
        Ok(Self {
            norm1_weight: Parameter::new(n1_w),
            norm1_bias: Parameter::new(n1_b),
            q_proj: Linear::new(q_w, None)?,
            k_proj: Linear::new(k_w, None)?,
            v_proj: Linear::new(v_w, None)?,
            out_proj: Linear::new(o_w, None)?,
            norm2_weight: Parameter::new(n2_w),
            norm2_bias: Parameter::new(n2_b),
            fc1: Linear::new(fc1_w, fc1_b)?,
            fc2: Linear::new(fc2_w, fc2_b)?,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_mask(x, None)
    }

    pub fn forward_with_mask(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let w1 = self.norm1_weight.tensor();
        let b1 = self.norm1_bias.tensor();
        let h = layer_norm(x, Some(&w1), Some(&b1), 1e-5)?;

        let q = self.q_proj.forward(&h)?;
        let k = self.k_proj.forward(&h)?;
        let v = self.v_proj.forward(&h)?;

        let q = reshape_for_attn(&q, self.num_heads, self.head_dim)?;
        let k = reshape_for_attn(&k, self.num_heads, self.head_dim)?;
        let v = reshape_for_attn(&v, self.num_heads, self.head_dim)?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn = gqa_attention(&q, &k, &v, scale, mask)?; // [B, H, S, D]

        let attn = reshape_from_attn(&attn, self.num_heads, self.head_dim)?;
        let attn_out = self.out_proj.forward(&attn)?;
        let x = x.add(&attn_out)?;

        let w2 = self.norm2_weight.tensor();
        let b2 = self.norm2_bias.tensor();
        let h2 = layer_norm(&x, Some(&w2), Some(&b2), 1e-5)?;

        let fc1_out = gelu_tanh(&self.fc1.forward(&h2)?)?;
        let fc2_out = self.fc2.forward(&fc1_out)?;
        x.add(&fc2_out)
    }
}

fn reshape_for_attn(x: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let rank = dims.len();
    let mut new_dims: Vec<usize> = dims[..rank - 1].to_vec();
    new_dims.push(num_heads);
    new_dims.push(head_dim);
    let x = x.reshape(new_dims)?;
    let r = x.rank();
    let mut perm: Vec<usize> = (0..r - 3).collect();
    perm.extend_from_slice(&[r - 2, r - 3, r - 1]);
    x.permute(perm)?.contiguous()
}

fn reshape_from_attn(x: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let r = x.rank();
    let mut perm: Vec<usize> = (0..r - 3).collect();
    perm.extend_from_slice(&[r - 2, r - 3, r - 1]);
    let x = x.permute(perm)?.contiguous()?;
    let dims = x.dims().to_vec();
    let rank = dims.len();
    let mut new_dims: Vec<usize> = dims[..rank - 2].to_vec();
    new_dims.push(num_heads * head_dim);
    x.reshape(new_dims)
}
