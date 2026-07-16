use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::layer_norm;

use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

pub struct ConformerBlock {
    pub ff1_n_w: Parameter, pub ff1_n_b: Parameter,
    pub ff1_in: Linear, pub ff1_out: Linear,
    pub attn_n_w: Parameter, pub attn_n_b: Parameter,
    pub q_proj: Linear, pub k_proj: Linear, pub v_proj: Linear, pub o_proj: Linear,
    pub ff2_n_w: Parameter, pub ff2_n_b: Parameter,
    pub ff2_in: Linear, pub ff2_out: Linear,
    pub final_n_w: Parameter, pub final_n_b: Parameter,
    pub num_heads: usize,
    pub hidden_size: usize,
}

impl ConformerBlock {
    pub fn from_weights(
        ff1_n_w: Tensor, ff1_n_b: Tensor,
        ff1_in_w: Tensor, ff1_in_b: Tensor,
        ff1_out_w: Tensor, ff1_out_b: Tensor,
        attn_n_w: Tensor, attn_n_b: Tensor,
        q_w: Tensor, k_w: Tensor, v_w: Tensor, o_w: Tensor,
        ff2_n_w: Tensor, ff2_n_b: Tensor,
        ff2_in_w: Tensor, ff2_in_b: Tensor,
        ff2_out_w: Tensor, ff2_out_b: Tensor,
        final_n_w: Tensor, final_n_b: Tensor,
        num_heads: usize,
    ) -> Result<Self> {
        let hidden_size = q_w.dims()[0];
        Ok(Self {
            ff1_n_w: Parameter::new(ff1_n_w), ff1_n_b: Parameter::new(ff1_n_b),
            ff1_in: Linear::new(ff1_in_w, Some(ff1_in_b))?,
            ff1_out: Linear::new(ff1_out_w, Some(ff1_out_b))?,
            attn_n_w: Parameter::new(attn_n_w), attn_n_b: Parameter::new(attn_n_b),
            q_proj: Linear::new(q_w, None)?,
            k_proj: Linear::new(k_w, None)?,
            v_proj: Linear::new(v_w, None)?,
            o_proj: Linear::new(o_w, None)?,
            ff2_n_w: Parameter::new(ff2_n_w), ff2_n_b: Parameter::new(ff2_n_b),
            ff2_in: Linear::new(ff2_in_w, Some(ff2_in_b))?,
            ff2_out: Linear::new(ff2_out_w, Some(ff2_out_b))?,
            final_n_w: Parameter::new(final_n_w), final_n_b: Parameter::new(final_n_b),
            num_heads,
            hidden_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let head_dim = self.hidden_size / self.num_heads;

        let h = layer_norm(x, Some(&self.ff1_n_w.tensor()), Some(&self.ff1_n_b.tensor()), 1e-5)?;
        let ff1 = self.ff1_out.forward(&self.ff1_in.forward(&h)?.silu()?)?;
        let x = x.add(&ff1.affine(0.5, 0.0)?)?;

        let h = layer_norm(&x, Some(&self.attn_n_w.tensor()), Some(&self.attn_n_b.tensor()), 1e-5)?;
        let q = self.q_proj.forward(&h)?;
        let k = self.k_proj.forward(&h)?;
        let v = self.v_proj.forward(&h)?;
        let b = q.dims()[0];
        let s = q.dims()[1];
        let q = q.reshape(vec![b, s, self.num_heads, head_dim])?.permute(vec![0, 2, 1, 3])?.contiguous()?;
        let k = k.reshape(vec![b, s, self.num_heads, head_dim])?.permute(vec![0, 2, 1, 3])?.contiguous()?;
        let v = v.reshape(vec![b, s, self.num_heads, head_dim])?.permute(vec![0, 2, 1, 3])?.contiguous()?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn = scaled_dot_attention(&q, &k, &v, scale, None)?;
        let attn = attn.permute(vec![0, 2, 1, 3])?.contiguous()?.reshape(vec![b, s, self.hidden_size])?;
        let attn = self.o_proj.forward(&attn)?;
        let x = x.add(&attn)?;

        let h = layer_norm(&x, Some(&self.ff2_n_w.tensor()), Some(&self.ff2_n_b.tensor()), 1e-5)?;
        let ff2 = self.ff2_out.forward(&self.ff2_in.forward(&h)?.silu()?)?;
        let x = x.add(&ff2.affine(0.5, 0.0)?)?;

        layer_norm(&x, Some(&self.final_n_w.tensor()), Some(&self.final_n_b.tensor()), 1e-5)
    }
}

pub struct ConformerEnc {
    pub blocks: Vec<ConformerBlock>,
    pub hidden_size: usize,
}

impl ConformerEnc {
    pub fn new(in_channels: usize, hidden_size: usize, _device: Device, _dtype: DType) -> Result<Self> {
        let _ = in_channels;
        Ok(Self { blocks: Vec::new(), hidden_size })
    }

    pub fn from_weights(blocks: Vec<ConformerBlock>) -> Result<Self> {
        if blocks.is_empty() {
            return Err(SynaptixError::Unsupported("conformer_enc: need at least 1 block"));
        }
        let hidden_size = blocks[0].hidden_size;
        Ok(Self { blocks, hidden_size })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        Ok(h)
    }
}
