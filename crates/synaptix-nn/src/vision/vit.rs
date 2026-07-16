use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::activation::gelu_tanh;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::layer_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

pub struct ViTBlock {
    pub norm1_w: Parameter,
    pub norm1_b: Parameter,
    pub norm2_w: Parameter,
    pub norm2_b: Parameter,
    pub attn_q: Linear,
    pub attn_k: Linear,
    pub attn_v: Linear,
    pub attn_out: Linear,
    pub ff1: Linear,
    pub ff2: Linear,
    pub num_heads: usize,
    pub hidden_size: usize,
}

impl ViTBlock {
    pub fn from_weights(
        n1_w: Tensor, n1_b: Tensor, n2_w: Tensor, n2_b: Tensor,
        q_w: Tensor, k_w: Tensor, v_w: Tensor, o_w: Tensor,
        ff1_w: Tensor, ff1_b: Option<Tensor>,
        ff2_w: Tensor, ff2_b: Option<Tensor>,
        num_heads: usize,
    ) -> Result<Self> {
        let hidden_size = q_w.dims()[0];
        Ok(Self {
            norm1_w: Parameter::new(n1_w),
            norm1_b: Parameter::new(n1_b),
            norm2_w: Parameter::new(n2_w),
            norm2_b: Parameter::new(n2_b),
            attn_q: Linear::new(q_w, None)?,
            attn_k: Linear::new(k_w, None)?,
            attn_v: Linear::new(v_w, None)?,
            attn_out: Linear::new(o_w, None)?,
            ff1: Linear::new(ff1_w, ff1_b)?,
            ff2: Linear::new(ff2_w, ff2_b)?,
            num_heads,
            hidden_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = layer_norm(x, Some(&self.norm1_w.tensor()), Some(&self.norm1_b.tensor()), 1e-6)?;
        let head_dim = self.hidden_size / self.num_heads;
        let q = self.attn_q.forward(&h)?;
        let k = self.attn_k.forward(&h)?;
        let v = self.attn_v.forward(&h)?;
        let b = q.dims()[0];
        let s = q.dims()[1];
        let q = q.reshape(vec![b, s, self.num_heads, head_dim])?.permute(vec![0, 2, 1, 3])?.contiguous()?;
        let k = k.reshape(vec![b, s, self.num_heads, head_dim])?.permute(vec![0, 2, 1, 3])?.contiguous()?;
        let v = v.reshape(vec![b, s, self.num_heads, head_dim])?.permute(vec![0, 2, 1, 3])?.contiguous()?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn = scaled_dot_attention(&q, &k, &v, scale, None)?;
        let attn = attn.permute(vec![0, 2, 1, 3])?.contiguous()?.reshape(vec![b, s, self.hidden_size])?;
        let attn_out = self.attn_out.forward(&attn)?;
        let x = x.add(&attn_out)?;

        let h2 = layer_norm(&x, Some(&self.norm2_w.tensor()), Some(&self.norm2_b.tensor()), 1e-6)?;
        let mlp = self.ff2.forward(&gelu_tanh(&self.ff1.forward(&h2)?)?)?;
        x.add(&mlp)
    }
}

pub struct VisionTransformer {
    pub patch_embed: Linear,
    pub blocks: Vec<ViTBlock>,
    pub norm_w: Parameter,
    pub norm_b: Parameter,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub num_patches: usize,
    pub in_channels: usize,
}

impl VisionTransformer {
    pub fn new(
        in_channels: usize,
        patch_size: usize,
        image_size: usize,
        hidden_size: usize,
        num_heads: usize,
        num_layers: usize,
        ffn_dim: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let patch_dim = patch_size * patch_size * in_channels;
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let norm_w = crate::init::init_tensor(&[hidden_size], InitMethod::Ones, dtype, 0, device)?;
        let norm_b = crate::init::init_tensor(&[hidden_size], InitMethod::Zeros, dtype, 1, device)?;
        let _ = (num_layers, ffn_dim);
        Ok(Self {
            patch_embed: Linear::from_init(patch_dim, hidden_size, true, InitMethod::XavierUniform { fan_in: patch_dim, fan_out: hidden_size }, InitMethod::Zeros, device, dtype, 0)?,
            blocks: Vec::new(),
            norm_w: Parameter::new(norm_w),
            norm_b: Parameter::new(norm_b),
            hidden_size,
            num_heads,
            patch_size,
            num_patches,
            in_channels,
        })
    }

    pub fn from_weights(
        in_channels: usize,
        patch_size: usize,
        patch_w: Tensor, patch_b: Option<Tensor>,
        blocks: Vec<ViTBlock>,
        norm_w: Tensor, norm_b: Tensor,
        num_heads: usize,
    ) -> Result<Self> {
        let patch_embed = Linear::new(patch_w, patch_b)?;
        let hidden_size = patch_embed.out_features();
        Ok(Self {
            patch_embed,
            blocks,
            norm_w: Parameter::new(norm_w),
            norm_b: Parameter::new(norm_b),
            hidden_size,
            num_heads,
            patch_size,
            num_patches: 0,
            in_channels,
        })
    }

    fn patchify(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 4 {
            return Err(SynaptixError::Unsupported("vit: input must be [B, C, H, W]"));
        }
        let (b, c, h, w) = (x.dims()[0], x.dims()[1], x.dims()[2], x.dims()[3]);
        let p = self.patch_size;
        if h % p != 0 || w % p != 0 {
            return Err(SynaptixError::Unsupported("vit: H and W must divide by patch_size"));
        }
        let nh = h / p;
        let nw = w / p;
        let reshaped = x.reshape(vec![b, c, nh, p, nw, p])?;
        let permuted = reshaped.permute(vec![0, 2, 4, 1, 3, 5])?.contiguous()?;
        permuted.reshape(vec![b, nh * nw, c * p * p])
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let tokens = self.patchify(x)?;
        let mut h = self.patch_embed.forward(&tokens)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        layer_norm(&h, Some(&self.norm_w.tensor()), Some(&self.norm_b.tensor()), 1e-6)
    }
}
