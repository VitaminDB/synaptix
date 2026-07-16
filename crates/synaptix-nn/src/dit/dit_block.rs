use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use synaptix_ops::activation::gelu_tanh;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::layer_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

pub struct DitBlock {
    pub norm1: Parameter,
    pub norm2: Parameter,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub ff1: Linear,
    pub ff2: Linear,
    pub adaln_modulation: Linear,
    pub hidden_size: usize,
    pub num_heads: usize,
}

impl DitBlock {
    pub fn new(hidden_size: usize, num_heads: usize, ffn_dim: usize, cond_dim: usize, device: Device, dtype: DType) -> Result<Self> {
        let n1 = crate::init::init_tensor(&[hidden_size], InitMethod::Ones, dtype, 0, device)?;
        let n2 = crate::init::init_tensor(&[hidden_size], InitMethod::Ones, dtype, 1, device)?;
        Ok(Self {
            norm1: Parameter::new(n1),
            norm2: Parameter::new(n2),
            q_proj: Linear::from_init(hidden_size, hidden_size, false, InitMethod::XavierUniform { fan_in: hidden_size, fan_out: hidden_size }, InitMethod::Zeros, device, dtype, 10)?,
            k_proj: Linear::from_init(hidden_size, hidden_size, false, InitMethod::XavierUniform { fan_in: hidden_size, fan_out: hidden_size }, InitMethod::Zeros, device, dtype, 11)?,
            v_proj: Linear::from_init(hidden_size, hidden_size, false, InitMethod::XavierUniform { fan_in: hidden_size, fan_out: hidden_size }, InitMethod::Zeros, device, dtype, 12)?,
            out_proj: Linear::from_init(hidden_size, hidden_size, false, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 13)?,
            ff1: Linear::from_init(hidden_size, ffn_dim, true, InitMethod::XavierUniform { fan_in: hidden_size, fan_out: ffn_dim }, InitMethod::Zeros, device, dtype, 20)?,
            ff2: Linear::from_init(ffn_dim, hidden_size, true, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 21)?,
            adaln_modulation: Linear::from_init(cond_dim, 6 * hidden_size, true, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 30)?,
            hidden_size,
            num_heads,
        })
    }

    pub fn from_weights(
        q_w: Tensor, k_w: Tensor, v_w: Tensor, o_w: Tensor,
        ff1_w: Tensor, ff1_b: Option<Tensor>,
        ff2_w: Tensor, ff2_b: Option<Tensor>,
        adaln_w: Tensor, adaln_b: Option<Tensor>,
        num_heads: usize,
    ) -> Result<Self> {
        let hidden_size = q_w.dims()[0];
        let device = q_w.device();
        let dtype = q_w.dtype();
        let n1 = crate::init::init_tensor(&[hidden_size], InitMethod::Ones, dtype, 0, device)?;
        let n2 = crate::init::init_tensor(&[hidden_size], InitMethod::Ones, dtype, 1, device)?;
        Ok(Self {
            norm1: Parameter::new(n1),
            norm2: Parameter::new(n2),
            q_proj: Linear::new(q_w, None)?,
            k_proj: Linear::new(k_w, None)?,
            v_proj: Linear::new(v_w, None)?,
            out_proj: Linear::new(o_w, None)?,
            ff1: Linear::new(ff1_w, ff1_b)?,
            ff2: Linear::new(ff2_w, ff2_b)?,
            adaln_modulation: Linear::new(adaln_w, adaln_b)?,
            hidden_size,
            num_heads,
        })
    }

    pub fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            compute_adaln(&self.adaln_modulation, cond, self.hidden_size)?;

        let h = layer_norm(x, None, None, 1e-6)?;
        let h = modulate(&h, &shift_msa, &scale_msa)?;
        let attn = self_attention(
            &h, &self.q_proj, &self.k_proj, &self.v_proj, &self.out_proj,
            self.num_heads, self.hidden_size,
        )?;
        let x = x.add(&gate_msa.unsqueeze(1)?.broadcast_mul(&attn)?)?;

        let h2 = layer_norm(&x, None, None, 1e-6)?;
        let h2 = modulate(&h2, &shift_mlp, &scale_mlp)?;
        let mlp_out = self.ff2.forward(&gelu_tanh(&self.ff1.forward(&h2)?)?)?;
        x.add(&gate_mlp.unsqueeze(1)?.broadcast_mul(&mlp_out)?)
    }
}

pub fn compute_adaln(
    modulation: &Linear,
    cond: &Tensor,
    hidden_size: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let cond_silu = cond.silu()?;
    let mod_out = modulation.forward(&cond_silu)?;
    let s1 = mod_out.narrow(1, 0, hidden_size)?.contiguous()?;
    let s2 = mod_out.narrow(1, hidden_size, hidden_size)?.contiguous()?;
    let s3 = mod_out.narrow(1, 2 * hidden_size, hidden_size)?.contiguous()?;
    let s4 = mod_out.narrow(1, 3 * hidden_size, hidden_size)?.contiguous()?;
    let s5 = mod_out.narrow(1, 4 * hidden_size, hidden_size)?.contiguous()?;
    let s6 = mod_out.narrow(1, 5 * hidden_size, hidden_size)?.contiguous()?;
    Ok((s1, s2, s3, s4, s5, s6))
}

pub fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let scale_b = scale.unsqueeze(1)?.affine(1.0, 1.0)?;
    let shift_b = shift.unsqueeze(1)?;
    x.broadcast_mul(&scale_b)?.broadcast_add(&shift_b)
}

fn self_attention(
    h: &Tensor,
    q_proj: &Linear,
    k_proj: &Linear,
    v_proj: &Linear,
    out_proj: &Linear,
    num_heads: usize,
    hidden_size: usize,
) -> Result<Tensor> {
    let head_dim = hidden_size / num_heads;
    let q = q_proj.forward(h)?;
    let k = k_proj.forward(h)?;
    let v = v_proj.forward(h)?;
    let b = q.dims()[0];
    let s = q.dims()[1];
    let q = q.reshape(vec![b, s, num_heads, head_dim])?.permute(vec![0, 2, 1, 3])?.contiguous()?;
    let k = k.reshape(vec![b, s, num_heads, head_dim])?.permute(vec![0, 2, 1, 3])?.contiguous()?;
    let v = v.reshape(vec![b, s, num_heads, head_dim])?.permute(vec![0, 2, 1, 3])?.contiguous()?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let attn = scaled_dot_attention(&q, &k, &v, scale, None)?;
    let attn = attn.permute(vec![0, 2, 1, 3])?.contiguous()?.reshape(vec![b, s, hidden_size])?;
    out_proj.forward(&attn)
}
