
use synaptix_core::{dtype::DType, tensor::Tensor};
use synaptix_nn::linear::Linear;
use synaptix_nn::module::Module;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::pos::rope_cache::RopeCache;

use crate::loader::CompLoader;
use crate::AceError;

type R<T> = Result<T, AceError>;

pub(crate) fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> R<Tensor> {
    let r = x.rank();
    let ms = x.sqr()?.mean_keepdim(r - 1)?;
    let inv = ms.affine(1.0, eps)?.sqrt()?.recip()?;
    Ok(x.broadcast_mul(&inv)?.broadcast_mul(w)?)
}

/// Bidirectional sliding-window additive attention mask `[1,1,s,s]`:
/// `abs(i-j) <= window -> 0`, else `-inf`. Used on the even-indexed lyric/timbre
/// encoder layers (Python layer_types[i]=="sliding_attention" for i%2==0), which
/// the synaptix port had dropped (full attention everywhere → wrong receptive
/// field for sequences > window, degrading lyric/timbre conditioning).
pub(crate) fn build_sliding_mask(
    s: usize,
    window: usize,
    device: synaptix_core::device::Device,
) -> R<Tensor> {
    let mut data = Vec::with_capacity(s * s);
    for i in 0..s {
        for j in 0..s {
            let d = if i > j { i - j } else { j - i };
            data.push(if d <= window { 0.0f32 } else { f32::NEG_INFINITY });
        }
    }
    Ok(Tensor::from_vec(data, vec![1usize, 1, s, s], device)?)
}

pub(crate) fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> R<Tensor> {
    let r = x.rank();
    let hd = x.dims()[r - 1];
    let half = hd / 2;
    let x1 = x.narrow(r - 1, 0, half)?.contiguous()?;
    let x2 = x.narrow(r - 1, half, half)?.contiguous()?;
    let rot = Tensor::cat(&[&x2.neg()?, &x1], r - 1)?;
    Ok(x.broadcast_mul(cos)?.broadcast_add(&rot.broadcast_mul(sin)?)?)
}

pub(crate) fn rope_tables(head_dim: usize, s: usize, theta: f32, device: synaptix_core::device::Device) -> R<(Tensor, Tensor)> {
    let cache = RopeCache::new(head_dim, s, theta, device)?;
    let (c, sn) = cache.select_range(0, s)?;
    let cos = Tensor::cat(&[&c, &c], 1)?;
    let sin = Tensor::cat(&[&sn, &sn], 1)?;
    Ok((cos, sin))
}

pub(crate) fn repeat_kv(x: &Tensor, g: usize) -> R<Tensor> {
    if g == 1 {
        return Ok(x.clone());
    }
    let d = x.dims().to_vec();
    let (b, nkv, s, hd) = (d[0], d[1], d[2], d[3]);
    let reps = Tensor::zeros(vec![b, nkv, g, s, hd], x.dtype(), x.device())?;
    Ok(x.unsqueeze(2)?.broadcast_add(&reps)?.reshape(vec![b, nkv * g, s, hd])?)
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(ck: &CompLoader, prefix: &str) -> R<Self> {
        let lin = |k: &str| -> R<Linear> {
            let w = ck.f32(&format!("{prefix}.{k}.weight"))?;
            Linear::new(w, None).map_err(AceError::Tensor)
        };
        Ok(Self { gate: lin("gate_proj")?, up: lin("up_proj")?, down: lin("down_proj")? })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let g = self.gate.forward(x).map_err(AceError::Tensor)?;
        let u = self.up.forward(x).map_err(AceError::Tensor)?;
        let act = g.silu()?.broadcast_mul(&u)?;
        self.down.forward(&act).map_err(AceError::Tensor)
    }
}

pub struct EncoderLayer {
    input_ln: Tensor,
    post_ln: Tensor,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    mlp: Mlp,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
}

impl EncoderLayer {
    pub fn load(
        ck: &CompLoader,
        prefix: &str,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> R<Self> {
        let lin = |k: &str| -> R<Linear> {
            let w = ck.f32(&format!("{prefix}.self_attn.{k}.weight"))?;
            Linear::new(w, None).map_err(AceError::Tensor)
        };
        Ok(Self {
            input_ln: ck.f32(&format!("{prefix}.input_layernorm.weight"))?,
            post_ln: ck.f32(&format!("{prefix}.post_attention_layernorm.weight"))?,
            q_proj: lin("q_proj")?,
            k_proj: lin("k_proj")?,
            v_proj: lin("v_proj")?,
            o_proj: lin("o_proj")?,
            q_norm: ck.f32(&format!("{prefix}.self_attn.q_norm.weight"))?,
            k_norm: ck.f32(&format!("{prefix}.self_attn.k_norm.weight"))?,
            mlp: Mlp::load(ck, &format!("{prefix}.mlp"))?,
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> R<Tensor> {
        let d = x.dims().to_vec();
        let (n, s) = (d[0], d[1]);
        let (hq, hkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);

        let h = rms_norm(x, &self.input_ln, self.eps)?;

        let q = self.q_proj.forward(&h).map_err(AceError::Tensor)?.contiguous()?.reshape(vec![n, s, hq, hd])?;
        let q = rms_norm(&q, &self.q_norm, self.eps)?.transpose(1, 2)?.contiguous()?;
        let k = self.k_proj.forward(&h).map_err(AceError::Tensor)?.contiguous()?.reshape(vec![n, s, hkv, hd])?;
        let k = rms_norm(&k, &self.k_norm, self.eps)?.transpose(1, 2)?.contiguous()?;
        let v = self
            .v_proj
            .forward(&h)
            .map_err(AceError::Tensor)?
            .contiguous()?
            .reshape(vec![n, s, hkv, hd])?
            .transpose(1, 2)?
            .contiguous()?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;
        let k = repeat_kv(&k, hq / hkv)?;
        let v = repeat_kv(&v, hq / hkv)?;

        let scale = 1.0 / (hd as f32).sqrt();
        let attn = scaled_dot_attention(&q, &k, &v, scale, mask)?;
        let attn = attn.transpose(1, 2)?.contiguous()?.reshape(vec![n, s, hq * hd])?;
        let attn = self.o_proj.forward(&attn).map_err(AceError::Tensor)?;
        let x = x.broadcast_add(&attn.to_dtype(DType::F32)?)?;

        let h2 = rms_norm(&x, &self.post_ln, self.eps)?;
        let mlp = self.mlp.forward(&h2)?;
        Ok(x.broadcast_add(&mlp)?)
    }
}
