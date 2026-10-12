use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::mask::causal_mask;
use synaptix_ops::norm::rms_norm;
use synaptix_ops::pos::longrope::{longrope_cache, LongRopeConfig};
use synaptix_ops::pos::rope::{apply_rope_range, RopeLayout};
use synaptix_ops::pos::rope_cache::RopeCache;

use crate::config::{LmConfig, SubTransformerConfig};
use crate::loader::{Lin, VoxCheckpoint};
use crate::VoxError;

#[derive(Debug, Clone, Copy)]
pub struct Dims {
    pub hidden: usize,
    pub intermediate: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
}

struct Attn {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
}

struct Mlp {
    gate: Lin,
    up: Lin,
    down: Lin,
}

struct Layer {
    input_ln: Tensor,
    post_ln: Tensor,
    attn: Attn,
    mlp: Mlp,
}

pub struct MiniCpm {
    layers: Vec<Layer>,
    norm: Tensor,
    rope: Option<RopeCache>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    scale: f32,
}

pub struct KvCache {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    pub len: usize,
}

/// KV-кэш постоянной ёмкости для NPU: буферы `[1, kv, cap, hd]`, позиция и
/// длина — тензоры. Шаг декода — один и тот же граф на любой позиции (иначе
/// граф собирается заново на каждом шаге с новой длиной кэша).
pub struct StaticKv {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    pub cap: usize,
    pub len: usize,
    pos: Tensor,
    t_cache: Tensor,
    rope: Option<(Tensor, Tensor)>,
}

impl MiniCpm {
    pub fn load(
        ck: &VoxCheckpoint,
        prefix: &str,
        dims: Dims,
        rope: Option<RopeCache>,
    ) -> Result<Self, VoxError> {
        let mut layers = Vec::with_capacity(dims.num_layers);
        for i in 0..dims.num_layers {
            let lp = format!("{prefix}.layers.{i}");
            let attn = Attn {
                q: Lin::load(ck, &format!("{lp}.self_attn"), "q_proj", false)?,
                k: Lin::load(ck, &format!("{lp}.self_attn"), "k_proj", false)?,
                v: Lin::load(ck, &format!("{lp}.self_attn"), "v_proj", false)?,
                o: Lin::load(ck, &format!("{lp}.self_attn"), "o_proj", false)?,
            };
            let mlp = Mlp {
                gate: Lin::load(ck, &format!("{lp}.mlp"), "gate_proj", false)?,
                up: Lin::load(ck, &format!("{lp}.mlp"), "up_proj", false)?,
                down: Lin::load(ck, &format!("{lp}.mlp"), "down_proj", false)?,
            };
            layers.push(Layer {
                input_ln: ck.get(&format!("{lp}.input_layernorm.weight"))?,
                post_ln: ck.get(&format!("{lp}.post_attention_layernorm.weight"))?,
                attn,
                mlp,
            });
        }
        let norm = ck.get(&format!("{prefix}.norm.weight"))?;
        Ok(Self {
            layers,
            norm,
            rope,
            num_heads: dims.num_heads,
            num_kv_heads: dims.num_kv_heads,
            head_dim: dims.head_dim,
            eps: dims.eps,
            scale: 1.0 / (dims.head_dim as f32).sqrt(),
        })
    }

    pub fn build_longrope(
        lm: &LmConfig,
        head_dim: usize,
        max_seq: usize,
        device: synaptix_core::device::Device,
    ) -> Result<RopeCache, VoxError> {
        let theta = lm.rope_theta;
        match &lm.rope_scaling {
            Some(rs) => {
                let cfg = LongRopeConfig {
                    long_factors: rs.long_factor.clone(),
                    short_factors: rs.short_factor.clone(),
                    original_max_seq: rs.original_max_position_embeddings,
                };
                Ok(longrope_cache(head_dim, max_seq, theta, &cfg, device)?)
            }
            None => Ok(RopeCache::new(head_dim, max_seq, theta, device)?),
        }
    }

    fn project_heads(&self, x: &Tensor, lin: &Lin, n_heads: usize) -> Result<Tensor, VoxError> {
        let dims = x.dims();
        let (b, s) = (dims[0], dims[1]);
        let y = lin.forward(x)?;
        let y = y
            .reshape((b, s, n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        Ok(y)
    }

    fn attention(
        &self,
        layer: &Layer,
        x: &Tensor,
        start: usize,
        mask: Option<&Tensor>,
        cache: Option<(&mut Tensor, &mut Tensor)>,
    ) -> Result<Tensor, VoxError> {
        let dims = x.dims();
        let (b, s, h) = (dims[0], dims[1], dims[2]);
        let mut q = self.project_heads(x, &layer.attn.q, self.num_heads)?;
        let mut k = self.project_heads(x, &layer.attn.k, self.num_kv_heads)?;
        let v = self.project_heads(x, &layer.attn.v, self.num_kv_heads)?;

        if let Some(rope) = &self.rope {
            q = apply_rope_range(&q, rope, start, s, RopeLayout::Split)?;
            k = apply_rope_range(&k, rope, start, s, RopeLayout::Split)?;
        }

        let (kk, vv) = match cache {
            Some((ck, cv)) => {
                let new_k = if ck.dims()[2] == 0 {
                    k.clone()
                } else {
                    Tensor::cat(&[ck as &Tensor, &k], 2)?
                };
                let new_v = if cv.dims()[2] == 0 {
                    v.clone()
                } else {
                    Tensor::cat(&[cv as &Tensor, &v], 2)?
                };
                *ck = new_k.clone();
                *cv = new_v.clone();
                (new_k, new_v)
            }
            None => (k, v),
        };

        let groups = self.num_heads / self.num_kv_heads;
        let (kk, vv) = if groups > 1 {
            (
                kk.repeat_interleave(1, groups)?.contiguous()?,
                vv.repeat_interleave(1, groups)?.contiguous()?,
            )
        } else {
            (kk, vv)
        };
        let attn = scaled_dot_attention(&q, &kk, &vv, self.scale, mask)?;
        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, self.num_heads * self.head_dim))?;
        let out = layer.attn.o.forward(&attn)?;
        debug_assert_eq!(out.dims(), &[b, s, h]);
        Ok(out)
    }

    fn mlp(&self, layer: &Layer, x: &Tensor) -> Result<Tensor, VoxError> {
        let gate = layer.mlp.gate.forward(x)?;
        let up = layer.mlp.up.forward(x)?;
        let act = match gate.silu_and_mul(&up) {
            Ok(a) => a,
            Err(_) => gate.silu()?.mul(&up)?,
        };
        layer.mlp.down.forward(&act)
    }

    fn layer_forward(
        &self,
        layer: &Layer,
        x: &Tensor,
        start: usize,
        mask: Option<&Tensor>,
        cache: Option<(&mut Tensor, &mut Tensor)>,
    ) -> Result<Tensor, VoxError> {
        let normed = rms_norm(x, &layer.input_ln, self.eps)?;
        let attn = self.attention(layer, &normed, start, mask, cache)?;
        let x = x.add(&attn)?;
        let normed = rms_norm(&x, &layer.post_ln, self.eps)?;
        let mlp = self.mlp(layer, &normed)?;
        Ok(x.add(&mlp)?)
    }

    pub fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor, VoxError> {
        let s = x.dims()[1];
        let mask = if causal {
            Some(causal_mask(s, x.device())?.to_dtype(x.dtype())?)
        } else {
            None
        };
        let mut h = x.clone();
        for layer in &self.layers {
            h = self.layer_forward(layer, &h, 0, mask.as_ref(), None)?;
        }
        Ok(rms_norm(&h, &self.norm, self.eps)?)
    }

    pub fn make_cache(&self, batch: usize) -> Result<KvCache, VoxError> {
        let empty = || -> Result<Tensor, VoxError> {
            Ok(Tensor::zeros(
                vec![batch, self.num_kv_heads, 0usize, self.head_dim],
                self.norm.dtype(),
                self.norm.device(),
            )?)
        };
        let mut k = Vec::with_capacity(self.layers.len());
        let mut v = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            k.push(empty()?);
            v.push(empty()?);
        }
        Ok(KvCache { k, v, len: 0 })
    }

    pub fn prefill(&self, x: &Tensor, cache: &mut KvCache) -> Result<Tensor, VoxError> {
        let s = x.dims()[1];
        let mask = causal_mask(s, x.device())?.to_dtype(x.dtype())?;
        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = self.layer_forward(layer, &h, 0, Some(&mask), Some((&mut cache.k[i], &mut cache.v[i])))?;
        }
        cache.len = s;
        Ok(rms_norm(&h, &self.norm, self.eps)?)
    }

    /// Пустой [`StaticKv`] ёмкости `cap`.
    pub fn static_cache(&self, cap: usize, dtype: synaptix_core::dtype::DType, dev: synaptix_core::device::Device) -> Result<StaticKv, VoxError> {
        let zeros = || Tensor::zeros(vec![1usize, self.num_kv_heads, cap, self.head_dim], dtype, dev);
        let mut k = Vec::with_capacity(self.layers.len());
        let mut v = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            k.push(zeros()?);
            v.push(zeros()?);
        }
        let rope = match &self.rope {
            Some(r) => {
                let (cos, sin) = (r.cos(), r.sin());
                Some((Tensor::cat(&[cos, cos], 1)?.to_dtype(dtype)?.to_device(dev)?, Tensor::cat(&[sin, sin], 1)?.to_dtype(dtype)?.to_device(dev)?))
            }
            None => None,
        };
        let pos = Tensor::from_vec(vec![0u32], vec![1usize], dev)?;
        let t_cache = Tensor::from_vec(vec![0u32], vec![1usize], dev)?;
        Ok(StaticKv { k, v, cap, len: 0, pos, t_cache, rope })
    }

    /// `T` строк `x` (`[1, T, H]`) с позиции `start` по [`StaticKv`]: префилл
    /// куском и шаг декода (T = 1) — графы одной формы на любой позиции.
    pub fn forward_static(&self, x: &Tensor, cache: &mut StaticKv, start: usize) -> Result<Tensor, VoxError> {
        let t = x.dims()[1];
        if start + t > cache.cap {
            return Err(VoxError::Other(format!("forward_static: {} строк больше ёмкости кэша {}", start + t, cache.cap)));
        }
        cache.pos.write_host_u32(&[start as u32])?;
        cache.t_cache.write_host_u32(&[(start + t) as u32])?;
        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let normed = rms_norm(&h, &layer.input_ln, self.eps)?;
            let mut q = self.project_heads(&normed, &layer.attn.q, self.num_heads)?;
            let mut k = self.project_heads(&normed, &layer.attn.k, self.num_kv_heads)?;
            let v = self.project_heads(&normed, &layer.attn.v, self.num_kv_heads)?;
            if let Some((cos, sin)) = &cache.rope {
                q = q.rope_apply_dev(cos, sin, &cache.pos, self.head_dim)?;
                k = k.rope_apply_dev(cos, sin, &cache.pos, self.head_dim)?;
            }
            cache.k[i].kv_append_dev(&k, &cache.pos)?;
            cache.v[i].kv_append_dev(&v, &cache.pos)?;
            let a = q.flash_attention_dev(&cache.k[i], &cache.v[i], &cache.t_cache, self.scale, true)?;
            let a = a.transpose(1, 2)?.contiguous()?.reshape((1usize, t, self.num_heads * self.head_dim))?;
            let hx = h.add(&layer.attn.o.forward(&a)?)?;
            let normed = rms_norm(&hx, &layer.post_ln, self.eps)?;
            h = hx.add(&self.mlp(layer, &normed)?)?;
        }
        cache.len = cache.len.max(start + t);
        Ok(rms_norm(&h, &self.norm, self.eps)?)
    }

    pub fn step(&self, x: &Tensor, cache: &mut KvCache) -> Result<Tensor, VoxError> {
        let start = cache.len;
        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = self.layer_forward(layer, &h, start, None, Some((&mut cache.k[i], &mut cache.v[i])))?;
        }
        cache.len += x.dims()[1];
        Ok(rms_norm(&h, &self.norm, self.eps)?)
    }
}

pub fn dims_from_lm(lm: &LmConfig, num_layers: usize) -> Dims {
    Dims {
        hidden: lm.hidden_size,
        intermediate: lm.intermediate_size,
        num_layers,
        num_heads: lm.num_attention_heads,
        num_kv_heads: lm.num_key_value_heads,
        head_dim: lm.kv_channels,
        eps: lm.rms_norm_eps,
    }
}

pub fn dims_from_sub(sub: &SubTransformerConfig, lm: &LmConfig) -> Dims {
    Dims {
        hidden: sub.hidden_dim,
        intermediate: sub.ffn_dim,
        num_layers: sub.num_layers,
        num_heads: sub.num_heads,
        num_kv_heads: lm.num_key_value_heads,
        head_dim: sub.kv_channels,
        eps: lm.rms_norm_eps,
    }
}
