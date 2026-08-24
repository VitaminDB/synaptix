use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::rms_norm::rms_norm;
use synaptix_ops::pos::rope::{apply_rope_range, RopeLayout};
use synaptix_ops::pos::rope_cache::RopeCache;

use crate::config::DecoderConfig;
use crate::loader::WeightSource;
use crate::{err, Result, VibeVoiceError};

const NEG_LARGE: f32 = -1.0e30;

struct Attn {
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    o_w: Tensor,
}

struct Mlp {
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

struct Layer {
    input_ln: Tensor,
    post_ln: Tensor,
    attn: Attn,
    mlp: Mlp,
}

pub struct KvCache {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    len: usize,
    cap: usize,
}

impl KvCache {
    pub fn new(
        layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cap: usize,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let mut k = Vec::with_capacity(layers);
        let mut v = Vec::with_capacity(layers);
        for _ in 0..layers {
            k.push(
                Tensor::zeros(vec![1usize, num_kv_heads, cap, head_dim], dtype, device)
                    .map_err(err)?,
            );
            v.push(
                Tensor::zeros(vec![1usize, num_kv_heads, cap, head_dim], dtype, device)
                    .map_err(err)?,
            );
        }
        Ok(Self { k, v, len: 0, cap })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}

fn repeat_kv(x: &Tensor, group: usize) -> Result<Tensor> {
    if group == 1 {
        return x.contiguous().map_err(err);
    }
    let d = x.dims().to_vec();
    let (b, nkv, s, hd) = (d[0], d[1], d[2], d[3]);
    x.unsqueeze(2)
        .and_then(|t| t.broadcast_as(vec![b, nkv, group, s, hd]))
        .and_then(|t| t.contiguous())
        .and_then(|t| t.reshape(vec![b, nkv * group, s, hd]))
        .map_err(err)
}

fn causal_bias(q_len: usize, kv_len: usize, device: Device) -> Result<Tensor> {
    let past = kv_len - q_len;
    let mut data = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        for j in 0..kv_len {
            if j > past + i {
                data[i * kv_len + j] = NEG_LARGE;
            }
        }
    }
    Tensor::from_vec(data, vec![1usize, 1, q_len, kv_len], device).map_err(err)
}

pub struct Qwen2Model {
    embed: Tensor,
    layers: Vec<Layer>,
    final_norm: Tensor,
    lm_head: Tensor,
    rope: RopeCache,
    pub cfg: DecoderConfig,
    pub device: Device,
    pub dtype: DType,
}

impl Qwen2Model {
    pub fn load(
        src: &dyn WeightSource,
        cfg: &DecoderConfig,
        prefix: &str,
        lm_head_name: &str,
        rope_capacity: usize,
    ) -> Result<Self> {
        let embed = src.get(&format!("{prefix}.embed_tokens.weight"))?;
        let final_norm = src.get(&format!("{prefix}.norm.weight"))?;
        let lm_head = if src.has(lm_head_name) {
            src.get(lm_head_name)?
        } else {
            embed.clone()
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let k = |s: &str| format!("{prefix}.layers.{i}.{s}");
            layers.push(Layer {
                input_ln: src.get(&k("input_layernorm.weight"))?,
                post_ln: src.get(&k("post_attention_layernorm.weight"))?,
                attn: Attn {
                    q_w: src.get(&k("self_attn.q_proj.weight"))?,
                    q_b: src.get(&k("self_attn.q_proj.bias"))?,
                    k_w: src.get(&k("self_attn.k_proj.weight"))?,
                    k_b: src.get(&k("self_attn.k_proj.bias"))?,
                    v_w: src.get(&k("self_attn.v_proj.weight"))?,
                    v_b: src.get(&k("self_attn.v_proj.bias"))?,
                    o_w: src.get(&k("self_attn.o_proj.weight"))?,
                },
                mlp: Mlp {
                    gate: src.get(&k("mlp.gate_proj.weight"))?,
                    up: src.get(&k("mlp.up_proj.weight"))?,
                    down: src.get(&k("mlp.down_proj.weight"))?,
                },
            });
        }
        let device = embed.device();
        let dtype = embed.dtype();
        let rope = RopeCache::new(
            cfg.head_dim(),
            rope_capacity.max(1),
            cfg.rope_theta as f32,
            device,
        )
        .map_err(err)?;
        Ok(Self {
            embed,
            layers,
            final_norm,
            lm_head,
            rope,
            cfg: cfg.clone(),
            device,
            dtype,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    pub fn new_cache(&self, cap: usize) -> Result<KvCache> {
        KvCache::new(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim(),
            cap,
            self.dtype,
            self.device,
        )
    }

    pub fn embed_tokens(&self, ids: &[i64]) -> Result<Tensor> {
        let n = ids.len();
        let idx = Tensor::from_vec(ids.to_vec(), vec![n], self.device).map_err(err)?;
        self.embed
            .index_select(0, &idx)
            .and_then(|t| t.reshape(vec![1usize, n, self.cfg.hidden_size]))
            .map_err(err)
    }

    pub fn lm_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        hidden.linear(&self.lm_head).map_err(err)
    }

    pub fn lm_head_rows(&self, ids: &[i64]) -> Result<Tensor> {
        let idx = Tensor::from_vec(ids.to_vec(), vec![ids.len()], self.device).map_err(err)?;
        self.lm_head
            .index_select(0, &idx)
            .and_then(|t| t.contiguous())
            .map_err(err)
    }

    fn attention(
        &self,
        layer_idx: usize,
        h: &Tensor,
        cache: &mut KvCache,
        past: usize,
        s: usize,
    ) -> Result<Tensor> {
        let layer = &self.layers[layer_idx];
        let a = &layer.attn;
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim();
        let scale = 1.0f32 / (hd as f32).sqrt();

        let q = h
            .linear_bias_residual(&a.q_w, Some(&a.q_b), None)
            .and_then(|t| t.reshape(vec![1usize, s, nh, hd]))
            .and_then(|t| t.permute(vec![0usize, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)?;
        let k = h
            .linear_bias_residual(&a.k_w, Some(&a.k_b), None)
            .and_then(|t| t.reshape(vec![1usize, s, nkv, hd]))
            .and_then(|t| t.permute(vec![0usize, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)?;
        let v = h
            .linear_bias_residual(&a.v_w, Some(&a.v_b), None)
            .and_then(|t| t.reshape(vec![1usize, s, nkv, hd]))
            .and_then(|t| t.permute(vec![0usize, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)?;

        let q = apply_rope_range(&q, &self.rope, past, s, RopeLayout::Split).map_err(err)?;
        let k = apply_rope_range(&k, &self.rope, past, s, RopeLayout::Split).map_err(err)?;

        cache.k[layer_idx].kv_append_inplace(&k, past).map_err(err)?;
        cache.v[layer_idx].kv_append_inplace(&v, past).map_err(err)?;

        let total = past + s;
        let k_all = cache.k[layer_idx].narrow(2, 0, total).map_err(err)?;
        let v_all = cache.v[layer_idx].narrow(2, 0, total).map_err(err)?;

        let attn = match q.flash_attention(&k_all, &v_all, scale, true) {
            Ok(o) => o,
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                let group = nh / nkv;
                let kr = repeat_kv(&k_all, group)?;
                let vr = repeat_kv(&v_all, group)?;
                let bias = if s == 1 {
                    None
                } else {
                    Some(causal_bias(s, total, self.device)?.to_dtype(q.dtype()).map_err(err)?)
                };
                scaled_dot_attention(&q, &kr, &vr, scale, bias.as_ref()).map_err(err)?
            }
            Err(e) => return Err(err(e)),
        };

        attn.permute(vec![0usize, 2, 1, 3])
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![1usize, s, nh * hd]))
            .and_then(|t| t.linear(&a.o_w))
            .map_err(err)
    }

    fn mlp(&self, layer_idx: usize, h: &Tensor) -> Result<Tensor> {
        let m = &self.layers[layer_idx].mlp;
        let gate = h.linear(&m.gate).map_err(err)?;
        let up = h.linear(&m.up).map_err(err)?;
        let act = match gate.silu_and_mul(&up) {
            Ok(t) => t,
            Err(_) => gate.silu().and_then(|g| g.mul(&up)).map_err(err)?,
        };
        act.linear(&m.down).map_err(err)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_pair(
        &self,
        layer_idx: usize,
        h: &Tensor,
        cache_a: &mut KvCache,
        cache_b: &mut KvCache,
    ) -> Result<Tensor> {
        let layer = &self.layers[layer_idx];
        let a = &layer.attn;
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim();
        let scale = 1.0f32 / (hd as f32).sqrt();

        let proj = |w: &Tensor, b: &Tensor, heads: usize| -> Result<Tensor> {
            h.linear_bias_residual(w, Some(b), None)
                .and_then(|t| t.reshape(vec![2usize, 1, heads, hd]))
                .and_then(|t| t.permute(vec![0usize, 2, 1, 3]))
                .and_then(|t| t.contiguous())
                .map_err(err)
        };
        let q = proj(&a.q_w, &a.q_b, nh)?;
        let k = proj(&a.k_w, &a.k_b, nkv)?;
        let v = proj(&a.v_w, &a.v_b, nkv)?;

        let mut outs: Vec<Tensor> = Vec::with_capacity(2);
        for slot in 0..2usize {
            let cache = if slot == 0 { &mut *cache_a } else { &mut *cache_b };
            let past = cache.len;
            let qi = q.narrow(0, slot, 1).and_then(|t| t.contiguous()).map_err(err)?;
            let ki = k.narrow(0, slot, 1).and_then(|t| t.contiguous()).map_err(err)?;
            let vi = v.narrow(0, slot, 1).and_then(|t| t.contiguous()).map_err(err)?;
            let qi = apply_rope_range(&qi, &self.rope, past, 1, RopeLayout::Split).map_err(err)?;
            let ki = apply_rope_range(&ki, &self.rope, past, 1, RopeLayout::Split).map_err(err)?;
            cache.k[layer_idx].kv_append_inplace(&ki, past).map_err(err)?;
            cache.v[layer_idx].kv_append_inplace(&vi, past).map_err(err)?;
            let total = past + 1;
            let k_all = cache.k[layer_idx].narrow(2, 0, total).map_err(err)?;
            let v_all = cache.v[layer_idx].narrow(2, 0, total).map_err(err)?;
            let attn = match qi.flash_attention(&k_all, &v_all, scale, true) {
                Ok(o) => o,
                Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                    let group = nh / nkv;
                    let kr = repeat_kv(&k_all, group)?;
                    let vr = repeat_kv(&v_all, group)?;
                    scaled_dot_attention(&qi, &kr, &vr, scale, None).map_err(err)?
                }
                Err(e) => return Err(err(e)),
            };
            outs.push(attn);
        }
        let merged = Tensor::cat(&[&outs[0], &outs[1]], 0).map_err(err)?;
        merged
            .permute(vec![0usize, 2, 1, 3])
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![2usize, 1, nh * hd]))
            .and_then(|t| t.linear(&a.o_w))
            .map_err(err)
    }

    pub fn forward_pair(
        &self,
        x_a: &Tensor,
        x_b: &Tensor,
        cache_a: &mut KvCache,
        cache_b: &mut KvCache,
    ) -> Result<(Tensor, Tensor)> {
        if cache_a.len + 1 > cache_a.cap || cache_b.len + 1 > cache_b.cap {
            return Err(VibeVoiceError::Inference("kv cache overflow (pair)".into()));
        }
        let eps = self.cfg.rms_norm_eps;
        let hidden_size = self.cfg.hidden_size;
        let a = x_a.to_dtype(self.dtype).and_then(|t| t.reshape(vec![1usize, 1, hidden_size])).map_err(err)?;
        let b = x_b.to_dtype(self.dtype).and_then(|t| t.reshape(vec![1usize, 1, hidden_size])).map_err(err)?;
        let mut hidden = Tensor::cat(&[&a, &b], 0).map_err(err)?;

        for i in 0..self.layers.len() {
            let residual = hidden.clone();
            let h = rms_norm(&hidden, &self.layers[i].input_ln, eps).map_err(err)?;
            let mixed = self.attention_pair(i, &h, cache_a, cache_b)?;
            hidden = residual.add(&mixed).map_err(err)?;

            let residual = hidden.clone();
            let h = rms_norm(&hidden, &self.layers[i].post_ln, eps).map_err(err)?;
            let m = self.mlp(i, &h)?;
            hidden = residual.add(&m).map_err(err)?;
        }
        cache_a.len += 1;
        cache_b.len += 1;
        let hidden = rms_norm(&hidden, &self.final_norm, eps).map_err(err)?;
        let ha = hidden.narrow(0, 0, 1).and_then(|t| t.contiguous()).map_err(err)?;
        let hb = hidden.narrow(0, 1, 1).and_then(|t| t.contiguous()).map_err(err)?;
        Ok((ha, hb))
    }

    pub fn rollback(&self, cache: &mut KvCache, n: usize) {
        cache.len = cache.len.saturating_sub(n);
    }

    pub fn forward(&self, inputs_embeds: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let dims = inputs_embeds.dims().to_vec();
        let s = dims[1];
        let past = cache.len;
        if past + s > cache.cap {
            return Err(VibeVoiceError::Inference(format!(
                "kv cache overflow: {past}+{s} > {}",
                cache.cap
            )));
        }
        let eps = self.cfg.rms_norm_eps;
        let mut hidden = inputs_embeds.to_dtype(self.dtype).map_err(err)?;
        for i in 0..self.layers.len() {
            let residual = hidden.clone();
            let h = rms_norm(&hidden, &self.layers[i].input_ln, eps).map_err(err)?;
            let mixed = self.attention(i, &h, cache, past, s)?;
            hidden = residual.add(&mixed).map_err(err)?;

            let residual = hidden.clone();
            let h = rms_norm(&hidden, &self.layers[i].post_ln, eps).map_err(err)?;
            let m = self.mlp(i, &h)?;
            hidden = residual.add(&m).map_err(err)?;
        }
        cache.len = past + s;
        rms_norm(&hidden, &self.final_norm, eps).map_err(err)
    }
}
