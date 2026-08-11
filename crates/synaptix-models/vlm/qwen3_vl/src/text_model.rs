use std::path::Path;

use serde::Deserialize;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::softmax::scaled_dot_attention;

use crate::model::{VisionError, VisionWeights};

pub const LM: &str = "model.language_model";

type R<T> = Result<T, VisionError>;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub mrope_section: Vec<usize>,
    pub mrope_interleaved: bool,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 5120,
            intermediate_size: 25600,
            num_hidden_layers: 64,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            mrope_section: vec![24, 20, 20],
            mrope_interleaved: true,
        }
    }
}

impl TextConfig {
    pub fn from_hf_bytes(bytes: &[u8]) -> Result<Self, VisionError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| VisionError::Load(format!("config.json: {e}")))?;
        let tc = root
            .get("text_config")
            .cloned()
            .ok_or_else(|| VisionError::Load("нет text_config".into()))?;
        let mut cfg: Self = serde_json::from_value(tc.clone())
            .map_err(|e| VisionError::Load(format!("text_config: {e}")))?;
        if let Some(rs) = tc.get("rope_scaling") {
            if let Some(sec) = rs.get("mrope_section").and_then(|v| v.as_array()) {
                cfg.mrope_section =
                    sec.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect();
            }
            if let Some(b) = rs.get("mrope_interleaved").and_then(|v| v.as_bool()) {
                cfg.mrope_interleaved = b;
            }
        }
        Ok(cfg)
    }

    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, VisionError> {
        let p = dir.as_ref().join("config.json");
        let bytes = std::fs::read(&p)
            .map_err(|e| VisionError::Load(format!("{}: {e}", p.display())))?;
        Self::from_hf_bytes(&bytes)
    }

    pub fn group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

struct Lin {
    wt: Tensor,
}

impl Lin {
    fn load(w: &dyn VisionWeights, key: &str, device: Device, dtype: DType) -> R<Self> {
        let raw = w.tensor(&format!("{key}.weight"), device, dtype)?;
        Ok(Self {
            wt: raw
                .transpose(0, 1)
                .and_then(|t| t.contiguous())
                .map_err(|e| VisionError::Load(e.to_string()))?,
        })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        x.matmul(&self.wt).map_err(|e| VisionError::Forward(e.to_string()))
    }
}

fn rms(x: &Tensor, w: &Tensor, eps: f32) -> R<Tensor> {
    if let Ok(y) = x.rms_norm_fused(w, eps, false) {
        return Ok(y);
    }
    synaptix_ops::norm::rms_norm::rms_norm(x, w, eps)
        .map_err(|e| VisionError::Forward(e.to_string()))
}

struct Layer {
    input_norm: Tensor,
    post_norm: Tensor,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: Lin,
    up: Lin,
    down: Lin,
}

impl Layer {
    fn load(
        w: &dyn VisionWeights,
        idx: usize,
        device: Device,
        dtype: DType,
    ) -> R<Self> {
        let p = format!("{LM}.layers.{idx}");
        Ok(Self {
            input_norm: w.tensor(&format!("{p}.input_layernorm.weight"), device, dtype)?,
            post_norm: w.tensor(&format!("{p}.post_attention_layernorm.weight"), device, dtype)?,
            q: Lin::load(w, &format!("{p}.self_attn.q_proj"), device, dtype)?,
            k: Lin::load(w, &format!("{p}.self_attn.k_proj"), device, dtype)?,
            v: Lin::load(w, &format!("{p}.self_attn.v_proj"), device, dtype)?,
            o: Lin::load(w, &format!("{p}.self_attn.o_proj"), device, dtype)?,
            q_norm: w.tensor(&format!("{p}.self_attn.q_norm.weight"), device, dtype)?,
            k_norm: w.tensor(&format!("{p}.self_attn.k_norm.weight"), device, dtype)?,
            gate: Lin::load(w, &format!("{p}.mlp.gate_proj"), device, dtype)?,
            up: Lin::load(w, &format!("{p}.mlp.up_proj"), device, dtype)?,
            down: Lin::load(w, &format!("{p}.mlp.down_proj"), device, dtype)?,
        })
    }
}

pub struct MRopeTables {
    pub cos: Tensor,
    pub sin: Tensor,
}

pub fn build_mrope(
    positions: &[[u32; 3]],
    cfg: &TextConfig,
    device: Device,
) -> Result<MRopeTables, VisionError> {
    let half = cfg.head_dim / 2;
    let s = positions.len();
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f32 / cfg.head_dim as f32))
        .collect();

    let mut axis_of = vec![0usize; half];
    if cfg.mrope_interleaved {
        let sec = &cfg.mrope_section;
        for (ax, offset) in [(1usize, 1usize), (2, 2)] {
            let limit = (sec.get(ax).copied().unwrap_or(0) * 3).min(half);
            let mut i = offset;
            while i < limit {
                axis_of[i] = ax;
                i += 3;
            }
        }
    } else {
        let sec = &cfg.mrope_section;
        let mut off = 0usize;
        for (ax, n) in sec.iter().enumerate() {
            for i in off..(off + n).min(half) {
                axis_of[i] = ax;
            }
            off += n;
        }
    }

    let mut cos = vec![0f32; s * half];
    let mut sin = vec![0f32; s * half];
    for (si, p) in positions.iter().enumerate() {
        for i in 0..half {
            let ang = p[axis_of[i]] as f32 * inv[i];
            cos[si * half + i] = ang.cos();
            sin[si * half + i] = ang.sin();
        }
    }
    Ok(MRopeTables {
        cos: Tensor::from_vec(cos, vec![s, half], device)
            .map_err(|e| VisionError::Load(e.to_string()))?,
        sin: Tensor::from_vec(sin, vec![s, half], device)
            .map_err(|e| VisionError::Load(e.to_string()))?,
    })
}

pub struct VisionSpan {
    pub start: usize,
    pub len: usize,
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

pub fn rope_positions(seq_len: usize, spans: &[VisionSpan]) -> Vec<[u32; 3]> {
    let mut out = vec![[0u32; 3]; seq_len];
    let mut cursor = 0u32;
    let mut i = 0usize;
    let mut span_idx = 0usize;
    while i < seq_len {
        if span_idx < spans.len() && spans[span_idx].start == i {
            let sp = &spans[span_idx];
            let base = cursor;
            let mut maxp = base;
            for t in 0..sp.grid_t {
                for h in 0..sp.grid_h {
                    for w in 0..sp.grid_w {
                        let idx = i + t * sp.grid_h * sp.grid_w + h * sp.grid_w + w;
                        if idx >= seq_len {
                            continue;
                        }
                        let p = [base + t as u32, base + h as u32, base + w as u32];
                        maxp = maxp.max(p[0]).max(p[1]).max(p[2]);
                        out[idx] = p;
                    }
                }
            }
            i += sp.len;
            cursor = maxp + 1;
            span_idx += 1;
            continue;
        }
        out[i] = [cursor, cursor, cursor];
        cursor += 1;
        i += 1;
    }
    out
}

pub struct TextEncoder {
    pub config: TextConfig,
    embed: Tensor,
    layers: Vec<Layer>,
    device: Device,
    dtype: DType,
}

impl TextEncoder {
    pub fn build(
        config: TextConfig,
        weights: &dyn VisionWeights,
        device: Device,
        dtype: DType,
        num_layers: usize,
    ) -> Result<Self, VisionError> {
        let n = num_layers.min(config.num_hidden_layers);
        let embed = weights.tensor(&format!("{LM}.embed_tokens.weight"), device, dtype)?;
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            layers.push(Layer::load(weights, i, device, dtype)?);
        }
        Ok(Self { config, embed, layers, device, dtype })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor, VisionError> {
        let idx = Tensor::from_vec(ids.to_vec(), vec![ids.len()], self.device)
            .map_err(|e| VisionError::Forward(e.to_string()))?;
        self.embed
            .embed_gather(&idx)
            .map_err(|e| VisionError::Forward(e.to_string()))
    }

    pub fn forward(
        &self,
        hidden: &Tensor,
        rope: &MRopeTables,
        deepstack: &[(Tensor, Vec<usize>)],
    ) -> Result<Tensor, VisionError> {
        let cfg = &self.config;
        let s = hidden.dims()[0];
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let group = cfg.group_size();
        let scale = 1.0 / (hd as f32).sqrt();
        let e = |r: Result<Tensor, SynaptixError>| r.map_err(|x| VisionError::Forward(x.to_string()));

        let mut x = hidden.clone();
        for (li, layer) in self.layers.iter().enumerate() {
            if li < deepstack.len() {
                let (feat, rows) = &deepstack[li];
                x = scatter_add_rows(&x, feat, rows)?;
            }

            let h = rms(&x, &layer.input_norm, cfg.rms_norm_eps)?;
            let q = layer.q.forward(&h)?;
            let k = layer.k.forward(&h)?;
            let v = layer.v.forward(&h)?;

            let q = rms(&e(q.reshape(vec![s, nh, hd]))?, &layer.q_norm, cfg.rms_norm_eps)?;
            let k = rms(&e(k.reshape(vec![s, nkv, hd]))?, &layer.k_norm, cfg.rms_norm_eps)?;

            let q = e(e(q.transpose(0, 1))?.contiguous())?;
            let k = e(e(k.transpose(0, 1))?.contiguous())?;
            let v = e(e(e(v.reshape(vec![s, nkv, hd]))?.transpose(0, 1))?.contiguous())?;

            let q = apply_rope(&q, rope, hd)?;
            let k = apply_rope(&k, rope, hd)?;
            let k = repeat_kv(&k, group)?;
            let v = repeat_kv(&v, group)?;

            let q = e(q.reshape(vec![1, nh, s, hd]))?;
            let k = e(k.reshape(vec![1, nh, s, hd]))?;
            let v = e(v.reshape(vec![1, nh, s, hd]))?;
            let attn = match q.dtype() {
                DType::BF16 | DType::F16 => match q.flash_attention(&k, &v, scale, true) {
                    Ok(a) => a,
                    Err(_) => {
                        let m = causal_mask(s, q.dtype(), self.device)?;
                        e(scaled_dot_attention(&q, &k, &v, scale, Some(&m)))?
                    }
                },
                _ => {
                    let m = causal_mask(s, q.dtype(), self.device)?;
                    e(scaled_dot_attention(&q, &k, &v, scale, Some(&m)))?
                }
            };
            let attn = e(e(e(e(attn.reshape(vec![nh, s, hd]))?.transpose(0, 1))?.contiguous())?
                .reshape(vec![s, nh * hd]))?;
            x = e(x.add(&layer.o.forward(&attn)?))?;

            let h = rms(&x, &layer.post_norm, cfg.rms_norm_eps)?;
            let g = layer.gate.forward(&h)?;
            let u = layer.up.forward(&h)?;
            let act = e(g.silu_and_mul(&u))?;
            x = e(x.add(&layer.down.forward(&act)?))?;
        }
        Ok(x)
    }
}

fn apply_rope(x: &Tensor, rope: &MRopeTables, head_dim: usize) -> R<Tensor> {
    if matches!(x.device(), Device::Cuda(_)) {
        if let Ok(y) = x.rope_split_partial_fused(&rope.cos, &rope.sin, head_dim) {
            return Ok(y);
        }
    }
    let dims = x.dims().to_vec();
    let last = dims.len() - 1;
    let half = head_dim / 2;
    let dt = x.dtype();
    let cos = rope.cos.to_dtype(dt).map_err(|e| VisionError::Forward(e.to_string()))?;
    let sin = rope.sin.to_dtype(dt).map_err(|e| VisionError::Forward(e.to_string()))?;
    let e = |r: Result<Tensor, SynaptixError>| r.map_err(|x| VisionError::Forward(x.to_string()));
    let x0 = e(e(x.narrow(last, 0, half))?.contiguous())?;
    let x1 = e(e(x.narrow(last, half, half))?.contiguous())?;
    let o0 = e(e(x0.mul(&cos))?.sub(&e(x1.mul(&sin))?))?;
    let o1 = e(e(x1.mul(&cos))?.add(&e(x0.mul(&sin))?))?;
    e(Tensor::cat(&[&o0, &o1], last))
}

fn repeat_kv(x: &Tensor, group: usize) -> R<Tensor> {
    if group == 1 {
        return Ok(x.clone());
    }
    let d = x.dims().to_vec();
    let (nkv, s, hd) = (d[0], d[1], d[2]);
    let e = |r: Result<Tensor, SynaptixError>| r.map_err(|x| VisionError::Forward(x.to_string()));
    let expanded = e(x.reshape(vec![nkv, 1, s, hd]))?;
    let mut parts = Vec::with_capacity(group);
    for _ in 0..group {
        parts.push(expanded.clone());
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    let cat = e(Tensor::cat(&refs, 1))?;
    e(e(cat.reshape(vec![nkv * group, s, hd]))?.contiguous())
}

fn causal_mask(n: usize, dtype: DType, device: Device) -> R<Tensor> {
    let mut v = vec![0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            v[i * n + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(v, vec![1, 1, n, n], device)
        .and_then(|t| t.to_dtype(dtype))
        .map_err(|e| VisionError::Forward(e.to_string()))
}

fn scatter_add_rows(x: &Tensor, feat: &Tensor, rows: &[usize]) -> R<Tensor> {
    if rows.is_empty() {
        return Ok(x.clone());
    }
    let e = |r: Result<Tensor, SynaptixError>| r.map_err(|x| VisionError::Forward(x.to_string()));
    let hidden = x.dims()[1];
    let idx: Vec<u32> = rows.iter().map(|r| *r as u32).collect();
    let idx = e(Tensor::from_vec(idx, vec![rows.len(), 1], x.device()))?;
    let idx = e(idx.broadcast_mul(&e(Tensor::ones(vec![1, hidden], idx.dtype(), x.device()))?))?;
    let feat = if feat.dtype() == x.dtype() {
        feat.clone()
    } else {
        e(feat.to_dtype(x.dtype()))?
    };
    e(x.scatter_add(0, &idx, &feat))
}
