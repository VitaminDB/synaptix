//! Кодер условий, flow-matching трансформер и сэмплер по окнам (эталон —
//! diffusers `condition_embedder_minimax_music3.py`,
//! `transformer_minimax_music3.py`, `denoise.py`).

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::conv1d::conv1d;
use synaptix_ops::norm::layer_norm::layer_norm;

use crate::loader::Comp;
use crate::{Music3Error, R};

const COMPUTE: DType = DType::BF16;
pub const LATENT: usize = 128;
const COND: usize = 2048;
const DIM: usize = 2048;
const HEADS: usize = 32;
const HEAD_DIM: usize = 64;
const ROTARY: usize = 32;
const FF: usize = 8192;
pub const CHUNK_FRAMES: usize = 200;
pub const CHUNK_HOP: usize = 100;
pub const OVERLAP: usize = 172;
const CFG_SCALE: f32 = 1.7;

fn host(t: &Tensor) -> R<Vec<f32>> {
    Ok(t.contiguous()?.to_dtype(DType::F32)?.to_device(Device::Cpu)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
}

/// Латентных кадров на `frames` кадров LM: `int(F · 44100/24000 · 960/512)`.
pub fn latent_len(frames: usize) -> usize {
    ((frames as f64 * 44_100.0 / 24_000.0 * 960.0 / 512.0) as usize).max(1)
}

pub struct ConditionEncoder {
    weights: Tensor,
    scale: f32,
    w: Tensor,
    b: Tensor,
    device: Device,
}

impl ConditionEncoder {
    pub fn load(c: &Comp, device: Device) -> R<Self> {
        let logits = host(&c.get("layer_weight_logits", DType::F32)?)?;
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|x| (x - m).exp()).collect();
        let z: f32 = e.iter().sum();
        let n = e.len();
        Ok(Self {
            weights: Tensor::from_vec(e.iter().map(|x| x / z).collect::<Vec<f32>>(), vec![1, n, 1], device)?,
            scale: host(&c.get("layer_scale", DType::F32)?)?[0],
            w: c.get("proj.weight", DType::F32)?,
            b: c.get("proj.bias", DType::F32)?,
            device,
        })
    }

    /// Состояния кадров `[1, F, 8·4096]` → условие `[L, 2048]` (F32) на сетке латентов.
    pub fn forward(&self, frames: &Tensor) -> R<Tensor> {
        let f = frames.dims()[1];
        let layers = self.weights.dims()[1];
        let h = frames.dims()[2] / layers;
        let x = frames.to_dtype(DType::F32)?.contiguous()?.reshape(vec![f, layers, h])?;
        let x = x.broadcast_mul(&self.weights)?.sum_keepdim(1)?.reshape(vec![f, h])?.mul_scalar(self.scale)?;
        let x = x.transpose(0, 1)?.contiguous()?.reshape(vec![1, h, f])?;
        let y = conv1d(&x, &self.w, Some(&self.b), 1, 1)?; // [1, 2048, F]
        let v = host(&y)?;
        let l = latent_len(f);
        // Ближайший сосед, как `F.interpolate(mode="nearest")`: src = ⌊dst · F / L⌋.
        let mut out = vec![0f32; l * COND];
        for i in 0..l {
            let src = ((i as f64 * f as f64 / l as f64) as usize).min(f - 1);
            for c in 0..COND {
                out[i * COND + c] = v[c * f + src];
            }
        }
        Ok(Tensor::from_vec(out, vec![l, COND], self.device)?)
    }
}

/// `y = x·Wᵀ + b`; вес — транспонированным `[in, out]`.
struct Lin {
    w: Tensor,
    b: Option<Tensor>,
}

impl Lin {
    fn load(c: &Comp, name: &str, bias: bool) -> R<Self> {
        let w = c.get(&format!("{name}.weight"), COMPUTE)?;
        let d = w.dims().to_vec();
        let w = w.reshape(vec![d[0], d[1]])?.transpose(0, 1)?.contiguous()?;
        Ok(Self { w, b: if bias { Some(c.get(&format!("{name}.bias"), COMPUTE)?) } else { None } })
    }
    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        let y = x.contiguous()?.matmul(&self.w)?;
        Ok(match &self.b {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        })
    }
}

struct Block {
    n1: (Tensor, Tensor),
    q: Lin,
    k: Lin,
    v: Lin,
    out: Lin,
    n2: (Tensor, Tensor),
    ff_in: Lin,
    ff_out: Lin,
}

pub struct FlowTransformer {
    time_w: Vec<f32>,
    t1: Lin,
    t2: Lin,
    pre: Lin,
    proj_in: Lin,
    blocks: Vec<Block>,
    proj_out: Lin,
    post: Lin,
    device: Device,
}

impl FlowTransformer {
    pub fn load(c: &Comp, device: Device) -> R<Self> {
        let ln = |p: &str| -> R<(Tensor, Tensor)> { Ok((c.get(&format!("{p}.weight"), COMPUTE)?, c.get(&format!("{p}.bias"), COMPUTE)?)) };
        let mut blocks = Vec::with_capacity(36);
        let mut i = 0;
        while c.has(&format!("transformer_blocks.{i}.norm1.weight")) {
            let p = format!("transformer_blocks.{i}");
            blocks.push(Block {
                n1: ln(&format!("{p}.norm1"))?,
                q: Lin::load(c, &format!("{p}.attn.to_q"), false)?,
                k: Lin::load(c, &format!("{p}.attn.to_k"), false)?,
                v: Lin::load(c, &format!("{p}.attn.to_v"), false)?,
                out: Lin::load(c, &format!("{p}.attn.to_out.0"), false)?,
                n2: ln(&format!("{p}.norm2"))?,
                ff_in: Lin::load(c, &format!("{p}.ff_in"), true)?,
                ff_out: Lin::load(c, &format!("{p}.ff_out"), true)?,
            });
            i += 1;
        }
        if blocks.is_empty() {
            return Err(Music3Error::Load("в компоненте transformer нет блоков".into()));
        }
        Ok(Self {
            time_w: host(&c.get("time_proj.weight", DType::F32)?)?,
            t1: Lin::load(c, "time_embed.linear_1", true)?,
            t2: Lin::load(c, "time_embed.linear_2", true)?,
            pre: Lin::load(c, "preprocess_conv", false)?,
            proj_in: Lin::load(c, "proj_in", false)?,
            blocks,
            proj_out: Lin::load(c, "proj_out", false)?,
            post: Lin::load(c, "postprocess_conv", false)?,
            device,
        })
    }

    /// Таблицы частичного RoPE для `s` позиций: `(cos, sin)` `[1, 1, s, 16]`.
    pub fn rope(&self, s: usize) -> R<(Tensor, Tensor)> {
        let half = ROTARY / 2;
        let (mut cs, mut sn) = (vec![0f32; s * half], vec![0f32; s * half]);
        for p in 0..s {
            for j in 0..half {
                let a = p as f32 * (1.0 / 10000f32.powf(2.0 * j as f32 / ROTARY as f32));
                cs[p * half + j] = a.cos();
                sn[p * half + j] = a.sin();
            }
        }
        let t = |v: Vec<f32>| -> R<Tensor> { Ok(Tensor::from_vec(v, vec![1, 1, s, half], self.device)?.to_dtype(COMPUTE)?) };
        Ok((t(cs)?, t(sn)?))
    }

    /// Скорость потока. `lat` `[L, 128]`, `cond` `[L, 2048]` (нули — безусловная ветка), `t ∈ [0, 1]`.
    pub fn forward(&self, lat: &Tensor, cond: &Tensor, t: f32, rope: &(Tensor, Tensor)) -> R<Tensor> {
        let l = lat.dims()[0];
        let zeros = Tensor::zeros(vec![l, LATENT], COMPUTE, self.device)?;
        let x = Tensor::cat(&[&lat.to_dtype(COMPUTE)?, &zeros, &cond.to_dtype(COMPUTE)?], 1)?.contiguous()?;
        let x = self.pre.fwd(&x)?.add(&x)?;
        let x = self.proj_in.fwd(&x)?;
        // Время: случайные признаки Фурье [cos, sin] → MLP.
        let half = self.time_w.len();
        let mut emb = vec![0f32; 2 * half];
        for (i, w) in self.time_w.iter().enumerate() {
            let a = 2.0 * std::f32::consts::PI * t * w;
            emb[i] = a.cos();
            emb[half + i] = a.sin();
        }
        let temb = self.t2.fwd(&self.t1.fwd(&Tensor::from_vec(emb, vec![1, 2 * half], self.device)?.to_dtype(COMPUTE)?)?.silu()?)?;
        let mut x = Tensor::cat(&[&temb, &x], 0)?.contiguous()?; // [S, DIM]
        let s = l + 1;
        let (cs, sn) = rope;
        let heads = |t: Tensor| -> R<Tensor> { Ok(t.reshape(vec![1, s, HEADS, HEAD_DIM])?.permute(vec![0, 2, 1, 3])?.contiguous()?) };
        let rot = |t: &Tensor| -> R<Tensor> {
            let half = ROTARY / 2;
            let a = t.narrow(3, 0, half)?.contiguous()?;
            let b = t.narrow(3, half, half)?.contiguous()?;
            let rest = t.narrow(3, ROTARY, HEAD_DIM - ROTARY)?.contiguous()?;
            let ra = a.broadcast_mul(cs)?.sub(&b.broadcast_mul(sn)?)?;
            let rb = b.broadcast_mul(cs)?.add(&a.broadcast_mul(sn)?)?;
            Ok(Tensor::cat(&[&ra, &rb, &rest], 3)?.contiguous()?)
        };
        for b in &self.blocks {
            let n = layer_norm(&x, Some(&b.n1.0), Some(&b.n1.1), 1e-5)?;
            let (q, k, v) = (rot(&heads(b.q.fwd(&n)?)?)?, rot(&heads(b.k.fwd(&n)?)?)?, heads(b.v.fwd(&n)?)?);
            let a = scaled_dot_attention(&q, &k, &v, 1.0 / (HEAD_DIM as f32).sqrt(), None)?;
            let a = a.permute(vec![0, 2, 1, 3])?.contiguous()?.reshape(vec![s, DIM])?;
            x = x.add(&b.out.fwd(&a)?)?;
            let y = b.ff_in.fwd(&layer_norm(&x, Some(&b.n2.0), Some(&b.n2.1), 1e-5)?)?;
            let (ya, yg) = (y.narrow(1, 0, FF)?.contiguous()?, y.narrow(1, FF, FF)?.contiguous()?);
            let act = yg.silu_and_mul(&ya).or_else(|_| yg.silu().and_then(|g| g.mul(&ya)))?;
            x = x.add(&b.ff_out.fwd(&act)?)?;
        }
        let o = self.proj_out.fwd(&x.narrow(0, 1, l)?.contiguous()?)?;
        Ok(self.post.fwd(&o)?.add(&o)?.to_dtype(DType::F32)?)
    }
}

/// Окна по 200 кадров с шагом 100.
pub fn chunk_starts(frames: usize) -> Vec<usize> {
    if frames <= CHUNK_FRAMES {
        vec![0]
    } else {
        (0..frames - CHUNK_HOP).step_by(CHUNK_HOP).collect()
    }
}

/// Flow-matching по окнам. Возвращает латенты окон `[L, 128]` (F32).
/// `progress(шаг, всего)`; `false` — остановить.
pub fn sample(enc: &ConditionEncoder, tr: &FlowTransformer, frames: &Tensor, steps: usize, seed: u64, progress: &mut dyn FnMut(usize, usize) -> bool) -> R<Vec<Tensor>> {
    let dev = tr.device;
    let f = frames.dims()[1];
    let starts = chunk_starts(f);
    let total = starts.len() * steps;
    let mut out = Vec::with_capacity(starts.len());
    let mut prev: Option<(Tensor, Tensor)> = None; // (латенты, условие) перекрытия
    for (ci, &start) in starts.iter().enumerate() {
        let end = (start + CHUNK_FRAMES).min(f);
        let mut cond = enc.forward(&frames.narrow(1, start, end - start)?.contiguous()?)?;
        let l = cond.dims()[0];
        let mut ov = 0;
        if let Some((pl, pc)) = &prev {
            ov = pl.dims()[0].min(l);
            cond = Tensor::cat(&[&pc.narrow(0, 0, ov)?.contiguous()?, &cond.narrow(0, ov, l - ov)?.contiguous()?], 0)?.contiguous()?;
        }
        let uncond = Tensor::zeros(vec![l, COND], DType::F32, dev)?;
        let noise = Tensor::randn_seeded(vec![l, LATENT], seed.wrapping_add(ci as u64 * 7919), Device::Cpu)?.to_device(dev)?.to_dtype(DType::F32)?;
        let mut lat = noise.clone();
        let rope = tr.rope(l + 1)?;
        let splice = |lat: &Tensor, head: &Tensor| -> R<Tensor> { Ok(Tensor::cat(&[head, &lat.narrow(0, ov, l - ov)?.contiguous()?], 0)?.contiguous()?) };
        for i in 0..steps {
            let t = i as f32 / steps as f32;
            if let (true, Some((pl, _))) = (ov > 0, &prev) {
                // Перекрытие тянется к хвосту прошлого окна: (1 − (1−1e-6)·t)·шум + t·прежние латенты.
                let head = noise.narrow(0, 0, ov)?.contiguous()?.mul_scalar(1.0 - (1.0 - 1e-6) * t)?.add(&pl.narrow(0, 0, ov)?.contiguous()?.mul_scalar(t)?)?;
                lat = splice(&lat, &head)?;
            }
            let vc = tr.forward(&lat, &cond, t, &rope)?;
            let vu = tr.forward(&lat, &uncond, t, &rope)?;
            let v = vu.add(&vc.sub(&vu)?.mul_scalar(CFG_SCALE)?)?;
            lat = lat.add(&v.mul_scalar(1.0 / steps as f32)?)?;
            if !progress(ci * steps + i + 1, total) {
                return Err(Music3Error::Cancelled);
            }
        }
        if let (true, Some((pl, _))) = (ov > 0, &prev) {
            lat = splice(&lat, &pl.narrow(0, 0, ov)?.contiguous()?)?;
        }
        let a = l.saturating_sub(2 * OVERLAP);
        let b = l.saturating_sub(OVERLAP).max(a);
        prev = Some((lat.narrow(0, a, b - a)?.contiguous()?, cond.narrow(0, a, b - a)?.contiguous()?));
        out.push(lat);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows() {
        assert_eq!(chunk_starts(150), vec![0]);
        assert_eq!(chunk_starts(200), vec![0]);
        assert_eq!(chunk_starts(450), vec![0, 100, 200, 300]);
        assert_eq!(latent_len(200), 689);
        assert_eq!(latent_len(1), 3);
    }
}
