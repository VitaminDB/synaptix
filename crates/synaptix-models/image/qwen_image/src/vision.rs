//! Башня зрения Qwen2.5-VL (`Qwen2_5_VisionTransformerPretrainedModel`):
//! патчи 14×14×2 → 32 блока ViT (RMSNorm, SwiGLU с bias, 2D-RoPE по (h, w))
//! с оконным вниманием (окна 8×8 патчей) и полным — в блоках
//! `fullatt_block_indexes`, затем слияние 2×2 → токены LLM (3584).
//!
//! Токены внутри считаются в порядке окон (как у HF: `window_index`), на
//! выходе возвращаются в порядок блоков слияния. Голова 80 не подходит
//! flash-ядру (кратно 128), поэтому q/k/v дополняются нулями до 128 — на
//! скалярные произведения это не влияет, лишние столбцы выхода отрезаются.

use synaptix_core::{
    device::Device,
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};
use synaptix_nn::module::Module;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_ops::attention::softmax::scaled_dot_attention;

use crate::config::VisionConfig;
use crate::memory;
use crate::preprocess::VisionPatches;
use crate::source::Weights;

fn rms(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    if let Ok(y) = x.rms_norm_fused(w, eps, false) {
        return Ok(y);
    }
    synaptix_ops::norm::rms_norm::rms_norm(x, w, eps)
}

struct VBlock {
    norm1: Tensor,
    norm2: Tensor,
    qkv: QuantLinear,
    proj: QuantLinear,
    gate: QuantLinear,
    up: QuantLinear,
    down: QuantLinear,
}

impl VBlock {
    fn load(w: &Weights, i: usize, dev: Device, dt: DType) -> Result<Self> {
        let p = format!("visual.blocks.{i}");
        let t = |n: &str| w.get(&format!("{p}.{n}"), dev, dt);
        let l = |n: &str| QuantLinear::build(t(&format!("{n}.weight"))?, Some(t(&format!("{n}.bias"))?), dt, dt);
        Ok(Self {
            norm1: t("norm1.weight")?,
            norm2: t("norm2.weight")?,
            qkv: l("attn.qkv")?,
            proj: l("attn.proj")?,
            gate: l("mlp.gate_proj")?,
            up: l("mlp.up_proj")?,
            down: l("mlp.down_proj")?,
        })
    }
}

pub struct VisionTower {
    cfg: VisionConfig,
    device: Device,
    dtype: DType,
    patch_embed: QuantLinear,
    blocks: Vec<VBlock>,
    merger_norm: Tensor,
    fc1: QuantLinear,
    fc2: QuantLinear,
}

/// Порядок окон и границы отрезков (в патчах) — `get_vision_window_index`.
pub fn window_index(gh: usize, gw: usize, merge: usize, win: usize) -> (Vec<usize>, Vec<usize>) {
    let (lh, lw) = (gh / merge, gw / merge);
    let unit = merge * merge;
    let pad_h = win - lh % win;
    let pad_w = win - lw % win;
    let (nwh, nww) = ((lh + pad_h) / win, (lw + pad_w) / win);
    let mut index = Vec::with_capacity(lh * lw);
    let mut cu = vec![0usize];
    for wy in 0..nwh {
        for wx in 0..nww {
            let mut n = 0usize;
            for y in wy * win..(wy + 1) * win {
                for x in wx * win..(wx + 1) * win {
                    if y < lh && x < lw {
                        index.push(y * lw + x);
                        n += 1;
                    }
                }
            }
            let last = *cu.last().unwrap_or(&0);
            if n > 0 {
                cu.push(last + n * unit);
            }
        }
    }
    (index, cu)
}

/// Позиции (h, w) каждого патча в порядке блоков слияния.
fn patch_positions(gh: usize, gw: usize, merge: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(gh * gw);
    for bh in 0..gh / merge {
        for bw in 0..gw / merge {
            for mh in 0..merge {
                for mw in 0..merge {
                    out.push((bh * merge + mh, bw * merge + mw));
                }
            }
        }
    }
    out
}

/// Внимание `[B, H, L, D]` без маски; D дополняется до кратного 128 для
/// flash-ядра, иначе — явный softmax порциями по запросам.
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let d = q.dims().to_vec();
    let (b, h, l, hd) = (d[0], d[1], d[2], d[3]);
    if q.device().is_cuda() && matches!(q.dtype(), DType::BF16 | DType::F16) {
        let pd = hd.div_ceil(128) * 128;
        let pad = |t: &Tensor| -> Result<Tensor> {
            if pd == hd {
                return t.contiguous();
            }
            let z = Tensor::zeros((b, h, l, pd - hd), t.dtype(), t.device())?;
            Tensor::cat(&[t, &z], 3)?.contiguous()
        };
        if let Ok(o) = pad(q)?.flash_attention(&pad(k)?, &pad(v)?, scale, false) {
            return if pd == hd { Ok(o) } else { o.narrow(3, 0, hd)?.contiguous() };
        }
    }
    const Q_CHUNK: usize = 1024;
    if l <= Q_CHUNK {
        return scaled_dot_attention(q, k, v, scale, None);
    }
    let mut parts = Vec::with_capacity(l.div_ceil(Q_CHUNK));
    let mut s = 0;
    while s < l {
        let n = Q_CHUNK.min(l - s);
        parts.push(scaled_dot_attention(&q.narrow(2, s, n)?.contiguous()?, k, v, scale, None)?);
        s += n;
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 2)
}

/// Внимание по отрезкам `cu` (длины в токенах): подряд идущие отрезки
/// одной длины считаются одним батчем. q/k/v `[S, H, D]` → `[S, H·D]`.
fn segmented_attention(q: &Tensor, k: &Tensor, v: &Tensor, cu: &[usize], scale: f32) -> Result<Tensor> {
    let d = q.dims().to_vec();
    let (s, h, hd) = (d[0], d[1], d[2]);
    let mut outs: Vec<Tensor> = Vec::new();
    let mut i = 0usize;
    while i + 1 < cu.len() {
        let len = cu[i + 1] - cu[i];
        let mut j = i + 1;
        while j + 1 < cu.len() && cu[j + 1] - cu[j] == len {
            j += 1;
        }
        let (start, cnt) = (cu[i], j - i);
        let seg = |t: &Tensor| -> Result<Tensor> {
            t.narrow(0, start, cnt * len)?.contiguous()?.reshape((cnt, len, h, hd))?.permute([0, 2, 1, 3])?.contiguous()
        };
        let o = attend(&seg(q)?, &seg(k)?, &seg(v)?, scale)?; // [cnt, H, len, D]
        outs.push(o.permute([0, 2, 1, 3])?.contiguous()?.reshape((cnt * len, h * hd))?);
        i = j;
    }
    if outs.len() == 1 {
        return Ok(outs.pop().expect("one"));
    }
    let refs: Vec<&Tensor> = outs.iter().collect();
    let out = Tensor::cat(&refs, 0)?;
    if out.dims()[0] != s {
        return Err(SynaptixError::Other(format!("vision: отрезки покрыли {} из {s} токенов", out.dims()[0])));
    }
    Ok(out)
}

/// Split-half RoPE (`rotate_half`) по `[H, S, D]`, cos/sin `[S, D/2]`.
fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor, hd: usize) -> Result<Tensor> {
    if x.device().is_cuda() {
        if let Ok(y) = x.rope_split_partial_fused(cos, sin, hd) {
            return Ok(y);
        }
    }
    let half = hd / 2;
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let x0 = xf.narrow(2, 0, half)?.contiguous()?;
    let x1 = xf.narrow(2, half, half)?.contiguous()?;
    let o0 = x0.broadcast_mul(cos)?.sub(&x1.broadcast_mul(sin)?)?;
    let o1 = x1.broadcast_mul(cos)?.add(&x0.broadcast_mul(sin)?)?;
    Tensor::cat(&[&o0, &o1], 2)?.to_dtype(dt)
}

impl VisionTower {
    /// Загрузить башню целиком (~0,7 млрд параметров, 1,3 ГБ в BF16).
    pub fn load(w: &Weights, cfg: &VisionConfig, device: Device, dtype: DType) -> Result<Self> {
        let _g = memory::weights_guard(device);
        let pe = w.get("visual.patch_embed.proj.weight", device, dtype)?;
        let n = pe.dims()[0];
        let k: usize = pe.dims()[1..].iter().product();
        let patch_embed = QuantLinear::build(pe.reshape((n, k))?, None, dtype, dtype)?;
        let blocks = (0..cfg.depth).map(|i| VBlock::load(w, i, device, dtype)).collect::<Result<Vec<_>>>()?;
        let lin = |p: &str| -> Result<QuantLinear> {
            QuantLinear::build(
                w.get(&format!("{p}.weight"), device, dtype)?,
                Some(w.get(&format!("{p}.bias"), device, dtype)?),
                dtype,
                dtype,
            )
        };
        Ok(Self {
            cfg: cfg.clone(),
            device,
            dtype,
            patch_embed,
            blocks,
            merger_norm: w.get("visual.merger.ln_q.weight", device, dtype)?,
            fc1: lin("visual.merger.mlp.0")?,
            fc2: lin("visual.merger.mlp.2")?,
        })
    }

    /// Патчи одной картинки → эмбеддинги токенов LLM `[N/4, out_hidden]` в
    /// порядке блоков слияния.
    pub fn forward(&self, p: &VisionPatches) -> Result<Tensor> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let cfg = &self.cfg;
        let (gt, gh, gw) = p.grid;
        if gt != 1 {
            return Err(SynaptixError::Other("vision: поддерживается только одна картинка (t = 1)".into()));
        }
        let (dev, dt) = (self.device, self.dtype);
        let m = cfg.merge_size;
        let unit = m * m;
        let s = gh * gw;
        let (heads, hd) = (cfg.num_heads, cfg.head_dim());
        let (win_idx, cu_win) = window_index(gh, gw, m, cfg.window_units());
        // Порядок окон по блокам слияния → по патчам.
        let perm: Vec<u32> =
            win_idx.iter().flat_map(|&u| (0..unit).map(move |j| (u * unit + j) as u32)).collect();
        let perm_t = Tensor::from_vec(perm.clone(), (s,), dev)?;

        let x = p.patches.to_device(dev)?.to_dtype(dt)?;
        let x = self.patch_embed.forward(&x)?; // [S, hidden]
        let mut x = x.index_select(0, &perm_t)?.contiguous()?;

        // 2D-RoPE: половина головы — частоты по h, половина — по w.
        let half = hd / 2; // 40
        let nf = half / 2; // 20 частот на ось
        let inv: Vec<f32> = (0..nf).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / half as f32)).collect();
        let pos = patch_positions(gh, gw, m);
        let mut cos = vec![0f32; s * half];
        let mut sin = vec![0f32; s * half];
        for (row, &pi) in perm.iter().enumerate() {
            let (ph, pw) = pos[pi as usize];
            for i in 0..nf {
                let (ah, aw) = (ph as f32 * inv[i], pw as f32 * inv[i]);
                cos[row * half + i] = (ah as f64).cos() as f32;
                sin[row * half + i] = (ah as f64).sin() as f32;
                cos[row * half + nf + i] = (aw as f64).cos() as f32;
                sin[row * half + nf + i] = (aw as f64).sin() as f32;
            }
        }
        let cos = Tensor::from_vec(cos, (s, half), dev)?;
        let sin = Tensor::from_vec(sin, (s, half), dev)?;
        let cu_full = vec![0usize, s];
        let scale = 1.0 / (hd as f32).sqrt();

        for (li, b) in self.blocks.iter().enumerate() {
            let h = rms(&x, &b.norm1, 1e-6)?;
            let qkv = b.qkv.forward(&h)?; // [S, 3·hidden]
            drop(h);
            let part = |i: usize| -> Result<Tensor> {
                qkv.narrow(1, i * heads * hd, heads * hd)?.contiguous()?.reshape((s, heads, hd))
            };
            let to_hsd = |t: Tensor| -> Result<Tensor> { t.transpose(0, 1)?.contiguous() };
            let q = rope(&to_hsd(part(0)?)?, &cos, &sin, hd)?.transpose(0, 1)?.contiguous()?;
            let k = rope(&to_hsd(part(1)?)?, &cos, &sin, hd)?.transpose(0, 1)?.contiguous()?;
            let v = part(2)?;
            drop(qkv);
            let cu = if cfg.fullatt_blocks.contains(&li) { &cu_full } else { &cu_win };
            let a = segmented_attention(&q, &k, &v, cu, scale)?;
            drop((q, k, v));
            x = x.add(&b.proj.forward(&a)?)?;
            let h = rms(&x, &b.norm2, 1e-6)?;
            let g = b.gate.forward(&h)?;
            let u = b.up.forward(&h)?;
            let act = match g.silu_and_mul(&u) {
                Ok(a) => a,
                Err(_) => g.silu()?.mul(&u)?,
            };
            x = x.add(&b.down.forward(&act)?)?;
        }

        let h = rms(&x, &self.merger_norm, 1e-6)?;
        let h = h.reshape((s / unit, cfg.hidden * unit))?;
        let h = self.fc1.forward(&h)?.gelu_exact()?;
        let out = self.fc2.forward(&h)?; // [S/4, out] в порядке окон
        // Обратно в порядок блоков слияния.
        let mut rev = vec![0u32; win_idx.len()];
        for (i, &u) in win_idx.iter().enumerate() {
            rev[u] = i as u32;
        }
        let rev = Tensor::from_vec(rev, (win_idx.len(),), dev)?;
        out.index_select(0, &rev)?.contiguous()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_index_matches_hf() {
        // grid 12×20 патчей → 6×10 блоков слияния, окна 4×4.
        let (idx, cu) = window_index(12, 20, 2, 4);
        assert_eq!(idx.len(), 60);
        // Первое окно: строки 0..4, столбцы 0..4.
        assert_eq!(&idx[..4], &[0, 1, 2, 3]);
        assert_eq!(idx[4], 10);
        // Окна: по высоте 4 + 2, по ширине 4 + 4 + 2.
        let lens: Vec<usize> = cu.windows(2).map(|w| (w[1] - w[0]) / 4).collect();
        assert_eq!(lens, vec![16, 16, 8, 8, 8, 4]);
        assert_eq!(*cu.last().unwrap(), 240);
        let mut s = idx.clone();
        s.sort();
        assert_eq!(s, (0..60).collect::<Vec<_>>());
    }

    #[test]
    fn window_index_exact_multiple() {
        // 8×8 блоков — ровно 4 окна, пустые окна паддинга выкинуты.
        let (idx, cu) = window_index(16, 16, 2, 4);
        assert_eq!(idx.len(), 64);
        assert_eq!(cu, vec![0, 64, 128, 192, 256]);
    }
}
