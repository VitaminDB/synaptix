//! Башня зрения Qwen3-VL (`Qwen3VLVisionModel`): патчи 16×16×2 → линейный
//! патч-эмбеддинг + учёная позиционная таблица 48×48 (билинейная
//! интерполяция под сетку картинки) → 27 блоков ViT (LayerNorm с bias,
//! 2D-RoPE по (h, w), полное внимание, MLP GELU-tanh с bias) → слияние 2×2
//! (`merger`: LayerNorm по патчу → Linear → GELU → Linear) в токены LLM
//! (4096). С блоков `deepstack_visual_indexes` снимаются отводы через свои
//! мерджеры (норма после склейки 2×2) — LLM прибавляет их к строкам картинки
//! после первых слоёв.
//!
//! Считается в F32: башня чувствительна, а её ошибку LLM размножает.

use synaptix_core::{
    device::Device,
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};
use synaptix_image_qwen::memory;
use synaptix_image_qwen::source::Weights;
use synaptix_nn::module::Module;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::layer_norm;

use crate::config::Qwen3VlVisionConfig;
use crate::image::VisionPatches;

pub const VIS: &str = "model.visual";
const ROPE_THETA: f64 = 10_000.0;

struct Norm {
    w: Tensor,
    b: Tensor,
}

impl Norm {
    fn load(w: &Weights, p: &str, dev: Device, dt: DType) -> Result<Self> {
        Ok(Self { w: w.get(&format!("{p}.weight"), dev, dt)?, b: w.get(&format!("{p}.bias"), dev, dt)? })
    }

    fn forward(&self, x: &Tensor, eps: f32) -> Result<Tensor> {
        layer_norm(x, Some(&self.w), Some(&self.b), eps)
    }
}

fn lin(w: &Weights, p: &str, dev: Device, dt: DType) -> Result<QuantLinear> {
    QuantLinear::build(w.get(&format!("{p}.weight"), dev, dt)?, Some(w.get(&format!("{p}.bias"), dev, dt)?), dt, dt)
}

struct VBlock {
    norm1: Norm,
    norm2: Norm,
    qkv: QuantLinear,
    proj: QuantLinear,
    fc1: QuantLinear,
    fc2: QuantLinear,
}

impl VBlock {
    fn load(w: &Weights, i: usize, dev: Device, dt: DType) -> Result<Self> {
        let p = format!("{VIS}.blocks.{i}");
        Ok(Self {
            norm1: Norm::load(w, &format!("{p}.norm1"), dev, dt)?,
            norm2: Norm::load(w, &format!("{p}.norm2"), dev, dt)?,
            qkv: lin(w, &format!("{p}.attn.qkv"), dev, dt)?,
            proj: lin(w, &format!("{p}.attn.proj"), dev, dt)?,
            fc1: lin(w, &format!("{p}.mlp.linear_fc1"), dev, dt)?,
            fc2: lin(w, &format!("{p}.mlp.linear_fc2"), dev, dt)?,
        })
    }
}

/// `Qwen3VLVisionPatchMerger`: норма (по патчу — у финального, по склейке
/// 2×2 — у deepstack), Linear → GELU (точный) → Linear.
struct Merger {
    norm: Norm,
    fc1: QuantLinear,
    fc2: QuantLinear,
    /// Норма после склейки 2×2 (`use_postshuffle_norm`).
    post: bool,
}

impl Merger {
    fn load(w: &Weights, p: &str, dev: Device, dt: DType, post: bool) -> Result<Self> {
        Ok(Self {
            norm: Norm::load(w, &format!("{p}.norm"), dev, dt)?,
            fc1: lin(w, &format!("{p}.linear_fc1"), dev, dt)?,
            fc2: lin(w, &format!("{p}.linear_fc2"), dev, dt)?,
            post,
        })
    }

    /// `[N, hidden]` → `[N/unit, out]`.
    fn forward(&self, x: &Tensor, unit: usize, eps: f32) -> Result<Tensor> {
        let d = x.dims();
        let (n, hid) = (d[0], d[1]);
        let h = if self.post {
            self.norm.forward(&x.reshape((n / unit, hid * unit))?, eps)?
        } else {
            self.norm.forward(x, eps)?.reshape((n / unit, hid * unit))?
        };
        let h = self.fc1.forward(&h)?.gelu_exact()?;
        self.fc2.forward(&h)
    }
}

pub struct VisionTower {
    cfg: Qwen3VlVisionConfig,
    device: Device,
    dtype: DType,
    patch_embed: QuantLinear,
    /// Таблица позиций `[side², hidden]` на CPU (F32) — интерполируется под
    /// сетку каждой картинки.
    pos_embed: Vec<f32>,
    blocks: Vec<VBlock>,
    merger: Merger,
    deepstack: Vec<Merger>,
}

/// Патчи в порядке блоков слияния → координаты `(h, w)`.
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

/// `torch.linspace(0, side − 1, n)` в f32 (симметричная схема torch).
fn linspace(side: usize, n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![0.0];
    }
    let end = (side - 1) as f32;
    let step = end / (n - 1) as f32;
    let half = n / 2;
    (0..n).map(|i| if i < half { i as f32 * step } else { end - (n - 1 - i) as f32 * step }).collect()
}

/// Внимание `[B, H, L, D]` без маски: flash (D дополняется до кратного 128)
/// на BF16/F16, иначе явный softmax порциями по запросам.
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
    /// Загрузить башню целиком (~0,6 млрд параметров: 2,4 ГБ в F32).
    pub fn load(w: &Weights, cfg: &Qwen3VlVisionConfig, device: Device, dtype: DType) -> Result<Self> {
        let _g = memory::weights_guard(device);
        let pe = w.get(&format!("{VIS}.patch_embed.proj.weight"), device, dtype)?;
        let n = pe.dims()[0];
        let k: usize = pe.dims()[1..].iter().product();
        if k != cfg.patch_features() {
            return Err(SynaptixError::Other(format!("vision: патч-эмбеддинг {k} ≠ {}", cfg.patch_features())));
        }
        let patch_embed = QuantLinear::build(
            pe.reshape((n, k))?,
            Some(w.get(&format!("{VIS}.patch_embed.proj.bias"), device, dtype)?),
            dtype,
            dtype,
        )?;
        let pos = w.get(&format!("{VIS}.pos_embed.weight"), Device::Cpu, DType::F32)?;
        let pos_embed = pos.flatten_all()?.to_vec1::<f32>()?;
        let blocks = (0..cfg.depth).map(|i| VBlock::load(w, i, device, dtype)).collect::<Result<Vec<_>>>()?;
        let deepstack = (0..cfg.deepstack_indexes.len())
            .map(|i| Merger::load(w, &format!("{VIS}.deepstack_merger_list.{i}"), device, dtype, true))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cfg: cfg.clone(),
            device,
            dtype,
            patch_embed,
            pos_embed,
            blocks,
            merger: Merger::load(w, &format!("{VIS}.merger"), device, dtype, false)?,
            deepstack,
        })
    }

    pub fn deepstack_len(&self) -> usize {
        self.deepstack.len()
    }

    /// Билинейная интерполяция таблицы `side×side` под сетку `gh×gw`
    /// (`fast_pos_embed_interpolate`: `linspace(0, side−1, n)` по осям), в
    /// порядке блоков слияния.
    fn interpolated_pos(&self, gh: usize, gw: usize) -> Vec<f32> {
        let side = self.cfg.pos_grid();
        let dim = self.cfg.hidden;
        let hs = linspace(side, gh);
        let ws = linspace(side, gw);
        let pos = patch_positions(gh, gw, self.cfg.merge_size);
        let mut out = vec![0f32; pos.len() * dim];
        for (i, &(ph, pw)) in pos.iter().enumerate() {
            let (hy, wx) = (hs[ph], ws[pw]);
            let (y0, x0) = (hy as usize, wx as usize);
            let (y1, x1) = ((y0 + 1).min(side - 1), (x0 + 1).min(side - 1));
            let (dy, dx) = (hy - y0 as f32, wx - x0 as f32);
            let w00 = (1.0 - dy) * (1.0 - dx);
            let w01 = (1.0 - dy) * dx;
            let w10 = dy * (1.0 - dx);
            let w11 = dy * dx;
            let (r00, r01, r10, r11) =
                ((y0 * side + x0) * dim, (y0 * side + x1) * dim, (y1 * side + x0) * dim, (y1 * side + x1) * dim);
            let o = &mut out[i * dim..(i + 1) * dim];
            for d in 0..dim {
                o[d] = w00 * self.pos_embed[r00 + d]
                    + w01 * self.pos_embed[r01 + d]
                    + w10 * self.pos_embed[r10 + d]
                    + w11 * self.pos_embed[r11 + d];
            }
        }
        out
    }

    /// Патчи одной картинки → (эмбеддинги токенов LLM `[N/4, out]`, отводы
    /// deepstack по слотам `[N/4, out]`) в порядке блоков слияния.
    pub fn forward(&self, p: &VisionPatches) -> Result<(Tensor, Vec<Tensor>)> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let cfg = &self.cfg;
        let (gt, gh, gw) = p.grid;
        if gt != 1 {
            return Err(SynaptixError::Other("vision: поддерживается только одна картинка (t = 1)".into()));
        }
        let (dev, dt) = (self.device, self.dtype);
        let unit = cfg.merge_unit();
        let s = gh * gw;
        let (heads, hd) = (cfg.num_heads, cfg.head_dim());
        let eps = cfg.layer_norm_eps;

        let x = p.patches.to_device(dev)?.to_dtype(dt)?;
        let x = self.patch_embed.forward(&x)?; // [S, hidden]
        let pos = Tensor::from_vec(self.interpolated_pos(gh, gw), (s, cfg.hidden), dev)?.to_dtype(dt)?;
        let mut x = x.add(&pos)?;

        // 2D-RoPE: половина головы — частоты по h, половина — по w
        // (`Qwen3VLVisionRotaryEmbedding(head_dim/2)`, theta 10000).
        let half = hd / 2;
        let nf = half / 2;
        let inv: Vec<f64> = (0..nf).map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f64 / half as f64)).collect();
        let coords = patch_positions(gh, gw, cfg.merge_size);
        let mut cos = vec![0f32; s * half];
        let mut sin = vec![0f32; s * half];
        for (row, &(ph, pw)) in coords.iter().enumerate() {
            for i in 0..nf {
                let ah = (ph as f32 * inv[i] as f32) as f64;
                let aw = (pw as f32 * inv[i] as f32) as f64;
                cos[row * half + i] = ah.cos() as f32;
                sin[row * half + i] = ah.sin() as f32;
                cos[row * half + nf + i] = aw.cos() as f32;
                sin[row * half + nf + i] = aw.sin() as f32;
            }
        }
        let cos = Tensor::from_vec(cos, (s, half), dev)?;
        let sin = Tensor::from_vec(sin, (s, half), dev)?;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut taps: Vec<Tensor> = Vec::with_capacity(self.deepstack.len());
        for (li, b) in self.blocks.iter().enumerate() {
            let h = b.norm1.forward(&x, eps)?;
            let qkv = b.qkv.forward(&h)?; // [S, 3·hidden]
            drop(h);
            let part = |i: usize| -> Result<Tensor> {
                qkv.narrow(1, i * heads * hd, heads * hd)?.contiguous()?.reshape((s, heads, hd))
            };
            let to_hsd = |t: Tensor| -> Result<Tensor> { t.transpose(0, 1)?.contiguous() };
            let q = rope(&to_hsd(part(0)?)?, &cos, &sin, hd)?.reshape((1, heads, s, hd))?;
            let k = rope(&to_hsd(part(1)?)?, &cos, &sin, hd)?.reshape((1, heads, s, hd))?;
            let v = to_hsd(part(2)?)?.reshape((1, heads, s, hd))?;
            drop(qkv);
            let a = attend(&q, &k, &v, scale)?; // [1, H, S, D]
            drop((q, k, v));
            let a = a.reshape((heads, s, hd))?.transpose(0, 1)?.contiguous()?.reshape((s, heads * hd))?;
            x = x.add(&b.proj.forward(&a)?)?;
            let h = b.norm2.forward(&x, eps)?;
            let h = b.fc1.forward(&h)?.gelu_tanh()?;
            x = x.add(&b.fc2.forward(&h)?)?;
            if let Some(slot) = cfg.deepstack_indexes.iter().position(|&d| d == li) {
                let feat = self.deepstack[slot].forward(&x, unit, eps)?;
                while taps.len() <= slot {
                    taps.push(feat.clone());
                }
                taps[slot] = feat;
            }
        }
        let out = self.merger.forward(&x, unit, eps)?;
        Ok((out, taps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_like_torch() {
        assert_eq!(linspace(48, 1), vec![0.0]);
        let l = linspace(48, 3);
        assert_eq!(l, vec![0.0, 23.5, 47.0]);
        let l = linspace(48, 48);
        assert_eq!(l[0], 0.0);
        assert_eq!(l[47], 47.0);
        assert!((l[10] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn positions_follow_merge_blocks() {
        let p = patch_positions(4, 4, 2);
        assert_eq!(p[..4], [(0, 0), (0, 1), (1, 0), (1, 1)]);
        assert_eq!(p[4], (0, 2));
        assert_eq!(p[15], (3, 3));
    }
}
