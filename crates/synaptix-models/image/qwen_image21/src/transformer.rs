//! `QwenImage21Transformer2DModel` — single-stream DiT Qwen-Image 2.1 (32
//! блока, 7 млрд параметров), по diffusers `transformer_qwenimage21.py`:
//! - текст и латенты живут в одной последовательности: токены референсов
//!   подставляются в текстовый поток на слоты `<|image_pad|>` энкодера (один
//!   слот = 2×2 латентных токена), генерируемая картинка — в конец;
//! - одна общая модуляция `Linear(d → 4d)` от времени на все блоки:
//!   `[scale₁, gate₁, scale₂, gate₂]`, `x·(1 + scale)`, остаток `tanh(gate)·y`;
//!   `causal_condition`: текст и референсы модулируются временем 0;
//! - внимание блочно-причинное: `q ≥ k` или та же картинка — текст видит
//!   только прошлое, картинка внутри себя двунаправленна, генерируемая видит
//!   всё; q/k/v/out без bias, RMSNorm по голове для q/k, FF — SwiGLU на 3d;
//! - RoPE по трём осям (кадр 16, высота 56, ширина 56), theta 10000: текст
//!   двигает общую позицию, картинка замораживает ось кадра и раскладывает
//!   токены по центрированной сетке (h, w);
//! - KV-кэш: K/V текста и референсов не зависят от шага (время 0), поэтому
//!   первый шаг их считает и запоминает, остальные считают только токены
//!   генерируемой картинки.
//!
//! Память: блоки, не влезшие в VRAM, стримятся в forward — из пиннованной
//! копии на хосте или из mmap-источника (схема Qwen-Image / FLUX.2).

use synaptix_core::{
    device::Device,
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};
use synaptix_image_qwen::memory;
use synaptix_image_qwen::source::Weights;
pub use synaptix_image_qwen::transformer::{build_rope, weight_bytes, Placement, Residency};
use synaptix_nn::module::Module;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::{layer_norm, rms_norm};

use crate::config::Qwen21Config;

/// Слот картинки у энкодера (`<|image_pad|>`) = 2×2 латентных токена.
pub const TOKENS_PER_SLOT: usize = 4;

type Getter<'a> = dyn Fn(&str, DType) -> Result<Tensor> + 'a;

/// Линейный слой без bias: квантуемый вес читается сразу в F16, плотный — в
/// `compute`.
fn lin_g(g: &Getter<'_>, name: &str, cuda: bool, compute: DType, quant: DType) -> Result<QuantLinear> {
    let q = quant.is_quantized() && cuda;
    let raw = g(&format!("{name}.weight"), if q { DType::F16 } else { compute })?;
    QuantLinear::build(raw, None, if q { quant } else { compute }, compute)
}

fn dense(w: &Weights, name: &str, dev: Device, compute: DType) -> Result<QuantLinear> {
    lin_g(&|n, dt| w.get(n, dev, dt), name, dev.is_cuda(), compute, compute)
}

/// `QwenImage21TemporalTimesteps`: `t·1000` (F32) × частоты
/// `exp(−ln 10000·i/half)` (округлены до типа модели), выход `[cos, sin]`.
fn timestep_embedding(t: f32, dim: usize, dt: DType, device: Device) -> Result<Tensor> {
    let half = dim / 2;
    let round = |v: f32| -> f32 {
        match dt {
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            DType::F16 => half::f16::from_f32(v).to_f32(),
            _ => v,
        }
    };
    let ln_max = (10000.0_f32).ln();
    let t1000 = 1000.0 * t;
    let mut v = vec![0f32; dim];
    for i in 0..half {
        let freq = round((-ln_max * i as f32 / half as f32).exp());
        let a = t1000 * freq;
        v[i] = a.cos();
        v[half + i] = a.sin();
    }
    Tensor::from_vec(v, (1, dim), device)
}

/// Позиции RoPE склейки (`QwenImage21Rope`): `img_shapes` — сетки картинок в
/// латентных токенах (референсы, затем генерируемая), `image_pad_mask` — где
/// в склейке стоят токены картинок. Текст двигает позицию по всем осям,
/// картинка замораживает ось кадра и центрирует (h, w); после неё позиция
/// сдвигается на `max(h, w)`.
pub fn rope_positions(img_shapes: &[(usize, usize)], image_pad_mask: &[bool]) -> Result<Vec<[i32; 3]>> {
    let total = image_pad_mask.len();
    let mut out: Vec<[i32; 3]> = Vec::with_capacity(total);
    let (mut cursor, mut position) = (0usize, 0i32);
    for &(h, w) in img_shapes {
        let block_start = (cursor..total)
            .find(|&i| image_pad_mask[i])
            .ok_or_else(|| SynaptixError::Other("rope: картинок больше, чем блоков в маске".into()))?;
        let text_len = block_start - cursor;
        for l in 0..text_len as i32 {
            out.push([position + l; 3]);
        }
        position += text_len as i32;
        cursor = block_start + h * w;
        if cursor > total || image_pad_mask[block_start..cursor].iter().any(|&m| !m) {
            return Err(SynaptixError::Other(format!("rope: блок {h}×{w} не помещается в маску картинок")));
        }
        let (oh, ow) = ((h - h / 2) as i32, (w - w / 2) as i32);
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                out.push([position, y - oh, x - ow]);
            }
        }
        position += h.max(w) as i32;
    }
    for l in 0..(total - cursor) as i32 {
        out.push([position + l; 3]);
    }
    Ok(out)
}

/// Раскладка совместной последовательности одного прогона — считается один
/// раз на сэмпл и общая для всех шагов.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// Длина выхода энкодера (без системной части).
    pub enc_len: usize,
    /// Где в склейке стоят токены картинок (референсы и генерируемая).
    pub image_pad_mask: Vec<bool>,
    /// Позиция склейки → строка в `cat(txt, img)`: текст — `i`, картинки —
    /// `enc_len + k` по порядку токенов латента.
    pub gather: Vec<u32>,
    /// Отрезки префикса `(start, end, текст?)` для блочно-причинного
    /// внимания; генерируемая картинка — всё после `prefix_len`.
    pub segments: Vec<(usize, usize, bool)>,
    pub prefix_len: usize,
    pub n_target: usize,
    /// Позиции RoPE каждого токена склейки.
    pub positions: Vec<[i32; 3]>,
}

impl Layout {
    /// `enc_pad_mask` — `<|image_pad|>` в выходе энкодера, `img_shapes` —
    /// сетки референсов и (последней) генерируемой картинки.
    pub fn build(enc_pad_mask: &[bool], img_shapes: &[(usize, usize)]) -> Result<Self> {
        let (&(th, tw), conds) =
            img_shapes.split_last().ok_or_else(|| SynaptixError::Other("layout: нет генерируемой картинки".into()))?;
        let n_target = th * tw;
        if n_target % TOKENS_PER_SLOT != 0 {
            return Err(SynaptixError::Other(format!("layout: {th}×{tw} токенов не делится на слоты по 4")));
        }
        let cond_tokens: usize = conds.iter().map(|&(h, w)| h * w).sum();
        let slots = enc_pad_mask.iter().filter(|&&m| m).count();
        if slots * TOKENS_PER_SLOT != cond_tokens {
            return Err(SynaptixError::Other(format!(
                "layout: у энкодера {slots} слотов картинок ({} токенов), латенты референсов дают {cond_tokens}",
                slots * TOKENS_PER_SLOT
            )));
        }
        let enc_len = enc_pad_mask.len();
        let mut image_pad_mask = Vec::with_capacity(enc_len + cond_tokens + n_target);
        let mut gather = Vec::with_capacity(image_pad_mask.capacity());
        let mut img_row = enc_len as u32;
        for (i, &m) in enc_pad_mask.iter().enumerate() {
            if m {
                for _ in 0..TOKENS_PER_SLOT {
                    image_pad_mask.push(true);
                    gather.push(img_row);
                    img_row += 1;
                }
            } else {
                image_pad_mask.push(false);
                gather.push(i as u32);
            }
        }
        for _ in 0..n_target {
            image_pad_mask.push(true);
            gather.push(img_row);
            img_row += 1;
        }
        let total = image_pad_mask.len();
        let prefix_len = total - n_target;
        // Блоки картинок — по длинам сеток, а не по сериям True: два соседних
        // референса без текста между ними должны остаться разными блоками.
        let mut ids: Vec<i64> = vec![-1; total];
        let mut positions_of_images = image_pad_mask.iter().enumerate().filter(|(_, &m)| m).map(|(i, _)| i);
        for (b, &(h, w)) in img_shapes.iter().enumerate() {
            for _ in 0..h * w {
                let p = positions_of_images.next().ok_or_else(|| SynaptixError::Other("layout: маска короче".into()))?;
                ids[p] = b as i64;
            }
        }
        let mut segments = Vec::new();
        let mut start = 0usize;
        for i in 1..=prefix_len {
            if i == prefix_len || ids[i] != ids[start] {
                segments.push((start, i, ids[start] < 0));
                start = i;
            }
        }
        let positions = rope_positions(img_shapes, &image_pad_mask)?;
        Ok(Self { enc_len, image_pad_mask, gather, segments, prefix_len, n_target, positions })
    }

    pub fn total(&self) -> usize {
        self.image_pad_mask.len()
    }
}

/// `apply_rotary_emb_qwen` (комплексное умножение пар) по `[B,S,H,D]`.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    match x.rope_interleaved_fused(cos, sin) {
        Ok(out) => return Ok(out),
        Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
        Err(e) => return Err(e),
    }
    let dd = x.dims();
    let (b, s, h, d) = (dd[0], dd[1], dd[2], dd[3]);
    let xf = x.to_dtype(DType::F32)?;
    let pairs = xf.reshape((b, s, h, d / 2, 2))?;
    let even = pairs.narrow(4, 0, 1)?.contiguous()?;
    let odd = pairs.narrow(4, 1, 1)?.contiguous()?;
    let rot = Tensor::cat(&[&odd.neg()?, &even], 4)?.contiguous()?.reshape((b, s, h, d))?;
    xf.broadcast_mul(cos)?.add(&rot.broadcast_mul(sin)?)?.to_dtype(x.dtype())
}

/// SDPA `[1,H,Sq,D]` × `[1,H,Sk,D]` → `[1,H,Sq,D]`; при `causal` диагональ
/// выровнена по концу (запрос `i` видит ключи `≤ Sk − Sq + i`): так работает
/// flash-ядро, так же строится явная маска.
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    let hd = q.dims()[3];
    let scale = 1.0 / (hd as f32).sqrt();
    if q.device().is_cuda() && matches!(q.dtype(), DType::BF16 | DType::F16) {
        match q.flash_attention(k, v, scale, causal) {
            Ok(o) => return Ok(o),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
            Err(e) => return Err(e),
        }
    }
    let (sq, sk) = (q.dims()[2], k.dims()[2]);
    let mask = if causal {
        let mut m = vec![0f32; sq * sk];
        let off = sk - sq;
        for i in 0..sq {
            for j in (off + i + 1)..sk {
                m[i * sk + j] = f32::NEG_INFINITY;
            }
        }
        Some(Tensor::from_vec(m, (1, 1, sq, sk), q.device())?.to_dtype(DType::F32)?)
    } else {
        None
    };
    let (qf, kf, vf) = (q.to_dtype(DType::F32)?, k.to_dtype(DType::F32)?, v.to_dtype(DType::F32)?);
    let kc = kf.contiguous()?;
    let vc = vf.contiguous()?;
    scaled_dot_attention(&qf.contiguous()?, &kc, &vc, scale, mask.as_ref())?.to_dtype(q.dtype())
}

/// Проекция: MXFP8 считает GEMM с выходом в F16 — вход приводится к
/// максимуму 1, чтобы остаточный поток не переполнял F16 (как у Qwen-Image).
fn qf(l: &QuantLinear, x: &Tensor) -> Result<Tensor> {
    if l.is_mxfp8() {
        l.forward_target(x, 1.0)
    } else {
        l.forward(x)
    }
}

fn to_heads(t: &Tensor, heads: usize, hd: usize) -> Result<Tensor> {
    let d = t.dims();
    t.reshape((d[0], d[1], heads, hd))
}

/// `[B,S,H,D]` → `[B,H,S,D]` contiguous.
fn heads_first(t: &Tensor) -> Result<Tensor> {
    t.transpose(1, 2)?.contiguous()
}

fn res_add(acc: &Tensor, contrib: &Tensor) -> Result<Tensor> {
    if acc.dtype() == contrib.dtype() {
        acc.add(contrib)
    } else {
        acc.add(&contrib.to_dtype(acc.dtype())?)
    }
}

/// Модуляция токенов: у префикса (текст и референсы) — строка времени 0, у
/// генерируемой картинки — строка шага. `[1, 1, D]` каждая.
struct Mod<'a> {
    scale: &'a Tensor,
    gate: &'a Tensor,
    /// `(scale, gate)` времени 0 для первых `prefix` токенов.
    zero: Option<(&'a Tensor, &'a Tensor, usize)>,
}

impl Mod<'_> {
    /// `x·(1 + scale)` по отрезкам.
    fn modulate(&self, x: &Tensor) -> Result<Tensor> {
        match self.zero {
            None | Some((_, _, 0)) => x.broadcast_mul(&self.scale.add_scalar(1.0)?),
            Some((zs, _, p)) => {
                let n = x.dims()[1];
                let a = x.narrow(1, 0, p)?.contiguous()?.broadcast_mul(&zs.add_scalar(1.0)?)?;
                if p == n {
                    return Ok(a);
                }
                let b = x.narrow(1, p, n - p)?.contiguous()?.broadcast_mul(&self.scale.add_scalar(1.0)?)?;
                Tensor::cat(&[&a, &b], 1)
            }
        }
    }

    /// `tanh(gate)·y` по отрезкам.
    fn gated(&self, y: &Tensor) -> Result<Tensor> {
        match self.zero {
            None | Some((_, _, 0)) => y.broadcast_mul(&self.gate.tanh()?),
            Some((_, zg, p)) => {
                let n = y.dims()[1];
                let a = y.narrow(1, 0, p)?.contiguous()?.broadcast_mul(&zg.tanh()?)?;
                if p == n {
                    return Ok(a);
                }
                let b = y.narrow(1, p, n - p)?.contiguous()?.broadcast_mul(&self.gate.tanh()?)?;
                Tensor::cat(&[&a, &b], 1)
            }
        }
    }
}

/// Веса блока.
struct Block {
    to_q: QuantLinear,
    to_k: QuantLinear,
    to_v: QuantLinear,
    to_out: QuantLinear,
    norm_q: Tensor,
    norm_k: Tensor,
    gate_layer: QuantLinear,
    proj: QuantLinear,
    out: QuantLinear,
}

fn block_prefix(idx: usize) -> String {
    format!("transformer_blocks.{idx}.")
}

impl Block {
    fn load(w: &Weights, idx: usize, dev: Device, compute: DType, quant: DType) -> Result<Self> {
        Self::load_g(&|n, dt| w.get(n, dev, dt), idx, dev.is_cuda(), compute, quant)
    }

    fn load_g(g: &Getter<'_>, idx: usize, cuda: bool, compute: DType, quant: DType) -> Result<Self> {
        let p = format!("transformer_blocks.{idx}");
        let lin = |n: &str| lin_g(g, &format!("{p}.{n}"), cuda, compute, quant);
        Ok(Self {
            to_q: lin("attn.to_q")?,
            to_k: lin("attn.to_k")?,
            to_v: lin("attn.to_v")?,
            to_out: lin("attn.to_out.0")?,
            norm_q: g(&format!("{p}.attn.norm_q.weight"), compute)?,
            norm_k: g(&format!("{p}.attn.norm_k.weight"), compute)?,
            gate_layer: lin("img_mlp.gate_layer")?,
            proj: lin("img_mlp.proj")?,
            out: lin("img_mlp.out")?,
        })
    }

    fn to_device(&self, dev: Device) -> Result<Self> {
        let q = |l: &QuantLinear| l.to_device(dev);
        Ok(Self {
            to_q: q(&self.to_q)?,
            to_k: q(&self.to_k)?,
            to_v: q(&self.to_v)?,
            to_out: q(&self.to_out)?,
            norm_q: self.norm_q.to_device(dev)?,
            norm_k: self.norm_k.to_device(dev)?,
            gate_layer: q(&self.gate_layer)?,
            proj: q(&self.proj)?,
            out: q(&self.out)?,
        })
    }

    /// Внимание блока. `x` — модулированный вход `[1, S, D]`: в prefill —
    /// вся склейка (`segments` префикса + генерируемая после `prefix`), в
    /// decode — только генерируемая, K/V префикса берутся из кэша.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        cfg: &Qwen21Config,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        layout: &Layout,
        cache: Option<&mut Option<(Tensor, Tensor)>>,
        decode: bool,
    ) -> Result<Tensor> {
        let (h, hd) = (cfg.num_heads, cfg.head_dim);
        let d = x.dims();
        let (b, s) = (d[0], d[1]);
        let q = rms_norm(&to_heads(&qf(&self.to_q, x)?, h, hd)?, &self.norm_q, cfg.eps)?;
        let k = rms_norm(&to_heads(&qf(&self.to_k, x)?, h, hd)?, &self.norm_k, cfg.eps)?;
        let v = to_heads(&qf(&self.to_v, x)?, h, hd)?;
        let q = heads_first(&apply_rope(&q, cos, sin)?)?;
        let k = heads_first(&apply_rope(&k, cos, sin)?)?;
        let v = heads_first(&v)?;

        let out = if decode {
            let (ck, cv) = match cache {
                Some(Some((ck, cv))) => (ck.clone(), cv.clone()),
                _ => return Err(SynaptixError::Other("qwen-image-2.1: decode без заполненного KV-кэша".into())),
            };
            let k_all = Tensor::cat(&[&ck, &k], 2)?;
            let v_all = Tensor::cat(&[&cv, &v], 2)?;
            attend(&q, &k_all, &v_all, false)?
        } else {
            let p = layout.prefix_len;
            let mut parts: Vec<Tensor> = Vec::with_capacity(layout.segments.len() + 1);
            for &(start, end, is_text) in &layout.segments {
                let qs = q.narrow(2, start, end - start)?.contiguous()?;
                let ks = k.narrow(2, 0, end)?;
                let vs = v.narrow(2, 0, end)?;
                parts.push(attend(&qs, &ks, &vs, is_text)?);
            }
            if p < s {
                let qt = q.narrow(2, p, s - p)?.contiguous()?;
                parts.push(attend(&qt, &k, &v, false)?);
            }
            if let Some(slot) = cache {
                // `clone`, а не view: кэш не должен держать K/V всей склейки.
                *slot = Some((k.narrow(2, 0, p)?.contiguous()?, v.narrow(2, 0, p)?.contiguous()?));
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            Tensor::cat(&refs, 2)?
        };
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, s, h * hd))?;
        qf(&self.to_out, &out)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        cfg: &Qwen21Config,
        x: &Tensor,
        m1: &Mod<'_>,
        m2: &Mod<'_>,
        cos: &Tensor,
        sin: &Tensor,
        layout: &Layout,
        cache: Option<&mut Option<(Tensor, Tensor)>>,
        decode: bool,
    ) -> Result<Tensor> {
        let n1 = m1.modulate(&layer_norm(x, None, None, cfg.eps)?)?;
        let a = self.attention(cfg, &n1, cos, sin, layout, cache, decode)?;
        drop(n1);
        let x = res_add(x, &m1.gated(&a)?)?;
        drop(a);
        let n2 = m2.modulate(&layer_norm(&x, None, None, cfg.eps)?)?;
        let g = qf(&self.gate_layer, &n2)?;
        let u = qf(&self.proj, &n2)?;
        drop(n2);
        let act = match g.silu_and_mul(&u) {
            Ok(a) => a,
            Err(_) => g.silu()?.mul(&u)?,
        };
        drop((g, u));
        let ff = qf(&self.out, &act)?;
        drop(act);
        res_add(&x, &m2.gated(&ff)?)
    }
}

enum Slot {
    Device(Block),
    Host(Block),
    Source,
}

/// Блок, доставленный потоком префетча: готовый (копия с хоста) или сырые
/// тензоры источника в типе файла — квант и приведение типа делаются в
/// основном потоке.
enum Staged {
    Ready(Block),
    Raw(std::collections::HashMap<String, Tensor>),
}

/// Модуляции на моменты расписания: строка `r` — время `sigmas[r]`,
/// последняя — время 0 (для текста и референсов при `causal_condition`).
pub struct ModTable {
    sigmas: Vec<f32>,
    /// `[R+1, 4D]`: scale₁ ‖ gate₁ ‖ scale₂ ‖ gate₂.
    rows: Tensor,
    /// `norm_out`: `[R+1, D]` (только scale).
    out: Tensor,
    zero_row: Option<usize>,
}

impl ModTable {
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    fn slice(&self, row: usize, k: usize, d: usize) -> Result<Tensor> {
        self.rows.narrow(0, row, 1)?.narrow(1, k * d, d)?.contiguous()?.reshape((1, 1, d))
    }
}

/// K/V префикса по блокам (`[1, H, prefix, D]`), заполняется первым шагом.
pub struct KvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
}

impl KvCache {
    pub fn is_filled(&self) -> bool {
        !self.layers.is_empty() && self.layers.iter().all(|l| l.is_some())
    }

    pub fn clear(&mut self) {
        for l in &mut self.layers {
            *l = None;
        }
    }

    /// Байт на кэш префикса в `prefix` токенов.
    pub fn bytes_for(cfg: &Qwen21Config, prefix: usize, compute: DType) -> usize {
        2 * cfg.num_layers * prefix * cfg.inner() * compute.bytes_for_numel(1).max(1)
    }
}

pub struct QwenImage21Transformer {
    cfg: Qwen21Config,
    device: Device,
    compute: DType,
    quant: DType,
    img_in: QuantLinear,
    /// `text_norm.weight + 1` в F32.
    txt_norm: Tensor,
    txt_in: QuantLinear,
    txt_out: QuantLinear,
    t_l1: QuantLinear,
    t_l2: QuantLinear,
    modulation: QuantLinear,
    norm_out: QuantLinear,
    proj_out: QuantLinear,
    slots: Vec<Slot>,
    weights: Weights,
}

impl QwenImage21Transformer {
    /// Загрузить DiT. `tokens` — длина самого крупного прогона (склейка:
    /// текст + референсы + латент): от неё запас VRAM под активации.
    pub fn load(
        w: &Weights,
        cfg: &Qwen21Config,
        device: Device,
        compute: DType,
        quant: DType,
        placement: Placement,
        tokens: usize,
    ) -> Result<Self> {
        let cuda = device.is_cuda();
        let quant = if cuda { quant } else { compute };
        let _g = memory::weights_guard(device);
        let img_in = dense(w, "img_in", device, compute)?;
        let txt_norm = w.get("txt_in.text_norm.weight", device, DType::F32)?.add_scalar(1.0)?;
        let txt_in = dense(w, "txt_in.in_layer", device, compute)?;
        let txt_out = dense(w, "txt_in.out_layer", device, compute)?;
        let t_l1 = dense(w, "time_text_embed.timestep_embedder.linear_1", device, compute)?;
        let t_l2 = dense(w, "time_text_embed.timestep_embedder.linear_2", device, compute)?;
        let modulation = dense(w, "modulation.1", device, compute)?;
        let norm_out = dense(w, "norm_out.linear", device, compute)?;
        let proj_out = dense(w, "proj_out", device, compute)?;

        let n = cfg.num_layers;
        let d = cfg.inner();
        let blk = weight_bytes(cfg.block_params(), quant, compute);
        // Сырой F16-вес крупнейшей проекции (d → 3d) перед квантованием.
        let raw_transient = if quant.is_quantized() { d * cfg.mlp_hidden() * 2 } else { 0 };
        let act = memory::dit_activation_bytes(tokens, d);
        let reserve = act + 2 * blk + raw_transient + memory::DESKTOP_MARGIN;
        let force_source = std::env::var("QWEN_IMAGE_STREAM_FROM").is_ok_and(|v| v == "source");
        let host_ok = |need: u64| {
            !force_source && memory::host_available().map(|a| a > need + (12u64 << 30)).unwrap_or(false)
        };

        let mut slots = Vec::with_capacity(n);
        let mut first_off: Option<usize> = match placement {
            Placement::Stream if cuda => Some(0),
            _ => None,
        };
        let mut use_host = false;
        for i in 0..n {
            if cuda && first_off.is_none() && placement == Placement::Auto && memory::free_vram(device) < reserve + blk {
                memory::release_pools(device);
                if memory::free_vram(device) < reserve + blk {
                    first_off = Some(i);
                }
            }
            if let Some(f) = first_off {
                if i == f {
                    let rest = ((n - i) * blk) as u64;
                    use_host = host_ok(rest);
                    eprintln!(
                        "[qwen-image-2.1] VRAM: блоки {i}..{n} ({:.1} ГБ) стримятся {} (свободно {:.1} ГБ, запас {:.1} ГБ)",
                        rest as f64 / 1e9,
                        if use_host { "из пиннованной копии на хосте" } else { "из источника (mmap)" },
                        memory::free_vram(device) as f64 / 1e9,
                        reserve as f64 / 1e9,
                    );
                }
                if use_host {
                    let b = Block::load(w, i, device, compute, quant)?;
                    synaptix_core::device::cuda::set_offload_pinned(true);
                    let host = b.to_device(Device::Cpu);
                    synaptix_core::device::cuda::set_offload_pinned(false);
                    drop(b);
                    slots.push(Slot::Host(host?));
                    memory::release_pools(device);
                } else {
                    slots.push(Slot::Source);
                }
            } else {
                slots.push(Slot::Device(Block::load(w, i, device, compute, quant)?));
            }
        }
        Ok(Self {
            cfg: cfg.clone(),
            device,
            compute,
            quant,
            img_in,
            txt_norm,
            txt_in,
            txt_out,
            t_l1,
            t_l2,
            modulation,
            norm_out,
            proj_out,
            slots,
            weights: w.clone(),
        })
    }

    pub fn config(&self) -> &Qwen21Config {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn compute_dtype(&self) -> DType {
        self.compute
    }

    pub fn quant_dtype(&self) -> DType {
        self.quant
    }

    pub fn residency(&self) -> Residency {
        let mut r = Residency::default();
        for s in &self.slots {
            match s {
                Slot::Device(_) => r.device += 1,
                Slot::Host(_) => r.host += 1,
                Slot::Source => r.source += 1,
            }
        }
        r
    }

    pub fn new_cache(&self) -> KvCache {
        KvCache { layers: (0..self.cfg.num_layers).map(|_| None).collect() }
    }

    /// Эмбеддинг времени `[1, D]`: как у пайплайна, время приводится к типу
    /// скрытых состояний (`bf16(σ·1000)/1000`), частоты — к типу модели.
    fn temb(&self, sigma: f32) -> Result<Tensor> {
        let dt = self.compute;
        let round = |v: f32| -> f32 {
            match dt {
                DType::BF16 => half::bf16::from_f32(v).to_f32(),
                DType::F16 => half::f16::from_f32(v).to_f32(),
                _ => v,
            }
        };
        let t = round(round(sigma * 1000.0) / 1000.0);
        let e = timestep_embedding(t, 256, dt, self.device)?.to_dtype(dt)?;
        self.t_l2.forward(&self.t_l1.forward(&e)?.silu()?)
    }

    /// Модуляции на моменты `sigmas` (+ время 0 при `causal_condition`).
    pub fn modulations(&self, sigmas: &[f32]) -> Result<ModTable> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let mut rows: Vec<Tensor> = sigmas.iter().map(|&s| self.temb(s)).collect::<Result<_>>()?;
        let zero_row = if self.cfg.causal_condition {
            rows.push(self.temb(0.0)?);
            Some(rows.len() - 1)
        } else {
            None
        };
        let refs: Vec<&Tensor> = rows.iter().collect();
        let act = Tensor::cat(&refs, 0)?.silu()?; // [R(+1), D]
        drop(rows);
        Ok(ModTable {
            sigmas: sigmas.to_vec(),
            rows: self.modulation.forward(&act)?,
            out: self.norm_out.forward(&act)?,
            zero_row,
        })
    }

    /// Текст энкодера `[1, St, context]` → вход текста `[1, St, D]`
    /// (`txt_in`: RMSNorm с нулецентрированным весом в F32, Linear, GELU-tanh,
    /// Linear): одинаков на всех шагах.
    pub fn embed_text(&self, txt: &Tensor) -> Result<Tensor> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let x = txt.to_device(self.device)?.to_dtype(DType::F32)?;
        let x = rms_norm(&x, &self.txt_norm, self.cfg.eps)?.to_dtype(self.compute)?;
        let x = self.txt_in.forward(&x)?.gelu_tanh()?;
        self.txt_out.forward(&x)
    }

    fn stage(&self, idx: usize) -> Result<Staged> {
        match &self.slots[idx] {
            Slot::Device(_) => Err(SynaptixError::Other(format!("qwen-image-2.1: блок {idx} и так на карте"))),
            Slot::Host(b) => b.to_device(self.device).map(Staged::Ready),
            Slot::Source => {
                let prefix = block_prefix(idx);
                let mut raw = std::collections::HashMap::new();
                for n in self.weights.names().into_iter().filter(|n| n.starts_with(&prefix)) {
                    let dt = self.weights.raw(&n).map(|(_, dt, _)| dt).unwrap_or(self.compute);
                    let t = self.weights.get(&n, self.device, dt)?;
                    raw.insert(n, t);
                }
                Ok(Staged::Raw(raw))
            }
        }
    }

    fn finish(&self, idx: usize, s: Staged) -> Result<Block> {
        match s {
            Staged::Ready(b) => Ok(b),
            Staged::Raw(raw) => {
                let get = |n: &str, dt: DType| -> Result<Tensor> {
                    let t = raw.get(n).ok_or_else(|| SynaptixError::Other(format!("qwen-image-2.1: нет тензора {n}")))?;
                    if t.dtype() == dt {
                        Ok(t.clone())
                    } else {
                        t.to_dtype(dt)
                    }
                };
                Block::load_g(&get, idx, self.device.is_cuda(), self.compute, self.quant)
            }
        }
    }

    /// Проход по блокам: резидентные как есть, остальные приезжают на карту,
    /// следующий нерезидентный — на loader-стриме параллельно счёту текущего.
    fn for_each_block<F>(&self, mut body: F) -> Result<()>
    where
        F: FnMut(usize, &Block) -> Result<()>,
    {
        let n = self.slots.len();
        let off: Vec<usize> = (0..n).filter(|&i| !matches!(self.slots[i], Slot::Device(_))).collect();
        if off.is_empty() {
            for (i, s) in self.slots.iter().enumerate() {
                if let Slot::Device(b) = s {
                    body(i, b)?;
                }
            }
            return Ok(());
        }
        let ord = match self.device {
            Device::Cuda(o) => o,
            _ => return Err(SynaptixError::Other("qwen-image-2.1: стриминг блоков только на CUDA".into())),
        };
        let ls = synaptix_core::device::cuda::loader_stream(ord)?;
        synaptix_core::device::cuda::set_offload_pinned(true);
        let mut staged = Some(self.stage(off[0]));
        let mut k = 0usize;
        let mut result = Ok(());
        for i in 0..n {
            if let Slot::Device(b) = &self.slots[i] {
                if let Err(e) = body(i, b) {
                    result = Err(e);
                    break;
                }
                continue;
            }
            let cur = match staged.take() {
                Some(Ok(b)) => match self.finish(i, b) {
                    Ok(b) => b,
                    Err(e) => {
                        result = Err(e);
                        break;
                    }
                },
                Some(Err(e)) => {
                    result = Err(e);
                    break;
                }
                None => {
                    result = Err(SynaptixError::Other("qwen-image-2.1: блок не доставлен".into()));
                    break;
                }
            };
            let next = off.get(k + 1).copied();
            k += 1;
            let lsc = ls.clone();
            let (step, nxt) = std::thread::scope(|sp| {
                let h = next.map(|j| {
                    sp.spawn(move || {
                        synaptix_core::device::cuda::set_alloc_stream(Some(lsc.clone()));
                        synaptix_core::device::cuda::set_offload_pinned(true);
                        let r = self.stage(j);
                        let _ = lsc.synchronize();
                        synaptix_core::device::cuda::set_offload_pinned(false);
                        synaptix_core::device::cuda::set_alloc_stream(None);
                        r
                    })
                });
                let step = body(i, &cur);
                let nxt = h.map(|h| {
                    h.join().unwrap_or_else(|_| {
                        Err(SynaptixError::Other("qwen-image-2.1: поток префетча блока упал".into()))
                    })
                });
                (step, nxt)
            });
            if let Ok(cs) = synaptix_core::device::cuda::default_stream(ord) {
                let _ = cs.synchronize();
            }
            drop(cur);
            if let Err(e) = step {
                result = Err(e);
                break;
            }
            staged = nxt;
        }
        synaptix_core::device::cuda::set_offload_pinned(false);
        result
    }

    /// Шаг денойза. `txt` — из [`Self::embed_text`] `[1, St, D]`;
    /// `img_tokens` `[1, N, in_channels]` — латенты референсов и следом
    /// генерируемая картинка (в порядке `layout`); `row` — строка [`ModTable`]
    /// шага; `cos/sin` — [`build_rope`] по `layout.positions`. `cache`:
    /// пустой — prefill с запоминанием K/V префикса, заполненный — только
    /// генерируемая картинка, `None` — полный prefill без кэша.
    /// → `[1, n_target, out_channels]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        txt: &Tensor,
        img_tokens: &Tensor,
        layout: &Layout,
        mods: &ModTable,
        row: usize,
        cos: &Tensor,
        sin: &Tensor,
        mut cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let cfg = &self.cfg;
        let dt = self.compute;
        let d = cfg.inner();
        let (p, nt) = (layout.prefix_len, layout.n_target);
        let total = layout.total();
        if txt.dims()[1] != layout.enc_len {
            return Err(SynaptixError::Other(format!(
                "qwen-image-2.1: текст {} токенов, раскладка ждёт {}",
                txt.dims()[1],
                layout.enc_len
            )));
        }
        let n_img_expected = layout.image_pad_mask.iter().filter(|&&m| m).count();
        if img_tokens.dims()[1] != n_img_expected {
            return Err(SynaptixError::Other(format!(
                "qwen-image-2.1: латентов {} токенов, раскладка ждёт {n_img_expected}",
                img_tokens.dims()[1]
            )));
        }
        let decode = matches!(cache.as_deref(), Some(c) if c.is_filled());
        if decode && !cfg.causal_condition {
            return Err(SynaptixError::Other("qwen-image-2.1: KV-кэш требует causal_condition".into()));
        }

        let img_dev = img_tokens.to_device(self.device)?.to_dtype(dt)?;
        let img_h = if decode {
            let n_img = img_dev.dims()[1];
            self.img_in.forward(&img_dev.narrow(1, n_img - nt, nt)?.contiguous()?)?
        } else {
            self.img_in.forward(&img_dev)?
        };
        drop(img_dev);
        // Склейка: текст и латенты по позициям слотов; в decode — только
        // генерируемая картинка.
        let mut x = if decode {
            img_h
        } else {
            let both = Tensor::cat(&[txt, &img_h], 1)?;
            drop(img_h);
            let idx = Tensor::from_vec(layout.gather.clone(), (total,), self.device)?;
            both.index_select(1, &idx)?.contiguous()?
        };
        let (cos_use, sin_use) = if decode {
            (cos.narrow(1, p, nt)?.contiguous()?, sin.narrow(1, p, nt)?.contiguous()?)
        } else {
            (cos.clone(), sin.clone())
        };
        // Модуляции: генерируемая картинка — строка шага, префикс — время 0.
        let s1 = mods.slice(row, 0, d)?;
        let g1 = mods.slice(row, 1, d)?;
        let s2 = mods.slice(row, 2, d)?;
        let g2 = mods.slice(row, 3, d)?;
        let zero = match mods.zero_row {
            Some(z) => Some((mods.slice(z, 0, d)?, mods.slice(z, 1, d)?, mods.slice(z, 2, d)?, mods.slice(z, 3, d)?)),
            None => None,
        };
        let prefix_in_x = if decode { 0 } else { p };
        let m1 = Mod { scale: &s1, gate: &g1, zero: zero.as_ref().map(|z| (&z.0, &z.1, prefix_in_x)) };
        let m2 = Mod { scale: &s2, gate: &g2, zero: zero.as_ref().map(|z| (&z.2, &z.3, prefix_in_x)) };

        let mut layers: Option<&mut Vec<Option<(Tensor, Tensor)>>> = cache.as_deref_mut().map(|c| &mut c.layers);
        self.for_each_block(|i, b| {
            let slot = layers.as_deref_mut().map(|l| &mut l[i]);
            x = b.forward(cfg, &x, &m1, &m2, &cos_use, &sin_use, layout, slot, decode)?;
            if std::env::var("QWEN_IMAGE_DEBUG_AMAX").is_ok() {
                let amax = |t: &Tensor| {
                    t.abs()
                        .and_then(|a| a.max_all())
                        .and_then(|a| a.to_dtype(DType::F32))
                        .and_then(|a| a.flatten_all())
                        .and_then(|a| a.to_vec1::<f32>())
                        .map(|v| v[0])
                        .unwrap_or(f32::NAN)
                };
                eprintln!("блок {i}: max|x| {:.1}", amax(&x));
            }
            Ok(())
        })?;
        // Выход — только генерируемая картинка (строка шага).
        let xt = if decode { x } else { x.narrow(1, p, nt)?.contiguous()? };
        let o = mods.out.narrow(0, row, 1)?.contiguous()?.reshape((1, 1, d))?;
        let y = layer_norm(&xt, None, None, cfg.eps)?.broadcast_mul(&o.add_scalar(1.0)?)?;
        self.proj_out.forward(&y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_positions_like_diffusers() {
        // Текст 3, референс 2×2, текст 2, генерируемая 2×3.
        let mask = [false, false, false, true, true, true, true, false, false, true, true, true, true, true, true];
        let p = rope_positions(&[(2, 2), (2, 3)], &mask).unwrap();
        assert_eq!(p[0], [0, 0, 0]);
        assert_eq!(p[2], [2, 2, 2]);
        // Референс: кадр 3, центрированные (h, w): −1..0.
        assert_eq!(p[3], [3, -1, -1]);
        assert_eq!(p[6], [3, 0, 0]);
        // Текст после референса: позиция 3 + max(2,2) = 5.
        assert_eq!(p[7], [5, 5, 5]);
        assert_eq!(p[8], [6, 6, 6]);
        // Генерируемая: кадр 7, h −1..0, w −2..0.
        assert_eq!(p[9], [7, -1, -2]);
        assert_eq!(p[14], [7, 0, 0]);
    }

    #[test]
    fn layout_segments_and_gather() {
        // Энкодер: 2 текста, 1 слот картинки (4 токена), 2 текста; референс
        // 2×2; генерируемая 2×2.
        let enc = [false, false, true, false, false];
        let l = Layout::build(&enc, &[(2, 2), (2, 2)]).unwrap();
        assert_eq!(l.total(), 2 + 4 + 2 + 4);
        assert_eq!(l.prefix_len, 8);
        assert_eq!(l.n_target, 4);
        assert_eq!(l.segments, vec![(0, 2, true), (2, 6, false), (6, 8, true)]);
        assert_eq!(l.gather, vec![0, 1, 5, 6, 7, 8, 3, 4, 9, 10, 11, 12]);
        assert_eq!(l.positions.len(), 12);
        // Без референсов: один текстовый отрезок.
        let l = Layout::build(&[false, false, false], &[(2, 2)]).unwrap();
        assert_eq!(l.segments, vec![(0, 3, true)]);
        assert_eq!(l.prefix_len, 3);
        // Два референса подряд без текста — два блока.
        let l = Layout::build(&[false, true, true], &[(2, 2), (2, 2), (2, 2)]).unwrap();
        assert_eq!(l.segments, vec![(0, 1, true), (1, 5, false), (5, 9, false)]);
        assert!(Layout::build(&[false, true], &[(2, 2)]).is_err());
    }

    #[test]
    fn causal_mask_is_bottom_right() {
        synaptix_kernels_cpu::ensure_registered();
        // 2 запроса на 4 ключа: запрос 0 видит ключи 0..=2, запрос 1 — все.
        let q = Tensor::from_vec(vec![1f32; 2 * 4], (1, 1, 2, 4), Device::Cpu).unwrap();
        let k = Tensor::from_vec(vec![1f32; 4 * 4], (1, 1, 4, 4), Device::Cpu).unwrap();
        let v = Tensor::from_vec((0..16).map(|i| (i / 4) as f32).collect::<Vec<f32>>(), (1, 1, 4, 4), Device::Cpu)
            .unwrap();
        let o = attend(&q, &k, &v, true).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Равные очки → среднее значений разрешённых ключей: (0+1+2)/3 = 1, (0+1+2+3)/4 = 1,5.
        assert!((o[0] - 1.0).abs() < 1e-5, "{o:?}");
        assert!((o[4] - 1.5).abs() < 1e-5, "{o:?}");
    }
}
