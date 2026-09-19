//! `Flux2Transformer2DModel` — DiT FLUX.2 (dev 8+48 блоков, klein-4B 5+20,
//! klein-9B 8+24). По diffusers `transformer_flux2.py`:
//! - модуляция общая: `double_stream_modulation_img/_txt` (по 6 векторов
//!   shift/scale/gate) и `single_stream_modulation` (3), считаются один раз
//!   на forward из эмбеддинга времени (+ guidance у dev);
//! - все линейные слои без bias, FF — SwiGLU (`linear_in` отдаёт gate‖up);
//! - single-блок: одна проекция `to_qkv_mlp_proj` на q‖k‖v‖gate‖up и одна
//!   `to_out` на attn‖mlp;
//! - RoPE interleaved по четырём осям (t, h, w, l) по 32, theta 2000;
//! - `norm_out` — AdaLayerNormContinuous (scale первым).
//!
//! Память: блоки, не влезшие в VRAM, стримятся в forward — из пиннованной
//! копии на хосте (квант делается один раз при загрузке) или прямо из
//! mmap-источника, если и хосту их не удержать. Следующий блок едет на
//! loader-стриме, пока считается текущий (схема `H3Dit::for_each_block`).

use synaptix_core::{
    device::Device,
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};
use synaptix_nn::module::Module;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_ops::attention::softmax_dim;
use synaptix_ops::norm::{layer_norm, rms_norm};

use crate::config::Flux2Config;
use crate::memory;
use crate::source::Weights;

/// Сколько весят веса блока с `params` параметрами в формате `quant`.
pub fn weight_bytes(params: usize, quant: DType, compute: DType) -> usize {
    match quant {
        DType::NVFP4 => params / 2 + params / 16,
        DType::MXFP8 => params + params / 32,
        _ => params * compute.bytes_for_numel(1).max(1),
    }
}

/// Линейный слой без bias: квантуемый вес читается сразу в F16 (без
/// промежуточной BF16-копии на карте), плотный — в `compute`.
fn lin(w: &Weights, name: &str, dev: Device, compute: DType, quant: DType) -> Result<QuantLinear> {
    let q = quant.is_quantized() && dev.is_cuda();
    let raw = w.get(&format!("{name}.weight"), dev, if q { DType::F16 } else { compute })?;
    QuantLinear::build(raw, None, if q { quant } else { compute }, compute)
}

fn dense(w: &Weights, name: &str, dev: Device, compute: DType) -> Result<QuantLinear> {
    lin(w, name, dev, compute, compute)
}

/// Sinusoidal-эмбеддинг времени (`Timesteps(256, flip_sin_to_cos=True,
/// shift=0)`): `cat[cos, sin]`, `t` уже ×1000.
fn timestep_embedding(t: f32, dim: usize, device: Device) -> Result<Tensor> {
    let half = dim / 2;
    let ln_max = (10000.0_f32).ln();
    let mut v = vec![0f32; dim];
    for i in 0..half {
        // torch: exponent = −ln(10000)·arange(half, f32)/half; emb = t·exp(exponent)
        let freq = (-ln_max * i as f32 / half as f32).exp();
        let a = t * freq;
        v[i] = a.cos();
        v[half + i] = a.sin();
    }
    Tensor::from_vec(v, (1, dim), device)
}

struct MlpEmbed {
    l1: QuantLinear,
    l2: QuantLinear,
}

impl MlpEmbed {
    fn load(w: &Weights, p: &str, dev: Device, compute: DType) -> Result<Self> {
        Ok(Self {
            l1: dense(w, &format!("{p}.linear_1"), dev, compute)?,
            l2: dense(w, &format!("{p}.linear_2"), dev, compute)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.l2.forward(&self.l1.forward(x)?.silu()?)
    }
}

/// cos/sin `[1, S, 1, D]` (F32) для interleaved RoPE по осям `axes`.
/// `ids` — координаты токенов (t, h, w, l) в порядке последовательности.
pub fn build_rope(ids: &[[f64; 4]], axes: &[usize], theta: f64, device: Device) -> Result<(Tensor, Tensor)> {
    let d: usize = axes.iter().sum();
    let s = ids.len();
    let mut cos = vec![0f32; s * d];
    let mut sin = vec![0f32; s * d];
    for (si, id) in ids.iter().enumerate() {
        let mut col = 0usize;
        for (ax, &dim_i) in axes.iter().enumerate() {
            let pos = id[ax];
            for j in 0..dim_i / 2 {
                // get_1d_rotary_pos_embed(freqs_dtype=float64, repeat_interleave_real)
                let freq = 1.0 / theta.powf((2 * j) as f64 / dim_i as f64);
                let ang = pos * freq;
                let (c, sn) = (ang.cos() as f32, ang.sin() as f32);
                cos[si * d + col + 2 * j] = c;
                cos[si * d + col + 2 * j + 1] = c;
                sin[si * d + col + 2 * j] = sn;
                sin[si * d + col + 2 * j + 1] = sn;
            }
            col += dim_i;
        }
    }
    Ok((
        Tensor::from_vec(cos, (1, s, 1, d), device)?,
        Tensor::from_vec(sin, (1, s, 1, d), device)?,
    ))
}

/// `apply_rotary_emb` (use_real, unbind_dim=−1): `x·cos + rot(x)·sin`,
/// rot = (−x1, x0, −x3, x2, …); x `[B,S,H,D]`.
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

/// SDPA без маски, scale 1/√D. q/k/v `[B,S,H,D]` → `[B,S,H·D]`.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d = q.dims();
    let (b, s, h, hd) = (d[0], d[1], d[2], d[3]);
    let scale = 1.0 / (hd as f32).sqrt();
    let qh = q.transpose(1, 2)?.contiguous()?;
    let kh = k.transpose(1, 2)?.contiguous()?;
    let vh = v.transpose(1, 2)?.contiguous()?;
    let out = match qh.flash_attention(&kh, &vh, scale, false) {
        Ok(o) => o,
        Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
            let (qf, kf, vf) = (qh.to_dtype(DType::F32)?, kh.to_dtype(DType::F32)?, vh.to_dtype(DType::F32)?);
            let kt = kf.transpose(2, 3)?.contiguous()?;
            let scores = qf.matmul(&kt)?.mul_scalar(scale)?;
            softmax_dim(&scores, 3)?.matmul(&vf)?.to_dtype(q.dtype())?
        }
        Err(e) => return Err(e),
    };
    out.transpose(1, 2)?.contiguous()?.reshape((b, s, h * hd))
}

/// LayerNorm без affine (eps из конфига) → `·(1+scale) + shift`.
fn ada_ln(x: &Tensor, scale: &Tensor, shift: &Tensor, eps: f32) -> Result<Tensor> {
    let n = layer_norm(x, None, None, eps)?;
    let n = if n.dtype() == scale.dtype() { n } else { n.to_dtype(scale.dtype())? };
    modulate(&n, scale, shift)
}

/// adaLN + (если вышло) fused-преквант входа следующей проекции.
fn ada_ln_quant(
    x: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    eps: f32,
    want: Option<DType>,
) -> Result<(Tensor, Option<(Tensor, Tensor, DType)>)> {
    // Fused LN+модуляция+квант рассчитан на F16-активации (как у FLUX.1):
    // на BF16-входе NVFP4-вариант даёт неверный результат, а MXFP8-GEMM по
    // преквант-паре не умеет BF16-выход.
    if x.dtype() == scale.dtype() && x.dtype() == DType::F16 && x.device().is_cuda() {
        let fused = match want {
            Some(DType::NVFP4) => x.ln_mod_quant_nvfp4(scale, shift, eps).ok(),
            Some(DType::MXFP8) => x.ln_mod_quant_mxfp8(scale, shift, eps).ok(),
            _ => None,
        };
        if let (Some(r), Some(fmt)) = (fused, want) {
            return Ok((r.0, Some((r.1, r.2, fmt))));
        }
    }
    Ok((ada_ln(x, scale, shift, eps)?, None))
}

fn lin_pq(lin: &QuantLinear, x: &Tensor, pq: Option<&(Tensor, Tensor, DType)>) -> Result<Tensor> {
    if let Some((p, sc, fmt)) = pq {
        if lin.quant_dtype() == Some(*fmt) {
            let d = x.dims();
            let (b, t) = (d[0], d[1]);
            let y = lin.forward_prequant(p, sc, b * t, x.dtype())?;
            return y.reshape((b, t, y.dims()[1]));
        }
    }
    lin.forward(x)
}

/// `x·(1+scale) + shift`; scale/shift `[1, D]` → broadcast по токенам.
fn modulate(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let d = scale.dims()[1];
    let sc = scale.add_scalar(1.0)?.reshape((1, 1, d))?;
    let sh = shift.reshape((1, 1, d))?;
    x.broadcast_mul(&sc)?.broadcast_add(&sh)
}

fn gated(gate: &Tensor, y: &Tensor) -> Result<Tensor> {
    let d = gate.dims()[1];
    gate.reshape((1, 1, d))?.broadcast_mul(y)
}

fn res_add(acc: &Tensor, contrib: &Tensor) -> Result<Tensor> {
    if acc.dtype() == contrib.dtype() {
        acc.add(contrib)
    } else {
        acc.add(&contrib.to_dtype(acc.dtype())?)
    }
}

/// `silu(x[..., :half]) · x[..., half:]` по последней оси.
fn swiglu(x: &Tensor) -> Result<Tensor> {
    let r = x.rank();
    let n = x.dims()[r - 1] / 2;
    let g = x.narrow(r - 1, 0, n)?.contiguous()?;
    let u = x.narrow(r - 1, n, n)?.contiguous()?;
    match g.silu_and_mul(&u) {
        Ok(y) => Ok(y),
        Err(_) => g.silu()?.mul(&u),
    }
}

/// Векторы модуляции одного вида: `[1, D]` каждый.
struct Mods(Vec<Tensor>);

impl Mods {
    fn split(m: &Tensor, n: usize) -> Result<Self> {
        let d = m.dims()[1] / n;
        Ok(Self((0..n).map(|i| m.narrow(1, i * d, d)?.contiguous()).collect::<Result<Vec<_>>>()?))
    }
    fn get(&self, i: usize) -> &Tensor {
        &self.0[i]
    }
}

fn to_heads(t: &Tensor, heads: usize, hd: usize) -> Result<Tensor> {
    let d = t.dims();
    t.reshape((d[0], d[1], heads, hd))
}

struct DoubleBlock {
    to_q: QuantLinear,
    to_k: QuantLinear,
    to_v: QuantLinear,
    norm_q: Tensor,
    norm_k: Tensor,
    add_q: QuantLinear,
    add_k: QuantLinear,
    add_v: QuantLinear,
    norm_aq: Tensor,
    norm_ak: Tensor,
    to_out: QuantLinear,
    to_add_out: QuantLinear,
    ff_in: QuantLinear,
    ff_out: QuantLinear,
    ffc_in: QuantLinear,
    ffc_out: QuantLinear,
}

struct SingleBlock {
    qkv_mlp: QuantLinear,
    norm_q: Tensor,
    norm_k: Tensor,
    to_out: QuantLinear,
}

enum Block {
    Double(DoubleBlock),
    Single(SingleBlock),
}

fn ql_to(l: &QuantLinear, dev: Device) -> Result<QuantLinear> {
    l.to_device(dev)
}

impl Block {
    fn load(w: &Weights, cfg: &Flux2Config, idx: usize, dev: Device, compute: DType, quant: DType) -> Result<Self> {
        let norm = |n: String| w.get(&n, dev, compute);
        if idx < cfg.num_layers {
            let p = format!("transformer_blocks.{idx}");
            let a = format!("{p}.attn");
            let l = |n: &str| lin(w, &format!("{a}.{n}"), dev, compute, quant);
            Ok(Block::Double(DoubleBlock {
                to_q: l("to_q")?,
                to_k: l("to_k")?,
                to_v: l("to_v")?,
                norm_q: norm(format!("{a}.norm_q.weight"))?,
                norm_k: norm(format!("{a}.norm_k.weight"))?,
                add_q: l("add_q_proj")?,
                add_k: l("add_k_proj")?,
                add_v: l("add_v_proj")?,
                norm_aq: norm(format!("{a}.norm_added_q.weight"))?,
                norm_ak: norm(format!("{a}.norm_added_k.weight"))?,
                to_out: l("to_out.0")?,
                to_add_out: l("to_add_out")?,
                ff_in: lin(w, &format!("{p}.ff.linear_in"), dev, compute, quant)?,
                ff_out: lin(w, &format!("{p}.ff.linear_out"), dev, compute, quant)?,
                ffc_in: lin(w, &format!("{p}.ff_context.linear_in"), dev, compute, quant)?,
                ffc_out: lin(w, &format!("{p}.ff_context.linear_out"), dev, compute, quant)?,
            }))
        } else {
            let a = format!("single_transformer_blocks.{}.attn", idx - cfg.num_layers);
            Ok(Block::Single(SingleBlock {
                qkv_mlp: lin(w, &format!("{a}.to_qkv_mlp_proj"), dev, compute, quant)?,
                norm_q: norm(format!("{a}.norm_q.weight"))?,
                norm_k: norm(format!("{a}.norm_k.weight"))?,
                to_out: lin(w, &format!("{a}.to_out"), dev, compute, quant)?,
            }))
        }
    }

    fn to_device(&self, dev: Device) -> Result<Self> {
        Ok(match self {
            Block::Double(b) => Block::Double(DoubleBlock {
                to_q: ql_to(&b.to_q, dev)?,
                to_k: ql_to(&b.to_k, dev)?,
                to_v: ql_to(&b.to_v, dev)?,
                norm_q: b.norm_q.to_device(dev)?,
                norm_k: b.norm_k.to_device(dev)?,
                add_q: ql_to(&b.add_q, dev)?,
                add_k: ql_to(&b.add_k, dev)?,
                add_v: ql_to(&b.add_v, dev)?,
                norm_aq: b.norm_aq.to_device(dev)?,
                norm_ak: b.norm_ak.to_device(dev)?,
                to_out: ql_to(&b.to_out, dev)?,
                to_add_out: ql_to(&b.to_add_out, dev)?,
                ff_in: ql_to(&b.ff_in, dev)?,
                ff_out: ql_to(&b.ff_out, dev)?,
                ffc_in: ql_to(&b.ffc_in, dev)?,
                ffc_out: ql_to(&b.ffc_out, dev)?,
            }),
            Block::Single(b) => Block::Single(SingleBlock {
                qkv_mlp: ql_to(&b.qkv_mlp, dev)?,
                norm_q: b.norm_q.to_device(dev)?,
                norm_k: b.norm_k.to_device(dev)?,
                to_out: ql_to(&b.to_out, dev)?,
            }),
        })
    }
}

/// Контекст одного forward: модуляции, RoPE, размеры.
struct Ctx<'a> {
    cfg: &'a Flux2Config,
    m_img: Mods,
    m_txt: Mods,
    m_single: Mods,
    cos: Tensor,
    sin: Tensor,
}

impl DoubleBlock {
    fn forward(&self, c: &Ctx<'_>, img: &Tensor, txt: &Tensor) -> Result<(Tensor, Tensor)> {
        let (h, hd, eps) = (c.cfg.num_heads, c.cfg.head_dim, c.cfg.eps);
        let st = txt.dims()[1];
        let (mi, mt) = (&c.m_img, &c.m_txt);
        let (nh, pq_h) = ada_ln_quant(img, mi.get(1), mi.get(0), eps, self.to_q.quant_dtype())?;
        let (ne, pq_e) = ada_ln_quant(txt, mt.get(1), mt.get(0), eps, self.add_q.quant_dtype())?;

        let q = rms_norm(&to_heads(&lin_pq(&self.to_q, &nh, pq_h.as_ref())?, h, hd)?, &self.norm_q, eps)?;
        let k = rms_norm(&to_heads(&lin_pq(&self.to_k, &nh, pq_h.as_ref())?, h, hd)?, &self.norm_k, eps)?;
        let v = to_heads(&lin_pq(&self.to_v, &nh, pq_h.as_ref())?, h, hd)?;
        let eq = rms_norm(&to_heads(&lin_pq(&self.add_q, &ne, pq_e.as_ref())?, h, hd)?, &self.norm_aq, eps)?;
        let ek = rms_norm(&to_heads(&lin_pq(&self.add_k, &ne, pq_e.as_ref())?, h, hd)?, &self.norm_ak, eps)?;
        let ev = to_heads(&lin_pq(&self.add_v, &ne, pq_e.as_ref())?, h, hd)?;
        drop((nh, ne, pq_h, pq_e));

        let q = apply_rope(&Tensor::cat(&[&eq, &q], 1)?, &c.cos, &c.sin)?;
        let k = apply_rope(&Tensor::cat(&[&ek, &k], 1)?, &c.cos, &c.sin)?;
        let v = Tensor::cat(&[&ev, &v], 1)?;
        let attn = attention(&q, &k, &v)?;
        drop((q, k, v));
        let total = attn.dims()[1];
        let ctx_attn = self.to_add_out.forward(&attn.narrow(1, 0, st)?.contiguous()?)?;
        let img_attn = self.to_out.forward(&attn.narrow(1, st, total - st)?.contiguous()?)?;
        drop(attn);

        let img = res_add(img, &gated(mi.get(2), &img_attn)?)?;
        let (n2, pq) = ada_ln_quant(&img, mi.get(4), mi.get(3), eps, self.ff_in.quant_dtype())?;
        let ff = self.ff_out.forward(&swiglu(&lin_pq(&self.ff_in, &n2, pq.as_ref())?)?)?;
        let img = res_add(&img, &gated(mi.get(5), &ff)?)?;

        let txt = res_add(txt, &gated(mt.get(2), &ctx_attn)?)?;
        let (n2c, pqc) = ada_ln_quant(&txt, mt.get(4), mt.get(3), eps, self.ffc_in.quant_dtype())?;
        let ffc = self.ffc_out.forward(&swiglu(&lin_pq(&self.ffc_in, &n2c, pqc.as_ref())?)?)?;
        let txt = res_add(&txt, &gated(mt.get(5), &ffc)?)?;
        Ok((img, txt))
    }
}

impl SingleBlock {
    fn forward(&self, c: &Ctx<'_>, x: &Tensor) -> Result<Tensor> {
        let (h, hd, eps) = (c.cfg.num_heads, c.cfg.head_dim, c.cfg.eps);
        let (inner, m) = (c.cfg.inner(), c.cfg.mlp_hidden());
        let ms = &c.m_single;
        let (n, pq) = ada_ln_quant(x, ms.get(1), ms.get(0), eps, self.qkv_mlp.quant_dtype())?;
        let p = lin_pq(&self.qkv_mlp, &n, pq.as_ref())?; // [1, S, 3·inner + 2·m]
        drop((n, pq));
        let part = |off: usize, len: usize| -> Result<Tensor> { p.narrow(2, off, len)?.contiguous() };
        let q = rms_norm(&to_heads(&part(0, inner)?, h, hd)?, &self.norm_q, eps)?;
        let k = rms_norm(&to_heads(&part(inner, inner)?, h, hd)?, &self.norm_k, eps)?;
        let v = to_heads(&part(2 * inner, inner)?, h, hd)?;
        let gate = part(3 * inner, m)?;
        let up = part(3 * inner + m, m)?;
        drop(p);
        let q = apply_rope(&q, &c.cos, &c.sin)?;
        let k = apply_rope(&k, &c.cos, &c.sin)?;
        let attn = attention(&q, &k, &v)?;
        drop((q, k, v));
        let act = match gate.silu_and_mul(&up) {
            Ok(y) => y,
            Err(_) => gate.silu()?.mul(&up)?,
        };
        drop((gate, up));
        let y = self.to_out.forward(&Tensor::cat(&[&attn, &act], 2)?)?;
        res_add(x, &gated(ms.get(2), &y)?)
    }
}

/// Где держать блоки DiT.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Placement {
    /// По свободной VRAM: сколько влезло с запасом под активации, остальное
    /// стримится.
    #[default]
    Auto,
    /// Все блоки на карте.
    Resident,
    /// Все блоки стримятся.
    Stream,
}

/// Блок: на карте, пиннованной копией на хосте или только в источнике.
enum Slot {
    Device(Block),
    Host(Block),
    Source,
}

pub struct Flux2Transformer {
    cfg: Flux2Config,
    device: Device,
    compute: DType,
    quant: DType,
    x_embedder: QuantLinear,
    context_embedder: QuantLinear,
    t_embed: MlpEmbed,
    g_embed: Option<MlpEmbed>,
    mod_img: QuantLinear,
    mod_txt: QuantLinear,
    mod_single: QuantLinear,
    norm_out: QuantLinear,
    proj_out: QuantLinear,
    slots: Vec<Slot>,
    weights: Weights,
}

/// Сводка размещения: сколько блоков где лежит.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Residency {
    pub device: usize,
    pub host: usize,
    pub source: usize,
}

impl Flux2Transformer {
    /// Загрузить DiT. `tokens` — длина самого крупного прогона (текст +
    /// латент + референсы): от неё запас VRAM под активации.
    pub fn load(
        w: &Weights,
        cfg: &Flux2Config,
        device: Device,
        compute: DType,
        quant: DType,
        placement: Placement,
        tokens: usize,
    ) -> Result<Self> {
        let cuda = device.is_cuda();
        let quant = if cuda { quant } else { compute };
        let _g = memory::weights_guard(device);
        let x_embedder = dense(w, "x_embedder", device, compute)?;
        // Модуляция считается из одной строки (эмбеддинг времени), и её
        // выход прямо масштабирует нормированные активации всех блоков:
        // NVFP4 здесь даёт косинус шага 0,81 к эталону, MXFP8 — 0,997 против
        // 0,9999 у BF16. Поэтому модуляция и norm_out всегда плотные; вход
        // текста терпит MXFP8 (NVFP4 заметно хуже).
        let q_ctx = if quant.is_quantized() { DType::MXFP8 } else { compute };
        let q_mod = compute;
        let context_embedder = lin(w, "context_embedder", device, compute, q_ctx)?;
        let t_embed = MlpEmbed::load(w, "time_guidance_embed.timestep_embedder", device, compute)?;
        let g_embed = if cfg.guidance_embeds {
            Some(MlpEmbed::load(w, "time_guidance_embed.guidance_embedder", device, compute)?)
        } else {
            None
        };
        let mod_img = lin(w, "double_stream_modulation_img.linear", device, compute, q_mod)?;
        let mod_txt = lin(w, "double_stream_modulation_txt.linear", device, compute, q_mod)?;
        let mod_single = lin(w, "single_stream_modulation.linear", device, compute, q_mod)?;
        let norm_out = lin(w, "norm_out.linear", device, compute, q_mod)?;
        let proj_out = dense(w, "proj_out", device, compute)?;

        let n = cfg.num_blocks();
        let (pd, ps) = cfg.block_params();
        let blk_bytes = |i: usize| weight_bytes(if i < cfg.num_layers { pd } else { ps }, quant, compute);
        let max_blk = blk_bytes(0).max(blk_bytes(n - 1));
        // Сырой F16-вес крупнейшей проекции перед квантованием.
        let raw_transient = if quant.is_quantized() {
            cfg.inner() * (3 * cfg.inner() + 2 * cfg.mlp_hidden()) * 2
        } else {
            0
        };
        let act = memory::dit_activation_bytes(tokens, cfg.inner(), cfg.mlp_hidden());
        // После резидентных блоков должно остаться: активации, два
        // стримящихся блока (текущий + префетч), транзиент квантования и
        // запас под рабочий стол.
        let reserve = act + 2 * max_blk + raw_transient + memory::DESKTOP_MARGIN;
        // Хост: пиннованные копии только если RAM хватает с запасом, иначе
        // блоки читаются из mmap-источника на каждом шаге.
        let host_ok = |need: u64| memory::host_available().map(|a| a > need + (12u64 << 30)).unwrap_or(false);

        let mut slots = Vec::with_capacity(n);
        let mut first_off: Option<usize> = match placement {
            Placement::Stream if cuda => Some(0),
            _ => None,
        };
        let mut use_host = false;
        for i in 0..n {
            if cuda && first_off.is_none() && placement == Placement::Auto {
                if memory::free_vram(device) < reserve + blk_bytes(i) {
                    memory::release_pools(device);
                    if memory::free_vram(device) < reserve + blk_bytes(i) {
                        first_off = Some(i);
                    }
                }
            }
            if let Some(f) = first_off {
                if i == f {
                    let rest: u64 = (i..n).map(|j| blk_bytes(j) as u64).sum();
                    use_host = host_ok(rest);
                    eprintln!(
                        "[flux2] VRAM: блоки {i}..{n} ({:.1} ГБ) стримятся {} (свободно {:.1} ГБ, запас {:.1} ГБ)",
                        rest as f64 / 1e9,
                        if use_host { "из пиннованной копии на хосте" } else { "из источника (mmap)" },
                        memory::free_vram(device) as f64 / 1e9,
                        reserve as f64 / 1e9,
                    );
                }
                if use_host {
                    let blk = Block::load(w, cfg, i, device, compute, quant)?;
                    synaptix_core::device::cuda::set_offload_pinned(true);
                    let host = blk.to_device(Device::Cpu);
                    synaptix_core::device::cuda::set_offload_pinned(false);
                    drop(blk);
                    slots.push(Slot::Host(host?));
                    memory::release_pools(device);
                } else {
                    slots.push(Slot::Source);
                }
            } else {
                slots.push(Slot::Device(Block::load(w, cfg, i, device, compute, quant)?));
            }
        }
        Ok(Self {
            cfg: cfg.clone(),
            device,
            compute,
            quant,
            x_embedder,
            context_embedder,
            t_embed,
            g_embed,
            mod_img,
            mod_txt,
            mod_single,
            norm_out,
            proj_out,
            slots,
            weights: w.clone(),
        })
    }

    pub fn config(&self) -> &Flux2Config {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn compute_dtype(&self) -> DType {
        self.compute
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

    /// Копия нерезидентного блока на карте.
    fn stage(&self, idx: usize) -> Result<Block> {
        match &self.slots[idx] {
            Slot::Device(_) => Err(SynaptixError::Other(format!("flux2: блок {idx} и так на карте"))),
            Slot::Host(b) => b.to_device(self.device),
            Slot::Source => Block::load(&self.weights, &self.cfg, idx, self.device, self.compute, self.quant),
        }
    }

    /// Проход по блокам: резидентные как есть, остальные приезжают на
    /// карту, следующий нерезидентный — на loader-стриме параллельно счёту
    /// текущего.
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
            _ => return Err(SynaptixError::Other("flux2: стриминг блоков только на CUDA".into())),
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
                Some(Ok(b)) => b,
                Some(Err(e)) => {
                    result = Err(e);
                    break;
                }
                None => {
                    result = Err(SynaptixError::Other("flux2: блок не доставлен".into()));
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
                    h.join().unwrap_or_else(|_| Err(SynaptixError::Other("flux2: поток префетча блока упал".into())))
                });
                (step, nxt)
            });
            // Освобождение буферов блока стоит в хвосте compute-стрима: без
            // синка пул берёт под следующий блок новые сегменты.
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

    /// Шаг денойза. `img` `[1, Si, in_channels]` — латент (и референсы
    /// следом), `txt` `[1, St, joint]`, `sigma` — момент расписания (в
    /// трансформер идёт ×1000), `guidance` — только у dev. `cos/sin` — из
    /// [`build_rope`] по координатам `[txt…, img…]`. → `[1, Si, in_channels]`.
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        sigma: f32,
        guidance: Option<f32>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let dev = self.device;
        let dt = self.compute;
        // Как diffusers: timestep приводится к dtype скрытых состояний и
        // умножается на 1000 в нём же.
        let round = |v: f32| -> f32 {
            match dt {
                DType::BF16 => half::bf16::from_f32(v).to_f32(),
                DType::F16 => half::f16::from_f32(v).to_f32(),
                _ => v,
            }
        };
        let t = round(round(sigma) * 1000.0);
        let mut temb = self
            .t_embed
            .forward(&timestep_embedding(t, self.cfg.timestep_channels, dev)?.to_dtype(dt)?)?;
        if let (Some(g), Some(gv)) = (&self.g_embed, guidance) {
            let g1000 = round(round(gv) * 1000.0);
            let ge = g.forward(&timestep_embedding(g1000, self.cfg.timestep_channels, dev)?.to_dtype(dt)?)?;
            temb = temb.add(&ge)?;
        }
        let act = temb.silu()?;
        let ctx = Ctx {
            cfg: &self.cfg,
            m_img: Mods::split(&self.mod_img.forward(&act)?, 6)?,
            m_txt: Mods::split(&self.mod_txt.forward(&act)?, 6)?,
            m_single: Mods::split(&self.mod_single.forward(&act)?, 3)?,
            cos: cos.clone(),
            sin: sin.clone(),
        };

        let mut img_h = self.x_embedder.forward(&img.to_dtype(dt)?)?;
        let mut txt_h = self.context_embedder.forward(&txt.to_dtype(dt)?)?;
        let st = txt_h.dims()[1];
        let mut joint: Option<Tensor> = None;
        self.for_each_block(|_, b| {
            match b {
                Block::Double(d) => {
                    let (a, t) = d.forward(&ctx, &img_h, &txt_h)?;
                    img_h = a;
                    txt_h = t;
                }
                Block::Single(s) => {
                    if joint.is_none() {
                        joint = Some(Tensor::cat(&[&txt_h, &img_h], 1)?);
                    }
                    let x = joint.take().expect("joint");
                    joint = Some(s.forward(&ctx, &x)?);
                }
            }
            Ok(())
        })?;
        let x = match joint {
            Some(j) => {
                let total = j.dims()[1];
                j.narrow(1, st, total - st)?.contiguous()?
            }
            None => img_h,
        };
        let m = Mods::split(&self.norm_out.forward(&act)?, 2)?;
        let out = ada_ln(&x, m.get(0), m.get(1), self.cfg.eps)?;
        self.proj_out.forward(&out)
    }
}
