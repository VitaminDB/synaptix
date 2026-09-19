//! `QwenImageTransformer2DModel` — MMDiT Qwen-Image: 60 double-stream блоков
//! (внимание по склейке [текст, картинка], свои веса у потоков), по
//! diffusers `transformer_qwenimage.py`:
//! - у каждого блока своя модуляция `img_mod`/`txt_mod` (Linear d → 6d с
//!   bias) из эмбеддинга времени: shift, scale, gate для внимания и FF;
//! - q/k/v/out с bias, RMSNorm по голове для q/k обоих потоков, FF —
//!   GELU-tanh на 4d;
//! - RoPE interleaved по трём осям (кадр 16, высота 56, ширина 56), theta
//!   10000; высота/ширина центрированы (`scale_rope`), картинки нумеруются
//!   по оси кадра (0 — генерируемая, 1, 2, … — референсы), текст идёт
//!   после максимума `max(h/2, w/2)` по всем осям сразу;
//! - 2511 (`zero_cond_t`): токены референсов модулируются временем 0.
//!
//! Модуляции зависят только от времени, поэтому считаются один раз на всё
//! расписание ([`ModTable`]): их веса (~40 % блока) не лежат в VRAM, а
//! читаются из источника на время подготовки. Остальные веса блоков: что не
//! влезло в VRAM, стримится в forward — из пиннованной копии на хосте или из
//! mmap-источника (схема FLUX.2).

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

use crate::config::QwenImageConfig;
use crate::memory;
use crate::source::Weights;

const EPS: f32 = 1e-6;

/// Сколько весят `params` параметров в формате `quant`.
pub fn weight_bytes(params: usize, quant: DType, compute: DType) -> usize {
    match quant {
        DType::NVFP4 => params / 2 + params / 16,
        DType::MXFP8 => params + params / 32,
        _ => params * compute.bytes_for_numel(1).max(1),
    }
}

type Getter<'a> = dyn Fn(&str, DType) -> Result<Tensor> + 'a;

/// Линейный слой с bias: квантуемый вес читается сразу в F16, плотный — в
/// `compute`; bias всегда в `compute`.
fn lin_g(g: &Getter<'_>, name: &str, cuda: bool, compute: DType, quant: DType) -> Result<QuantLinear> {
    let q = quant.is_quantized() && cuda;
    let raw = g(&format!("{name}.weight"), if q { DType::F16 } else { compute })?;
    let bias = g(&format!("{name}.bias"), compute)?;
    QuantLinear::build(raw, Some(bias), if q { quant } else { compute }, compute)
}

fn dense(w: &Weights, name: &str, dev: Device, compute: DType) -> Result<QuantLinear> {
    lin_g(&|n, dt| w.get(n, dev, dt), name, dev.is_cuda(), compute, compute)
}

/// `get_timestep_embedding(t, 256, flip_sin_to_cos=True, shift=0,
/// scale=1000)` как у diffusers: частоты округляются до типа времени (BF16),
/// произведение — в F32, затем ×1000; выход `[cos, sin]`.
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
    let mut v = vec![0f32; dim];
    for i in 0..half {
        let freq = round((-ln_max * i as f32 / half as f32).exp());
        let a = 1000.0 * (t * freq);
        v[i] = a.cos();
        v[half + i] = a.sin();
    }
    Tensor::from_vec(v, (1, dim), device)
}

/// cos/sin `[1, S, 1, D]` (F32) для interleaved RoPE: `pos[i]` — позиции
/// токена по осям (кадр, высота, ширина).
pub fn build_rope(pos: &[[i32; 3]], axes: &[usize], device: Device) -> Result<(Tensor, Tensor)> {
    let d: usize = axes.iter().sum();
    let s = pos.len();
    let invs: Vec<Vec<f32>> = axes
        .iter()
        .map(|&dim| (0..dim / 2).map(|j| (1.0 / 10000f64.powf((2 * j) as f64 / dim as f64)) as f32).collect())
        .collect();
    let mut cos = vec![0f32; s * d];
    let mut sin = vec![0f32; s * d];
    for (si, p) in pos.iter().enumerate() {
        let mut col = 0usize;
        for (ax, inv) in invs.iter().enumerate() {
            for (j, f) in inv.iter().enumerate() {
                // torch.outer(index, inv_freq) в f32, затем polar.
                let a = (p[ax] as f32 * f) as f64;
                let (c, sn) = (a.cos() as f32, a.sin() as f32);
                cos[si * d + col + 2 * j] = c;
                cos[si * d + col + 2 * j + 1] = c;
                sin[si * d + col + 2 * j] = sn;
                sin[si * d + col + 2 * j + 1] = sn;
            }
            col += axes[ax];
        }
    }
    Ok((Tensor::from_vec(cos, (1, s, 1, d), device)?, Tensor::from_vec(sin, (1, s, 1, d), device)?))
}

/// Позиции RoPE склейки `[текст…, картинки…]`: `images[i] = (h, w)` в
/// токенах (латент/2), картинка i — кадр i; текст длины `txt` — после
/// максимума `max(h/2, w/2)`.
pub fn rope_positions(images: &[(usize, usize)], txt: usize) -> Vec<[i32; 3]> {
    let max_vid = images.iter().map(|&(h, w)| (h / 2).max(w / 2)).max().unwrap_or(0) as i32;
    let mut out: Vec<[i32; 3]> = (0..txt as i32).map(|l| [max_vid + l; 3]).collect();
    for (idx, &(h, w)) in images.iter().enumerate() {
        let (oh, ow) = ((h - h / 2) as i32, (w - w / 2) as i32);
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                out.push([idx as i32, y - oh, x - ow]);
            }
        }
    }
    out
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

/// Проекция: MXFP8 считает GEMM с выходом в F16, а остаточный поток
/// Qwen-Image доходит до ~10⁹ — с обычной подготовкой входа (максимум 64)
/// выход FF последних блоков переполнял F16 (белая картинка). Вход MXFP8
/// приводится к максимуму 1: |y| ≤ ‖w_row‖₁. NVFP4 на BF16 (выход BF16) и
/// плотные слои — как обычно.
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

fn res_add(acc: &Tensor, contrib: &Tensor) -> Result<Tensor> {
    if acc.dtype() == contrib.dtype() {
        acc.add(contrib)
    } else {
        acc.add(&contrib.to_dtype(acc.dtype())?)
    }
}

/// Векторы модуляции одного потока на шаг: `shift, scale, gate` × 2
/// (внимание, FF), каждый `[1, 1, D]`.
#[derive(Clone)]
struct Mod6 {
    v: Vec<Tensor>,
}

impl Mod6 {
    /// Строка `[6D]` → шесть `[1, 1, D]`.
    fn from_row(row: &Tensor, d: usize) -> Result<Self> {
        Ok(Self { v: (0..6).map(|i| row.narrow(0, i * d, d)?.contiguous()?.reshape((1, 1, d))).collect::<Result<_>>()? })
    }
    fn shift(&self, k: usize) -> &Tensor {
        &self.v[3 * k]
    }
    fn scale(&self, k: usize) -> &Tensor {
        &self.v[3 * k + 1]
    }
    fn gate(&self, k: usize) -> &Tensor {
        &self.v[3 * k + 2]
    }
}

/// Модуляция потока картинки: одна на все токены или (2511) своя у
/// генерируемой картинки (первые `split` токенов) и у референсов.
struct ImgMod<'a> {
    main: &'a Mod6,
    cond: Option<(&'a Mod6, usize)>,
}

impl ImgMod<'_> {
    /// `x·(1 + scale) + shift` по отрезкам.
    fn modulate(&self, x: &Tensor, k: usize) -> Result<Tensor> {
        let one = |x: &Tensor, m: &Mod6| -> Result<Tensor> {
            x.broadcast_mul(&m.scale(k).add_scalar(1.0)?)?.broadcast_add(m.shift(k))
        };
        match self.cond {
            None => one(x, self.main),
            Some((c, split)) => {
                let n = x.dims()[1];
                let a = one(&x.narrow(1, 0, split)?.contiguous()?, self.main)?;
                let b = one(&x.narrow(1, split, n - split)?.contiguous()?, c)?;
                Tensor::cat(&[&a, &b], 1)
            }
        }
    }

    fn gated(&self, y: &Tensor, k: usize) -> Result<Tensor> {
        match self.cond {
            None => y.broadcast_mul(self.main.gate(k)),
            Some((c, split)) => {
                let n = y.dims()[1];
                let a = y.narrow(1, 0, split)?.contiguous()?.broadcast_mul(self.main.gate(k))?;
                let b = y.narrow(1, split, n - split)?.contiguous()?.broadcast_mul(c.gate(k))?;
                Tensor::cat(&[&a, &b], 1)
            }
        }
    }
}

fn modulate(x: &Tensor, m: &Mod6, k: usize) -> Result<Tensor> {
    x.broadcast_mul(&m.scale(k).add_scalar(1.0)?)?.broadcast_add(m.shift(k))
}

/// Веса блока без модуляций.
struct Block {
    to_q: QuantLinear,
    to_k: QuantLinear,
    to_v: QuantLinear,
    add_q: QuantLinear,
    add_k: QuantLinear,
    add_v: QuantLinear,
    norm_q: Tensor,
    norm_k: Tensor,
    norm_aq: Tensor,
    norm_ak: Tensor,
    to_out: QuantLinear,
    to_add_out: QuantLinear,
    img_ff_in: QuantLinear,
    img_ff_out: QuantLinear,
    txt_ff_in: QuantLinear,
    txt_ff_out: QuantLinear,
}

fn block_prefix(idx: usize) -> String {
    format!("transformer_blocks.{idx}.")
}

/// Тензор модуляций блока — их веса не входят в [`Block`].
fn is_mod_tensor(name: &str) -> bool {
    name.contains(".img_mod.") || name.contains(".txt_mod.")
}

impl Block {
    fn load(w: &Weights, idx: usize, dev: Device, compute: DType, quant: DType) -> Result<Self> {
        Self::load_g(&|n, dt| w.get(n, dev, dt), idx, dev.is_cuda(), compute, quant)
    }

    fn load_g(g: &Getter<'_>, idx: usize, cuda: bool, compute: DType, quant: DType) -> Result<Self> {
        let p = format!("transformer_blocks.{idx}");
        let lin = |n: &str| lin_g(g, &format!("{p}.{n}"), cuda, compute, quant);
        let norm = |n: &str| g(&format!("{p}.attn.{n}.weight"), compute);
        Ok(Self {
            to_q: lin("attn.to_q")?,
            to_k: lin("attn.to_k")?,
            to_v: lin("attn.to_v")?,
            add_q: lin("attn.add_q_proj")?,
            add_k: lin("attn.add_k_proj")?,
            add_v: lin("attn.add_v_proj")?,
            norm_q: norm("norm_q")?,
            norm_k: norm("norm_k")?,
            norm_aq: norm("norm_added_q")?,
            norm_ak: norm("norm_added_k")?,
            to_out: lin("attn.to_out.0")?,
            to_add_out: lin("attn.to_add_out")?,
            img_ff_in: lin("img_mlp.net.0.proj")?,
            img_ff_out: lin("img_mlp.net.2")?,
            txt_ff_in: lin("txt_mlp.net.0.proj")?,
            txt_ff_out: lin("txt_mlp.net.2")?,
        })
    }

    fn to_device(&self, dev: Device) -> Result<Self> {
        let q = |l: &QuantLinear| l.to_device(dev);
        Ok(Self {
            to_q: q(&self.to_q)?,
            to_k: q(&self.to_k)?,
            to_v: q(&self.to_v)?,
            add_q: q(&self.add_q)?,
            add_k: q(&self.add_k)?,
            add_v: q(&self.add_v)?,
            norm_q: self.norm_q.to_device(dev)?,
            norm_k: self.norm_k.to_device(dev)?,
            norm_aq: self.norm_aq.to_device(dev)?,
            norm_ak: self.norm_ak.to_device(dev)?,
            to_out: q(&self.to_out)?,
            to_add_out: q(&self.to_add_out)?,
            img_ff_in: q(&self.img_ff_in)?,
            img_ff_out: q(&self.img_ff_out)?,
            txt_ff_in: q(&self.txt_ff_in)?,
            txt_ff_out: q(&self.txt_ff_out)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        cfg: &QwenImageConfig,
        img: &Tensor,
        txt: &Tensor,
        im: &ImgMod<'_>,
        tm: &Mod6,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (h, hd) = (cfg.num_heads, cfg.head_dim);
        let st = txt.dims()[1];
        let ni = im.modulate(&layer_norm(img, None, None, EPS)?, 0)?;
        let nt = modulate(&layer_norm(txt, None, None, EPS)?, tm, 0)?;

        let q = rms_norm(&to_heads(&qf(&self.to_q, &ni)?, h, hd)?, &self.norm_q, EPS)?;
        let k = rms_norm(&to_heads(&qf(&self.to_k, &ni)?, h, hd)?, &self.norm_k, EPS)?;
        let v = to_heads(&qf(&self.to_v, &ni)?, h, hd)?;
        let eq = rms_norm(&to_heads(&qf(&self.add_q, &nt)?, h, hd)?, &self.norm_aq, EPS)?;
        let ek = rms_norm(&to_heads(&qf(&self.add_k, &nt)?, h, hd)?, &self.norm_ak, EPS)?;
        let ev = to_heads(&qf(&self.add_v, &nt)?, h, hd)?;
        drop((ni, nt));

        let q = apply_rope(&Tensor::cat(&[&eq, &q], 1)?, cos, sin)?;
        let k = apply_rope(&Tensor::cat(&[&ek, &k], 1)?, cos, sin)?;
        let v = Tensor::cat(&[&ev, &v], 1)?;
        drop((eq, ek, ev));
        let attn = attention(&q, &k, &v)?;
        drop((q, k, v));
        let total = attn.dims()[1];
        let txt_attn = qf(&self.to_add_out, &attn.narrow(1, 0, st)?.contiguous()?)?;
        let img_attn = qf(&self.to_out, &attn.narrow(1, st, total - st)?.contiguous()?)?;
        drop(attn);

        let img = res_add(img, &im.gated(&img_attn, 0)?)?;
        let txt = res_add(txt, &txt_attn.broadcast_mul(tm.gate(0))?)?;
        drop((img_attn, txt_attn));

        let n2 = im.modulate(&layer_norm(&img, None, None, EPS)?, 1)?;
        let ff = qf(&self.img_ff_out, &qf(&self.img_ff_in, &n2)?.gelu_tanh()?)?;
        drop(n2);
        let img = res_add(&img, &im.gated(&ff, 1)?)?;
        drop(ff);

        let n2 = modulate(&layer_norm(&txt, None, None, EPS)?, tm, 1)?;
        let ff = qf(&self.txt_ff_out, &qf(&self.txt_ff_in, &n2)?.gelu_tanh()?)?;
        let txt = res_add(&txt, &ff.broadcast_mul(tm.gate(1))?)?;
        Ok((img, txt))
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

enum Slot {
    Device(Block),
    Host(Block),
    Source,
}

/// Блок, доставленный потоком префетча: готовый (копия с хоста) или сырые
/// тензоры источника в типе файла — квант и приведение типа делаются уже в
/// основном потоке (ядра из потока префетча не ждут копию на loader-стриме).
enum Staged {
    Ready(Block),
    Raw(std::collections::HashMap<String, Tensor>),
}

/// Сводка размещения: сколько блоков где лежит.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Residency {
    pub device: usize,
    pub host: usize,
    pub source: usize,
}

/// Модуляции всех блоков на заданные моменты расписания: строка `r` —
/// время `sigmas[r]`; при `zero_cond_t` последняя строка — время 0.
pub struct ModTable {
    sigmas: Vec<f32>,
    /// По блокам: `[R, 6D]` (BF16) для потоков картинки и текста.
    img: Vec<Tensor>,
    txt: Vec<Tensor>,
    /// `norm_out`: `[R, 2D]` (scale ‖ shift).
    out: Tensor,
    zero_row: Option<usize>,
}

impl ModTable {
    pub fn row_of(&self, sigma: f32) -> Option<usize> {
        self.sigmas.iter().position(|s| *s == sigma)
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
}

pub struct QwenImageTransformer {
    cfg: QwenImageConfig,
    device: Device,
    compute: DType,
    quant: DType,
    img_in: QuantLinear,
    txt_norm: Tensor,
    txt_in: QuantLinear,
    t_l1: QuantLinear,
    t_l2: QuantLinear,
    norm_out: QuantLinear,
    proj_out: QuantLinear,
    slots: Vec<Slot>,
    weights: Weights,
}

impl QwenImageTransformer {
    /// Загрузить DiT. `tokens` — длина самого крупного прогона (текст +
    /// латент + референсы): от неё запас VRAM под активации.
    pub fn load(
        w: &Weights,
        cfg: &QwenImageConfig,
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
        let txt_norm = w.get("txt_norm.weight", device, compute)?;
        // Вход текста терпит MXFP8 (как у FLUX.2), остальное вне блоков — плотное.
        let q_txt = if quant.is_quantized() { DType::MXFP8 } else { compute };
        let txt_in = lin_g(&|n, dt| w.get(n, device, dt), "txt_in", cuda, compute, q_txt)?;
        let t_l1 = dense(w, "time_text_embed.timestep_embedder.linear_1", device, compute)?;
        let t_l2 = dense(w, "time_text_embed.timestep_embedder.linear_2", device, compute)?;
        let norm_out = dense(w, "norm_out.linear", device, compute)?;
        let proj_out = dense(w, "proj_out", device, compute)?;

        let n = cfg.num_layers;
        let d = cfg.inner();
        let blk_params = cfg.block_params() - 2 * 6 * d * d;
        let blk = weight_bytes(blk_params, quant, compute);
        // Сырой F16-вес крупнейшей проекции (FF d → 4d) перед квантованием.
        let raw_transient = if quant.is_quantized() { d * cfg.mlp_hidden() * 2 } else { 0 };
        let act = memory::dit_activation_bytes(tokens, d);
        // После резидентных блоков: активации, два стримящихся блока, транзиент
        // квантования, подготовка модуляций (два блока модуляций в BF16) и
        // запас под рабочий стол.
        let mods = 2 * 2 * 6 * d * d * 2;
        let reserve = act + 2 * blk + raw_transient + mods + memory::DESKTOP_MARGIN;
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
                        "[qwen-image] VRAM: блоки {i}..{n} ({:.1} ГБ) стримятся {} (свободно {:.1} ГБ, запас {:.1} ГБ)",
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
            t_l1,
            t_l2,
            norm_out,
            proj_out,
            slots,
            weights: w.clone(),
        })
    }

    pub fn config(&self) -> &QwenImageConfig {
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

    /// Эмбеддинг времени `[1, D]`. `sigma` — момент расписания; как у
    /// пайплайна, время приводится к типу скрытых состояний: `bf16(σ·1000)/1000`.
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

    /// Модуляции всех блоков на моменты `sigmas` (+ время 0 при
    /// `zero_cond_t`). Веса модуляций читаются из источника по блоку, следующий
    /// — в потоке префетча.
    pub fn modulations(&self, sigmas: &[f32]) -> Result<ModTable> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let (dev, dt) = (self.device, self.compute);
        let mut rows: Vec<Tensor> = sigmas.iter().map(|&s| self.temb(s)).collect::<Result<_>>()?;
        let zero_row = if self.cfg.zero_cond_t {
            rows.push(self.temb(0.0)?);
            Some(rows.len() - 1)
        } else {
            None
        };
        let refs: Vec<&Tensor> = rows.iter().collect();
        let act = Tensor::cat(&refs, 0)?.silu()?; // [R, D]
        drop(rows);
        let out = self.norm_out.forward(&act)?;

        let n = self.cfg.num_layers;
        let load = |i: usize, on_loader: bool| -> Result<(Tensor, Tensor, Tensor, Tensor)> {
            let p = format!("transformer_blocks.{i}");
            let get = |name: &str| self.weights.get(&format!("{p}.{name}"), dev, dt);
            let run = || -> Result<(Tensor, Tensor, Tensor, Tensor)> {
                Ok((get("img_mod.1.weight")?, get("img_mod.1.bias")?, get("txt_mod.1.weight")?, get("txt_mod.1.bias")?))
            };
            match (on_loader, dev) {
                (true, Device::Cuda(ord)) => {
                    let ls = synaptix_core::device::cuda::loader_stream(ord)?;
                    synaptix_core::device::cuda::set_alloc_stream(Some(ls.clone()));
                    let r = run();
                    let _ = ls.synchronize();
                    synaptix_core::device::cuda::set_alloc_stream(None);
                    r
                }
                _ => run(),
            }
        };
        let mut img = Vec::with_capacity(n);
        let mut txt = Vec::with_capacity(n);
        let mut staged = Some(load(0, false));
        for i in 0..n {
            let (wi, bi, wt, bt) = staged.take().expect("staged")?;
            let (step, next) = std::thread::scope(|sp| {
                let h = (i + 1 < n).then(|| sp.spawn(move || load(i + 1, true)));
                let step = (|| -> Result<(Tensor, Tensor)> {
                    let li = QuantLinear::dense(wi.clone(), Some(bi.clone()))?;
                    let lt = QuantLinear::dense(wt.clone(), Some(bt.clone()))?;
                    Ok((li.forward(&act)?, lt.forward(&act)?))
                })();
                let next = h.map(|h| {
                    h.join().unwrap_or_else(|_| Err(SynaptixError::Other("поток префетча модуляций упал".into())))
                });
                (step, next)
            });
            if let Device::Cuda(ord) = dev {
                if let Ok(cs) = synaptix_core::device::cuda::default_stream(ord) {
                    let _ = cs.synchronize();
                }
            }
            drop((wi, bi, wt, bt));
            let (a, b) = step?;
            img.push(a);
            txt.push(b);
            staged = next;
        }
        memory::release_pools(dev);
        Ok(ModTable { sigmas: sigmas.to_vec(), img, txt, out, zero_row })
    }

    fn stage(&self, idx: usize) -> Result<Staged> {
        match &self.slots[idx] {
            Slot::Device(_) => Err(SynaptixError::Other(format!("qwen-image: блок {idx} и так на карте"))),
            Slot::Host(b) => b.to_device(self.device).map(Staged::Ready),
            Slot::Source => {
                let prefix = block_prefix(idx);
                let mut raw = std::collections::HashMap::new();
                for n in self.weights.names().into_iter().filter(|n| n.starts_with(&prefix) && !is_mod_tensor(n)) {
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
                    let t = raw.get(n).ok_or_else(|| SynaptixError::Other(format!("qwen-image: нет тензора {n}")))?;
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
            _ => return Err(SynaptixError::Other("qwen-image: стриминг блоков только на CUDA".into())),
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
                    result = Err(SynaptixError::Other("qwen-image: блок не доставлен".into()));
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
                    h.join()
                        .unwrap_or_else(|_| Err(SynaptixError::Other("qwen-image: поток префетча блока упал".into())))
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

    /// Текст энкодера `[1, St, joint]` → вход потока текста `[1, St, D]`
    /// (`txt_norm` + `txt_in`): одинаков на всех шагах, считается один раз.
    pub fn embed_text(&self, txt: &Tensor) -> Result<Tensor> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let x = txt.to_device(self.device)?.to_dtype(self.compute)?;
        let x = rms_norm(&x, &self.txt_norm, EPS)?;
        qf(&self.txt_in, &x)
    }

    /// Шаг денойза. `img` `[1, Si, in_channels]` — латент и следом токены
    /// референсов; `n_target` — сколько первых токенов у генерируемой
    /// картинки. `txt` — из [`Self::embed_text`]. `row` — строка
    /// [`ModTable`] текущего шага. `cos/sin` — из [`build_rope`] по
    /// [`rope_positions`]. → `[1, Si, in_channels]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Tensor,
        n_target: usize,
        txt: &Tensor,
        mods: &ModTable,
        row: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let dt = self.compute;
        let d = self.cfg.inner();
        let si = img.dims()[1];
        let mut img_h = self.img_in.forward(&img.to_device(self.device)?.to_dtype(dt)?)?;
        let mut txt_h = txt.clone();
        let cond_row = match (mods.zero_row, n_target < si) {
            (Some(z), true) => Some(z),
            _ => None,
        };
        self.for_each_block(|i, b| {
            let main = Mod6::from_row(&mods.img[i].narrow(0, row, 1)?.contiguous()?.reshape((6 * d,))?, d)?;
            let cond = match cond_row {
                Some(z) => Some(Mod6::from_row(&mods.img[i].narrow(0, z, 1)?.contiguous()?.reshape((6 * d,))?, d)?),
                None => None,
            };
            let tm = Mod6::from_row(&mods.txt[i].narrow(0, row, 1)?.contiguous()?.reshape((6 * d,))?, d)?;
            let im = ImgMod { main: &main, cond: cond.as_ref().map(|c| (c, n_target)) };
            let (a, t) = b.forward(&self.cfg, &img_h, &txt_h, &im, &tm, cos, sin)?;
            img_h = a;
            txt_h = t;
            if std::env::var("QWEN_IMAGE_DEBUG_AMAX").is_ok() {
                let amax = |t: &Tensor| t.abs().and_then(|a| a.max_all()).and_then(|a| a.to_dtype(DType::F32)).and_then(|a| a.flatten_all()).and_then(|a| a.to_vec1::<f32>()).map(|v| v[0]).unwrap_or(f32::NAN);
                eprintln!("блок {i}: max|img| {:.1} max|txt| {:.1}", amax(&img_h), amax(&txt_h));
            }
            Ok(())
        })?;
        drop(txt_h);
        let o = mods.out.narrow(0, row, 1)?.contiguous()?;
        let scale = o.narrow(1, 0, d)?.contiguous()?.reshape((1, 1, d))?;
        let shift = o.narrow(1, d, d)?.contiguous()?.reshape((1, 1, d))?;
        let x = layer_norm(&img_h, None, None, EPS)?.broadcast_mul(&scale.add_scalar(1.0)?)?.broadcast_add(&shift)?;
        self.proj_out.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_positions_centered() {
        // 4×6 токенов картинки, 3 токена текста; max(h/2, w/2) = 3.
        let p = rope_positions(&[(4, 6)], 3);
        assert_eq!(p[0], [3, 3, 3]);
        assert_eq!(p[2], [5, 5, 5]);
        // Первая строка картинки: y = −2, x от −3.
        assert_eq!(p[3], [0, -2, -3]);
        assert_eq!(p[3 + 5], [0, -2, 2]);
        assert_eq!(p[3 + 23], [0, 1, 2]);
        // Референс — кадр 1; нечётная сторона 5 → от −3 до 1.
        let p = rope_positions(&[(2, 2), (5, 5)], 0);
        assert_eq!(p[4], [1, -3, -3]);
        assert_eq!(p[4 + 24], [1, 1, 1]);
    }

    #[test]
    fn block_param_split() {
        let cfg = QwenImageConfig::from_json(
            br#"{"num_layers":60,"num_attention_heads":24,"attention_head_dim":128,"in_channels":64,"out_channels":16}"#,
        )
        .unwrap();
        let d = cfg.inner();
        assert_eq!(cfg.block_params() - 12 * d * d, 8 * d * d + 16 * d * d);
    }
}
