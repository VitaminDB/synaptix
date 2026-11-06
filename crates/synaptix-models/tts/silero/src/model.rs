//! Сеть Silero TTS v5 (`silero_vocoder.jit_model.MultiTTSModel`): предсказатели длительности и высоты
//! тона (трансформеры FFT 64), такотрон FastPitch (кодировщик FFT 128 ×4, высота тона — свёртка k3,
//! регулятор длины, декодер «песочные часы» с понижением ×3), вокодер Vocos-подобный (ConvNeXt ×8,
//! ISTFT `same` на 48 кГц) и PQMF для 24/8 кГц.
//!
//! Повторены особенности оригинала: в декодере первый «короткий» слой считается и отбрасывается
//! (второй берёт тот же понижённый вход); обнуление высоты на пунктуации в корне не срабатывает
//! (cumsum по оси батча) — его нет и здесь.

use std::collections::HashMap;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::WeightLoader;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_nn::Module;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::{conv1d, depthwise_conv};
use synaptix_ops::norm::layer_norm::layer_norm;

use crate::config::ModelConfig;
use crate::{Result, SileroError};

type TResult<T> = std::result::Result<T, SynaptixError>;

pub struct Weights<'a> {
    pub l: &'a dyn WeightLoader,
    pub device: Device,
    pub dtype: DType,
}

impl Weights<'_> {
    pub fn get(&self, n: &str) -> Result<Tensor> {
        self.l.load_to(n, self.device, self.dtype).map_err(|e| SileroError::Load(format!("{n}: {e}")))
    }

    pub fn host(&self, n: &str) -> Result<Vec<f32>> {
        Ok(self.l.load_to(n, Device::Cpu, DType::F32).map_err(|e| SileroError::Load(format!("{n}: {e}")))?.flatten_all()?.to_vec1::<f32>()?)
    }

    pub fn has(&self, n: &str) -> bool {
        self.l.load_to(n, Device::Cpu, DType::F32).is_ok()
    }

    fn lin(&self, p: &str) -> Result<QuantLinear> {
        Ok(QuantLinear::dense(self.get(&format!("{p}.weight"))?, Some(self.get(&format!("{p}.bias"))?))?)
    }

    fn pair(&self, p: &str) -> Result<(Tensor, Tensor)> {
        Ok((self.get(&format!("{p}.weight"))?, self.get(&format!("{p}.bias"))?))
    }
}

fn t12(x: &Tensor) -> TResult<Tensor> {
    x.transpose(1, 2)?.contiguous()
}

/// Таблица эмбеддингов на хосте: строки по индексам.
struct Table {
    w: Vec<f32>,
    dim: usize,
}

impl Table {
    fn load(w: &Weights, n: &str) -> Result<Self> {
        let t = w.l.load_to(n, Device::Cpu, DType::F32).map_err(|e| SileroError::Load(format!("{n}: {e}")))?;
        let dim = *t.dims().last().unwrap_or(&0);
        Ok(Self { w: t.flatten_all()?.to_vec1::<f32>()?, dim })
    }

    fn rows(&self) -> usize {
        self.w.len() / self.dim.max(1)
    }

    fn row(&self, i: usize) -> Result<&[f32]> {
        self.w.get(i * self.dim..(i + 1) * self.dim).ok_or_else(|| SileroError::Other(format!("индекс эмбеддинга {i} ≥ {}", self.rows())))
    }

    /// Сумма строк по индексам в `out` (`[T, dim]`).
    fn add_into(&self, ids: &[i64], out: &mut [f32]) -> Result<()> {
        for (t, &i) in ids.iter().enumerate() {
            let r = self.row(i as usize)?;
            out[t * self.dim..(t + 1) * self.dim].iter_mut().zip(r).for_each(|(o, v)| *o += v);
        }
        Ok(())
    }
}

/// Блок FFT (пост-норма): внимание → +x → LN; свёртка k9 → ReLU → свёртка k1 → +x → LN.
struct Fft {
    in_w: QuantLinear,
    out: QuantLinear,
    heads: usize,
    conv1: (Tensor, Tensor, usize),
    conv2: (Tensor, Tensor, usize),
    n1: (Tensor, Tensor),
    n2: (Tensor, Tensor),
    pitch_pred: bool,
}

impl Fft {
    fn load(w: &Weights, cfg: &ModelConfig, p: &str) -> Result<Self> {
        let at = cfg.module(p);
        let heads = cfg.module(&format!("{p}.self_attn")).and_then(|m| m.num_heads).unwrap_or(2);
        let conv = |n: &str| -> Result<(Tensor, Tensor, usize)> {
            let (a, b) = w.pair(&format!("{p}.{n}"))?;
            let pad = cfg.module(&format!("{p}.{n}")).and_then(|m| m.conv_pad()).unwrap_or(a.dims()[2] / 2);
            Ok((a, b, pad))
        };
        Ok(Self {
            in_w: QuantLinear::dense(w.get(&format!("{p}.self_attn.in_proj_weight"))?, Some(w.get(&format!("{p}.self_attn.in_proj_bias"))?))?,
            out: w.lin(&format!("{p}.self_attn.out_proj"))?,
            heads,
            conv1: conv("conv1")?,
            conv2: conv("conv2")?,
            n1: w.pair(&format!("{p}.norm1"))?,
            n2: w.pair(&format!("{p}.norm2"))?,
            pitch_pred: at.and_then(|m| m.pitch_pred).unwrap_or(false),
        })
    }

    /// `x` `[B, T, C]`; `mask` — `[B, 1, T, T]` аддитивная маска паддинга ключей.
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> TResult<Tensor> {
        let (b, t, c) = (x.dims()[0], x.dims()[1], x.dims()[2]);
        let hd = c / self.heads;
        let qkv = self.in_w.forward(x)?;
        let part = |i: usize| -> TResult<Tensor> { qkv.narrow(2, i * c, c)?.contiguous()?.reshape(vec![b, t, self.heads, hd])?.permute(&[0, 2, 1, 3])?.contiguous() };
        let a = scaled_dot_attention(&part(0)?, &part(1)?, &part(2)?, 1.0 / (hd as f32).sqrt(), mask)?;
        let a = self.out.forward(&a.permute(&[0, 2, 1, 3])?.contiguous()?.reshape(vec![b, t, c])?)?;
        let x = layer_norm(&x.add(&a)?, Some(&self.n1.0), Some(&self.n1.1), 1e-5)?;
        let y = t12(&x)?;
        let h = conv1d(&y, &self.conv1.0, Some(&self.conv1.1), 1, self.conv1.2)?.relu()?;
        let h = conv1d(&h, &self.conv2.0, Some(&self.conv2.1), 1, self.conv2.2)?;
        let y = if self.pitch_pred {
            let h = h.relu()?;
            y.add(&h)?.add(&h)?
        } else {
            y.add(&h)?
        };
        layer_norm(&t12(&y)?, Some(&self.n2.0), Some(&self.n2.1), 1e-5)
    }
}

/// Синусоидальное позиционное: `x + scale · pe[:T]`.
struct PosEnc {
    pe: Vec<f32>,
    dim: usize,
    scale: f32,
}

impl PosEnc {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        let t = w.l.load_to(&format!("{p}.pe"), Device::Cpu, DType::F32).map_err(|e| SileroError::Load(format!("{p}.pe: {e}")))?;
        let dim = *t.dims().last().unwrap_or(&1);
        Ok(Self { pe: t.flatten_all()?.to_vec1::<f32>()?, dim, scale: w.host(&format!("{p}.scale"))?[0] })
    }

    fn add_host(&self, x: &mut [f32], t: usize) -> Result<()> {
        if t * self.dim > self.pe.len() {
            return Err(SileroError::Other(format!("фраза длиннее позиционной таблицы ({t} > {})", self.pe.len() / self.dim)));
        }
        for (i, v) in x.iter_mut().enumerate() {
            *v += self.scale * self.pe[i % (t * self.dim)];
        }
        Ok(())
    }

    fn tensor(&self, t: usize, dev: Device, dt: DType) -> Result<Tensor> {
        if t * self.dim > self.pe.len() {
            return Err(SileroError::Other(format!("фраза длиннее позиционной таблицы ({t} > {})", self.pe.len() / self.dim)));
        }
        let v: Vec<f32> = self.pe[..t * self.dim].iter().map(|x| x * self.scale).collect();
        Ok(Tensor::from_vec(v, vec![1usize, t, self.dim], Device::Cpu)?.to_dtype(dt)?.to_device(dev)?)
    }
}

/// `ForwardTransformer`: позиционное → FFT ×N → LN.
struct Encoder {
    pos: PosEnc,
    layers: Vec<Fft>,
    norm: (Tensor, Tensor),
}

impl Encoder {
    fn load(w: &Weights, cfg: &ModelConfig, p: &str) -> Result<Self> {
        let mut layers = Vec::new();
        while w.has(&format!("{p}.layers.{}.norm1.weight", layers.len())) {
            layers.push(Fft::load(w, cfg, &format!("{p}.layers.{}", layers.len()))?);
        }
        Ok(Self { pos: PosEnc::load(w, &format!("{p}.pos_encoder"))?, layers, norm: w.pair(&format!("{p}.norm"))? })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let t = x.dims()[1];
        let mut h = x.broadcast_add(&self.pos.tensor(t, x.device(), x.dtype())?)?;
        for l in &self.layers {
            h = l.forward(&h, mask)?;
        }
        Ok(layer_norm(&h, Some(&self.norm.0), Some(&self.norm.1), 1e-5)?)
    }
}

/// `TransformerSeriesPredictor` (длительность или высота тона).
struct Predictor {
    emb: Table,
    spk: Table,
    types: Option<Table>,
    focus: Option<Table>,
    emphasis: Option<Table>,
    enc: Encoder,
    lin: QuantLinear,
}

impl Predictor {
    fn load(w: &Weights, cfg: &ModelConfig, p: &str) -> Result<Self> {
        let at = cfg.module(p);
        let flag = |f: fn(&crate::config::ModuleAttrs) -> Option<bool>| at.and_then(f).unwrap_or(false);
        let opt = |on: bool, n: &str| -> Result<Option<Table>> { if on { Ok(Some(Table::load(w, &format!("{p}.{n}"))?)) } else { Ok(None) } };
        Ok(Self {
            emb: Table::load(w, &format!("{p}.embedding.weight"))?,
            spk: Table::load(w, &format!("{p}.speaker_embedding.weight"))?,
            types: opt(flag(|m| m.utt_type_emb), "type_embedding.weight")?,
            focus: opt(flag(|m| m.focus_emb), "focus_embedding.weight")?,
            emphasis: opt(flag(|m| m.emphasis_emb), "emphasis_embedding.weight")?,
            enc: Encoder::load(w, cfg, &format!("{p}.transformer"))?,
            lin: w.lin(&format!("{p}.lin"))?,
        })
    }

    /// → `[T]` (выход линейного слоя).
    fn forward(&self, seq: &[i64], speaker: usize, extra: &Extra, dev: Device, dt: DType) -> Result<Vec<f32>> {
        let t = seq.len();
        let d = self.emb.dim;
        let mut x = vec![0f32; t * d];
        self.emb.add_into(seq, &mut x)?;
        let s = self.spk.row(speaker)?;
        x.chunks_mut(d).for_each(|r| r.iter_mut().zip(s).for_each(|(o, v)| *o += v));
        if let (Some(tab), Some(ids)) = (&self.types, &extra.type_ids) {
            tab.add_into(ids, &mut x)?;
        }
        if let (Some(tab), Some(ids)) = (&self.focus, &extra.focus) {
            tab.add_into(ids, &mut x)?;
        }
        if let (Some(tab), Some(ids)) = (&self.emphasis, &extra.emphasis) {
            tab.add_into(ids, &mut x)?;
        }
        let x = Tensor::from_vec(x, vec![1usize, t, d], Device::Cpu)?.to_dtype(dt)?.to_device(dev)?;
        let h = self.enc.forward(&x, None)?;
        Ok(self.lin.forward(&h)?.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?)
    }
}

/// Доп. входы предсказателей (v5_4+): тип предложения, фокус, эмфаза — по токену.
#[derive(Default, Clone, Debug)]
pub struct Extra {
    pub type_ids: Option<Vec<i64>>,
    pub focus: Option<Vec<i64>>,
    pub emphasis: Option<Vec<i64>>,
}

/// Декодер «песочные часы».
struct HourGlass {
    pos: PosEnc,
    pre: Fft,
    short1: Fft,
    post: Fft,
    up: QuantLinear,
    factor: usize,
    norm: (Tensor, Tensor),
}

impl HourGlass {
    fn load(w: &Weights, cfg: &ModelConfig, p: &str) -> Result<Self> {
        Ok(Self {
            pos: PosEnc::load(w, &format!("{p}.pos_encoder"))?,
            pre: Fft::load(w, cfg, &format!("{p}.pre_vanilla_layers.0"))?,
            // shorten_layers.0 в оригинале считается, но его выход не используется
            short1: Fft::load(w, cfg, &format!("{p}.shorten_layers.1"))?,
            post: Fft::load(w, cfg, &format!("{p}.post_vanilla_layers.0"))?,
            up: w.lin(&format!("{p}.upsample.proj"))?,
            factor: cfg.module(p).and_then(|m| m.shorten_factor).unwrap_or(3),
            norm: w.pair(&format!("{p}.norm"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, c) = (x.dims()[0], x.dims()[1], x.dims()[2]);
        let h = x.broadcast_add(&self.pos.tensor(l, x.device(), x.dtype())?)?;
        let h = self.pre.forward(&h, None)?;
        let f = self.factor;
        let lp = l.div_ceil(f) * f;
        let h = if lp > l { Tensor::cat(&[&h, &Tensor::zeros(vec![b, lp - l, c], h.dtype(), h.device())?], 1)? } else { h };
        // AvgPool1d(k=f, s=f) по времени
        let down = h.reshape(vec![b, lp / f, f, c])?.sum_keepdim(2)?.affine(1.0 / f as f32, 0.0)?.reshape(vec![b, lp / f, c])?;
        let s = self.short1.forward(&down, None)?;
        let up = self.up.forward(&s)?.reshape(vec![b, lp, c])?;
        let y = up.add(&h)?.narrow(1, 0, l)?.contiguous()?;
        let y = self.post.forward(&y, None)?;
        Ok(layer_norm(&y, Some(&self.norm.0), Some(&self.norm.1), 1e-5)?)
    }
}

struct Tacotron {
    emb: Table,
    spk: Table,
    enc: Encoder,
    pitch_proj: (Tensor, Tensor),
    pitch_strength: f32,
    pitch_frame: bool,
    dec: HourGlass,
    lin: QuantLinear,
}

impl Tacotron {
    fn load(w: &Weights, cfg: &ModelConfig) -> Result<Self> {
        Ok(Self {
            emb: Table::load(w, "tacotron.embedding.weight")?,
            spk: Table::load(w, "tacotron.speaker_embedding.weight")?,
            enc: Encoder::load(w, cfg, "tacotron.encoder")?,
            pitch_proj: w.pair("tacotron.pitch_proj")?,
            pitch_strength: cfg.pitch_strength.unwrap_or(1.0) as f32,
            pitch_frame: cfg.pitch_emb_type.as_deref() == Some("mel_frame"),
            dec: HourGlass::load(w, cfg, "tacotron.decoder")?,
            lin: w.lin("tacotron.lin")?,
        })
    }

    /// → мел `[1, 192, кадров]`.
    fn forward(&self, seq: &[i64], speaker: usize, dur: &[f32], pitch: &[f32], dev: Device, dt: DType) -> Result<Tensor> {
        let t = seq.len();
        let d = self.emb.dim;
        let mut x = vec![0f32; t * d];
        self.emb.add_into(seq, &mut x)?;
        let xt = Tensor::from_vec(x, vec![1usize, t, d], Device::Cpu)?.to_dtype(dt)?.to_device(dev)?;
        let enc = self.enc.forward(&xt, None)?;
        let s = Tensor::from_vec(self.spk.row(speaker)?.to_vec(), vec![1usize, 1, d], Device::Cpu)?.to_dtype(dt)?.to_device(dev)?;
        let mut h = enc.broadcast_add(&s)?;
        let p = Tensor::from_vec(pitch.to_vec(), vec![1usize, 1, t], Device::Cpu)?.to_dtype(dt)?.to_device(dev)?;
        let pp = t12(&conv1d(&p, &self.pitch_proj.0, Some(&self.pitch_proj.1), 1, 1)?)?.affine(self.pitch_strength, 0.0)?;
        if !self.pitch_frame {
            h = h.add(&pp)?;
        }
        // регулятор длины: повтор каждого токена int(dur + 0.5) раз
        let frames: Vec<i64> = dur.iter().enumerate().flat_map(|(i, &dv)| std::iter::repeat_n(i as i64, (dv.max(0.0) + 0.5) as usize)).collect();
        let n = frames.len();
        if n == 0 {
            return Err(SileroError::Other("нулевая длительность фразы".into()));
        }
        let it = Tensor::from_vec(frames, vec![n], dev)?;
        let e = h.reshape(vec![t, d])?.index_select(0, &it)?.reshape(vec![1usize, n, d])?;
        let e = if self.pitch_frame { e.add(&pp)? } else { e };
        let y = self.dec.forward(&e)?;
        Ok(t12(&self.lin.forward(&y)?)?)
    }
}

struct Vocoder {
    embed: (Tensor, Tensor),
    norm: (Tensor, Tensor),
    blocks: Vec<((Tensor, Tensor), (Tensor, Tensor), QuantLinear, QuantLinear, Tensor)>,
    final_norm: (Tensor, Tensor),
    out: QuantLinear,
    window: Vec<f32>,
    n_fft: usize,
    hop: usize,
    win: usize,
    pqmf: HashMap<usize, Vec<f32>>,
    taps: usize,
}

impl Vocoder {
    fn load(w: &Weights, cfg: &ModelConfig) -> Result<Self> {
        let mut blocks = Vec::new();
        while w.has(&format!("vocoder.backbone.convnext.{}.gamma", blocks.len())) {
            let p = format!("vocoder.backbone.convnext.{}", blocks.len());
            let g = w.get(&format!("{p}.gamma"))?;
            let c = g.dims()[0];
            blocks.push((w.pair(&format!("{p}.dwconv"))?, w.pair(&format!("{p}.norm"))?, w.lin(&format!("{p}.pwconv1"))?, w.lin(&format!("{p}.pwconv2"))?, g.reshape(vec![1usize, 1, c])?));
        }
        let ist = cfg.module("vocoder.head.istft").cloned().unwrap_or_default();
        let mut pqmf = HashMap::new();
        for n in [2usize, 6] {
            pqmf.insert(n, w.host(&format!("vocoder.pqmf_{n}.H"))?);
        }
        Ok(Self {
            embed: w.pair("vocoder.backbone.embed")?,
            norm: w.pair("vocoder.backbone.norm")?,
            blocks,
            final_norm: w.pair("vocoder.backbone.final_layer_norm")?,
            out: w.lin("vocoder.head.out")?,
            window: w.host("vocoder.head.istft.window")?,
            n_fft: ist.n_fft.unwrap_or(2400),
            hop: ist.hop_length.unwrap_or(600),
            win: ist.win_length.unwrap_or(2400),
            taps: cfg.module("vocoder.pqmf_2").and_then(|m| m.taps).unwrap_or(62),
            pqmf,
        })
    }

    /// Мел `[1, 192, T]` → звук 48 кГц (`T · hop` отсчётов).
    fn forward(&self, mel: &Tensor) -> Result<Vec<f32>> {
        let x = conv1d(mel, &self.embed.0, Some(&self.embed.1), 1, 3)?;
        let mut h = layer_norm(&t12(&x)?, Some(&self.norm.0), Some(&self.norm.1), 1e-6)?; // [1, T, C]
        for (dw, ln, p1, p2, g) in &self.blocks {
            let c = h.dims()[2];
            let y = t12(&depthwise_conv(&t12(&h)?, &dw.0, Some(&dw.1), 1, 3, c)?)?;
            let y = layer_norm(&y, Some(&ln.0), Some(&ln.1), 1e-6)?;
            let y = p2.forward(&p1.forward(&y)?.gelu_exact()?)?;
            h = h.add(&y.broadcast_mul(g)?)?;
        }
        let h = layer_norm(&h, Some(&self.final_norm.0), Some(&self.final_norm.1), 1e-6)?;
        let o = self.out.forward(&h)?.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let nf = self.n_fft / 2 + 1;
        let t = o.len() / (2 * nf);
        Ok(self.istft_same(&o, t, nf))
    }

    /// ISTFT `same`: irfft кадра · окно, наложение со сдвигом hop, срез pad = (win − hop)/2 с краёв,
    /// деление на огибающую окна².
    fn istft_same(&self, o: &[f32], t: usize, nf: usize) -> Vec<f32> {
        use rustfft::num_complex::Complex32;
        let n = self.n_fft;
        let mut planner = rustfft::FftPlanner::<f32>::new();
        let fft = planner.plan_fft_inverse(n);
        let out_len = (t.max(1) - 1) * self.hop + self.win;
        let mut y = vec![0f32; out_len];
        let mut env = vec![0f32; out_len];
        let mut buf = vec![Complex32::new(0.0, 0.0); n];
        for f in 0..t {
            let r = &o[f * 2 * nf..(f + 1) * 2 * nf];
            for k in 0..nf {
                let m = r[k].exp().min(100.0);
                let p = r[nf + k];
                let (re, im) = (m * p.cos(), m * p.sin());
                // irfft: мнимые части DC и Найквиста отбрасываются
                let im = if k == 0 || k == nf - 1 { 0.0 } else { im };
                buf[k] = Complex32::new(re, im);
                if k > 0 && k < nf - 1 {
                    buf[n - k] = Complex32::new(re, -im);
                }
            }
            fft.process(&mut buf);
            let s = f * self.hop;
            for i in 0..self.win {
                y[s + i] += buf[i].re / n as f32 * self.window[i];
                env[s + i] += self.window[i] * self.window[i];
            }
        }
        let pad = (self.win - self.hop) / 2;
        (pad..out_len - pad).map(|i| y[i] / env[i]).collect()
    }

    /// PQMF-анализ: свёртка с полосой `H` (шаг N, паддинг taps/2), первая полоса; если хоть
    /// одна полоса выходит за ±1 — tanh (как в оригинале).
    fn pqmf(&self, x: &[f32], n: usize) -> Vec<f32> {
        let h = &self.pqmf[&n];
        let k = h.len() / n;
        let pad = self.taps / 2;
        let len = x.len();
        let out_len = (len + 2 * pad - k) / n + 1;
        let mut bands = vec![vec![0f32; out_len]; n];
        let mut maxabs = 0f32;
        for (b, band) in bands.iter_mut().enumerate() {
            let hb = &h[b * k..(b + 1) * k];
            for (o, v) in band.iter_mut().enumerate() {
                let start = o * n;
                let mut acc = 0f32;
                for (j, &c) in hb.iter().enumerate() {
                    let i = start + j;
                    if i >= pad && i - pad < len {
                        acc += c * x[i - pad];
                    }
                }
                *v = acc;
                maxabs = maxabs.max(acc.abs());
            }
        }
        let mut y = std::mem::take(&mut bands[0]);
        if maxabs > 1.0 {
            y.iter_mut().for_each(|v| *v = v.tanh());
        }
        y
    }

    fn resample_out(&self, audio48: Vec<f32>, sr: u32) -> Vec<f32> {
        match sr {
            24000 => self.pqmf(&audio48, 2),
            8000 => self.pqmf(&audio48, 6),
            _ => audio48,
        }
    }
}

/// Вариант логики корня: v5_6 — колпак начала и эмфаза (`fx_durs`), остальные — прежние правки.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RootKind {
    Classic,
    Emph,
}

pub struct SileroNet {
    dur: Predictor,
    pitch: Predictor,
    tac: Tacotron,
    voc: Vocoder,
    pub mean_std: Vec<f32>,
    pub kind: RootKind,
    pub device: Device,
    pub dtype: DType,
}

/// Результат: звук, длительности по токенам, высота тона.
pub struct NetOut {
    pub audio: Vec<f32>,
    pub dur: Vec<f32>,
    pub pitch: Vec<f32>,
    pub mel: Vec<f32>,
}

impl SileroNet {
    pub fn load(w: &Weights, cfg: &ModelConfig) -> Result<Self> {
        let kind = if cfg.modules.get("pitch_predictor.pitch_pred").and_then(|m| m.emphasis_emb).is_some() { RootKind::Emph } else { RootKind::Classic };
        Ok(Self {
            dur: Predictor::load(w, cfg, "dur_predictor.dur_pred")?,
            pitch: Predictor::load(w, cfg, "pitch_predictor.pitch_pred")?,
            tac: Tacotron::load(w, cfg)?,
            voc: Vocoder::load(w, cfg)?,
            mean_std: cfg.mean_std_coef.iter().map(|&v| v as f32).collect(),
            kind,
            device: w.device,
            dtype: w.dtype,
        })
    }

    /// Длительности (кадров) как в корне: exp(x) − 1, ≥ 0, округление, колпаки начала/конца.
    pub fn durations(&self, seq: &[i64], speaker: usize, extra: &Extra, rate: f32) -> Result<Vec<f32>> {
        let raw = self.dur.forward(seq, speaker, extra, self.device, self.dtype)?;
        let mut d: Vec<f32> = raw.iter().map(|v| (v.exp() - 1.0).max(0.0).round_ties_even()).collect();
        let n = d.len();
        match self.kind {
            RootKind::Classic => {
                if d[0] > 5.0 {
                    d[0] = 5.0;
                }
                if (rate - 1.0).abs() > 0.0 {
                    d.iter_mut().for_each(|v| *v = (*v / rate).round_ties_even());
                }
                if d[0] > 5.0 {
                    d[0] = 5.0;
                }
            }
            RootKind::Emph => {
                let emph = extra.type_ids.as_ref().and_then(|t| t.first()).is_some_and(|&t| t >= 6);
                let cap = if emph { 20.0 } else { 5.0 };
                if d[0] > cap {
                    d[0] = cap;
                }
                if (rate - 1.0).abs() > 0.0 {
                    d.iter_mut().for_each(|v| *v = (*v / rate).round_ties_even());
                }
                if d[0] > cap {
                    d[0] = cap;
                }
                if emph {
                    let m = 5.min(n - 1);
                    d[1..m].iter_mut().for_each(|v| *v = v.max(5.0));
                }
            }
        }
        if n >= 3 {
            if d[n - 1] > 7.0 {
                d[n - 1] = 7.0;
            }
            d[n - 2] = 13.0;
            if d[n - 3] > 13.0 {
                d[n - 3] = 13.0;
            }
        }
        Ok(d)
    }

    /// Высота тона: предсказатель, |p| < 0.001 → 0, коэффициент высоты (как update_pitch_coef).
    pub fn pitch(&self, seq: &[i64], speaker: usize, extra: &Extra, coef: f32) -> Result<Vec<f32>> {
        let mut p = self.pitch.forward(seq, speaker, extra, self.device, self.dtype)?;
        p.iter_mut().for_each(|v| {
            if v.abs() < 0.001 {
                *v = 0.0;
            }
        });
        if coef == 1.0 {
            return Ok(p);
        }
        // coef 0 («robot»): всё в ноль; иначе сдвиг (coef − 1)·σ диктора там, где высота не ноль
        let shift = (coef - 1.0) * self.mean_std.get(speaker).copied().unwrap_or(0.0);
        Ok(p.into_iter().map(|v| v * coef).map(|v| if v == 0.0 { 0.0 } else { v + shift }).collect())
    }

    pub fn forward(&self, seq: &[i64], speaker: usize, extra: &Extra, sr: u32, rate: f32, pitch_coef: f32) -> Result<NetOut> {
        let dur = self.durations(seq, speaker, extra, rate)?;
        let pitch = self.pitch(seq, speaker, extra, pitch_coef)?;
        let mel = self.tac.forward(seq, speaker, &dur, &pitch, self.device, self.dtype)?;
        let audio = self.voc.forward(&mel.to_dtype(DType::F32)?)?;
        let mel_host = mel.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(NetOut { audio: self.voc.resample_out(audio, sr), dur, pitch, mel: mel_host })
    }

    /// Вокодер отдельно (для сверки): мел `[192 × T]` построчно по каналам.
    pub fn vocode(&self, mel: &[f32], channels: usize, sr: u32) -> Result<Vec<f32>> {
        let t = mel.len() / channels;
        let m = Tensor::from_vec(mel.to_vec(), vec![1usize, channels, t], Device::Cpu)?.to_device(self.device)?;
        Ok(self.voc.resample_out(self.voc.forward(&m)?, sr))
    }

    pub fn speakers(&self) -> usize {
        self.tac.spk.rows()
    }

    #[doc(hidden)]
    pub fn pos_add_host(&self, x: &mut [f32], t: usize) -> Result<()> {
        self.tac.enc.pos.add_host(x, t)
    }
}
