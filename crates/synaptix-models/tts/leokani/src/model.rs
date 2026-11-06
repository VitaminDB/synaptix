//! Акустическая модель и вокодер Leokani-tts (`leokani.py` upstream).
//!
//! Акустика: id фонем → эмбеддинг 192 → 3× (conv k5 → GELU → LayerNorm) → двунаправленный
//! MLGRU (f = σ, c = SiLU, h = f·h + (1−f)·c, o = (σ(g)·h)·Wo) → длительности (40 Гц, ×2 → 80 Гц)
//! → покадровый декодер (позиция внутри фонемы, 2× MLGRU с преднормой) → conv k5 → GELU → mel 100.
//! Вокодер (Vocos): conv k7 (replicate-паддинг 4 слева / 2 справа) → LN → 8 ConvNeXt (каузальная
//! depthwise k7 с нулями) → LN → голова 1026 = log|S| (513) + фаза (513) → обратное STFT
//! (n_fft 1024, hop 300, окно Ханна, «same»-обрезка 362) → 24 кГц.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::WeightLoader;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_nn::Module;
use synaptix_ops::conv::{conv1d, depthwise_conv};
use synaptix_ops::norm::layer_norm::layer_norm;

use crate::{LkError, Result};

pub const SR: u32 = 24_000;
const N_FFT: usize = 1024;
const HOP: usize = 300;
const PAD: usize = (N_FFT - HOP) / 2;
const LOOKAHEAD: usize = 2;

pub struct W<'a> {
    pub l: &'a dyn WeightLoader,
    pub prefix: String,
    pub device: Device,
}

impl W<'_> {
    pub fn get(&self, n: &str) -> Result<Tensor> {
        let name = if self.prefix.is_empty() { n.to_string() } else { format!("{}.{n}", self.prefix) };
        self.l.load_to(&name, self.device, DType::F32).map_err(|e| LkError::Load(format!("{name}: {e}")))
    }

    fn lin(&self, p: &str) -> Result<QuantLinear> {
        Ok(QuantLinear::dense(self.get(&format!("{p}.weight"))?, Some(self.get(&format!("{p}.bias"))?))?)
    }

    fn ln(&self, p: &str) -> Result<(Tensor, Tensor)> {
        Ok((self.get(&format!("{p}.weight"))?, self.get(&format!("{p}.bias"))?))
    }
}

fn t12(x: &Tensor) -> Result<Tensor> {
    Ok(x.transpose(1, 2)?.contiguous()?)
}

fn host(x: &Tensor) -> Result<Vec<f32>> {
    Ok(x.flatten_all()?.to_vec1::<f32>()?)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// MLGRU: смешивание по времени рекуррентным сканом (на хосте — каналов 256…384).
struct Mlgru {
    wfcg: QuantLinear,
    wo: QuantLinear,
}

impl Mlgru {
    fn load(w: &W, p: &str) -> Result<Self> {
        Ok(Self { wfcg: w.lin(&format!("{p}.wfcg"))?, wo: w.lin(&format!("{p}.wo"))? })
    }

    /// `x` `[1, T, C]`; `reverse` — скан с конца (обратное направление без переворотов).
    fn forward(&self, x: &Tensor, reverse: bool) -> Result<Tensor> {
        let t = x.dims()[1];
        let z = host(&self.wfcg.forward(x)?)?;
        let d = z.len() / t / 3;
        let mut out = vec![0f32; t * d];
        let mut h = vec![0f32; d];
        for s in 0..t {
            let i = if reverse { t - 1 - s } else { s };
            let row = &z[i * 3 * d..(i + 1) * 3 * d];
            for k in 0..d {
                let f = sigmoid(row[k]);
                let c = row[d + k];
                let c = c * sigmoid(c);
                h[k] = f * h[k] + (1.0 - f) * c;
                out[i * d + k] = sigmoid(row[2 * d + k]) * h[k];
            }
        }
        let o = Tensor::from_vec(out, vec![1usize, t, d], x.device())?;
        Ok(self.wo.forward(&o)?)
    }
}

pub struct Acoustic {
    embed: Tensor,
    convs: Vec<(Tensor, Tensor, (Tensor, Tensor))>,
    fwd: Mlgru,
    bwd: Mlgru,
    enc_out: QuantLinear,
    enc_norm: (Tensor, Tensor),
    dur: QuantLinear,
    dec_in: QuantLinear,
    dec: Vec<(Mlgru, (Tensor, Tensor))>,
    post: (Tensor, Tensor),
    head: QuantLinear,
    n_mels: usize,
    device: Device,
}

impl Acoustic {
    pub fn load(w: &W, n_mels: usize) -> Result<Self> {
        let p = |s: &str| format!("acoustic.{s}");
        let mut convs = Vec::new();
        for i in 0..3 {
            convs.push((w.get(&p(&format!("convs.{i}.weight")))?, w.get(&p(&format!("convs.{i}.bias")))?, w.ln(&p(&format!("norms.{i}")))?));
        }
        let mut dec = Vec::new();
        for i in 0..2 {
            dec.push((Mlgru::load(w, &p(&format!("dec.{i}")))?, w.ln(&p(&format!("dec_norms.{i}")))?));
        }
        Ok(Self {
            embed: w.get(&p("embed.weight"))?,
            convs,
            fwd: Mlgru::load(w, &p("fwd"))?,
            bwd: Mlgru::load(w, &p("bwd"))?,
            enc_out: w.lin(&p("enc_out"))?,
            enc_norm: w.ln(&p("enc_norm"))?,
            dur: w.lin(&p("dur"))?,
            dec_in: w.lin(&p("dec_in"))?,
            dec,
            post: (w.get(&p("post.weight"))?, w.get(&p("post.bias"))?),
            head: w.lin(&p("head"))?,
            n_mels,
            device: w.device,
        })
    }

    /// id `[0, ids…, 0]` → (кодировка `[1, N, 256]`, лог-длительности).
    pub fn encode(&self, ids: &[i64]) -> Result<(Tensor, Vec<f32>)> {
        let n = ids.len();
        let it = Tensor::from_vec(ids.to_vec(), vec![n], self.device)?;
        let mut x = self.embed.index_select(0, &it)?.reshape(vec![1usize, n, self.embed.dims()[1]])?;
        for (cw, cb, ln) in &self.convs {
            let y = conv1d(&t12(&x)?, cw, Some(cb), 1, 2)?.gelu_exact()?;
            x = layer_norm(&t12(&y)?, Some(&ln.0), Some(&ln.1), 1e-5)?;
        }
        let h = Tensor::cat(&[&self.fwd.forward(&x, false)?, &self.bwd.forward(&x, true)?], 2)?;
        let e = layer_norm(&self.enc_out.forward(&h)?, Some(&self.enc_norm.0), Some(&self.enc_norm.1), 1e-5)?;
        let logd = host(&self.dur.forward(&e)?)?;
        Ok((e, logd))
    }

    /// Кадров на id при 80 Гц: 2·max(1, round_even((e^d − 1)/speed)).
    pub fn durations(logd: &[f32], speed: f32) -> Vec<usize> {
        logd.iter().map(|&d| 2 * ((((d.exp() - 1.0) / speed).round_ties_even()).max(1.0) as usize)).collect()
    }

    /// Покадровый декодер → log-mel `[1, n_mels, T]`.
    pub fn decode(&self, e: &Tensor, d80: &[usize]) -> Result<Tensor> {
        let total: usize = d80.iter().sum();
        let mut idx = Vec::with_capacity(total);
        let mut pos = Vec::with_capacity(total);
        for (i, &d) in d80.iter().enumerate() {
            for k in 0..d {
                idx.push(i as i64);
                pos.push((k as f32 + 0.5) / d as f32);
            }
        }
        let c = e.dims()[2];
        let it = Tensor::from_vec(idx, vec![total], self.device)?;
        let ex = e.reshape(vec![e.dims()[1], c])?.index_select(0, &it)?;
        let pt = Tensor::from_vec(pos, vec![total, 1], self.device)?;
        let mut x = self.dec_in.forward(&Tensor::cat(&[&ex, &pt], 1)?.reshape(vec![1usize, total, c + 1])?)?;
        for (g, ln) in &self.dec {
            let y = g.forward(&layer_norm(&x, Some(&ln.0), Some(&ln.1), 1e-5)?, false)?;
            x = x.add(&y)?;
        }
        let y = t12(&conv1d(&t12(&x)?, &self.post.0, Some(&self.post.1), 1, 2)?.gelu_exact()?)?;
        let mel = self.head.forward(&y)?.narrow(2, 0, self.n_mels)?.contiguous()?;
        t12(&mel)
    }

    pub fn forward(&self, ids: &[i64], speed: f32) -> Result<Tensor> {
        let (e, logd) = self.encode(ids)?;
        self.decode(&e, &Self::durations(&logd, speed))
    }
}

struct Block {
    dw: (Tensor, Tensor),
    ln: (Tensor, Tensor),
    pw1: QuantLinear,
    pw2: QuantLinear,
    gamma: Tensor,
}

pub struct Vocoder {
    embed: (Tensor, Tensor),
    norm: (Tensor, Tensor),
    blocks: Vec<Block>,
    fin: (Tensor, Tensor),
    head: QuantLinear,
    /// Синтез кадра обратным rFFT с окном: `[513, 1024]` для Re и Im.
    icos: Tensor,
    isin: Tensor,
    window: Vec<f32>,
    device: Device,
}

fn pad_replicate(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    let d = x.dims().to_vec();
    let mut parts = Vec::new();
    if left > 0 {
        parts.push(x.narrow(2, 0, 1)?.broadcast_as(vec![d[0], d[1], left])?.contiguous()?);
    }
    parts.push(x.clone());
    if right > 0 {
        parts.push(x.narrow(2, d[2] - 1, 1)?.broadcast_as(vec![d[0], d[1], right])?.contiguous()?);
    }
    let r: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&r, 2)?)
}

fn pad_zeros_left(x: &Tensor, left: usize) -> Result<Tensor> {
    let d = x.dims().to_vec();
    let z = Tensor::zeros(vec![d[0], d[1], left], x.dtype(), x.device())?;
    Ok(Tensor::cat(&[&z, x], 2)?)
}

impl Vocoder {
    pub fn load(w: &W, blocks: usize) -> Result<Self> {
        let p = |s: &str| format!("vocoder.{s}");
        let mut bl = Vec::new();
        for i in 0..blocks {
            let b = |s: &str| p(&format!("blocks.{i}.{s}"));
            let gamma = w.get(&b("gamma"))?;
            bl.push(Block {
                dw: (w.get(&b("dwconv.weight"))?, w.get(&b("dwconv.bias"))?),
                ln: w.ln(&b("norm"))?,
                pw1: w.lin(&b("pwconv1"))?,
                pw2: w.lin(&b("pwconv2"))?,
                gamma,
            });
        }
        // hann_window(1024) (периодическое), irfft «backward»: 1/N, двойные веса у 1…511.
        let window: Vec<f32> = (0..N_FFT).map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos()) as f32).collect();
        let bins = N_FFT / 2 + 1;
        let (mut c, mut s) = (vec![0f32; bins * N_FFT], vec![0f32; bins * N_FFT]);
        for k in 0..bins {
            let wk = if k == 0 || k == N_FFT / 2 { 1.0 } else { 2.0 } / N_FFT as f64;
            for n in 0..N_FFT {
                let a = 2.0 * std::f64::consts::PI * ((k * n) % N_FFT) as f64 / N_FFT as f64;
                c[k * N_FFT + n] = (wk * a.cos() * window[n] as f64) as f32;
                s[k * N_FFT + n] = (-wk * a.sin() * window[n] as f64) as f32;
            }
        }
        Ok(Self {
            embed: (w.get(&p("embed.weight"))?, w.get(&p("embed.bias"))?),
            norm: w.ln(&p("norm"))?,
            blocks: bl,
            fin: w.ln(&p("final"))?,
            head: w.lin(&p("head"))?,
            icos: Tensor::from_vec(c, vec![bins, N_FFT], w.device)?,
            isin: Tensor::from_vec(s, vec![bins, N_FFT], w.device)?,
            window,
            device: w.device,
        })
    }

    /// log-mel `[1, n_mels, T]` → PCM `T·300`.
    pub fn forward(&self, mel: &Tensor) -> Result<Vec<f32>> {
        let x = conv1d(&pad_replicate(mel, 6 - LOOKAHEAD, LOOKAHEAD)?, &self.embed.0, Some(&self.embed.1), 1, 0)?;
        let mut x = t12(&layer_norm(&t12(&x)?, Some(&self.norm.0), Some(&self.norm.1), 1e-6)?)?;
        for b in &self.blocks {
            let c = x.dims()[1];
            let y = depthwise_conv(&pad_zeros_left(&x, 6)?, &b.dw.0, Some(&b.dw.1), 1, 0, c)?;
            let y = layer_norm(&t12(&y)?, Some(&b.ln.0), Some(&b.ln.1), 1e-6)?;
            let y = b.pw2.forward(&b.pw1.forward(&y)?.gelu_exact()?)?.broadcast_mul(&b.gamma)?;
            x = x.add(&t12(&y)?)?;
        }
        let h = self.head.forward(&layer_norm(&t12(&x)?, Some(&self.fin.0), Some(&self.fin.1), 1e-6)?)?; // [1, T, 1026]
        let t = h.dims()[1];
        let bins = N_FFT / 2 + 1;
        let h = h.reshape(vec![t, 2 * bins])?;
        let logmag = h.narrow(1, 0, bins)?.contiguous()?;
        let phase = h.narrow(1, bins, bins)?.contiguous()?;
        // |S| = min(e^logmag, 100)
        let mag = host(&logmag)?.into_iter().map(|v| v.exp().min(100.0)).collect::<Vec<f32>>();
        let mag = Tensor::from_vec(mag, vec![t, bins], self.device)?;
        let re = mag.mul(&phase.cos()?)?;
        let im = mag.mul(&phase.sin()?)?;
        let frames = host(&re.matmul(&self.icos)?.add(&im.matmul(&self.isin)?)?)?; // [T, 1024], уже с окном
        // overlap-add и деление на сумму квадратов окна, «same»-обрезка
        let size = (t - 1) * HOP + N_FFT;
        let (mut y, mut env) = (vec![0f32; size], vec![0f32; size]);
        for f in 0..t {
            let o = f * HOP;
            for n in 0..N_FFT {
                y[o + n] += frames[f * N_FFT + n];
                env[o + n] += self.window[n] * self.window[n];
            }
        }
        Ok((PAD..size - PAD).map(|i| y[i] / env[i].max(1e-11)).collect())
    }
}
