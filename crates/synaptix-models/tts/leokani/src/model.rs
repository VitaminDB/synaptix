//! Акустическая модель и вокодер Leokani-tts (`leokani.py` upstream).
//!
//! Акустика: id фонем → эмбеддинг 192 → 3× (conv k5 → GELU → LayerNorm) → двунаправленный
//! MLGRU (f = σ, c = SiLU, h = f·h + (1−f)·c, o = (σ(g)·h)·Wo) → длительности (40 Гц, ×2 → 80 Гц)
//! → покадровый декодер (позиция внутри фонемы, 2× MLGRU с преднормой) → conv k5 → GELU → mel 100.
//! Вокодер (Vocos): conv k7 (replicate-паддинг 4 слева / 2 справа) → LN → 8 ConvNeXt (каузальная
//! depthwise k7 с нулями) → LN → голова 1026 = log|S| (513) + фаза (513) → обратное STFT
//! (n_fft 1024, hop 300, окно Ханна, «same»-обрезка 362) → 24 кГц.
//!
//! Корзины длины (NPU: один граф на корзину). Фонемы и кадры добиваются до корзины так, что
//! результат не меняется: после каждого слоя акустики хвост зануляется (свёртки видят нули, как
//! паддинг оригинала), MLGRU сканирует на хосте только реальную длину, перед свёрткой декодера
//! хвост тоже зануляется, вход вокодера добивается повтором последнего кадра (дальше всё каузально),
//! обратное STFT — только по реальным кадрам. Скан, эмбеддинг, развёртка по длительностям,
//! exp/cos/sin спектра и overlap-add — на хосте; свёртки, нормы, линейные и синтез кадров
//! (матмулы cos/sin) — на устройстве модели в её типе (F16 на NPU).

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
/// Масштаб матриц синтеза кадра (их элементы ~1e-3·окно — в F16 ушли бы в субнормали).
const ISTFT_SCALE: f32 = 1024.0;

/// Корзины длины: фонемы (с нулями по краям) и кадры mel.
pub const ID_BUCKETS: [usize; 6] = [32, 64, 128, 192, 288, 416];
pub const FRAME_BUCKETS: [usize; 8] = [128, 256, 384, 512, 768, 1024, 1536, 2048];

/// Наименьшая корзина ≥ n (длиннее последней — кратно ей).
pub fn bucket(n: usize, b: &[usize]) -> usize {
    b.iter().copied().find(|&x| x >= n).unwrap_or_else(|| n.div_ceil(*b.last().unwrap()) * b.last().unwrap())
}

pub struct W<'a> {
    pub l: &'a dyn WeightLoader,
    pub prefix: String,
    pub device: Device,
    /// Тип вычислений на устройстве (F32; F16 на NPU).
    pub dtype: DType,
}

impl W<'_> {
    fn name(&self, n: &str) -> String {
        if self.prefix.is_empty() { n.to_string() } else { format!("{}.{n}", self.prefix) }
    }

    pub fn get(&self, n: &str) -> Result<Tensor> {
        let name = self.name(n);
        self.l.load_to(&name, self.device, self.dtype).map_err(|e| LkError::Load(format!("{name}: {e}")))
    }

    pub fn get_cpu(&self, n: &str) -> Result<Tensor> {
        let name = self.name(n);
        self.l.load_to(&name, Device::Cpu, DType::F32).map_err(|e| LkError::Load(format!("{name}: {e}")))
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
    Ok(x.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?)
}

fn upload(v: Vec<f32>, dims: Vec<usize>, dev: Device, dtype: DType) -> Result<Tensor> {
    Ok(Tensor::from_vec(v, dims, Device::Cpu)?.to_dtype(dtype)?.to_device(dev)?)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// MLGRU: смешивание по времени рекуррентным сканом на хосте (каналов 256…384).
struct Mlgru {
    wfcg: QuantLinear,
    wo: QuantLinear,
}

impl Mlgru {
    fn load(w: &W, p: &str) -> Result<Self> {
        Ok(Self { wfcg: w.lin(&format!("{p}.wfcg"))?, wo: w.lin(&format!("{p}.wo"))? })
    }

    /// `x` `[1, T, C]`; скан по первым `real` позициям (хвост корзины — нули до `wo`);
    /// `reverse` — скан с конца реальной части.
    fn forward(&self, x: &Tensor, real: usize, reverse: bool) -> Result<Tensor> {
        let t = x.dims()[1];
        let z = host(&self.wfcg.forward(x)?)?;
        let d = z.len() / t / 3;
        let mut out = vec![0f32; t * d];
        let mut h = vec![0f32; d];
        for s in 0..real {
            let i = if reverse { real - 1 - s } else { s };
            let row = &z[i * 3 * d..(i + 1) * 3 * d];
            for k in 0..d {
                let f = sigmoid(row[k]);
                let c = row[d + k];
                let c = c * sigmoid(c);
                h[k] = f * h[k] + (1.0 - f) * c;
                out[i * d + k] = sigmoid(row[2 * d + k]) * h[k];
            }
        }
        let o = upload(out, vec![1usize, t, d], x.device(), x.dtype())?;
        Ok(self.wo.forward(&o)?)
    }
}

/// Маска `[1, T, 1]`: единицы на первых `real` позициях.
fn mask(real: usize, t: usize, dev: Device, dtype: DType) -> Result<Tensor> {
    let m: Vec<f32> = (0..t).map(|i| if i < real { 1.0 } else { 0.0 }).collect();
    upload(m, vec![1usize, t, 1], dev, dtype)
}

pub struct Acoustic {
    /// Таблица эмбеддингов — на хосте (строки выбираются там).
    embed: Vec<f32>,
    emb_dim: usize,
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
    dtype: DType,
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
        let e = w.get_cpu(&p("embed.weight"))?;
        let emb_dim = e.dims()[1];
        Ok(Self {
            embed: e.flatten_all()?.to_vec1::<f32>()?,
            emb_dim,
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
            dtype: w.dtype,
        })
    }

    /// id `[0, ids…, 0]` → (кодировка фонем на хосте `[n, enc]`, лог-длительности `[n]`);
    /// `nb` — длина корзины (≥ n).
    pub fn encode(&self, ids: &[i64], nb: usize) -> Result<(Vec<f32>, usize, Vec<f32>)> {
        let n = ids.len();
        let d = self.emb_dim;
        let mut x = vec![0f32; nb * d];
        for (i, &id) in ids.iter().enumerate() {
            let r = id as usize * d;
            x[i * d..(i + 1) * d].copy_from_slice(&self.embed[r..r + d]);
        }
        let mut x = upload(x, vec![1usize, nb, d], self.device, self.dtype)?;
        let m = (nb > n).then(|| mask(n, nb, self.device, self.dtype)).transpose()?;
        for (cw, cb, ln) in &self.convs {
            let y = conv1d(&t12(&x)?, cw, Some(cb), 1, 2)?.gelu_exact()?;
            x = layer_norm(&t12(&y)?, Some(&ln.0), Some(&ln.1), 1e-5)?;
            if let Some(m) = &m {
                x = x.broadcast_mul(m)?;
            }
        }
        let h = Tensor::cat(&[&self.fwd.forward(&x, n, false)?, &self.bwd.forward(&x, n, true)?], 2)?;
        let e = layer_norm(&self.enc_out.forward(&h)?, Some(&self.enc_norm.0), Some(&self.enc_norm.1), 1e-5)?;
        let logd = host(&self.dur.forward(&e)?)?[..n].to_vec();
        let eh = host(&e)?;
        let c = eh.len() / nb;
        Ok((eh[..n * c].to_vec(), c, logd))
    }

    /// Кадров на id при 80 Гц: 2·max(1, round_even((e^d − 1)/speed)).
    pub fn durations(logd: &[f32], speed: f32) -> Vec<usize> {
        logd.iter().map(|&d| 2 * ((((d.exp() - 1.0) / speed).round_ties_even()).max(1.0) as usize)).collect()
    }

    /// Покадровый декодер → log-mel на хосте `[T, n_mels]` (кадр за кадром); `tb` — корзина кадров.
    pub fn decode(&self, e: &[f32], c: usize, d80: &[usize], tb: usize) -> Result<Vec<f32>> {
        let total: usize = d80.iter().sum();
        let tb = tb.max(total);
        let mut x = vec![0f32; tb * (c + 1)];
        let mut f = 0;
        for (i, &d) in d80.iter().enumerate() {
            for k in 0..d {
                x[f * (c + 1)..f * (c + 1) + c].copy_from_slice(&e[i * c..(i + 1) * c]);
                x[f * (c + 1) + c] = (k as f32 + 0.5) / d as f32;
                f += 1;
            }
        }
        let mut x = self.dec_in.forward(&upload(x, vec![1usize, tb, c + 1], self.device, self.dtype)?)?;
        for (g, ln) in &self.dec {
            let y = g.forward(&layer_norm(&x, Some(&ln.0), Some(&ln.1), 1e-5)?, total, false)?;
            x = x.add(&y)?;
        }
        if tb > total {
            x = x.broadcast_mul(&mask(total, tb, self.device, self.dtype)?)?;
        }
        let y = t12(&conv1d(&t12(&x)?, &self.post.0, Some(&self.post.1), 1, 2)?.gelu_exact()?)?;
        let mel = host(&self.head.forward(&y)?)?;
        let w = mel.len() / tb;
        Ok((0..total).flat_map(|t| mel[t * w..t * w + self.n_mels].to_vec()).collect())
    }

    /// id → log-mel `[T, n_mels]` (хост); `buckets` — добивать до корзин длины.
    pub fn forward(&self, ids: &[i64], speed: f32, buckets: bool) -> Result<Vec<f32>> {
        let nb = if buckets { bucket(ids.len(), &ID_BUCKETS) } else { ids.len() };
        let (e, c, logd) = self.encode(ids, nb)?;
        let d80 = Self::durations(&logd, speed);
        let total: usize = d80.iter().sum();
        let tb = if buckets { bucket(total, &FRAME_BUCKETS) } else { total };
        self.decode(&e, c, &d80, tb)
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
    /// Синтез кадра обратным rFFT с окном ×`ISTFT_SCALE`: `[513, 1024]` для Re и Im.
    icos: Tensor,
    isin: Tensor,
    window: Vec<f32>,
    n_mels: usize,
    device: Device,
    dtype: DType,
}

fn pad_zeros_left(x: &Tensor, left: usize) -> Result<Tensor> {
    let d = x.dims().to_vec();
    let z = Tensor::zeros(vec![d[0], d[1], left], x.dtype(), x.device())?;
    Ok(Tensor::cat(&[&z, x], 2)?)
}

impl Vocoder {
    pub fn load(w: &W, blocks: usize, n_mels: usize) -> Result<Self> {
        let p = |s: &str| format!("vocoder.{s}");
        let mut bl = Vec::new();
        for i in 0..blocks {
            let b = |s: &str| p(&format!("blocks.{i}.{s}"));
            bl.push(Block {
                dw: (w.get(&b("dwconv.weight"))?, w.get(&b("dwconv.bias"))?),
                ln: w.ln(&b("norm"))?,
                pw1: w.lin(&b("pwconv1"))?,
                pw2: w.lin(&b("pwconv2"))?,
                gamma: w.get(&b("gamma"))?,
            });
        }
        // hann_window(1024) (периодическое), irfft «backward»: 1/N, двойные веса у 1…511.
        let window: Vec<f32> = (0..N_FFT).map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos()) as f32).collect();
        let bins = N_FFT / 2 + 1;
        let (mut c, mut s) = (vec![0f32; bins * N_FFT], vec![0f32; bins * N_FFT]);
        for k in 0..bins {
            let wk = if k == 0 || k == N_FFT / 2 { 1.0 } else { 2.0 } / N_FFT as f64 * ISTFT_SCALE as f64;
            for n in 0..N_FFT {
                let a = 2.0 * std::f64::consts::PI * ((k * n) % N_FFT) as f64 / N_FFT as f64;
                c[k * N_FFT + n] = (wk * a.cos() * window[n] as f64) as f32;
                s[k * N_FFT + n] = (-wk * a.sin() * window[n] as f64) as f32;
            }
        }
        // Матрицы синтеза — веса графа (статика), не входы.
        let mk = |v: Vec<f32>| -> Result<Tensor> {
            let t = Tensor::from_vec(v, vec![bins, N_FFT], Device::Cpu)?.to_dtype(w.dtype)?;
            Ok(synaptix_core::backend::with_param_upload(|| t.to_device(w.device))?)
        };
        Ok(Self {
            embed: (w.get(&p("embed.weight"))?, w.get(&p("embed.bias"))?),
            norm: w.ln(&p("norm"))?,
            blocks: bl,
            fin: w.ln(&p("final"))?,
            head: w.lin(&p("head"))?,
            icos: mk(c)?,
            isin: mk(s)?,
            window,
            n_mels,
            device: w.device,
            dtype: w.dtype,
        })
    }

    /// log-mel `[T, n_mels]` (хост) → PCM `T·300`; `tb` — корзина кадров (≥ T).
    pub fn forward(&self, mel: &[f32], tb: usize) -> Result<Vec<f32>> {
        let m = self.n_mels;
        let t = mel.len() / m;
        let tb = tb.max(t);
        // [1, m, 4 + tb + 2]: replicate слева 4, справа — до корзины и ещё 2 (lookahead).
        let w = tb + 6;
        let mut x = vec![0f32; m * w];
        for j in 0..w {
            let src = j.saturating_sub(6 - LOOKAHEAD).min(t - 1);
            for ch in 0..m {
                x[ch * w + j] = mel[src * m + ch];
            }
        }
        let x = upload(x, vec![1usize, m, w], self.device, self.dtype)?;
        let x = conv1d(&x, &self.embed.0, Some(&self.embed.1), 1, 0)?;
        let mut x = t12(&layer_norm(&t12(&x)?, Some(&self.norm.0), Some(&self.norm.1), 1e-6)?)?;
        for b in &self.blocks {
            let c = x.dims()[1];
            let y = depthwise_conv(&pad_zeros_left(&x, 6)?, &b.dw.0, Some(&b.dw.1), 1, 0, c)?;
            let y = layer_norm(&t12(&y)?, Some(&b.ln.0), Some(&b.ln.1), 1e-6)?;
            let y = b.pw2.forward(&b.pw1.forward(&y)?.gelu_exact()?)?.broadcast_mul(&b.gamma)?;
            x = x.add(&t12(&y)?)?;
        }
        let h = host(&self.head.forward(&layer_norm(&t12(&x)?, Some(&self.fin.0), Some(&self.fin.1), 1e-6)?)?)?; // [tb, 1026]
        let bins = N_FFT / 2 + 1;
        let (mut re, mut im) = (vec![0f32; tb * bins], vec![0f32; tb * bins]);
        for f in 0..t {
            let row = &h[f * 2 * bins..(f + 1) * 2 * bins];
            for k in 0..bins {
                let mag = row[k].exp().min(100.0);
                let ph = row[bins + k];
                re[f * bins + k] = mag * ph.cos();
                im[f * bins + k] = mag * ph.sin();
            }
        }
        let re = upload(re, vec![tb, bins], self.device, self.dtype)?;
        let im = upload(im, vec![tb, bins], self.device, self.dtype)?;
        let frames = host(&re.matmul(&self.icos)?.add(&im.matmul(&self.isin)?)?)?; // [tb, 1024], с окном
        // overlap-add и деление на сумму квадратов окна, «same»-обрезка
        let size = (t - 1) * HOP + N_FFT;
        let (mut y, mut env) = (vec![0f32; size], vec![0f32; size]);
        for f in 0..t {
            let o = f * HOP;
            for n in 0..N_FFT {
                y[o + n] += frames[f * N_FFT + n] / ISTFT_SCALE;
                env[o + n] += self.window[n] * self.window[n];
            }
        }
        Ok((PAD..size - PAD).map(|i| y[i] / env[i].max(1e-11)).collect())
    }
}
