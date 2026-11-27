//! GigaAM-v4 CTC (`ai-sage/GigaAM-v4`, ветка `ctc`) — кодер Zapformer с двумя
//! CTC-головами: `raw` (строчные буквы без пунктуации) и `e2e` (регистр,
//! пунктуация, цифры). Полный контекст, одна запись за раз.
//!
//! Путь (эталон — `modeling_gigaam.py` из репозитория модели):
//! log-mel 64 полосы (как у v3: окно 320, шаг 160, без центрирования) →
//! фронтенд (3 свёртки 3×3 с прореживанием ×4 по времени, ConvNeXt, проекция
//! 960 → 768) → три стека слоёв (768 / 1024 на вдвое реже сетке / 768; у стека
//! свой ортогональный вход-выход `proj` и взвешенная сумма выходов слоёв) →
//! выходная норма → две головы (линейный слой по кадру) → жадный CTC.
//!
//! Слой: `x += FF1(x)`; внимание с RoPE (q, k — от нормированного ВХОДА слоя,
//! v и два гейта — от x после FF1); у стеков 0 и 2 ещё свёрточный модуль
//! (гейт, depthwise k=5, глобальное среднее по времени) и `FF2`; выход —
//! `RMSNorm(вход + 0,25·(x − вход))`.
//!
//! Счёт — F32, как эталон на процессоре (веса кодера в файле — F16).

use std::path::Path;

use synaptix_audio::resample;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_nn::Module;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::conv2d::conv2d;
use synaptix_ops::conv::depthwise_conv;
use synaptix_ops::norm::rms_norm::rms_norm;
use synaptix_ops::pos::rope::{apply_rope_with_cossin, RopeLayout};

use crate::config::PreprocessorConfig;
use crate::mel::log_mel;
use crate::pipeline::{CtcWord, SR};
use crate::{GigaAmError, Result};

const DT: DType = DType::F32;
const HEAD_DIM: usize = 32;
const NORM_EPS: f32 = 1e-6;
/// Предел длины одного прохода, как у эталона (`transcribe` — до 30 с).
pub const MAX_SECONDS: usize = 30;

fn le(e: impl std::fmt::Display) -> GigaAmError {
    GigaAmError::Load(e.to_string())
}

/// Веса: `.syn`-бандл или каталог HF (`model.safetensors` + `config.json`).
struct Src {
    loader: Box<dyn WeightLoader>,
    device: Device,
}

impl Src {
    fn get(&self, name: &str) -> Result<Tensor> {
        self.loader.load_to(&format!("model.{name}"), self.device, DT).map_err(|e| GigaAmError::Load(format!("{name}: {e}")))
    }
    fn has(&self, name: &str) -> bool {
        self.loader.contains(&format!("model.{name}"))
    }
    fn host(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.loader.load_to(&format!("model.{name}"), Device::Cpu, DT).map_err(|e| GigaAmError::Load(format!("{name}: {e}")))?;
        Ok(t.flatten_all()?.to_vec1::<f32>()?)
    }
    fn lin(&self, base: &str, bias: bool) -> Result<QuantLinear> {
        let w = self.get(&format!("{base}.weight"))?;
        let b = if bias { Some(self.get(&format!("{base}.bias"))?) } else { None };
        Ok(QuantLinear::dense(w, b)?)
    }
}

struct Ff {
    a: QuantLinear,
    b: QuantLinear,
}

impl Ff {
    fn load(s: &Src, p: &str) -> Result<Self> {
        Ok(Self { a: s.lin(&format!("{p}.in_proj"), true)?, b: s.lin(&format!("{p}.out_proj"), true)? })
    }
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.b.forward(&self.a.forward(x)?.silu()?)?)
    }
}

struct ConvModule {
    in_proj: QuantLinear,
    dw_w: Tensor,
    dw_b: Tensor,
    mean_w: Tensor,
    out_proj: QuantLinear,
}

struct Layer {
    ff1: Ff,
    in_norm: Tensor,
    qk: QuantLinear,
    vg: QuantLinear,
    attn_out: QuantLinear,
    conv: Option<(ConvModule, Ff)>,
    norm: Tensor,
    heads: usize,
}

struct Stack {
    /// Вход: `x @ Wpᵀ` (`Wp = proj · in^-½ / 0,3`).
    proj_in: QuantLinear,
    /// Выход: `bypass @ Wp`.
    proj_out: QuantLinear,
    residual_scales: Vec<f32>,
    input_scale: f32,
    layers: Vec<Layer>,
    ds: usize,
}

struct Frontend {
    convs: Vec<(Tensor, Tensor)>,
    /// Depthwise 3×3 ConvNeXt: девять тапов `[1, 64, 1, 1]` и смещение.
    dw_taps: Vec<Tensor>,
    dw_b: Tensor,
    pw1: (Tensor, Tensor),
    pw2: (Tensor, Tensor),
    out: QuantLinear,
}

pub struct Model {
    fe: Frontend,
    stacks: Vec<Stack>,
    out_norm_w: Tensor,
    out_norm_eps: f32,
    cos: Tensor,
    sin: Tensor,
    rope_len: usize,
    head_raw: QuantLinear,
    head_e2e: QuantLinear,
    dim: usize,
    device: Device,
}

/// Нули вокруг `[B, C, H, W]`: слева/справа по W, сверху/снизу по H.
fn pad4(x: &Tensor, wl: usize, wr: usize, ht: usize, hb: usize) -> Result<Tensor> {
    let mut x = x.clone();
    for (dim, before, after) in [(3usize, wl, wr), (2, ht, hb)] {
        if before == 0 && after == 0 {
            continue;
        }
        let d = x.dims().to_vec();
        let mut parts: Vec<Tensor> = Vec::with_capacity(3);
        let z = |n: usize| -> Result<Tensor> {
            let mut s = d.clone();
            s[dim] = n;
            Ok(Tensor::zeros(s, x.dtype(), x.device())?)
        };
        if before > 0 {
            parts.push(z(before)?);
        }
        parts.push(x.clone());
        if after > 0 {
            parts.push(z(after)?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        x = Tensor::cat(&refs, dim)?.contiguous()?;
    }
    Ok(x)
}

fn column(v: Vec<f32>, device: Device) -> Result<Tensor> {
    let n = v.len();
    Ok(Tensor::from_vec(v, vec![n, 1], device)?)
}

impl Frontend {
    fn load(s: &Src) -> Result<Self> {
        let p = "encoder.encoder_embed";
        let wb = |n: &str| -> Result<(Tensor, Tensor)> { Ok((s.get(&format!("{p}.{n}.weight"))?, s.get(&format!("{p}.{n}.bias"))?)) };
        let dw = s.host(&format!("{p}.convnext.depthwise_conv.weight"))?; // [C, 1, 3, 3]
        let c = dw.len() / 9;
        let dw_taps = (0..9)
            .map(|k| Ok(Tensor::from_vec((0..c).map(|ch| dw[ch * 9 + k]).collect::<Vec<f32>>(), vec![1, c, 1, 1], s.device)?))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            convs: vec![wb("conv1")?, wb("conv2")?, wb("conv3")?],
            dw_taps,
            dw_b: s.get(&format!("{p}.convnext.depthwise_conv.bias"))?.reshape(vec![1, c, 1, 1])?,
            pw1: wb("convnext.pointwise_conv1")?,
            pw2: wb("convnext.pointwise_conv2")?,
            out: s.lin(&format!("{p}.out"), true)?,
        })
    }

    /// Признаки `[1, 1, T, 64]` (высота — время) → `[T', 768]`, `T' ≈ T/4`.
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        let one = (1usize, 1usize);
        // conv1: частота по 1 с краёв, время — 2 кадра слева (каузально).
        let x = conv2d(&pad4(x, 1, 1, 2, 0)?, &self.convs[0].0, Some(&self.convs[0].1), one, (0, 0), one)?.silu()?;
        let x = conv2d(&x, &self.convs[1].0, Some(&self.convs[1].1), (2, 2), (1, 0), one)?.silu()?;
        let x = conv2d(&x, &self.convs[2].0, Some(&self.convs[2].1), (2, 2), (1, 0), one)?.silu()?;
        // ConvNeXt: depthwise 3×3 (суммой девяти сдвигов) → 1×1 → SiLU → 1×1, плюс остаток.
        let (t, f) = (x.dims()[2], x.dims()[3]);
        let xp = pad4(&x, 1, 1, 2, 0)?;
        let mut y: Option<Tensor> = None;
        for i in 0..3 {
            for j in 0..3 {
                let tap = xp.narrow(2, i, t)?.narrow(3, j, f)?.contiguous()?.broadcast_mul(&self.dw_taps[i * 3 + j])?;
                y = Some(match y {
                    Some(a) => a.add(&tap)?,
                    None => tap,
                });
            }
        }
        let y = y.expect("девять тапов").broadcast_add(&self.dw_b)?;
        let y = conv2d(&y, &self.pw1.0, Some(&self.pw1.1), one, (0, 0), one)?.silu()?;
        let x = x.add(&conv2d(&y, &self.pw2.0, Some(&self.pw2.1), one, (0, 0), one)?)?;
        // [1, C, T', F] → [T', C·F], признак `c·F + f`.
        let c = x.dims()[1];
        let x = x.transpose(1, 2)?.contiguous()?.reshape(vec![t, c * f])?;
        Ok(self.out.forward(&x)?)
    }
}

impl Layer {
    fn load(s: &Src, p: &str, heads: usize) -> Result<Self> {
        let conv = if s.has(&format!("{p}.conv_module.in_proj.weight")) {
            let c = format!("{p}.conv_module");
            Some((
                ConvModule {
                    in_proj: s.lin(&format!("{c}.in_proj"), true)?,
                    dw_w: s.get(&format!("{c}.depthwise_conv.weight"))?,
                    dw_b: s.get(&format!("{c}.depthwise_conv.bias"))?,
                    mean_w: s.get(&format!("{c}.weighted_mean.weights"))?,
                    out_proj: s.lin(&format!("{c}.out_proj"), true)?,
                },
                Ff::load(s, &format!("{p}.feed_forward2"))?,
            ))
        } else {
            None
        };
        Ok(Self {
            ff1: Ff::load(s, &format!("{p}.feed_forward1"))?,
            in_norm: s.get(&format!("{p}.self_attn.in_norm.weight"))?,
            qk: s.lin(&format!("{p}.self_attn.qk_in_proj"), true)?,
            vg: s.lin(&format!("{p}.self_attn.vg_in_proj"), true)?,
            attn_out: s.lin(&format!("{p}.self_attn.out_proj"), true)?,
            conv,
            norm: s.get(&format!("{p}.norm.weight"))?,
            heads,
        })
    }

    /// `x` `[n, d]`; `valid_col` `[n, 1]` и число валидных кадров; `mask` — запрет
    /// ключей-паддинга `[1, 1, n, n]` (или `None`); таблицы RoPE `[n, 16]`.
    fn fwd(&self, x: &Tensor, valid_col: &Tensor, valid_n: usize, mask: Option<&Tensor>, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (n, d) = (x.dims()[0], x.dims()[1]);
        let (h, w) = (self.heads, self.heads * HEAD_DIM);
        let res = x.clone();
        let mut x = x.add(&self.ff1.fwd(x)?)?;
        // Внимание: q, k — от нормированного входа слоя; v и гейты — от x после FF1.
        let qk = self.qk.forward(&rms_norm(&res, &self.in_norm, NORM_EPS)?)?;
        let heads = |t: Tensor| -> Result<Tensor> { Ok(t.contiguous()?.reshape(vec![n, h, HEAD_DIM])?.permute(vec![1, 0, 2])?.contiguous()?.reshape(vec![1, h, n, HEAD_DIM])?) };
        let q = apply_rope_with_cossin(&heads(qk.narrow(1, 0, w)?)?, cos, sin, RopeLayout::Interleaved)?;
        let k = apply_rope_with_cossin(&heads(qk.narrow(1, w, w)?)?, cos, sin, RopeLayout::Interleaved)?;
        let vg = self.vg.forward(&x)?;
        let v = heads(vg.narrow(1, 0, w)?.contiguous()?.mul(&vg.narrow(1, w, w)?.contiguous()?.sigmoid()?)?)?;
        let o = scaled_dot_attention(&q, &k, &v, 1.0 / (HEAD_DIM as f32).sqrt(), mask)?;
        let o = o.reshape(vec![h, n, HEAD_DIM])?.permute(vec![1, 0, 2])?.contiguous()?.reshape(vec![n, w])?;
        x = x.add(&self.attn_out.forward(&o.mul(&vg.narrow(1, 2 * w, w)?.contiguous()?.sigmoid()?)?)?)?;
        if let Some((c, ff2)) = &self.conv {
            let hg = c.in_proj.forward(&x)?;
            let hh = hg.narrow(1, 0, d)?.contiguous()?.broadcast_mul(valid_col)?;
            let gate = hg.narrow(1, d, d)?.contiguous()?.sigmoid()?;
            // Глобальное среднее по валидным кадрам, одно на все кадры.
            let mean = hh.sum_keepdim(0)?.mul_scalar(1.0 / valid_n.max(1) as f32)?.broadcast_mul(&c.mean_w.reshape(vec![1, d])?)?;
            let hc = hh.transpose(0, 1)?.contiguous()?.reshape(vec![1, d, n])?;
            let y = depthwise_conv(&hc, &c.dw_w, Some(&c.dw_b), 1, c.dw_w.dims()[2] / 2, d)?;
            let y = y.reshape(vec![d, n])?.transpose(0, 1)?.contiguous()?;
            let y = y.broadcast_add(&mean)?.mul(&gate)?.silu()?;
            x = x.add(&c.out_proj.forward(&y)?)?;
            x = x.add(&ff2.fwd(&x)?)?;
        }
        // RMSNorm(вход + 0,25·(x − вход)).
        let mix = res.add(&x.sub(&res)?.mul_scalar(0.25)?)?;
        Ok(rms_norm(&mix, &self.norm, NORM_EPS)?)
    }
}

impl Stack {
    fn load(s: &Src, idx: usize, ds: usize, in_dim: usize, heads: usize, layers: usize) -> Result<Self> {
        let p = format!("encoder.encoder.encoders.{idx}");
        let scale = (in_dim as f32).powf(-0.5) / 0.3;
        let wp = s.get(&format!("{p}.proj.weight"))?.mul_scalar(scale)?; // [d, in]
        let proj_out = QuantLinear::dense(wp.transpose(0, 1)?.contiguous()?, None)?;
        let proj_in = QuantLinear::dense(wp, None)?;
        let residual_scales = s.host(&format!("{p}.residual_scales"))?;
        if residual_scales.len() != layers + 1 {
            return Err(GigaAmError::Load(format!("стек {idx}: весов выходов {}, слоёв {layers}", residual_scales.len())));
        }
        Ok(Self {
            proj_in,
            proj_out,
            residual_scales,
            input_scale: s.host(&format!("{p}.input_scale"))?[0],
            layers: (0..layers).map(|l| Layer::load(s, &format!("{p}.layers.{l}"), heads)).collect::<Result<Vec<_>>>()?,
            ds,
        })
    }
}

/// Список чисел из строки конфига (`"768,1024,768"`) или из числа.
fn list(v: &serde_json::Value) -> Vec<usize> {
    match v {
        serde_json::Value::String(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        serde_json::Value::Number(n) => n.as_u64().map(|x| vec![x as usize]).unwrap_or_default(),
        serde_json::Value::Array(a) => a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect(),
        _ => Vec::new(),
    }
}

impl Model {
    fn load(s: &Src, cfg: &serde_json::Value) -> Result<Self> {
        let enc = &cfg["encoder"];
        if !enc["_target_"].as_str().unwrap_or("").contains("Zapformer") {
            return Err(GigaAmError::Config("кодер не Zapformer — это не GigaAM-v4".into()));
        }
        let (dims, layers, ds, heads) = (list(&enc["encoder_dim"]), list(&enc["num_encoder_layers"]), list(&enc["downsampling_factor"]), list(&enc["num_heads"]));
        if dims.is_empty() || [layers.len(), ds.len(), heads.len()].iter().any(|n| *n != dims.len()) {
            return Err(GigaAmError::Config("кодер: списки размеров стеков разной длины".into()));
        }
        if list(&enc["query_head_dim"]).iter().chain(list(&enc["value_head_dim"]).iter()).any(|d| *d != HEAD_DIM) {
            return Err(GigaAmError::Config(format!("кодер: ширина головы внимания не {HEAD_DIM}")));
        }
        let dim = dims[0];
        let stacks = (0..dims.len()).map(|i| Stack::load(s, i, ds[i], dim * ds[i], heads[i], layers[i])).collect::<Result<Vec<_>>>()?;
        // Таблицы RoPE — из файла (F16): формула в f32 расходится с ними до 2,4e-4.
        let rp = "encoder.encoder.rope_bank.tables.ed32_S10000_b10000";
        let cos = s.get(&format!("{rp}.cos"))?;
        let rope_len = cos.dims()[2];
        let cos = cos.reshape(vec![rope_len, HEAD_DIM / 2])?;
        let sin = s.get(&format!("{rp}.sin"))?.reshape(vec![rope_len, HEAD_DIM / 2])?;
        let head = |n: &str| -> Result<QuantLinear> {
            let w = s.get(&format!("heads.{n}.decoder_layers.0.weight"))?;
            let d = w.dims().to_vec();
            Ok(QuantLinear::dense(w.reshape(vec![d[0], d[1]])?, Some(s.get(&format!("heads.{n}.decoder_layers.0.bias"))?))?)
        };
        let scale = s.host("encoder.encoder.out_norm.scale")?[0];
        let eps = s.host("encoder.encoder.out_norm.eps")?[0];
        Ok(Self {
            fe: Frontend::load(s)?,
            stacks,
            out_norm_w: Tensor::from_vec(vec![scale; dim], vec![dim], s.device)?,
            out_norm_eps: eps * eps,
            cos,
            sin,
            rope_len,
            head_raw: head("raw")?,
            head_e2e: head("e2e")?,
            dim,
            device: s.device,
        })
    }

    /// Признаки `[T, 64]` построчно (кадр — строка) → выход кодера `[T', 768]`.
    /// `tap(имя, тензор)` — контрольные точки сверки.
    pub fn encode(&self, feats: Vec<f32>, frames: usize, tap: &mut dyn FnMut(&str, &Tensor)) -> Result<Tensor> {
        let mels = feats.len() / frames.max(1);
        let x = Tensor::from_vec(feats, vec![1, 1, frames, mels], self.device)?;
        let mut x = self.fe.fwd(&x)?;
        tap("03_frontend_out", &x);
        let t0 = x.dims()[0];
        if t0 % 2 == 1 {
            x = Tensor::cat(&[&x, &Tensor::zeros(vec![1, self.dim], DT, self.device)?], 0)?.contiguous()?;
        }
        let tp = x.dims()[0];
        let valid: Vec<bool> = (0..tp).map(|t| t < t0).collect();
        for (si, st) in self.stacks.iter().enumerate() {
            let ds = st.ds;
            if ds > 1 {
                x = x.broadcast_mul(&column(valid.iter().map(|v| if *v { 1.0 } else { 0.0 }).collect(), self.device)?)?;
            }
            let n = tp / ds;
            if n > self.rope_len {
                return Err(GigaAmError::Audio(format!("запись длиннее таблицы позиций: {n} кадров при пределе {}", self.rope_len)));
            }
            let xin = x.contiguous()?.reshape(vec![n, self.dim * ds])?;
            let v: Vec<bool> = valid.iter().step_by(ds).copied().collect();
            let valid_n = v.iter().filter(|b| **b).count();
            let valid_col = column(v.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect(), self.device)?;
            let mask = if valid_n < n {
                let mut m = vec![0f32; n * n];
                for r in 0..n {
                    for (c, ok) in v.iter().enumerate() {
                        if !ok {
                            m[r * n + c] = -1.0e30;
                        }
                    }
                }
                Some(Tensor::from_vec(m, vec![1, 1, n, n], self.device)?)
            } else {
                None
            };
            let cos = self.cos.narrow(0, 0, n)?.contiguous()?;
            let sin = self.sin.narrow(0, 0, n)?.contiguous()?;
            let mut h = st.proj_in.forward(&xin)?;
            let mut bypass = h.mul_scalar(st.residual_scales[0])?;
            h = h.mul_scalar(st.input_scale)?;
            for (li, layer) in st.layers.iter().enumerate() {
                h = layer.fwd(&h, &valid_col, valid_n, mask.as_ref(), &cos, &sin)?;
                tap(&format!("04_stack{si}_layer{li:02}"), &h);
                bypass = bypass.add(&h.mul_scalar(st.residual_scales[li + 1])?)?;
            }
            let out = xin.add(&st.proj_out.forward(&bypass)?)?;
            tap(&format!("05_stack{si}_out"), &out);
            x = out.reshape(vec![tp, self.dim])?;
        }
        let x = x.narrow(0, 0, t0)?.contiguous()?;
        let x = rms_norm(&x, &self.out_norm_w, self.out_norm_eps)?;
        tap("06_encoder_out", &x);
        Ok(x)
    }

    /// Логиты голов `[T', C]` на хосте: (raw, e2e, число классов raw, e2e).
    fn heads(&self, enc: &Tensor) -> Result<(Vec<f32>, Vec<f32>)> {
        let host = |t: Tensor| -> Result<Vec<f32>> { Ok(t.contiguous()?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?) };
        Ok((host(self.head_raw.forward(enc)?)?, host(self.head_e2e.forward(enc)?)?))
    }
}

/// Результат распознавания.
#[derive(Debug, Clone, Default)]
pub struct Transcript {
    /// Строчные буквы без пунктуации.
    pub raw: String,
    /// С регистром, пунктуацией и цифрами.
    pub e2e: String,
    /// Слова `e2e` с таймкодами (границы — по пробелам) и уверенностью.
    pub words: Vec<CtcWord>,
}

pub struct GigaAm4 {
    model: Model,
    pre: PreprocessorConfig,
    vocab_raw: Vec<String>,
    vocab_e2e: Vec<String>,
    device: Device,
}

/// Серии одинаковых argmax-меток: (метка, первый кадр, конец, макс. вероятность).
fn runs(logits: &[f32], classes: usize) -> Vec<(usize, usize, usize, f32)> {
    let mut out: Vec<(usize, usize, usize, f32)> = Vec::new();
    for (t, row) in logits.chunks(classes).enumerate() {
        let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
        for (i, &v) in row.iter().enumerate() {
            if v > bv {
                best = i;
                bv = v;
            }
        }
        // Вероятность лучшего класса: softmax по строке.
        let z: f32 = row.iter().map(|v| (v - bv).exp()).sum();
        let p = 1.0 / z;
        match out.last_mut() {
            Some(r) if r.0 == best => {
                r.2 = t + 1;
                r.3 = r.3.max(p);
            }
            _ => out.push((best, t, t + 1, p)),
        }
    }
    out
}

impl GigaAm4 {
    /// Признак v4 по `config.json` (HF или из бандла).
    pub fn is_v4_config(bytes: &[u8]) -> bool {
        serde_json::from_slice::<serde_json::Value>(bytes).ok().is_some_and(|v| v["cfg"]["encoder"]["_target_"].as_str().is_some_and(|t| t.contains("Zapformer")))
    }

    /// `.syn`-бандл или каталог HF (`model.safetensors` + `config.json`).
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let (loader, cfg_bytes): (Box<dyn WeightLoader>, Vec<u8>) = if path.is_dir() {
            let l = SafetensorsLoader::open(path.join("model.safetensors")).map_err(le)?.with_device(*device);
            (Box::new(l), std::fs::read(path.join("config.json")).map_err(le)?)
        } else {
            let l = SynBundleLoader::open(path).map_err(|e| GigaAmError::Bundle(e.to_string()))?.with_device(*device);
            let c = l.read_file("config.json").ok_or_else(|| GigaAmError::Bundle("в бандле нет config.json".into()))?;
            (Box::new(l), c)
        };
        let root: serde_json::Value = serde_json::from_slice(&cfg_bytes).map_err(|e| GigaAmError::Config(e.to_string()))?;
        let cfg = &root["cfg"];
        let p = &cfg["preprocessor"];
        let num = |k: &str| -> Result<usize> { p[k].as_u64().map(|v| v as usize).ok_or_else(|| GigaAmError::Config(format!("preprocessor.{k}"))) };
        let pre = PreprocessorConfig {
            sample_rate: num("sample_rate")? as u32,
            features: num("features")?,
            win_length: num("win_length")?,
            hop_length: num("hop_length")?,
            n_fft: num("n_fft")?,
            mel_scale: p["mel_scale"].as_str().unwrap_or("htk").to_string(),
            mel_norm: p["mel_norm"].as_str().map(str::to_string),
            center: p["center"].as_bool().unwrap_or(false),
        };
        if pre.sample_rate != SR || pre.mel_scale != "htk" || pre.mel_norm.is_some() || pre.center || p["highfreq"].as_u64().is_some_and(|f| f != (SR / 2) as u64) {
            return Err(GigaAmError::Config("препроцессор не тот, под который написан порт (16 кГц, HTK, без нормы и центрирования, до 8 кГц)".into()));
        }
        let vocab = |n: &str| -> Result<Vec<String>> {
            let v: Vec<String> = cfg["decodings"][n]["vocabulary"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
            let classes = cfg["heads"][n]["num_classes"].as_u64().unwrap_or(0) as usize;
            if v.is_empty() || v.len() + 1 != classes {
                return Err(GigaAmError::Config(format!("голова {n}: словарь {} при {classes} классах", v.len())));
            }
            Ok(v)
        };
        let (vocab_raw, vocab_e2e) = (vocab("raw")?, vocab("e2e")?);
        let model = Model::load(&Src { loader, device: *device }, cfg)?;
        Ok(Self { model, pre, vocab_raw, vocab_e2e, device: *device })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Один проход (до 30 с, 16 кГц): логиты обеих голов и число кадров.
    fn pass(&self, audio: &[f32], tap: &mut dyn FnMut(&str, &Tensor)) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        if audio.len() < self.pre.win_length {
            return Ok((Vec::new(), Vec::new(), 0));
        }
        let (flat, mels, frames) = log_mel(audio, &self.pre); // [mels][frames]
        let mut rows = vec![0f32; flat.len()];
        for m in 0..mels {
            for t in 0..frames {
                rows[t * mels + m] = flat[m * frames + t];
            }
        }
        if let Ok(t) = Tensor::from_vec(flat, vec![1, mels, frames], Device::Cpu) {
            tap("01_features", &t);
        }
        let enc = self.model.encode(rows, frames, tap)?;
        let n = enc.dims()[0];
        let (raw, e2e) = self.model.heads(&enc)?;
        synaptix_core::memory::arena::release_cpu_pool();
        Ok((raw, e2e, n))
    }

    fn text(vocab: &[String], logits: &[f32]) -> String {
        let blank = vocab.len();
        runs(logits, blank + 1).iter().filter(|r| r.0 != blank).map(|r| vocab[r.0].as_str()).collect()
    }

    /// Куски записи не длиннее 30 с: разрез — в самом тихом месте последних
    /// десяти секунд окна (эталон режет длинные записи своим VAD; здесь —
    /// по энергии, без отдельной модели).
    fn segments(audio: &[f32]) -> Vec<(usize, usize)> {
        let max = MAX_SECONDS * SR as usize;
        let hop = SR as usize / 10;
        let mut out = Vec::new();
        let mut start = 0usize;
        while audio.len() - start > max {
            let (lo, hi) = (start + max * 2 / 3, start + max - hop);
            let mut best = (f32::INFINITY, hi);
            let mut p = lo;
            while p + hop <= hi {
                let e: f32 = audio[p..p + hop].iter().map(|v| v * v).sum();
                if e < best.0 {
                    best = (e, p + hop / 2);
                }
                p += hop / 2;
            }
            out.push((start, best.1));
            start = best.1;
        }
        out.push((start, audio.len()));
        out
    }

    /// PCM любой частоты → оба текста и слова `e2e` с таймкодами.
    pub fn transcribe(&self, pcm: &[f32], sample_rate: u32) -> Result<Transcript> {
        let resampled;
        let audio: &[f32] = if sample_rate == SR {
            pcm
        } else {
            resampled = resample(pcm, sample_rate, SR).map_err(|e| GigaAmError::Audio(e.to_string()))?;
            &resampled
        };
        let mut out = Transcript::default();
        let blank = self.vocab_e2e.len();
        for (a, b) in Self::segments(audio) {
            let (raw, e2e, n) = self.pass(&audio[a..b], &mut |_, _| {})?;
            if n == 0 {
                continue;
            }
            let join = |dst: &mut String, s: &str| {
                let s = s.trim();
                if !s.is_empty() {
                    if !dst.is_empty() {
                        dst.push(' ');
                    }
                    dst.push_str(s);
                }
            };
            join(&mut out.raw, &Self::text(&self.vocab_raw, &raw));
            join(&mut out.e2e, &Self::text(&self.vocab_e2e, &e2e));
            // Слова: кадр → секунды как в эталоне (длина куска / число кадров).
            let shift = (b - a) as f32 / SR as f32 / n as f32;
            let off = a as f32 / SR as f32;
            let mut cur: Option<(CtcWord, Vec<f32>)> = None;
            let flush = |cur: &mut Option<(CtcWord, Vec<f32>)>, words: &mut Vec<CtcWord>| {
                if let Some((mut w, p)) = cur.take() {
                    w.prob = p.iter().sum::<f32>() / p.len().max(1) as f32;
                    words.push(w);
                }
            };
            for &(id, s, e, p) in runs(&e2e, blank + 1).iter().filter(|r| r.0 != blank) {
                let ch = self.vocab_e2e[id].as_str();
                if ch == " " {
                    flush(&mut cur, &mut out.words);
                    continue;
                }
                match cur.as_mut() {
                    Some((w, ps)) => {
                        w.text.push_str(ch);
                        w.end = off + e as f32 * shift;
                        ps.push(p);
                    }
                    None => cur = Some((CtcWord { text: ch.to_string(), start: off + s as f32 * shift, end: off + e as f32 * shift, prob: 0.0 }, vec![p])),
                }
            }
            flush(&mut cur, &mut out.words);
        }
        Ok(out)
    }

    /// Один проход с контрольными точками (сверка с эталоном): логиты `raw` и
    /// `e2e` `[T', C]` и тексты. Запись — 16 кГц, до 30 с.
    pub fn debug_pass(&self, audio16k: &[f32], tap: &mut dyn FnMut(&str, &Tensor)) -> Result<(Vec<f32>, Vec<f32>, usize, String, String)> {
        let (raw, e2e, n) = self.pass(audio16k, tap)?;
        let (tr, te) = (Self::text(&self.vocab_raw, &raw), Self::text(&self.vocab_e2e, &e2e));
        Ok((raw, e2e, n, tr, te))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_collapse_and_probability() {
        // 3 класса (2 — пустой): a a _ b
        let l = [5.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 3.0, 0.0];
        let r = runs(&l, 3);
        assert_eq!(r.iter().map(|x| (x.0, x.1, x.2)).collect::<Vec<_>>(), vec![(0, 0, 2), (2, 2, 3), (1, 3, 4)]);
        assert!(r[0].3 > 0.98 && r[0].3 < 1.0);
        assert_eq!(GigaAm4::text(&["а".into(), "б".into()], &l), "аб");
    }

    #[test]
    fn segments_cover_audio_without_gaps() {
        let n = 75 * SR as usize;
        let mut a = vec![0.5f32; n];
        // Тихое место на 25-й секунде — туда и должен лечь первый разрез.
        for v in &mut a[25 * SR as usize..25 * SR as usize + 4000] {
            *v = 0.0;
        }
        let s = GigaAm4::segments(&a);
        assert!(s.len() >= 3);
        assert_eq!(s[0].0, 0);
        assert_eq!(s.last().unwrap().1, n);
        for w in s.windows(2) {
            assert_eq!(w[0].1, w[1].0);
        }
        assert!(s.iter().all(|(a, b)| b - a <= MAX_SECONDS * SR as usize));
        assert!((s[0].1 as i64 - (25 * SR as i64 + 2000)).abs() < 2000);
    }

    #[test]
    fn config_lists() {
        assert_eq!(list(&serde_json::json!("768,1024,768")), vec![768, 1024, 768]);
        assert_eq!(list(&serde_json::json!(4)), vec![4]);
    }
}
