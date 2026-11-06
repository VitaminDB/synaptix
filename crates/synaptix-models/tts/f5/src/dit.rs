//! DiT F5-TTS v1 (`backbones/dit.py`, `modules.py`): текстовые вложения (эмбеддинг + синусоиды +
//! 4 ConvNeXtV2 с GRN), вход `proj(x ‖ cond ‖ текст)` + свёрточное позиционное (групповая k31, Mish),
//! 22 блока DiT (AdaLN-zero, внимание 16×64 с RoPE x_transformers, FF ×2 с GELU-tanh), финальный
//! AdaLN и проекция в 100 мел-полос. CFG — батчем 2: условный и безусловный (без образца и текста).
//!
//! RoPE x_transformers — пары соседних измерений (2j, 2j+1); строки `to_q`/`to_k` переставляются по
//! головам в раскладку половинами — тогда работает слитое split-ядро (q·k от перестановки не меняется).

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
use synaptix_ops::pos::rope::{apply_rope_with_cossin, RopeLayout};

use crate::{F5Error, Result, N_MELS};

pub struct Cfg {
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub text_dim: usize,
    pub conv_layers: usize,
}

impl Default for Cfg {
    fn default() -> Self {
        Self { dim: 1024, depth: 22, heads: 16, text_dim: 512, conv_layers: 4 }
    }
}

pub struct W<'a> {
    pub l: &'a dyn WeightLoader,
    pub prefix: &'a str,
    pub device: Device,
    pub dtype: DType,
    /// Квант линейных слоёв блоков на лету (MXFP8/NVFP4); `None` — плотные.
    pub quant: Option<DType>,
}

impl W<'_> {
    fn name(&self, n: &str) -> String {
        format!("{}{n}", self.prefix)
    }

    pub fn get(&self, n: &str) -> Result<Tensor> {
        let name = self.name(n);
        self.l.load_to(&name, self.device, self.dtype).map_err(|e| F5Error::Load(format!("{name}: {e}")))
    }

    pub fn get_as(&self, n: &str, dev: Device, dt: DType) -> Result<Tensor> {
        let name = self.name(n);
        self.l.load_to(&name, dev, dt).map_err(|e| F5Error::Load(format!("{name}: {e}")))
    }

    fn lin(&self, p: &str, quant: bool) -> Result<QuantLinear> {
        let name = self.name(&format!("{p}.weight"));
        let b = Some(self.get(&format!("{p}.bias"))?);
        if let Some(q) = self.l.prequant(&name, self.device) {
            return Ok(QuantLinear::Quant { w: q.map_err(|e| F5Error::Load(format!("{name}: {e}")))?, bias: b });
        }
        let w = self.get(&format!("{p}.weight"))?;
        match self.quant.filter(|_| quant && self.device.is_cuda()) {
            Some(q) => Ok(QuantLinear::build(w, b, q, self.dtype)?),
            None => Ok(QuantLinear::dense(w, b)?),
        }
    }

    /// Линейный слой с перестановкой строк (и смещения) по `perm` — до кванта.
    fn lin_perm(&self, p: &str, perm: &[i64], quant: bool) -> Result<QuantLinear> {
        let idx = Tensor::from_vec(perm.to_vec(), vec![perm.len()], self.device)?;
        let w = self.get(&format!("{p}.weight"))?.index_select(0, &idx)?.contiguous()?;
        let b = self.get(&format!("{p}.bias"))?.index_select(0, &idx)?.contiguous()?;
        match self.quant.filter(|_| quant && self.device.is_cuda()) {
            Some(q) => Ok(QuantLinear::build(w, Some(b), q, self.dtype)?),
            None => Ok(QuantLinear::dense(w, Some(b))?),
        }
    }
}

fn t12(x: &Tensor) -> std::result::Result<Tensor, SynaptixError> {
    x.transpose(1, 2)?.contiguous()
}

fn mish(x: &Tensor) -> std::result::Result<Tensor, SynaptixError> {
    // x·tanh(softplus(x)), softplus = ln(1 + eˣ) (при x > 20 — x)
    let sp = x.exp()?.affine(1.0, 1.0)?.log()?;
    x.mul(&sp.tanh()?)
}

struct ConvNeXtV2 {
    dw: (Tensor, Tensor),
    ln: (Tensor, Tensor),
    pw1: QuantLinear,
    grn_g: Tensor,
    grn_b: Tensor,
    pw2: QuantLinear,
}

impl ConvNeXtV2 {
    fn load(w: &W, p: &str) -> Result<Self> {
        let n = |s: &str| format!("{p}.{s}");
        let g = w.get(&n("grn.gamma"))?;
        let c = g.dims()[2];
        Ok(Self {
            dw: (w.get(&n("dwconv.weight"))?, w.get(&n("dwconv.bias"))?),
            ln: (w.get(&n("norm.weight"))?, w.get(&n("norm.bias"))?),
            pw1: w.lin(&n("pwconv1"), false)?,
            grn_g: g.reshape(vec![1usize, 1, c])?,
            grn_b: w.get(&n("grn.beta"))?.reshape(vec![1usize, 1, c])?,
            pw2: w.lin(&n("pwconv2"), false)?,
        })
    }

    /// `x` `[B, N, C]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let c = x.dims()[2];
        let y = t12(&depthwise_conv(&t12(x)?, &self.dw.0, Some(&self.dw.1), 1, 3, c)?)?;
        let y = layer_norm(&y, Some(&self.ln.0), Some(&self.ln.1), 1e-6)?;
        let y = self.pw1.forward(&y)?.gelu_exact()?;
        // GRN: ‖y‖₂ по времени, нормировка средним по каналам
        let gx = y.to_dtype(DType::F32)?.sqr()?.sum_keepdim(1)?.sqrt()?; // [B, 1, D]
        let d = gx.dims()[2];
        let mean = gx.sum_keepdim(2)?.affine(1.0 / d as f32, 1e-6)?;
        let nx = gx.broadcast_div(&mean)?.to_dtype(y.dtype())?;
        let y = y.broadcast_mul(&nx)?.broadcast_mul(&self.grn_g)?.broadcast_add(&self.grn_b)?.add(&y)?;
        Ok(x.add(&self.pw2.forward(&y)?)?)
    }
}

struct Block {
    ada: QuantLinear,
    q: QuantLinear,
    k: QuantLinear,
    v: QuantLinear,
    o: QuantLinear,
    ff1: QuantLinear,
    ff2: QuantLinear,
}

pub struct Dit {
    cfg: Cfg,
    time_l0: QuantLinear,
    time_l1: QuantLinear,
    /// Таблица текстовых эмбеддингов на хосте.
    text_table: Vec<f32>,
    text_blocks: Vec<ConvNeXtV2>,
    proj: QuantLinear,
    pos_convs: Vec<(Tensor, Tensor)>,
    blocks: Vec<Block>,
    norm_out: QuantLinear,
    proj_out: QuantLinear,
    inv_freq: Vec<f32>,
    pub device: Device,
    pub dtype: DType,
}

/// Текстовое и временное состояние на весь синтез (кэш, как `text_cond`/`text_uncond` upstream).
pub struct Prepared {
    /// `[2, N, text_dim]`: условный и безусловный.
    text: Tensor,
    cos: Tensor,
    sin: Tensor,
    n: usize,
}

impl Dit {
    pub fn load(w: &W, cfg: Cfg) -> Result<Self> {
        let hd = cfg.dim / cfg.heads;
        let mut perm = Vec::with_capacity(cfg.dim);
        for h in 0..cfg.heads {
            perm.extend((0..hd / 2).map(|j| (h * hd + 2 * j) as i64));
            perm.extend((0..hd / 2).map(|j| (h * hd + 2 * j + 1) as i64));
        }
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let p = |s: &str| format!("transformer_blocks.{i}.{s}");
            blocks.push(Block {
                ada: w.lin(&p("attn_norm.linear"), true)?,
                q: w.lin_perm(&p("attn.to_q"), &perm, true)?,
                k: w.lin_perm(&p("attn.to_k"), &perm, true)?,
                v: w.lin(&p("attn.to_v"), true)?,
                o: w.lin(&p("attn.to_out.0"), true)?,
                ff1: w.lin(&p("ff.ff.0.0"), true)?,
                ff2: w.lin(&p("ff.ff.2"), true)?,
            });
        }
        let text_table = w.get_as("text_embed.text_embed.weight", Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let text_blocks = (0..cfg.conv_layers).map(|i| ConvNeXtV2::load(w, &format!("text_embed.text_blocks.{i}"))).collect::<Result<Vec<_>>>()?;
        let pos_convs = [0, 2]
            .iter()
            .map(|i| Ok((w.get(&format!("input_embed.conv_pos_embed.conv1d.{i}.weight"))?, w.get(&format!("input_embed.conv_pos_embed.conv1d.{i}.bias"))?)))
            .collect::<Result<Vec<_>>>()?;
        let inv_freq = w.get_as("rotary_embed.inv_freq", Device::Cpu, DType::F32)?.to_vec1::<f32>()?;
        Ok(Self {
            time_l0: w.lin("time_embed.time_mlp.0", false)?,
            time_l1: w.lin("time_embed.time_mlp.2", false)?,
            text_table,
            text_blocks,
            proj: w.lin("input_embed.proj", false)?,
            pos_convs,
            blocks,
            norm_out: w.lin("norm_out.linear", false)?,
            proj_out: w.lin("proj_out", false)?,
            inv_freq,
            device: w.device,
            dtype: w.dtype,
            cfg,
        })
    }

    fn up(&self, v: Vec<f32>, dims: Vec<usize>) -> Result<Tensor> {
        Ok(Tensor::from_vec(v, dims, Device::Cpu)?.to_dtype(self.dtype)?.to_device(self.device)?)
    }

    /// Текстовые вложения (условное и безусловное) и RoPE для `n` кадров; `ids` — индексы словаря.
    pub fn prepare(&self, ids: &[i64], n: usize) -> Result<Prepared> {
        let td = self.cfg.text_dim;
        // text + 1 (0 — заполнитель), обрезка/добивка до n
        let toks: Vec<i64> = (0..n).map(|i| ids.get(i).map_or(0, |&t| t + 1)).collect();
        let half = td / 2;
        let freq: Vec<f32> = (0..half).map(|j| 1.0 / 10000f32.powf((2 * j) as f32 / td as f32)).collect();
        let mut rows = Vec::with_capacity(2);
        let mask: Vec<f32> = toks.iter().map(|&t| if t == 0 { 0.0 } else { 1.0 }).collect();
        for drop in [false, true] {
            let mut e = vec![0f32; n * td];
            for (i, &t) in toks.iter().enumerate() {
                let r = if drop { 0 } else { t as usize };
                e[i * td..(i + 1) * td].copy_from_slice(&self.text_table[r * td..(r + 1) * td]);
                for j in 0..half {
                    let a = i as f32 * freq[j];
                    e[i * td + j] += a.cos();
                    e[i * td + half + j] += a.sin();
                }
                if mask[i] == 0.0 {
                    e[i * td..(i + 1) * td].iter_mut().for_each(|v| *v = 0.0);
                }
            }
            rows.push(e);
        }
        let flat: Vec<f32> = rows.concat();
        let mut x = self.up(flat, vec![2usize, n, td])?;
        let m = self.up([mask.clone(), mask].concat(), vec![2usize, n, 1])?;
        for b in &self.text_blocks {
            x = b.forward(&x)?.broadcast_mul(&m)?;
        }
        // RoPE x_transformers: углы pos·inv_freq
        let h = self.inv_freq.len();
        let (mut c, mut s) = (vec![0f32; n * h], vec![0f32; n * h]);
        for p in 0..n {
            for k in 0..h {
                let a = p as f32 * self.inv_freq[k];
                c[p * h + k] = a.cos();
                s[p * h + k] = a.sin();
            }
        }
        Ok(Prepared {
            text: x,
            cos: Tensor::from_vec(c, vec![n, h], self.device)?,
            sin: Tensor::from_vec(s, vec![n, h], self.device)?,
            n,
        })
    }

    /// Поток для CFG: `x` `[1, N, 100]`, `cond` `[1, N, 100]`, время `t` → (условный, безусловный) `[1, N, 100]`.
    pub fn forward(&self, x: &Tensor, cond: &Tensor, pre: &Prepared, t: f32) -> Result<(Tensor, Tensor)> {
        let n = pre.n;
        let dim = self.cfg.dim;
        // время: синусоиды 256 (масштаб 1000) → MLP
        let half = 128;
        let emb_f = (10000f32).ln() / (half - 1) as f32;
        let mut te: Vec<f32> = (0..half).map(|i| (1000.0 * t * (-(i as f32) * emb_f).exp()).sin()).collect();
        te.extend((0..half).map(|i| (1000.0 * t * (-(i as f32) * emb_f).exp()).cos()));
        let temb = self.time_l1.forward(&self.time_l0.forward(&self.up(te, vec![1usize, 256])?)?.silu()?)?; // [1, dim]
        let st = temb.silu()?;
        // вход: [x ‖ cond ‖ текст] → proj; безусловный — без образца
        let zeros = Tensor::zeros(cond.dims().to_vec(), cond.dtype(), cond.device())?;
        let xb = Tensor::cat(&[x, x], 0)?;
        let cb = Tensor::cat(&[cond, &zeros], 0)?;
        let inp = Tensor::cat(&[&xb.to_dtype(self.dtype)?, &cb.to_dtype(self.dtype)?, &pre.text], 2)?;
        let mut h = self.proj.forward(&inp)?; // [2, N, dim]
        // свёрточное позиционное: групповая k31 (16 групп), Mish ×2
        let mut y = t12(&h)?;
        for (cw, cb) in &self.pos_convs {
            let groups = 16;
            let gsz = dim / groups;
            let mut parts = Vec::with_capacity(groups);
            for g in 0..groups {
                let xi = y.narrow(1, g * gsz, gsz)?.contiguous()?;
                let wi = cw.narrow(0, g * gsz, gsz)?.contiguous()?;
                let bi = cb.narrow(0, g * gsz, gsz)?.contiguous()?;
                parts.push(conv1d(&xi, &wi, Some(&bi), 1, 15)?);
            }
            let r: Vec<&Tensor> = parts.iter().collect();
            y = mish(&Tensor::cat(&r, 1)?)?;
        }
        h = h.add(&t12(&y)?)?;
        let heads = self.cfg.heads;
        let hd = dim / heads;
        let split = |z: Tensor| -> Result<Tensor> { Ok(z.reshape(vec![2usize, n, heads, hd])?.permute(&[0, 2, 1, 3])?.contiguous()?) };
        for b in &self.blocks {
            let m = b.ada.forward(&st)?; // [1, 6·dim]
            let ch = |i: usize| -> Result<Tensor> { Ok(m.narrow(1, i * dim, dim)?.contiguous()?.reshape(vec![1usize, 1, dim])?) };
            let (sh_a, sc_a, g_a, sh_f, sc_f, g_f) = (ch(0)?, ch(1)?, ch(2)?, ch(3)?, ch(4)?, ch(5)?);
            let nrm = layer_norm(&h, None, None, 1e-6)?.broadcast_mul(&sc_a.affine(1.0, 1.0)?)?.broadcast_add(&sh_a)?;
            let q = apply_rope_with_cossin(&split(b.q.forward(&nrm)?)?, &pre.cos, &pre.sin, RopeLayout::Split)?;
            let k = apply_rope_with_cossin(&split(b.k.forward(&nrm)?)?, &pre.cos, &pre.sin, RopeLayout::Split)?;
            let v = split(b.v.forward(&nrm)?)?;
            let scale = 1.0 / (hd as f32).sqrt();
            let a = match q.flash_attention(&k, &v, scale, false) {
                Ok(o) => o,
                Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => scaled_dot_attention(&q, &k, &v, scale, None)?,
                Err(e) => return Err(e.into()),
            };
            let a = b.o.forward(&a.permute(&[0, 2, 1, 3])?.contiguous()?.reshape(vec![2usize, n, dim])?)?;
            h = h.add(&a.broadcast_mul(&g_a)?)?;
            let nrm = layer_norm(&h, None, None, 1e-6)?.broadcast_mul(&sc_f.affine(1.0, 1.0)?)?.broadcast_add(&sh_f)?;
            let f = b.ff2.forward(&b.ff1.forward(&nrm)?.gelu_tanh()?)?;
            h = h.add(&f.broadcast_mul(&g_f)?)?;
        }
        let m = self.norm_out.forward(&st)?;
        let scale = m.narrow(1, 0, dim)?.contiguous()?.reshape(vec![1usize, 1, dim])?;
        let shift = m.narrow(1, dim, dim)?.contiguous()?.reshape(vec![1usize, 1, dim])?;
        let h = layer_norm(&h, None, None, 1e-6)?.broadcast_mul(&scale.affine(1.0, 1.0)?)?.broadcast_add(&shift)?;
        let out = self.proj_out.forward(&h)?.to_dtype(DType::F32)?; // [2, N, 100]
        let _ = N_MELS;
        Ok((out.narrow(0, 0, 1)?.contiguous()?, out.narrow(0, 1, 1)?.contiguous()?))
    }
}
