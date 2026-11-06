//! Кодек Audio8 (`modeling_arktts_codec.py`, DAC-подобный, 44.1 кГц, 2048 семплов на кадр).
//!
//! Декод: коды `[10, T]` → RVQ (семантический кодбук 4096×8 + 9 остаточных 1024×8,
//! `out_proj` — weight-norm 1×1) → оконный трансформер (8 слоёв, окно 128) → 2× (ConvT
//! k2 s2 + ConvNeXt) → свёрточный декодер (1024→1536, ConvT 8·8·4·2, snake, residual
//! units с дилатацией 1/3/9) → tanh.
//! Энкод (образец голоса): свёртки 64→1024 (шаги 2·4·8·8, в последнем блоке —
//! трансформер 4 слоя с окном 512) → 2× (conv k2 s2 + ConvNeXt) → трансформер 8 слоёв
//! (окно 128) → RVQ по косинусной близости.
//! Все свёртки каузальные: слева `k_eff − stride`, справа добивка до целого кадра.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_nn::Module;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::{conv1d_dilated, conv_transpose1d, depthwise_conv};
use synaptix_ops::norm::layer_norm::layer_norm;
use synaptix_ops::norm::rms_norm::rms_norm;
use synaptix_ops::pos::rope::{apply_rope_with_cossin, RopeLayout};

use crate::dualar::rope_tables;
use crate::loader::Checkpoint;
use crate::{Audio8Error, Result, FRAME_SAMPLES};

/// weight-norm: w = g · v / ‖v‖ (норма по всем осям, кроме 0). Имена — новые
/// параметризации (`parametrizations.weight.original0/1`) или старые (`weight_g/v`).
fn wn_weight(ck: &Checkpoint, prefix: &str) -> Result<Tensor> {
    let (g, v) = if ck.codec_has(&format!("{prefix}.parametrizations.weight.original0")) {
        (
            ck.codec_f32(&format!("{prefix}.parametrizations.weight.original0"))?,
            ck.codec_f32(&format!("{prefix}.parametrizations.weight.original1"))?,
        )
    } else {
        (ck.codec_f32(&format!("{prefix}.weight_g"))?, ck.codec_f32(&format!("{prefix}.weight_v"))?)
    };
    let d = v.dims().to_vec();
    let rest: usize = d[1..].iter().product();
    let norm = v.reshape(vec![d[0], rest])?.sqr()?.sum_keepdim(1)?.sqrt()?;
    let mut shape = vec![d[0]];
    shape.extend(std::iter::repeat_n(1usize, d.len() - 1));
    let w = v.broadcast_mul(&g.reshape(shape.clone())?.div(&norm.reshape(shape)?)?)?;
    Ok(w.to_dtype(ck.codec_dtype)?)
}

fn pad_time(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let d = x.dims();
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if left > 0 {
        parts.push(Tensor::zeros(vec![d[0], d[1], left], x.dtype(), x.device())?);
    }
    parts.push(x.clone());
    if right > 0 {
        parts.push(Tensor::zeros(vec![d[0], d[1], right], x.dtype(), x.device())?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&refs, 2)?)
}

/// Каузальная свёртка `ArkttsCausalConv1d`.
struct CConv {
    w: Tensor,
    b: Option<Tensor>,
    stride: usize,
    dilation: usize,
    k_eff: usize,
    depthwise: bool,
}

impl CConv {
    fn load(ck: &Checkpoint, prefix: &str, wn: bool, stride: usize, dilation: usize) -> Result<Self> {
        Self::load_g(ck, prefix, wn, stride, dilation, false)
    }

    fn load_g(ck: &Checkpoint, prefix: &str, wn: bool, stride: usize, dilation: usize, groups_depthwise: bool) -> Result<Self> {
        let w = if wn { wn_weight(ck, prefix)? } else { ck.codec(&format!("{prefix}.weight"))? };
        let b = Some(ck.codec(&format!("{prefix}.bias"))?);
        let k = w.dims()[2];
        // depthwise (groups = C): вес `[C, 1, k]` при C входных каналах; вход
        // с одним каналом (`[64, 1, 7]` у первой свёртки энкодера) — обычная свёртка.
        let depthwise = groups_depthwise;
        Ok(Self { w, b, stride, dilation, k_eff: (k - 1) * dilation + 1, depthwise })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let len = x.dims()[2];
        let pad = self.k_eff - self.stride;
        // `_extra_padding`: добить справа до целого числа шагов.
        let frames = (len as f64 - self.k_eff as f64 + pad as f64) / self.stride as f64 + 1.0;
        let ideal = (frames.ceil() as i64 - 1) * self.stride as i64 + self.k_eff as i64 - pad as i64;
        let right = (ideal - len as i64).max(0) as usize;
        let xp = pad_time(x, pad, right)?;
        if self.depthwise {
            let c = xp.dims()[1];
            Ok(depthwise_conv(&xp, &self.w, self.b.as_ref(), self.stride, 0, c)?)
        } else {
            Ok(conv1d_dilated(&xp, &self.w, self.b.as_ref(), self.stride, 0, self.dilation)?)
        }
    }
}

/// Каузальная транспонированная свёртка: справа отрезается `k − stride`.
struct CConvT {
    w: Tensor,
    b: Option<Tensor>,
    stride: usize,
    crop: usize,
}

impl CConvT {
    fn load(ck: &Checkpoint, prefix: &str, wn: bool, stride: usize) -> Result<Self> {
        let w = if wn { wn_weight(ck, prefix)? } else { ck.codec(&format!("{prefix}.weight"))? };
        let b = Some(ck.codec(&format!("{prefix}.bias"))?);
        let k = w.dims()[2];
        Ok(Self { w, b, stride, crop: k - stride })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = conv_transpose1d(x, &self.w, self.b.as_ref(), self.stride, 0, 0, 1, 1)?;
        if self.crop == 0 {
            return Ok(y);
        }
        let len = y.dims()[2];
        Ok(y.narrow(2, 0, len - self.crop)?.contiguous()?)
    }
}

/// Snake: x + sin²(αx)/(α + 1e-9). При α > 0 — слитое ядро (`ln α` в обе
/// экспоненты), иначе по шагам.
struct Snake {
    alpha: Tensor,
    ln_alpha: Option<Tensor>,
}

impl Snake {
    fn load(ck: &Checkpoint, name: &str) -> Result<Self> {
        let a = ck.codec_f32(name)?;
        let host = a.flatten_all()?.to_vec1::<f32>()?;
        let positive = host.iter().all(|&v| v > 0.0) && std::env::var("SYN_AUDIO8_SNAKE").as_deref() != Ok("plain");
        let ln_alpha = if positive { Some(a.flatten_all()?.log()?) } else { None };
        Ok(Self { alpha: a.to_dtype(ck.codec_dtype)?, ln_alpha })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(la) = &self.ln_alpha {
            match x.snake(la, la, 1e-9) {
                Ok(y) => return Ok(y),
                Err(synaptix_core::error::SynaptixError::Unsupported(_)) => {}
                Err(e) => return Err(e.into()),
            }
        }
        let ax = x.broadcast_mul(&self.alpha)?;
        let s = ax.sin()?.sqr()?;
        let inv = self.alpha.affine(1.0, 1e-9)?.recip()?;
        Ok(x.add(&s.broadcast_mul(&inv)?)?)
    }
}

struct ResUnit {
    s1: Snake,
    c1: CConv,
    s2: Snake,
    c2: CConv,
}

impl ResUnit {
    fn load(ck: &Checkpoint, p: &str, dilation: usize) -> Result<Self> {
        Ok(Self {
            s1: Snake::load(ck, &format!("{p}.block.0.alpha"))?,
            c1: CConv::load(ck, &format!("{p}.block.1.conv"), true, 1, dilation)?,
            s2: Snake::load(ck, &format!("{p}.block.2.alpha"))?,
            c2: CConv::load(ck, &format!("{p}.block.3.conv"), true, 1, 1)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.c2.forward(&self.s2.forward(&self.c1.forward(&self.s1.forward(x)?)?)?)?;
        let (lx, ly) = (x.dims()[2], y.dims()[2]);
        let x = if lx > ly { x.narrow(2, 0, ly)?.contiguous()? } else { x.clone() };
        Ok(x.add(&y)?)
    }
}

fn lin(ck: &Checkpoint, p: &str, bias: bool) -> Result<QuantLinear> {
    let w = ck.codec(&format!("{p}.weight"))?;
    let b = if bias { Some(ck.codec(&format!("{p}.bias"))?) } else { None };
    Ok(QuantLinear::dense(w, b)?)
}

struct TBlock {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    wqkv: QuantLinear,
    wo: QuantLinear,
    w1: QuantLinear,
    w3: QuantLinear,
    w2: QuantLinear,
    attn_scale: Tensor,
    ffn_scale: Tensor,
}

/// `ArkttsCodecWindowTransformer` (вход/выход каналами вперёд `[1, C, T]`).
struct WindowTransformer {
    blocks: Vec<TBlock>,
    norm: Tensor,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    window: usize,
    dim: usize,
    device: Device,
}

impl WindowTransformer {
    fn load(ck: &Checkpoint, p: &str, n_layer: usize, n_head: usize, n_kv: usize, window: usize) -> Result<Self> {
        let hd = 64usize;
        let mut perm = Vec::with_capacity((n_head + 2 * n_kv) * hd);
        for h in 0..n_head + n_kv {
            perm.extend((0..hd / 2).map(|j| (h * hd + 2 * j) as i64));
            perm.extend((0..hd / 2).map(|j| (h * hd + 2 * j + 1) as i64));
        }
        let v0 = (n_head + n_kv) * hd;
        perm.extend((v0..v0 + n_kv * hd).map(|i| i as i64));
        let mut blocks = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let q = |s: &str| format!("{p}.layers.{i}.{s}");
            let w = ck.codec(&q("attention.wqkv.weight"))?;
            let idx = Tensor::from_vec(perm.clone(), vec![perm.len()], w.device())?;
            let w = w.index_select(0, &idx)?.contiguous()?;
            blocks.push(TBlock {
                attn_norm: ck.codec(&q("attention_norm.weight"))?,
                ffn_norm: ck.codec(&q("ffn_norm.weight"))?,
                wqkv: QuantLinear::dense(w, None)?,
                wo: lin(ck, &q("attention.wo"), false)?,
                w1: lin(ck, &q("feed_forward.w1"), false)?,
                w3: lin(ck, &q("feed_forward.w3"), false)?,
                w2: lin(ck, &q("feed_forward.w2"), false)?,
                attn_scale: ck.codec(&q("attention_layer_scale.gamma"))?,
                ffn_scale: ck.codec(&q("ffn_layer_scale.gamma"))?,
            });
        }
        let norm = ck.codec(&format!("{p}.norm.weight"))?;
        let dim = norm.dims()[0];
        Ok(Self { blocks, norm, n_head, n_kv, hd, window, dim, device: ck.device })
    }

    fn heads(&self, t: &Tensor, off: usize, n: usize, s: usize) -> Result<Tensor> {
        Ok(t.narrow(2, off * self.hd, n * self.hd)?
            .contiguous()?
            .reshape(vec![1usize, s, n, self.hd])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dims()[2];
        let mut h = x.transpose(1, 2)?.contiguous()?;
        let (cos, sin) = rope_tables(t, self.hd, 10_000.0, self.device)?;
        let mut mask = vec![0f32; t * t];
        for r in 0..t {
            let lo = (r + 1).saturating_sub(self.window);
            for c in 0..t {
                if c > r || c < lo {
                    mask[r * t + c] = -1.0e30;
                }
            }
        }
        let mask = Tensor::from_vec(mask, vec![1usize, 1, t, t], self.device)?.to_dtype(x.dtype())?;
        let scale = 1.0 / (self.hd as f32).sqrt();
        let g = self.n_head / self.n_kv;
        for b in &self.blocks {
            let a = rms_norm(&h, &b.attn_norm, 1e-5)?;
            let qkv = b.wqkv.forward(&a)?;
            let q = apply_rope_with_cossin(&self.heads(&qkv, 0, self.n_head, t)?, &cos, &sin, RopeLayout::Split)?;
            let k = apply_rope_with_cossin(&self.heads(&qkv, self.n_head, self.n_kv, t)?, &cos, &sin, RopeLayout::Split)?;
            let v = self.heads(&qkv, self.n_head + self.n_kv, self.n_kv, t)?;
            let (k, v) = if g > 1 {
                let rep = |x: &Tensor| -> Result<Tensor> {
                    Ok(x.unsqueeze(2)?
                        .broadcast_as(vec![1usize, self.n_kv, g, t, self.hd])?
                        .contiguous()?
                        .reshape(vec![1usize, self.n_head, t, self.hd])?)
                };
                (rep(&k)?, rep(&v)?)
            } else {
                (k, v)
            };
            let o = scaled_dot_attention(&q, &k, &v, scale, Some(&mask))?;
            let o = o.permute(&[0, 2, 1, 3])?.contiguous()?.reshape(vec![1usize, t, self.n_head * self.hd])?;
            let o = b.wo.forward(&o)?.broadcast_mul(&b.attn_scale)?;
            h = h.add(&o)?;
            let f = rms_norm(&h, &b.ffn_norm, 1e-5)?;
            let gte = b.w1.forward(&f)?;
            let up = b.w3.forward(&f)?;
            let act = match gte.silu_and_mul(&up) {
                Ok(t) => t,
                Err(_) => gte.silu()?.mul(&up)?,
            };
            let f = b.w2.forward(&act)?.broadcast_mul(&b.ffn_scale)?;
            h = h.add(&f)?;
        }
        let h = rms_norm(&h, &self.norm, 1e-5)?;
        let _ = self.dim;
        Ok(h.transpose(1, 2)?.contiguous()?)
    }
}

struct ConvNeXt {
    dw: CConv,
    norm_w: Tensor,
    norm_b: Tensor,
    pw1: QuantLinear,
    pw2: QuantLinear,
    gamma: Tensor,
}

impl ConvNeXt {
    fn load(ck: &Checkpoint, p: &str) -> Result<Self> {
        Ok(Self {
            dw: CConv::load_g(ck, &format!("{p}.dwconv.conv"), false, 1, 1, true)?,
            norm_w: ck.codec(&format!("{p}.norm.weight"))?,
            norm_b: ck.codec(&format!("{p}.norm.bias"))?,
            pw1: lin(ck, &format!("{p}.pwconv1"), true)?,
            pw2: lin(ck, &format!("{p}.pwconv2"), true)?,
            gamma: ck.codec(&format!("{p}.gamma"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.dw.forward(x)?.transpose(1, 2)?.contiguous()?;
        let y = layer_norm(&y, Some(&self.norm_w), Some(&self.norm_b), 1e-6)?;
        let y = self.pw2.forward(&self.pw1.forward(&y)?.gelu_exact()?)?;
        let y = y.broadcast_mul(&self.gamma)?.transpose(1, 2)?.contiguous()?;
        Ok(x.add(&y)?)
    }
}

struct Vq {
    /// Кодбук `[n, 8]` (F32).
    codebook: Tensor,
    /// Нормированный кодбук (для поиска).
    codebook_n: Tensor,
    /// ‖ĉ‖² строк нормированного кодбука `[1, n]`.
    cn2: Tensor,
    in_w: Option<QuantLinear>,
    out: QuantLinear,
    size: usize,
}

impl Vq {
    fn load(ck: &Checkpoint, p: &str, with_in: bool) -> Result<Self> {
        let codebook = ck.codec_f32(&format!("{p}.codebook.weight"))?;
        let size = codebook.dims()[0];
        let flat = |w: Tensor| -> Result<Tensor> {
            let d = w.dims().to_vec();
            Ok(w.reshape(vec![d[0], d[1]])?)
        };
        let out_w = flat(wn_weight(ck, &format!("{p}.out_proj"))?)?;
        let out = QuantLinear::dense(out_w, Some(ck.codec(&format!("{p}.out_proj.bias"))?))?;
        let (codebook_n, cn2, in_w) = if with_in {
            let n = codebook.sqr()?.sum_keepdim(1)?.sqrt()?;
            let n = n.affine(1.0, 1e-12)?;
            let cbn = codebook.broadcast_div(&n)?;
            let cn2 = cbn.sqr()?.sum_keepdim(1)?.reshape(vec![1usize, size])?;
            let in_w = flat(wn_weight(ck, &format!("{p}.in_proj"))?.to_dtype(DType::F32)?)?;
            let in_b = ck.codec_f32(&format!("{p}.in_proj.bias"))?;
            (cbn, cn2, Some(QuantLinear::dense(in_w, Some(in_b))?))
        } else {
            (codebook.clone(), codebook.clone(), None)
        };
        Ok(Self { codebook, codebook_n, cn2, in_w, out, size })
    }

    /// Коды `[T]` → вклад `[1, T, 1024]` (`out_proj(codebook[c])`).
    fn decode(&self, codes: &[u32], dtype: DType) -> Result<Tensor> {
        let t = codes.len();
        let idx: Vec<i64> = codes.iter().map(|&c| (c as usize).min(self.size - 1) as i64).collect();
        let idx = Tensor::from_vec(idx, vec![t], self.codebook.device())?;
        let e = self.codebook.index_select(0, &idx)?.reshape(vec![1usize, t, self.codebook.dims()[1]])?.to_dtype(dtype)?;
        Ok(self.out.forward(&e)?)
    }

    /// Квантование `[1, T, 1024]`: (коды, вклад `[1, T, 1024]`).
    fn encode(&self, z: &Tensor) -> Result<(Vec<u32>, Tensor)> {
        let in_w = self.in_w.as_ref().ok_or_else(|| Audio8Error::Other("VQ без in_proj".into()))?;
        let t = z.dims()[1];
        let p = in_w.forward(&z.to_dtype(DType::F32)?)?.reshape(vec![t, self.codebook.dims()[1]])?;
        let n = p.sqr()?.sum_keepdim(1)?.sqrt()?.affine(1.0, 1e-12)?;
        let pn = p.broadcast_div(&n)?;
        // argmax(−dist) = argmax(2·p̂·ĉ − ‖ĉ‖²) (‖p̂‖² у строки общий).
        let score = pn.matmul(&self.codebook_n.transpose(0, 1)?.contiguous()?)?.affine(2.0, 0.0)?.broadcast_sub(&self.cn2)?;
        let host = score.to_vec2::<f32>()?;
        let codes: Vec<u32> = host.iter().map(|r| crate::sample::argmax(r) as u32).collect();
        let q = self.decode(&codes, z.dtype())?;
        Ok((codes, q))
    }
}

/// Отладка: `SYN_AUDIO8_DUMP=каталог` — промежуточные тензоры энкода (f32, сырые).
fn dump(name: &str, x: &Tensor) {
    if let Ok(dir) = std::env::var("SYN_AUDIO8_DUMP") {
        if let Ok(v) = x.to_dtype(DType::F32).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()) {
            let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/{name}.f32"), bytes);
        }
    }
}

pub struct Codec {
    // декод
    sem_vq: Vq,
    res_vq: Vec<Vq>,
    post: WindowTransformer,
    up: Vec<(CConvT, ConvNeXt)>,
    dec_in: CConv,
    dec_blocks: Vec<(Snake, CConvT, Vec<ResUnit>)>,
    dec_snake: Snake,
    dec_out: CConv,
    // энкод (образец голоса): грузится при первом кодировании — ~0.8 ГБ F32,
    // синтезу без образца не нужен.
    enc: std::sync::OnceLock<Encoder>,
    enc_lock: std::sync::Mutex<()>,
    dtype: DType,
    device: Device,
}

pub struct Encoder {
    first: CConv,
    blocks: Vec<(Vec<ResUnit>, Snake, CConv, Option<WindowTransformer>)>,
    snake: Snake,
    last: CConv,
    down: Vec<(CConv, ConvNeXt)>,
    pre: WindowTransformer,
}

impl Encoder {
    fn load(ck: &Checkpoint) -> Result<Self> {
        let first = CConv::load(ck, "encoder.block.0.conv", true, 1, 1)?;
        let mut blocks = Vec::new();
        for (i, (&stride, &tl)) in [2usize, 4, 8, 8].iter().zip([0usize, 0, 0, 4].iter()).enumerate() {
            let p = format!("encoder.block.{}", i + 1);
            let res = [1usize, 3, 9]
                .iter()
                .enumerate()
                .map(|(j, &d)| ResUnit::load(ck, &format!("{p}.block.{j}"), d))
                .collect::<Result<Vec<_>>>()?;
            let snake = Snake::load(ck, &format!("{p}.block.3.alpha"))?;
            let conv = CConv::load(ck, &format!("{p}.block.4.conv"), true, stride, 1)?;
            let tr = if tl > 0 {
                let dim = conv.w.dims()[0];
                Some(WindowTransformer::load(ck, &format!("{p}.block.5"), tl, dim / 64, dim / 64, 512)?)
            } else {
                None
            };
            blocks.push((res, snake, conv, tr));
        }
        let snake = Snake::load(ck, "encoder.block.5.alpha")?;
        let last = CConv::load(ck, "encoder.block.6.conv", true, 1, 1)?;
        let down = (0..2)
            .map(|i| {
                Ok((
                    CConv::load(ck, &format!("quantizer.downsample.{i}.0.conv"), false, 2, 1)?,
                    ConvNeXt::load(ck, &format!("quantizer.downsample.{i}.1"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let pre = WindowTransformer::load(ck, "quantizer.pre_module", 8, 16, 16, 128)?;
        Ok(Self { first, blocks, snake, last, down, pre })
    }
}

impl Codec {
    pub fn load(ck: &Checkpoint, with_encoder: bool) -> Result<Self> {
        let c = &ck.config;
        let sem_vq = Vq::load(ck, "quantizer.semantic_quantizer.quantizers.0", true)?;
        let res_vq = (0..c.num_codebooks - 1)
            .map(|i| Vq::load(ck, &format!("quantizer.quantizer.quantizers.{i}"), true))
            .collect::<Result<Vec<_>>>()?;
        let post = WindowTransformer::load(ck, "quantizer.post_module", c.codec_post_n_layer, c.codec_post_n_head, c.codec_post_n_local_heads, 128)?;
        let up = (0..2)
            .map(|i| {
                Ok((
                    CConvT::load(ck, &format!("quantizer.upsample.{i}.0.conv"), false, 2)?,
                    ConvNeXt::load(ck, &format!("quantizer.upsample.{i}.1"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let dec_in = CConv::load(ck, "decoder.model.0.conv", true, 1, 1)?;
        let mut dec_blocks = Vec::new();
        for (i, &stride) in [8usize, 8, 4, 2].iter().enumerate() {
            let p = format!("decoder.model.{}", i + 1);
            let snake = Snake::load(ck, &format!("{p}.block.0.alpha"))?;
            let ct = CConvT::load(ck, &format!("{p}.block.1.conv"), true, stride)?;
            let res = [1usize, 3, 9]
                .iter()
                .enumerate()
                .map(|(j, &d)| ResUnit::load(ck, &format!("{p}.block.{}", j + 2), d))
                .collect::<Result<Vec<_>>>()?;
            dec_blocks.push((snake, ct, res));
        }
        let dec_snake = Snake::load(ck, "decoder.model.5.alpha")?;
        let dec_out = CConv::load(ck, "decoder.model.6.conv", true, 1, 1)?;
        let enc = std::sync::OnceLock::new();
        if with_encoder {
            let _ = enc.set(Encoder::load(ck)?);
        }
        Ok(Self { sem_vq, res_vq, post, up, dec_in, dec_blocks, dec_snake, dec_out, enc, enc_lock: Default::default(), dtype: ck.codec_dtype, device: ck.device })
    }

    pub fn has_encoder(&self) -> bool {
        self.enc.get().is_some()
    }

    /// Загрузить энкодер, если ещё не загружен.
    pub fn ensure_encoder(&self, ck: &Checkpoint) -> Result<()> {
        if self.enc.get().is_some() {
            return Ok(());
        }
        let _g = self.enc_lock.lock().map_err(|_| Audio8Error::Other("энкодер: замок отравлен".into()))?;
        if self.enc.get().is_none() {
            let _ = self.enc.set(Encoder::load(ck)?);
        }
        Ok(())
    }

    /// Коды `[num_codebooks][T]` → PCM (44.1 кГц, `T · 2048` семплов).
    pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        let t = codes[0].len();
        if t == 0 {
            return Ok(Vec::new());
        }
        let mut z = self.sem_vq.decode(&codes[0], self.dtype)?;
        for (i, vq) in self.res_vq.iter().enumerate() {
            z = z.add(&vq.decode(&codes[i + 1], self.dtype)?)?;
        }
        let mut x = z.transpose(1, 2)?.contiguous()?;
        x = self.post.forward(&x)?;
        for (ct, cn) in &self.up {
            x = cn.forward(&ct.forward(&x)?)?;
        }
        x = self.dec_in.forward(&x)?;
        for (snake, ct, res) in &self.dec_blocks {
            x = ct.forward(&snake.forward(&x)?)?;
            for r in res {
                x = r.forward(&x)?;
            }
        }
        x = self.dec_out.forward(&self.dec_snake.forward(&x)?)?.tanh()?;
        Ok(x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
    }

    /// PCM 44.1 кГц → коды `[num_codebooks][ceil(len/2048)]`.
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<Vec<u32>>> {
        let enc = self.enc.get().ok_or_else(|| Audio8Error::Other("энкодер кодека не загружен (ensure_encoder)".into()))?;
        if pcm.is_empty() {
            return Err(Audio8Error::Audio("пустой образец".into()));
        }
        let frames = pcm.len().div_ceil(FRAME_SAMPLES);
        let mut buf = pcm.to_vec();
        buf.resize(frames * FRAME_SAMPLES, 0.0);
        let n = buf.len();
        let mut x = Tensor::from_vec(buf, vec![1usize, 1, n], self.device)?.to_dtype(self.dtype)?;
        x = enc.first.forward(&x)?;
        dump("enc0", &x);
        for (bi, (res, snake, conv, tr)) in enc.blocks.iter().enumerate() {
            for r in res {
                x = r.forward(&x)?;
            }
            x = conv.forward(&snake.forward(&x)?)?;
            dump(&format!("encb{}", bi + 1), &x);
            if let Some(t) = tr {
                x = t.forward(&x)?;
                dump("enctr", &x);
            }
        }
        x = enc.last.forward(&enc.snake.forward(&x)?)?;
        dump("encout", &x);
        for (c, cn) in &enc.down {
            x = cn.forward(&c.forward(&x)?)?;
        }
        dump("down", &x);
        x = enc.pre.forward(&x)?;
        dump("pre", &x);
        let t = x.dims()[2];
        let z = x.transpose(1, 2)?.contiguous()?;
        let (sem_codes, sem_q) = self.sem_vq.encode(&z)?;
        let mut residual = z.sub(&sem_q)?;
        let mut codes = vec![sem_codes];
        for vq in &self.res_vq {
            let (c, q) = vq.encode(&residual)?;
            residual = residual.sub(&q)?;
            codes.push(c);
        }
        let keep = frames.min(t);
        for c in &mut codes {
            c.truncate(keep);
        }
        Ok(codes)
    }
}
