//! VAE Qwen-Image (`AutoencoderKLQwenImage` — каузальный 3D-VAE Wan 2.1) в
//! режиме одной картинки.
//!
//! Для одного кадра каузальная свёртка по времени сводится к 2D: перед
//! кадром — нули (паддинг `2·p` слева), и из ядра `[.., kt, kh, kw]` работает
//! только последний временной срез `kt − 1`. Временные свёртки ресэмплинга
//! (`time_conv`) на первом кадре пропускаются самим diffusers (кэш пуст),
//! поэтому здесь их нет. RMS-нормы — `F.normalize` по каналам × √C × γ.
//! Считается в F32.
//!
//! Нормировка латента пайплайна: `(z − mean) / std` по каналам.

use synaptix_core::{
    device::Device,
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::conv2d;

use crate::config::QwenVaeConfig;
use crate::memory;
use crate::source::Weights;

type Get<'a> = dyn Fn(&str) -> Result<Tensor> + 'a;

/// 2D-свёртка из 3D-ядра (последний временной срез).
struct Conv {
    w: Tensor,
    b: Option<Tensor>,
    stride: usize,
    pad: usize,
}

impl Conv {
    fn load(g: &Get<'_>, p: &str, stride: usize, pad: usize) -> Result<Self> {
        let w = g(&format!("{p}.weight"))?;
        let w = match w.rank() {
            5 => {
                let kt = w.dims()[2];
                let d = w.dims().to_vec();
                w.narrow(2, kt - 1, 1)?.contiguous()?.reshape((d[0], d[1], d[3], d[4]))?
            }
            4 => w,
            r => return Err(SynaptixError::Other(format!("VAE {p}: ядро ранга {r}"))),
        };
        let b = g(&format!("{p}.bias")).ok();
        Ok(Self { w, b, stride, pad })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        conv2d(x, &self.w, self.b.as_ref(), (self.stride, self.stride), (self.pad, self.pad), (1, 1))
    }
}

/// `QwenImageRMS_norm`: `x / ‖x‖₂(по каналам) · √C · γ`.
struct Norm {
    gamma: Tensor,
}

impl Norm {
    fn load(g: &Get<'_>, p: &str) -> Result<Self> {
        let gm = g(&format!("{p}.gamma"))?;
        let c = gm.dims()[0];
        Ok(Self { gamma: gm.reshape((1, c, 1, 1))? })
    }

    fn forward(&self, x: &Tensor, silu: bool) -> Result<Tensor> {
        // x/‖x‖·√C = x/√(mean x²): PixelNorm без eps (eps F.normalize = 1e-12).
        let n = match x.pixel_norm_fused(1e-24, false) {
            Ok(n) => n,
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                let ms = x.sqr()?.mean_keepdim(1)?.add_scalar(1e-24)?.sqrt()?;
                x.broadcast_div(&ms)?
            }
            Err(e) => return Err(e),
        };
        let y = n.broadcast_mul(&self.gamma)?;
        if silu {
            y.silu()
        } else {
            Ok(y)
        }
    }
}

struct ResBlock {
    norm1: Norm,
    conv1: Conv,
    norm2: Norm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

impl ResBlock {
    fn load(g: &Get<'_>, p: &str, cin: usize, cout: usize) -> Result<Self> {
        Ok(Self {
            norm1: Norm::load(g, &format!("{p}.norm1"))?,
            conv1: Conv::load(g, &format!("{p}.conv1"), 1, 1)?,
            norm2: Norm::load(g, &format!("{p}.norm2"))?,
            conv2: Conv::load(g, &format!("{p}.conv2"), 1, 1)?,
            shortcut: if cin != cout { Some(Conv::load(g, &format!("{p}.conv_shortcut"), 1, 0)?) } else { None },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = match &self.shortcut {
            Some(s) => s.forward(x)?,
            None => x.clone(),
        };
        let y = self.conv1.forward(&self.norm1.forward(x, true)?)?;
        let y = self.conv2.forward(&self.norm2.forward(&y, true)?)?;
        y.add(&h)
    }
}

/// Внимание mid-блока: одна голова на всё полотно, q/k/v из одной 1×1-свёртки.
struct Attn {
    norm: Norm,
    qkv: Conv,
    proj: Conv,
}

impl Attn {
    fn load(g: &Get<'_>, p: &str) -> Result<Self> {
        Ok(Self {
            norm: Norm::load(g, &format!("{p}.norm"))?,
            qkv: Conv::load(g, &format!("{p}.to_qkv"), 1, 0)?,
            proj: Conv::load(g, &format!("{p}.proj"), 1, 0)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.dims().to_vec();
        let (c, h, w) = (d[1], d[2], d[3]);
        let hw = h * w;
        let qkv = self.qkv.forward(&self.norm.forward(x, false)?)?; // [1, 3C, H, W]
        let seq = qkv.reshape((3 * c, hw))?.transpose(0, 1)?.contiguous()?; // [HW, 3C]
        let part = |i: usize| -> Result<Tensor> { seq.narrow(1, i * c, c)?.contiguous()?.reshape((1, 1, hw, c)) };
        let (q, k, v) = (part(0)?, part(1)?, part(2)?);
        drop(seq);
        let scale = 1.0 / (c as f32).sqrt();
        // Матрица очков [HW, HW] в F32 на 1024² — 1 ГБ: порции по запросам.
        const Q_CHUNK: usize = 2048;
        let attn = if hw > 2 * Q_CHUNK {
            let mut parts = Vec::with_capacity(hw.div_ceil(Q_CHUNK));
            let mut s = 0;
            while s < hw {
                let len = Q_CHUNK.min(hw - s);
                let qc = q.narrow(2, s, len)?.contiguous()?;
                parts.push(scaled_dot_attention(&qc, &k, &v, scale, None)?);
                s += len;
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            Tensor::cat(&refs, 2)?
        } else {
            scaled_dot_attention(&q, &k, &v, scale, None)?
        };
        let out = attn.reshape((hw, c))?.transpose(0, 1)?.contiguous()?.reshape((1, c, h, w))?;
        self.proj.forward(&out)?.add(x)
    }
}

struct Mid {
    r0: ResBlock,
    attn: Attn,
    r1: ResBlock,
}

impl Mid {
    fn load(g: &Get<'_>, p: &str, c: usize) -> Result<Self> {
        Ok(Self {
            r0: ResBlock::load(g, &format!("{p}.resnets.0"), c, c)?,
            attn: Attn::load(g, &format!("{p}.attentions.0"))?,
            r1: ResBlock::load(g, &format!("{p}.resnets.1"), c, c)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.r0.forward(x)?;
        let x = self.attn.forward(&x)?;
        self.r1.forward(&x)
    }
}

/// Паддинг справа и снизу на 1 (`ZeroPad2d((0, 1, 0, 1))`).
fn pad_br(x: &Tensor) -> Result<Tensor> {
    let d = x.dims().to_vec();
    let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
    let right = Tensor::zeros((b, c, h, 1), x.dtype(), x.device())?;
    let x = Tensor::cat(&[x, &right], 3)?;
    let bottom = Tensor::zeros((b, c, 1, w + 1), x.dtype(), x.device())?;
    Tensor::cat(&[&x, &bottom], 2)?.contiguous()
}

fn upsample2x(x: &Tensor) -> Result<Tensor> {
    match x.upsample_nearest2x() {
        Ok(o) => return Ok(o),
        Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
        Err(e) => return Err(e),
    }
    let d = x.dims().to_vec();
    let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
    let xw = x.reshape((b, c, h, w, 1))?;
    let xw = Tensor::cat(&[&xw, &xw], 4)?.contiguous()?.reshape((b, c, h, w * 2))?;
    let xh = xw.reshape((b, c, h, 1, w * 2))?;
    Tensor::cat(&[&xh, &xh], 3)?.contiguous()?.reshape((b, c, h * 2, w * 2))
}

enum DownLayer {
    Res(ResBlock),
    Down(Conv),
}

pub struct Encoder {
    conv_in: Conv,
    layers: Vec<DownLayer>,
    mid: Mid,
    norm_out: Norm,
    conv_out: Conv,
    quant_conv: Conv,
    z_dim: usize,
}

impl Encoder {
    pub fn load(g: &Get<'_>, cfg: &QwenVaeConfig) -> Result<Self> {
        let dims: Vec<usize> = std::iter::once(1).chain(cfg.dim_mult.iter().copied()).map(|m| m * cfg.base_dim).collect();
        let mut layers = Vec::new();
        let mut idx = 0usize;
        let stages = cfg.dim_mult.len();
        for i in 0..stages {
            let (mut cin, cout) = (dims[i], dims[i + 1]);
            for _ in 0..cfg.num_res_blocks {
                layers.push(DownLayer::Res(ResBlock::load(g, &format!("encoder.down_blocks.{idx}"), cin, cout)?));
                idx += 1;
                cin = cout;
            }
            if i != stages - 1 {
                layers.push(DownLayer::Down(Conv::load(g, &format!("encoder.down_blocks.{idx}.resample.1"), 2, 0)?));
                idx += 1;
            }
        }
        let top = *dims.last().unwrap_or(&cfg.base_dim);
        Ok(Self {
            conv_in: Conv::load(g, "encoder.conv_in", 1, 1)?,
            layers,
            mid: Mid::load(g, "encoder.mid_block", top)?,
            norm_out: Norm::load(g, "encoder.norm_out")?,
            conv_out: Conv::load(g, "encoder.conv_out", 1, 1)?,
            quant_conv: Conv::load(g, "quant_conv", 1, 0)?,
            z_dim: cfg.z_dim,
        })
    }

    /// `[1, 3, H, W]` в [−1, 1] → мода распределения `[1, z, H/8, W/8]`.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;
        for l in &self.layers {
            h = match l {
                DownLayer::Res(r) => r.forward(&h)?,
                DownLayer::Down(c) => c.forward(&pad_br(&h)?)?,
            };
        }
        let h = self.mid.forward(&h)?;
        let h = self.conv_out.forward(&self.norm_out.forward(&h, true)?)?;
        let m = self.quant_conv.forward(&h)?;
        m.narrow(1, 0, self.z_dim)?.contiguous()
    }
}

struct UpBlock {
    resnets: Vec<ResBlock>,
    up: Option<Conv>,
}

pub struct Decoder {
    post_quant_conv: Conv,
    conv_in: Conv,
    mid: Mid,
    blocks: Vec<UpBlock>,
    norm_out: Norm,
    conv_out: Conv,
}

impl Decoder {
    pub fn load(g: &Get<'_>, cfg: &QwenVaeConfig) -> Result<Self> {
        let mults: Vec<usize> =
            std::iter::once(*cfg.dim_mult.last().unwrap_or(&1)).chain(cfg.dim_mult.iter().rev().copied()).collect();
        let dims: Vec<usize> = mults.iter().map(|m| m * cfg.base_dim).collect();
        let n = cfg.dim_mult.len();
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            let (mut cin, cout) = (dims[i], dims[i + 1]);
            if i > 0 {
                cin /= 2;
            }
            let p = format!("decoder.up_blocks.{i}");
            let mut resnets = Vec::new();
            for j in 0..=cfg.num_res_blocks {
                resnets.push(ResBlock::load(g, &format!("{p}.resnets.{j}"), cin, cout)?);
                cin = cout;
            }
            let up = if i != n - 1 { Some(Conv::load(g, &format!("{p}.upsamplers.0.resample.1"), 1, 1)?) } else { None };
            blocks.push(UpBlock { resnets, up });
        }
        Ok(Self {
            post_quant_conv: Conv::load(g, "post_quant_conv", 1, 0)?,
            conv_in: Conv::load(g, "decoder.conv_in", 1, 1)?,
            mid: Mid::load(g, "decoder.mid_block", dims[0])?,
            blocks,
            norm_out: Norm::load(g, "decoder.norm_out")?,
            conv_out: Conv::load(g, "decoder.conv_out", 1, 1)?,
        })
    }

    /// `[1, z, h, w]` (денормированный) → `[1, 3, 8h, 8w]` в [−1, 1].
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let x = self.post_quant_conv.forward(z)?;
        let mut h = self.mid.forward(&self.conv_in.forward(&x)?)?;
        drop(x);
        for b in &self.blocks {
            for r in &b.resnets {
                h = r.forward(&h)?;
            }
            if let Some(c) = &b.up {
                h = c.forward(&upsample2x(&h)?)?;
            }
        }
        let h = self.conv_out.forward(&self.norm_out.forward(&h, true)?)?;
        h.clamp(-1.0, 1.0)
    }
}

/// `(mean, std)` латента `[1, z, 1, 1]` (F32).
fn latent_stats(cfg: &QwenVaeConfig, dev: Device) -> Result<(Tensor, Tensor)> {
    let z = cfg.z_dim;
    Ok((
        Tensor::from_vec(cfg.latents_mean.clone(), (1, z, 1, 1), dev)?,
        Tensor::from_vec(cfg.latents_std.clone(), (1, z, 1, 1), dev)?,
    ))
}

/// Картинка `[3, H, W]` в [0, 1] (стороны кратны 8) → нормированный латент
/// `[1, z, H/8, W/8]` (F32, на `dev`).
pub fn encode(w: &Weights, cfg: &QwenVaeConfig, dev: Device, image: &Tensor) -> Result<Tensor> {
    let d = image.dims().to_vec();
    let x = image.to_device(dev)?.to_dtype(DType::F32)?.reshape((1, d[0], d[1], d[2]))?.affine(2.0, -1.0)?;
    let enc = {
        let _g = memory::weights_guard(dev);
        Encoder::load(&|n| w.get(n, dev, DType::F32), cfg)?
    };
    let z = enc.encode(&x)?;
    drop((x, enc));
    let (mean, std) = latent_stats(cfg, dev)?;
    let out = z.broadcast_sub(&mean)?.broadcast_div(&std)?;
    drop(z);
    memory::release_pools(dev);
    Ok(out)
}

/// Нормированный латент `[1, z, h, w]` → картинка `[3, 8h, 8w]` в [0, 1]
/// (F32, CPU).
pub fn decode(w: &Weights, cfg: &QwenVaeConfig, dev: Device, latent: &Tensor) -> Result<Tensor> {
    let (mean, std) = latent_stats(cfg, dev)?;
    let z = latent.to_device(dev)?.to_dtype(DType::F32)?.broadcast_mul(&std)?.broadcast_add(&mean)?;
    let image = {
        let dec = {
            let _g = memory::weights_guard(dev);
            Decoder::load(&|n| w.get(n, dev, DType::F32), cfg)?
        };
        dec.decode(&z)?
    };
    drop(z);
    let image = image.affine(0.5, 0.5)?.clamp(0.0, 1.0)?;
    let d = image.dims().to_vec();
    let out = image.reshape((d[1], d[2], d[3]))?.contiguous()?.to_device(Device::Cpu)?;
    drop(image);
    memory::release_pools(dev);
    Ok(out)
}
