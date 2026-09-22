//! VAE Qwen-Image 2.1 (`AutoencoderKLQwenImage21` — резидуальный 3D-VAE
//! Wan 2.2) в режиме одной картинки: 4 канала RGBA, сжатие 16×, латент 64.
//!
//! У одного кадра каузальные 3D-свёртки уже сведены в 2D самим diffusers
//! (`QwenImage21CausalConv3d` — `Conv2d`), временные свёртки ресэмплинга
//! (`time_conv`) на первом кадре не вызываются. Резидуальные шорткаты:
//! - энкодер `AvgDown3D`: усреднение по группам каналов над блоком
//!   `(ft, fs, fs)`; при сжатии по времени кадр дополняется нулевым спереди,
//!   и половина выходных каналов шортката — нули;
//! - декодер `DupUp3D`: повтор каналов и раскладка `(ft, fs, fs)` в
//!   пространство; из двух временных кадров остаётся последний
//!   (`first_chunk`).
//! RMS-нормы — `F.normalize` по каналам × √C × γ. Считается в F32.
//!
//! Большие картинки кодируются/декодируются плитками с перекрытием и
//! линейным смешиванием швов, как `tiled_encode/tiled_decode` diffusers.

use synaptix_core::{
    device::Device,
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};
use synaptix_image_qwen::memory;
use synaptix_image_qwen::source::Weights;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::conv2d;

use crate::config::Qwen21VaeConfig;

type Get<'a> = dyn Fn(&str) -> Result<Tensor> + 'a;

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

/// `QwenImage21RMS_norm`: `x / ‖x‖₂(по каналам) · √C · γ`.
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

/// Внимание mid-блока: одна голова на всё полотно.
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

/// `QwenImage21AvgDown3D` для одного кадра: выходной канал `o` — среднее по
/// группе `g = in·factor/out` элементов блока `(ft, fs, fs)` канала
/// `o / R` (`R = factor/g`), где при сжатии по времени кадр `ft = 0` — нулевой
/// (паддинг спереди), реальный — `ft − 1`.
struct AvgDown {
    cin: usize,
    cout: usize,
    ft: usize,
    fs: usize,
}

impl AvgDown {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let factor = self.ft * self.fs * self.fs;
        if self.cin * factor % self.cout != 0 {
            return Err(SynaptixError::Other("VAE AvgDown: каналы не делятся".into()));
        }
        let g = self.cin * factor / self.cout;
        if g == 0 || factor % g != 0 {
            return Err(SynaptixError::Other(format!("VAE AvgDown: group {g} не делит factor {factor}")));
        }
        let r_count = factor / g;
        let d = x.dims().to_vec();
        let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
        let fs = self.fs;
        let (hs, ws) = (h / fs, w / fs);
        // Подвыборки со сдвигом (sh, sw): [1, C, fs, fs, H/fs, W/fs].
        let sub = if fs == 1 {
            x.reshape((b, c, 1, 1, h, w))?
        } else {
            x.reshape((b, c, hs, fs, ws, fs))?.permute([0, 1, 3, 5, 2, 4])?.contiguous()?
        };
        let real_ft = self.ft - 1;
        let mut sums: Vec<Tensor> = Vec::with_capacity(r_count);
        for r in 0..r_count {
            let mut acc: Option<Tensor> = None;
            for j in 0..g {
                let f = r * g + j;
                let (fti, sh, sw) = (f / (fs * fs), (f / fs) % fs, f % fs);
                if fti != real_ft {
                    continue;
                }
                let s = sub.narrow(2, sh, 1)?.narrow(3, sw, 1)?.contiguous()?.reshape((b, c, hs, ws))?;
                acc = Some(match acc {
                    Some(a) => a.add(&s)?,
                    None => s,
                });
            }
            let t = match acc {
                Some(a) => a.mul_scalar(1.0 / g as f32)?,
                None => Tensor::zeros((b, c, hs, ws), x.dtype(), x.device())?,
            };
            sums.push(t.reshape((b, c, 1, hs, ws))?);
        }
        let refs: Vec<&Tensor> = sums.iter().collect();
        Tensor::cat(&refs, 2)?.contiguous()?.reshape((b, c * r_count, hs, ws))
    }
}

/// `QwenImage21DupUp3D` (`first_chunk`): выходной пиксель `(sh, sw)` канала
/// `o` берётся из входного канала `(o·factor + k) / repeats`,
/// `k = (ft−1)·fs² + sh·fs + sw`, `repeats = out·factor/in`.
struct DupUp {
    cin: usize,
    cout: usize,
    ft: usize,
    fs: usize,
}

impl DupUp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let factor = self.ft * self.fs * self.fs;
        if self.cout * factor % self.cin != 0 {
            return Err(SynaptixError::Other("VAE DupUp: каналы не делятся".into()));
        }
        let repeats = self.cout * factor / self.cin;
        let d = x.dims().to_vec();
        let (b, _c, h, w) = (d[0], d[1], d[2], d[3]);
        let fs = self.fs;
        let mut rows: Vec<Tensor> = Vec::with_capacity(fs);
        for sh in 0..fs {
            let mut cols: Vec<Tensor> = Vec::with_capacity(fs);
            for sw in 0..fs {
                let k = (self.ft - 1) * fs * fs + sh * fs + sw;
                let idx: Vec<u32> = (0..self.cout).map(|o| ((o * factor + k) / repeats) as u32).collect();
                let idx = Tensor::from_vec(idx, (self.cout,), x.device())?;
                let g = x.index_select(1, &idx)?.contiguous()?.reshape((b, self.cout, h, w, 1))?;
                cols.push(g);
            }
            let refs: Vec<&Tensor> = cols.iter().collect();
            rows.push(Tensor::cat(&refs, 4)?.contiguous()?.reshape((b, self.cout, h, 1, w * fs))?);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        Tensor::cat(&refs, 3)?.contiguous()?.reshape((b, self.cout, h * fs, w * fs))
    }
}

struct DownBlock {
    resnets: Vec<ResBlock>,
    down: Option<Conv>,
    shortcut: AvgDown,
}

pub struct Encoder {
    conv_in: Conv,
    blocks: Vec<DownBlock>,
    mid: Mid,
    norm_out: Norm,
    conv_out: Conv,
    quant_conv: Conv,
    z_dim: usize,
}

impl Encoder {
    pub fn load(g: &Get<'_>, cfg: &Qwen21VaeConfig) -> Result<Self> {
        let dims: Vec<usize> = std::iter::once(1).chain(cfg.dim_mult.iter().copied()).map(|m| m * cfg.base_dim).collect();
        let stages = cfg.dim_mult.len();
        let mut blocks = Vec::with_capacity(stages);
        for i in 0..stages {
            let (mut cin, cout) = (dims[i], dims[i + 1]);
            let p = format!("encoder.down_blocks.{i}");
            let mut resnets = Vec::with_capacity(cfg.num_res_blocks);
            for j in 0..cfg.num_res_blocks {
                resnets.push(ResBlock::load(g, &format!("{p}.resnets.{j}"), cin, cout)?);
                cin = cout;
            }
            let last = i == stages - 1;
            let temporal = !last && cfg.temporal_downsample.get(i).copied().unwrap_or(false);
            let down = if last { None } else { Some(Conv::load(g, &format!("{p}.downsampler.resample.1"), 2, 0)?) };
            blocks.push(DownBlock {
                resnets,
                down,
                shortcut: AvgDown {
                    cin: dims[i],
                    cout,
                    ft: if temporal { 2 } else { 1 },
                    fs: if last { 1 } else { 2 },
                },
            });
        }
        let top = *dims.last().unwrap_or(&cfg.base_dim);
        Ok(Self {
            conv_in: Conv::load(g, "encoder.conv_in", 1, 1)?,
            blocks,
            mid: Mid::load(g, "encoder.mid_block", top)?,
            norm_out: Norm::load(g, "encoder.norm_out")?,
            conv_out: Conv::load(g, "encoder.conv_out", 1, 1)?,
            quant_conv: Conv::load(g, "quant_conv", 1, 0)?,
            z_dim: cfg.z_dim,
        })
    }

    /// `[1, 4, H, W]` в [−1, 1] → мода распределения `[1, z, H/16, W/16]`.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;
        for b in &self.blocks {
            let skip = b.shortcut.forward(&h)?;
            let mut y = h;
            for r in &b.resnets {
                y = r.forward(&y)?;
            }
            if let Some(c) = &b.down {
                y = c.forward(&pad_br(&y)?)?;
            }
            h = y.add(&skip)?;
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
    shortcut: Option<DupUp>,
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
    pub fn load(g: &Get<'_>, cfg: &Qwen21VaeConfig) -> Result<Self> {
        let mults: Vec<usize> =
            std::iter::once(*cfg.dim_mult.last().unwrap_or(&1)).chain(cfg.dim_mult.iter().rev().copied()).collect();
        let dims: Vec<usize> = mults.iter().map(|m| m * cfg.decoder_base_dim).collect();
        let n = cfg.dim_mult.len();
        let temporal_up: Vec<bool> = cfg.temporal_downsample.iter().rev().copied().collect();
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            let (mut cin, cout) = (dims[i], dims[i + 1]);
            let p = format!("decoder.up_blocks.{i}");
            let mut resnets = Vec::with_capacity(cfg.num_res_blocks + 1);
            for j in 0..=cfg.num_res_blocks {
                resnets.push(ResBlock::load(g, &format!("{p}.resnets.{j}"), cin, cout)?);
                cin = cout;
            }
            let last = i == n - 1;
            let temporal = !last && temporal_up.get(i).copied().unwrap_or(false);
            let up = if last { None } else { Some(Conv::load(g, &format!("{p}.upsampler.resample.1"), 1, 1)?) };
            let shortcut =
                if last { None } else { Some(DupUp { cin: dims[i], cout, ft: if temporal { 2 } else { 1 }, fs: 2 }) };
            blocks.push(UpBlock { resnets, up, shortcut });
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

    /// `[1, z, h, w]` (денормированный) → `[1, 4, 16h, 16w]` в [−1, 1].
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let x = self.post_quant_conv.forward(z)?;
        let mut h = self.mid.forward(&self.conv_in.forward(&x)?)?;
        drop(x);
        for b in &self.blocks {
            let skip = match &b.shortcut {
                Some(s) => Some(s.forward(&h)?),
                None => None,
            };
            let mut y = h;
            for r in &b.resnets {
                y = r.forward(&y)?;
            }
            if let Some(c) = &b.up {
                y = c.forward(&upsample2x(&y)?)?;
            }
            h = match skip {
                Some(s) => y.add(&s)?,
                None => y,
            };
        }
        let h = self.conv_out.forward(&self.norm_out.forward(&h, true)?)?;
        h.clamp(-1.0, 1.0)
    }
}

/// `(mean, std)` латента `[1, z, 1, 1]` (F32).
fn latent_stats(cfg: &Qwen21VaeConfig, dev: Device) -> Result<(Tensor, Tensor)> {
    let z = cfg.z_dim;
    Ok((
        Tensor::from_vec(cfg.latents_mean.clone(), (1, z, 1, 1), dev)?,
        Tensor::from_vec(cfg.latents_std.clone(), (1, z, 1, 1), dev)?,
    ))
}

/// Плитки с перекрытием (`tiled_encode/decode`): размер и шаг плитки в
/// пикселях картинки; `None` — целиком.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tiling {
    pub tile: usize,
    pub stride: usize,
}

impl Tiling {
    /// Умолчание diffusers.
    pub const DIFFUSERS: Tiling = Tiling { tile: 256, stride: 192 };
    /// Крупнее — меньше швов.
    pub const LARGE: Tiling = Tiling { tile: 1024, stride: 896 };
    pub const MEDIUM: Tiling = Tiling { tile: 512, stride: 448 };

    /// Оценка пика активаций декодера/энкодера на картинку `w×h` (F32):
    /// самый широкий слой на полном разрешении, несколько живых копий.
    pub fn activation_bytes(w: usize, h: usize, cfg: &Qwen21VaeConfig) -> usize {
        6 * cfg.decoder_base_dim.max(cfg.base_dim) * w * h * 4 + (256 << 20)
    }

    /// Плитки, при которых оценка пика влезает в `budget`; `None`, если
    /// влезает целиком.
    pub fn pick(w: usize, h: usize, cfg: &Qwen21VaeConfig, budget: usize) -> Option<Tiling> {
        if Self::activation_bytes(w, h, cfg) <= budget {
            return None;
        }
        for t in [Self::LARGE, Self::MEDIUM, Self::DIFFUSERS] {
            if t.tile < w.max(h) && Self::activation_bytes(t.tile, t.tile, cfg) <= budget {
                return Some(t);
            }
        }
        Some(Self::DIFFUSERS)
    }
}

/// Линейное смешивание вертикального шва: первые `extent` строк `b` берутся
/// из хвоста `a` с весом `1 − y/extent`.
fn blend_rows(a: &Tensor, b: &Tensor, extent: usize) -> Result<Tensor> {
    let extent = extent.min(a.dims()[2]).min(b.dims()[2]);
    if extent == 0 {
        return Ok(b.clone());
    }
    let ha = a.dims()[2];
    let hb = b.dims()[2];
    let wa = a.narrow(2, ha - extent, extent)?.contiguous()?;
    let wb = b.narrow(2, 0, extent)?.contiguous()?;
    let ramp: Vec<f32> = (0..extent).map(|y| y as f32 / extent as f32).collect();
    let ramp = Tensor::from_vec(ramp, (1, 1, extent, 1), b.device())?;
    let one_minus = ramp.affine(-1.0, 1.0)?;
    let mixed = wa.broadcast_mul(&one_minus)?.add(&wb.broadcast_mul(&ramp)?)?;
    if hb == extent {
        return Ok(mixed);
    }
    let rest = b.narrow(2, extent, hb - extent)?.contiguous()?;
    Tensor::cat(&[&mixed, &rest], 2)
}

fn blend_cols(a: &Tensor, b: &Tensor, extent: usize) -> Result<Tensor> {
    let extent = extent.min(a.dims()[3]).min(b.dims()[3]);
    if extent == 0 {
        return Ok(b.clone());
    }
    let wa_ = a.dims()[3];
    let wb_ = b.dims()[3];
    let wa = a.narrow(3, wa_ - extent, extent)?.contiguous()?;
    let wb = b.narrow(3, 0, extent)?.contiguous()?;
    let ramp: Vec<f32> = (0..extent).map(|x| x as f32 / extent as f32).collect();
    let ramp = Tensor::from_vec(ramp, (1, 1, 1, extent), b.device())?;
    let one_minus = ramp.affine(-1.0, 1.0)?;
    let mixed = wa.broadcast_mul(&one_minus)?.add(&wb.broadcast_mul(&ramp)?)?;
    if wb_ == extent {
        return Ok(mixed);
    }
    let rest = b.narrow(3, extent, wb_ - extent)?.contiguous()?;
    Tensor::cat(&[&mixed, &rest], 3)
}

/// Общая схема плиток: `run(y0, x0, th, tw)` считает плитку входа
/// `[y0, y0+th) × [x0, x0+tw)` (в единицах входа) и возвращает её выход;
/// `scale` — во сколько раз выход крупнее входа (декод 16, энкод 1/16 →
/// передаётся как `(mul, div)`).
fn tiled(
    h: usize,
    w: usize,
    tile: usize,
    stride: usize,
    out_mul: usize,
    out_div: usize,
    run: &mut dyn FnMut(usize, usize, usize, usize) -> Result<Tensor>,
) -> Result<Tensor> {
    let blend = (tile - stride) * out_mul / out_div;
    let out_stride = stride * out_mul / out_div;
    let (out_h, out_w) = (h * out_mul / out_div, w * out_mul / out_div);
    let mut rows: Vec<Vec<Tensor>> = Vec::new();
    let mut y0 = 0;
    while y0 < h {
        let mut row = Vec::new();
        let mut x0 = 0;
        while x0 < w {
            let th = tile.min(h - y0);
            let tw = tile.min(w - x0);
            row.push(run(y0, x0, th, tw)?);
            x0 += stride;
        }
        rows.push(row);
        y0 += stride;
    }
    let mut result_rows: Vec<Tensor> = Vec::with_capacity(rows.len());
    for i in 0..rows.len() {
        let mut result_row: Vec<Tensor> = Vec::with_capacity(rows[i].len());
        for j in 0..rows[i].len() {
            let mut t = rows[i][j].clone();
            if i > 0 {
                t = blend_rows(&rows[i - 1][j], &t, blend)?;
            }
            if j > 0 {
                t = blend_cols(&rows[i][j - 1], &t, blend)?;
            }
            let th = out_stride.min(t.dims()[2]);
            let tw = out_stride.min(t.dims()[3]);
            result_row.push(t.narrow(2, 0, th)?.narrow(3, 0, tw)?.contiguous()?);
        }
        let refs: Vec<&Tensor> = result_row.iter().collect();
        result_rows.push(Tensor::cat(&refs, 3)?);
    }
    let refs: Vec<&Tensor> = result_rows.iter().collect();
    let out = Tensor::cat(&refs, 2)?;
    out.narrow(2, 0, out_h)?.narrow(3, 0, out_w)?.contiguous()
}

/// Картинка `[4, H, W]` в [0, 1] (стороны кратны 16) → нормированный латент
/// `[1, z, H/16, W/16]` (F32, на `dev`).
pub fn encode(w: &Weights, cfg: &Qwen21VaeConfig, dev: Device, image: &Tensor, tiling: Option<Tiling>) -> Result<Tensor> {
    let d = image.dims().to_vec();
    if d.len() != 3 || d[0] != cfg.in_channels {
        return Err(SynaptixError::Other(format!("VAE encode: ожидалась картинка [{}, H, W], пришло {d:?}", cfg.in_channels)));
    }
    let f = cfg.scale_factor_spatial;
    if d[1] % f != 0 || d[2] % f != 0 {
        return Err(SynaptixError::Other(format!("VAE encode: стороны должны быть кратны {f}, пришло {}×{}", d[2], d[1])));
    }
    let x = image.to_device(dev)?.to_dtype(DType::F32)?.reshape((1, d[0], d[1], d[2]))?.affine(2.0, -1.0)?;
    let enc = {
        let _g = memory::weights_guard(dev);
        Encoder::load(&|n| w.get(n, dev, DType::F32), cfg)?
    };
    let z = match tiling.filter(|t| t.tile < d[1].max(d[2])) {
        None => enc.encode(&x)?,
        Some(t) => tiled(d[1], d[2], t.tile, t.stride, 1, f, &mut |y0, x0, th, tw| {
            enc.encode(&x.narrow(2, y0, th)?.narrow(3, x0, tw)?.contiguous()?)
        })?,
    };
    drop((x, enc));
    let (mean, std) = latent_stats(cfg, dev)?;
    let out = z.broadcast_sub(&mean)?.broadcast_div(&std)?;
    drop(z);
    memory::release_pools(dev);
    Ok(out)
}

/// Нормированный латент `[1, z, h, w]` → картинка `[4, 16h, 16w]` в [0, 1]
/// (F32, CPU).
pub fn decode(w: &Weights, cfg: &Qwen21VaeConfig, dev: Device, latent: &Tensor, tiling: Option<Tiling>) -> Result<Tensor> {
    let (mean, std) = latent_stats(cfg, dev)?;
    let z = latent.to_device(dev)?.to_dtype(DType::F32)?.broadcast_mul(&std)?.broadcast_add(&mean)?;
    let zd = z.dims().to_vec();
    let f = cfg.scale_factor_spatial;
    let image = {
        let dec = {
            let _g = memory::weights_guard(dev);
            Decoder::load(&|n| w.get(n, dev, DType::F32), cfg)?
        };
        match tiling.filter(|t| t.tile / f < zd[2].max(zd[3])) {
            None => dec.decode(&z)?,
            Some(t) => tiled(zd[2], zd[3], t.tile / f, t.stride / f, f, 1, &mut |y0, x0, th, tw| {
                dec.decode(&z.narrow(2, y0, th)?.narrow(3, x0, tw)?.contiguous()?)
            })?,
        }
    };
    drop(z);
    let image = image.affine(0.5, 0.5)?.clamp(0.0, 1.0)?;
    let d = image.dims().to_vec();
    let out = image.reshape((d[1], d[2], d[3]))?.contiguous()?.to_device(Device::Cpu)?;
    drop(image);
    memory::release_pools(dev);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avgdown_matches_derivation() {
        synaptix_kernels_cpu::ensure_registered();
        // 2 канала 2×2 → factor 8 (ft 2, fs 2), out 4: g = 4, R = 2 —
        // чётные каналы нули (кадр паддинга), нечётные — среднее 2×2.
        let x = Tensor::from_vec((1..=8).map(|v| v as f32).collect::<Vec<_>>(), (1, 2, 2, 2), Device::Cpu).unwrap();
        let y = AvgDown { cin: 2, cout: 4, ft: 2, fs: 2 }.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 4, 1, 1]);
        assert_eq!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![0.0, 2.5, 0.0, 6.5]);
        // Без времени: обычный avgpool.
        let y = AvgDown { cin: 2, cout: 2, ft: 1, fs: 2 }.forward(&x).unwrap();
        assert_eq!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![2.5, 6.5]);
        // Последняя стадия: тождество.
        let y = AvgDown { cin: 2, cout: 2, ft: 1, fs: 1 }.forward(&x).unwrap();
        assert_eq!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap(), (1..=8).map(|v| v as f32).collect::<Vec<_>>());
    }

    #[test]
    fn dupup_matches_derivation() {
        synaptix_kernels_cpu::ensure_registered();
        // 4 канала 1×1 → out 2, ft 2, fs 2: factor 8, repeats 4 → канал o
        // берёт входной 2o+1 (второй кадр), nearest ×2.
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 4, 1, 1), Device::Cpu).unwrap();
        let y = DupUp { cin: 4, cout: 2, ft: 2, fs: 2 }.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 2, 2, 2]);
        assert_eq!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![2.0; 4].into_iter().chain(vec![4.0; 4]).collect::<Vec<_>>());
        // Без времени (ft 1), 4 → 2: repeats 2 → пиксель (sh, sw) из канала 2o + sh.
        let y = DupUp { cin: 4, cout: 2, ft: 1, fs: 2 }.forward(&x).unwrap();
        assert_eq!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
        // Тот же канал: nearest ×2.
        let y = DupUp { cin: 4, cout: 4, ft: 2, fs: 2 }.forward(&x).unwrap();
        assert_eq!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap()[..4], [1.0; 4]);
    }

    #[test]
    fn tiled_identity_is_seamless() {
        synaptix_kernels_cpu::ensure_registered();
        // Тождественная «сеть»: плитки 4 с шагом 3 собираются в исходник.
        let v: Vec<f32> = (0..(2 * 7 * 9)).map(|i| i as f32).collect();
        let x = Tensor::from_vec(v.clone(), (1, 2, 7, 9), Device::Cpu).unwrap();
        let out = tiled(7, 9, 4, 3, 1, 1, &mut |y0, x0, th, tw| x.narrow(2, y0, th)?.narrow(3, x0, tw)?.contiguous()).unwrap();
        assert_eq!(out.dims(), &[1, 2, 7, 9]);
        assert_eq!(out.flatten_all().unwrap().to_vec1::<f32>().unwrap(), v);
    }

    #[test]
    fn tiling_pick_by_budget() {
        let cfg = Qwen21VaeConfig {
            base_dim: 96,
            decoder_base_dim: 144,
            z_dim: 64,
            dim_mult: vec![1, 2, 4, 8, 8],
            num_res_blocks: 2,
            temporal_downsample: vec![false, true, true, true],
            in_channels: 4,
            out_channels: 4,
            scale_factor_spatial: 16,
            latents_mean: vec![0.0; 64],
            latents_std: vec![1.0; 64],
        };
        assert_eq!(Tiling::pick(1024, 1024, &cfg, 64 << 30), None);
        assert_eq!(Tiling::pick(2048, 2048, &cfg, 6 << 30), Some(Tiling::LARGE));
        assert_eq!(Tiling::pick(2048, 2048, &cfg, 2 << 30), Some(Tiling::MEDIUM));
        assert_eq!(Tiling::pick(2048, 2048, &cfg, 1 << 30), Some(Tiling::DIFFUSERS));
    }
}
