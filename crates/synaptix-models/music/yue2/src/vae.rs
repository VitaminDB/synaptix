//! Oobleck-VAE YuE2: латент 64×25 Гц → 48 кГц стерео.
//!
//! Раскладка весов — `nn.Sequential` из релиза, то есть имена вида
//! `decoder.layers.1.layers.2.layers.3.weight_v`. Свёртки обёрнуты
//! `weight_norm` (`weight_g`/`weight_v`), активация — SnakeBeta в лог-шкале.
//!
//! Декодирование идёт тайлами: у тайла берётся только ядро, а по краям
//! добавляется «гало» — столько кадров контекста, сколько видит рецептивное
//! поле. Кроссфейда нет: каждый сэмпл ядра посчитан ровно так же, как считался
//! бы при полном проходе.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_ops::conv::{conv1d_dilated, conv_transpose1d};

use crate::config::Yue2VaeConfig;
use crate::loader::{read_bundle_file, CompLoader};
use crate::YueError;

type R<T> = Result<T, YueError>;

/// Текст ошибки прерванного декода.
pub const DECODE_CANCELLED: &str = "декодирование VAE прервано";

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Свёртка с `weight_norm`: `w = g * v / ||v||`, норма по всем осям кроме
/// нулевой (у ConvTranspose1d нулевая ось — входные каналы, у Conv1d —
/// выходные; правило то же).
fn load_wn(ck: &CompLoader, prefix: &str, bias: bool, dtype: DType) -> R<(Tensor, Option<Tensor>)> {
    let g = ck.f32(&format!("{prefix}.weight_g"))?;
    let v = ck.f32(&format!("{prefix}.weight_v"))?;
    let d = v.dims().to_vec();
    if d.len() != 3 {
        return Err(YueError::Load(format!("{prefix}.weight_v: ожидался ранг 3, а он {}", d.len())));
    }
    let (d0, d1, k) = (d[0], d[1], d[2]);
    let norm = v
        .reshape((d0, d1 * k))?
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .reshape((d0, 1usize, 1usize))?;
    let scale = g.broadcast_mul(&norm.recip()?)?;
    let w = v.broadcast_mul(&scale)?.to_dtype(dtype)?;
    let b = if bias {
        Some(ck.f32(&format!("{prefix}.bias"))?.to_dtype(dtype)?)
    } else {
        None
    };
    Ok((w, b))
}

/// SnakeBeta в лог-шкале: `y = x + sin(exp(α)·x)² / (exp(β) + ε)`.
struct Snake {
    alpha: Tensor,
    beta: Tensor,
}

impl Snake {
    fn load(ck: &CompLoader, prefix: &str, dtype: DType) -> R<Self> {
        Ok(Self {
            alpha: ck.f32(&format!("{prefix}.alpha"))?.to_dtype(dtype)?,
            beta: ck.f32(&format!("{prefix}.beta"))?.to_dtype(dtype)?,
        })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        // Слитое ядро; на бэкендах без него — разложение теми же операциями.
        match x.snake(&self.alpha, &self.beta, 1e-9) {
            Ok(y) => Ok(y),
            Err(synaptix_core::error::SynaptixError::Unsupported(_)) => {
                // Слитого ядра нет (CPU): считаем теми же операциями, но
                // параметры надо разложить по оси каналов — она у сигнала
                // средняя, а broadcast выравнивает формы по последней.
                let channels = self.alpha.numel();
                let shape = vec![1usize, channels, 1usize];
                let a = self.alpha.reshape(shape.clone())?.exp()?;
                let b = self.beta.reshape(shape)?.exp()?;
                let ax = x.broadcast_mul(&a)?;
                let s = ax.sin()?.sqr()?;
                let inv = b.affine(1.0, 1e-9)?.recip()?;
                Ok(x.broadcast_add(&s.broadcast_mul(&inv)?)?)
            }
            Err(e) => Err(e.into()),
        }
    }
}

/// ELU — запасная активация конфигурации (`use_snake: false`).
enum Act {
    Snake(Snake),
    Elu,
}

impl Act {
    fn load(ck: &CompLoader, prefix: &str, use_snake: bool, dtype: DType) -> R<Self> {
        if use_snake {
            Ok(Act::Snake(Snake::load(ck, prefix, dtype)?))
        } else {
            Ok(Act::Elu)
        }
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        match self {
            Act::Snake(s) => s.forward(x),
            Act::Elu => Ok(synaptix_ops::activation::elu::elu(x, 1.0)?),
        }
    }
}

struct Conv {
    w: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    pad: usize,
    dilation: usize,
}

impl Conv {
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        Ok(conv1d_dilated(x, &self.w, self.bias.as_ref(), self.stride, self.pad, self.dilation)?)
    }

    /// Длина выхода при данной длине входа.
    fn out_len(&self, len: usize) -> usize {
        let k = self.w.dims()[2];
        (len + 2 * self.pad - self.dilation * (k - 1) - 1) / self.stride + 1
    }
}

struct ConvT {
    w: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    pad: usize,
}

impl ConvT {
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        Ok(conv_transpose1d(x, &self.w, self.bias.as_ref(), self.stride, self.pad, 0, 1, 1)?)
    }

    fn out_len(&self, len: usize) -> usize {
        let k = self.w.dims()[2];
        (len - 1) * self.stride + k - 2 * self.pad
    }
}

/// `act → conv(k=7, dil) → act → conv(k=1)`, плюс вход.
struct ResidualUnit {
    act1: Act,
    conv1: Conv,
    act2: Act,
    conv2: Conv,
}

impl ResidualUnit {
    fn load(ck: &CompLoader, prefix: &str, dilation: usize, use_snake: bool, dtype: DType) -> R<Self> {
        let (w1, b1) = load_wn(ck, &format!("{prefix}.layers.1"), true, dtype)?;
        let (w2, b2) = load_wn(ck, &format!("{prefix}.layers.3"), true, dtype)?;
        Ok(Self {
            act1: Act::load(ck, &format!("{prefix}.layers.0"), use_snake, dtype)?,
            conv1: Conv { w: w1, bias: b1, stride: 1, pad: 3 * dilation, dilation },
            act2: Act::load(ck, &format!("{prefix}.layers.2"), use_snake, dtype)?,
            conv2: Conv { w: w2, bias: b2, stride: 1, pad: 0, dilation: 1 },
        })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let y = self.conv1.forward(&self.act1.forward(x)?)?;
        let y = self.conv2.forward(&self.act2.forward(&y)?)?;
        Ok(x.broadcast_add(&y)?)
    }
}

struct DecoderBlock {
    act: Act,
    up: ConvT,
    res: [ResidualUnit; 3],
}

impl DecoderBlock {
    fn load(ck: &CompLoader, prefix: &str, stride: usize, use_snake: bool, dtype: DType) -> R<Self> {
        let (w, b) = load_wn(ck, &format!("{prefix}.layers.1"), true, dtype)?;
        Ok(Self {
            act: Act::load(ck, &format!("{prefix}.layers.0"), use_snake, dtype)?,
            up: ConvT { w, bias: b, stride, pad: ceil_div(stride, 2) },
            res: [
                ResidualUnit::load(ck, &format!("{prefix}.layers.2"), 1, use_snake, dtype)?,
                ResidualUnit::load(ck, &format!("{prefix}.layers.3"), 3, use_snake, dtype)?,
                ResidualUnit::load(ck, &format!("{prefix}.layers.4"), 9, use_snake, dtype)?,
            ],
        })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut h = self.up.forward(&self.act.forward(x)?)?;
        for r in &self.res {
            h = r.forward(&h)?;
        }
        Ok(h)
    }

    fn out_len(&self, len: usize) -> usize {
        self.up.out_len(len)
    }
}

struct EncoderBlock {
    res: [ResidualUnit; 3],
    act: Act,
    down: Conv,
}

impl EncoderBlock {
    fn load(ck: &CompLoader, prefix: &str, stride: usize, use_snake: bool, dtype: DType) -> R<Self> {
        let (w, b) = load_wn(ck, &format!("{prefix}.layers.4"), true, dtype)?;
        Ok(Self {
            res: [
                ResidualUnit::load(ck, &format!("{prefix}.layers.0"), 1, use_snake, dtype)?,
                ResidualUnit::load(ck, &format!("{prefix}.layers.1"), 3, use_snake, dtype)?,
                ResidualUnit::load(ck, &format!("{prefix}.layers.2"), 9, use_snake, dtype)?,
            ],
            act: Act::load(ck, &format!("{prefix}.layers.3"), use_snake, dtype)?,
            down: Conv { w, bias: b, stride, pad: ceil_div(stride, 2), dilation: 1 },
        })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut h = x.clone();
        for r in &self.res {
            h = r.forward(&h)?;
        }
        self.down.forward(&self.act.forward(&h)?)
    }
}

struct Decoder {
    conv_in: Conv,
    blocks: Vec<DecoderBlock>,
    act: Act,
    conv_out: Conv,
    final_tanh: bool,
}

impl Decoder {
    fn load(ck: &CompLoader, cfg: &Yue2VaeConfig, dtype: DType) -> R<Self> {
        let (w_in, b_in) = load_wn(ck, "decoder.layers.0", true, dtype)?;
        let depth = cfg.c_mults.len() + 1;
        let mut blocks = Vec::with_capacity(cfg.strides.len());
        // Блоки идут от самого «узкого» уровня к выходу: страйды в обратном
        // порядке, как в `for i in range(depth-1, 0, -1)` у релиза.
        for (n, &stride) in cfg.strides.iter().rev().enumerate() {
            blocks.push(DecoderBlock::load(
                ck,
                &format!("decoder.layers.{}", n + 1),
                stride,
                cfg.use_snake,
                dtype,
            )?);
        }
        let (w_out, b_out) = load_wn(ck, &format!("decoder.layers.{}", depth + 1), false, dtype)?;
        Ok(Self {
            conv_in: Conv { w: w_in, bias: b_in, stride: 1, pad: 3, dilation: 1 },
            blocks,
            act: Act::load(ck, &format!("decoder.layers.{depth}"), cfg.use_snake, dtype)?,
            conv_out: Conv { w: w_out, bias: b_out, stride: 1, pad: 3, dilation: 1 },
            final_tanh: cfg.final_tanh,
        })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut h = self.conv_in.forward(x)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        h = self.conv_out.forward(&self.act.forward(&h)?)?;
        if self.final_tanh {
            h = h.tanh()?;
        }
        Ok(h)
    }

    /// Длина сигнала, которую даст полный проход по `frames` кадрам.
    fn out_len(&self, frames: usize) -> usize {
        let mut len = self.conv_in.out_len(frames);
        for b in &self.blocks {
            len = b.out_len(len);
        }
        self.conv_out.out_len(len)
    }
}

struct Encoder {
    conv_in: Conv,
    blocks: Vec<EncoderBlock>,
    act: Act,
    conv_out: Conv,
}

impl Encoder {
    fn load(ck: &CompLoader, cfg: &Yue2VaeConfig, dtype: DType) -> R<Self> {
        let (w_in, b_in) = load_wn(ck, "encoder.layers.0", true, dtype)?;
        let depth = cfg.c_mults.len() + 1;
        let mut blocks = Vec::with_capacity(cfg.strides.len());
        for (n, &stride) in cfg.strides.iter().enumerate() {
            blocks.push(EncoderBlock::load(
                ck,
                &format!("encoder.layers.{}", n + 1),
                stride,
                cfg.use_snake,
                dtype,
            )?);
        }
        let (w_out, b_out) = load_wn(ck, &format!("encoder.layers.{}", depth + 1), true, dtype)?;
        Ok(Self {
            conv_in: Conv { w: w_in, bias: b_in, stride: 1, pad: 3, dilation: 1 },
            blocks,
            act: Act::load(ck, &format!("encoder.layers.{depth}"), cfg.use_snake, dtype)?,
            conv_out: Conv { w: w_out, bias: b_out, stride: 1, pad: 1, dilation: 1 },
        })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut h = self.conv_in.forward(x)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        self.conv_out.forward(&self.act.forward(&h)?)
    }
}

pub struct Yue2Vae {
    decoder: Decoder,
    encoder: Option<Encoder>,
    cfg: Yue2VaeConfig,
    dtype: DType,
    device: Device,
}

impl Yue2Vae {
    /// Открыть бандл. `decoder_only` — не читать энкодер (он нужен только для
    /// каверов и переоценки латентов, а это половина весов).
    pub fn open(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        decoder_only: bool,
    ) -> R<Self> {
        let path = path.as_ref();
        let ck = CompLoader::open(path, None, device)?;
        let cfg = match read_bundle_file(path, "config.json") {
            Ok(bytes) => Yue2VaeConfig::from_json(&bytes)?,
            Err(_) => Yue2VaeConfig::default(),
        };
        let decoder = Decoder::load(&ck, &cfg, dtype)?;
        let encoder = if decoder_only {
            None
        } else {
            Some(Encoder::load(&ck, &cfg, dtype)?)
        };
        Ok(Self { decoder, encoder, cfg, dtype, device })
    }

    pub fn config(&self) -> &Yue2VaeConfig {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Длина сигнала полного декода `frames` кадров (у релиза — `1920·T − 64`).
    pub fn output_len(&self, frames: usize) -> usize {
        self.decoder.out_len(frames)
    }

    fn as_bcl(&self, z: &Tensor) -> R<Tensor> {
        let z = if z.rank() == 2 { z.unsqueeze(0)? } else { z.clone() };
        if z.rank() != 3 || z.dims()[1] != self.cfg.latent_dim {
            return Err(YueError::Other(format!(
                "ожидался латент [B,{},T], пришёл {:?}",
                self.cfg.latent_dim,
                z.dims()
            )));
        }
        Ok(z)
    }

    /// Полный декод — без тайлов. Память растёт линейно по длине, поэтому на
    /// песню целиком его звать не стоит.
    pub fn decode(&self, z: &Tensor) -> R<Tensor> {
        let z = self.as_bcl(z)?;
        Ok(self.decoder.forward(&z.to_dtype(self.dtype)?)?.to_dtype(DType::F32)?)
    }

    /// Декод тайлами: ядро `core_frames` кадров плюс `halo_frames` контекста с
    /// каждой стороны, наружу отдаётся только ядро. `on_progress(сделано,
    /// всего)` зовётся после каждого тайла; `cancel` опрашивается перед ним.
    pub fn decode_tiled(
        &self,
        z: &Tensor,
        core_frames: usize,
        halo_frames: usize,
        cancel: &dyn Fn() -> bool,
        on_progress: &dyn Fn(usize, usize),
    ) -> R<Tensor> {
        let z = self.as_bcl(z)?;
        let frames = z.dims()[2];
        let core = core_frames.max(1);
        let total = self.output_len(frames);
        let ratio = self.cfg.downsampling_ratio;
        let tiles = ceil_div(frames, core);
        if tiles <= 1 {
            let out = self.decode(&z)?;
            on_progress(1, 1);
            return Ok(out);
        }
        let mut cores: Vec<Tensor> = Vec::with_capacity(tiles);
        for (index, start) in (0..frames).step_by(core).enumerate() {
            if cancel() {
                return Err(YueError::Other(DECODE_CANCELLED.into()));
            }
            let end = (start + core).min(frames);
            let left = start.saturating_sub(halo_frames);
            let right = (end + halo_frames).min(frames);
            let tile = self.decode(&z.narrow(2, left, right - left)?.contiguous()?)?;
            let out_start = start * ratio;
            let out_end = (end * ratio).min(total);
            let crop_start = (start - left) * ratio;
            let want = out_end - out_start;
            if tile.dims()[2] < crop_start + want {
                return Err(YueError::Other(format!(
                    "тайл VAE короче своего ядра: {} < {}",
                    tile.dims()[2],
                    crop_start + want
                )));
            }
            cores.push(tile.narrow(2, crop_start, want)?.contiguous()?);
            on_progress(index + 1, tiles);
        }
        let refs: Vec<&Tensor> = cores.iter().collect();
        Ok(Tensor::cat(&refs, 2)?)
    }

    /// Латент по среднему апостериорного распределения (сэмплирование не
    /// нужно: генерации оно не участвует).
    pub fn encode_mean(&self, audio: &Tensor) -> R<Tensor> {
        let Some(encoder) = &self.encoder else {
            return Err(YueError::Other("энкодер не загружен (открыт decoder_only)".into()));
        };
        let x = if audio.rank() == 2 { audio.unsqueeze(0)? } else { audio.clone() };
        if x.rank() != 3 || x.dims()[1] != self.cfg.audio_channels {
            return Err(YueError::Other(format!(
                "ожидалось аудио [B,{},S], пришло {:?}",
                self.cfg.audio_channels,
                x.dims()
            )));
        }
        let x = x.to_dtype(self.dtype)?;
        let len = x.dims()[2];
        let hop = self.cfg.hop_length();
        let pad_to = ceil_div(len, hop) * hop;
        let x = if pad_to > len {
            let d = x.dims().to_vec();
            let z = Tensor::zeros(vec![d[0], d[1], pad_to - len], x.dtype(), x.device())?;
            Tensor::cat(&[&x, &z], 2)?
        } else {
            x
        };
        let h = encoder.forward(&x)?;
        Ok(h.narrow(1, 0, self.cfg.latent_dim)?.contiguous()?.to_dtype(DType::F32)?)
    }
}
