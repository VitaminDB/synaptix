
use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_ops::conv::{conv1d_dilated, conv_transpose1d};

use crate::config::VaeConfig;
use crate::loader::CompLoader;
use crate::AceError;

type R<T> = Result<T, AceError>;

fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

fn load_wn(ck: &CompLoader, prefix: &str, bias: bool) -> R<(Tensor, Option<Tensor>)> {
    let g = ck.f32(&format!("{prefix}.weight_g"))?;
    let v = ck.f32(&format!("{prefix}.weight_v"))?;
    let d = v.dims().to_vec();
    let (d0, d1, k) = (d[0], d[1], d[2]);
    let norm = v
        .reshape((d0, d1 * k))?
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .reshape((d0, 1usize, 1usize))?;
    let scale = g.broadcast_mul(&norm.recip()?)?;
    let w = v.broadcast_mul(&scale)?.to_dtype(DType::BF16)?;
    let b = if bias { Some(ck.f32(&format!("{prefix}.bias"))?.to_dtype(DType::BF16)?) } else { None };
    Ok((w, b))
}

struct Snake {
    alpha: Tensor,
    beta: Tensor,
}

impl Snake {
    fn load(ck: &CompLoader, prefix: &str) -> R<Self> {
        Ok(Self {
            alpha: ck.f32(&format!("{prefix}.alpha"))?.to_dtype(DType::BF16)?,
            beta: ck.f32(&format!("{prefix}.beta"))?.to_dtype(DType::BF16)?,
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        // Fused single-pass kernel: y = x + sin(exp(α)·x)²·1/(exp(β)+ε).
        // Replaces the 5-pass decomposition (mul/sin/sqr/mul/add); that
        // elementwise swarm dominated VAE decode once the convT copy-storm was
        // removed. Falls back to decomposed ops where the fused kernel is
        // unavailable (e.g. CPU backend).
        match x.snake(&self.alpha, &self.beta, 1e-9) {
            Ok(y) => Ok(y),
            Err(synaptix_core::error::SynaptixError::Unsupported(_)) => {
                let a = self.alpha.exp()?;
                let b = self.beta.exp()?;
                let ax = x.broadcast_mul(&a)?;
                let s = ax.sin()?.sqr()?;
                let inv = b.affine(1.0, 1e-9)?.recip()?;
                Ok(x.broadcast_add(&s.broadcast_mul(&inv)?)?)
            }
            Err(e) => Err(e.into()),
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
}

struct ResidualUnit {
    snake1: Snake,
    conv1: Conv,
    snake2: Snake,
    conv2: Conv,
}

impl ResidualUnit {
    fn load(ck: &CompLoader, prefix: &str, dilation: usize) -> R<Self> {
        let (w1, b1) = load_wn(ck, &format!("{prefix}.conv1"), true)?;
        let (w2, b2) = load_wn(ck, &format!("{prefix}.conv2"), true)?;
        Ok(Self {
            snake1: Snake::load(ck, &format!("{prefix}.snake1"))?,
            conv1: Conv { w: w1, bias: b1, stride: 1, pad: 3 * dilation, dilation },
            snake2: Snake::load(ck, &format!("{prefix}.snake2"))?,
            conv2: Conv { w: w2, bias: b2, stride: 1, pad: 0, dilation: 1 },
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let y = self.conv1.forward(&self.snake1.forward(x)?)?;
        let y = self.conv2.forward(&self.snake2.forward(&y)?)?;
        Ok(x.broadcast_add(&y)?)
    }
}

struct EncoderBlock {
    res: [ResidualUnit; 3],
    snake1: Snake,
    down: Conv,
}

impl EncoderBlock {
    fn load(ck: &CompLoader, prefix: &str, stride: usize) -> R<Self> {
        let (w, b) = load_wn(ck, &format!("{prefix}.conv1"), true)?;
        Ok(Self {
            res: [
                ResidualUnit::load(ck, &format!("{prefix}.res_unit1"), 1)?,
                ResidualUnit::load(ck, &format!("{prefix}.res_unit2"), 3)?,
                ResidualUnit::load(ck, &format!("{prefix}.res_unit3"), 9)?,
            ],
            snake1: Snake::load(ck, &format!("{prefix}.snake1"))?,
            down: Conv { w, bias: b, stride, pad: ceil_div(stride, 2), dilation: 1 },
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut h = x.clone();
        for r in &self.res {
            h = r.forward(&h)?;
        }
        h = self.snake1.forward(&h)?;
        self.down.forward(&h)
    }
}

struct DecoderBlock {
    snake1: Snake,
    up: ConvT,
    res: [ResidualUnit; 3],
}

impl DecoderBlock {
    fn load(ck: &CompLoader, prefix: &str, stride: usize) -> R<Self> {
        let (w, b) = load_wn(ck, &format!("{prefix}.conv_t1"), true)?;
        Ok(Self {
            snake1: Snake::load(ck, &format!("{prefix}.snake1"))?,
            up: ConvT { w, bias: b, stride, pad: ceil_div(stride, 2) },
            res: [
                ResidualUnit::load(ck, &format!("{prefix}.res_unit1"), 1)?,
                ResidualUnit::load(ck, &format!("{prefix}.res_unit2"), 3)?,
                ResidualUnit::load(ck, &format!("{prefix}.res_unit3"), 9)?,
            ],
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut h = self.snake1.forward(x)?;
        h = self.up.forward(&h)?;
        for r in &self.res {
            h = r.forward(&h)?;
        }
        Ok(h)
    }
}

struct Encoder {
    conv1: Conv,
    blocks: Vec<EncoderBlock>,
    snake1: Snake,
    conv2: Conv,
}

impl Encoder {
    fn load(ck: &CompLoader, cfg: &VaeConfig) -> R<Self> {
        let (w1, b1) = load_wn(ck, "encoder.conv1", true)?;
        let mut blocks = Vec::with_capacity(cfg.downsampling_ratios.len());
        for (i, &stride) in cfg.downsampling_ratios.iter().enumerate() {
            blocks.push(EncoderBlock::load(ck, &format!("encoder.block.{i}"), stride)?);
        }
        let (w2, b2) = load_wn(ck, "encoder.conv2", true)?;
        Ok(Self {
            conv1: Conv { w: w1, bias: b1, stride: 1, pad: 3, dilation: 1 },
            blocks,
            snake1: Snake::load(ck, "encoder.snake1")?,
            conv2: Conv { w: w2, bias: b2, stride: 1, pad: 1, dilation: 1 },
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut h = self.conv1.forward(x)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        h = self.snake1.forward(&h)?;
        self.conv2.forward(&h)
    }
}

struct Decoder {
    conv1: Conv,
    blocks: Vec<DecoderBlock>,
    snake1: Snake,
    conv2: Conv,
}

impl Decoder {
    fn load(ck: &CompLoader, cfg: &VaeConfig) -> R<Self> {
        let (w1, b1) = load_wn(ck, "decoder.conv1", true)?;
        let mut blocks = Vec::with_capacity(cfg.downsampling_ratios.len());
        let up: Vec<usize> = cfg.downsampling_ratios.iter().rev().copied().collect();
        for (i, &stride) in up.iter().enumerate() {
            blocks.push(DecoderBlock::load(ck, &format!("decoder.block.{i}"), stride)?);
        }
        let (w2, b2) = load_wn(ck, "decoder.conv2", false)?;
        Ok(Self {
            conv1: Conv { w: w1, bias: b1, stride: 1, pad: 3, dilation: 1 },
            blocks,
            snake1: Snake::load(ck, "decoder.snake1")?,
            conv2: Conv { w: w2, bias: b2, stride: 1, pad: 3, dilation: 1 },
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut h = self.conv1.forward(x)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        h = self.snake1.forward(&h)?;
        self.conv2.forward(&h)
    }
}

pub struct AceStepVae {
    encoder: Encoder,
    decoder: Decoder,
    cfg: VaeConfig,
}

impl AceStepVae {
    pub fn open(path: impl AsRef<Path>, device: Device) -> R<Self> {
        let ck = CompLoader::open(path, None, device)?;
        let cfg = VaeConfig::default();
        Ok(Self {
            encoder: Encoder::load(&ck, &cfg)?,
            decoder: Decoder::load(&ck, &cfg)?,
            cfg,
        })
    }

    pub fn config(&self) -> &VaeConfig {
        &self.cfg
    }

    fn as_bcl(audio: &Tensor) -> R<Tensor> {
        Ok(if audio.rank() == 2 { audio.unsqueeze(0)? } else { audio.clone() })
    }

    pub fn encode_mean(&self, audio: &Tensor) -> R<Tensor> {
        let x = Self::as_bcl(audio)?.to_dtype(DType::BF16)?;
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
        let h = self.encoder.forward(&x)?;
        let lat = self.cfg.decoder_input_channels;
        Ok(h.narrow(1, 0, lat)?.contiguous()?.to_dtype(DType::F32)?)
    }

    pub fn decode(&self, z: &Tensor) -> R<Tensor> {
        let z = if z.rank() == 2 { z.unsqueeze(0)? } else { z.clone() };
        Ok(self.decoder.forward(&z.to_dtype(DType::BF16)?)?.to_dtype(DType::F32)?)
    }

    pub fn decode_tiled(&self, z: &Tensor, chunk_frames: usize, overlap_frames: usize) -> R<Tensor> {
        let z = if z.rank() == 2 { z.unsqueeze(0)? } else { z.clone() };
        let t = z.dims()[2];
        if t <= chunk_frames {
            return self.decode(&z);
        }
        let mut overlap = overlap_frames;
        while chunk_frames <= 2 * overlap && overlap > 4 {
            overlap /= 2;
        }
        let stride = chunk_frames.saturating_sub(2 * overlap).max(1);
        let mut cores: Vec<Tensor> = Vec::new();
        let mut core_start = 0usize;
        while core_start < t {
            let core_end = (core_start + stride).min(t);
            let win_start = core_start.saturating_sub(overlap);
            let win_end = (core_end + overlap).min(t);
            let chunk = z.narrow(2, win_start, win_end - win_start)?.contiguous()?;
            let audio = self.decode(&chunk)?;
            let up = audio.dims()[2] / (win_end - win_start);
            let trim_start = (core_start - win_start) * up;
            let core_len = (core_end - core_start) * up;
            cores.push(audio.narrow(2, trim_start, core_len)?.contiguous()?);
            core_start += stride;
        }
        let refs: Vec<&Tensor> = cores.iter().collect();
        Ok(Tensor::cat(&refs, 2)?)
    }
}
