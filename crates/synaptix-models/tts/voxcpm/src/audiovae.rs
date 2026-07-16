use synaptix_core::{dtype::DType, tensor::Tensor};
use synaptix_ops::conv::{conv1d, conv_transpose1d, depthwise_conv};

use crate::loader::VoxCheckpoint;
use crate::VoxError;

fn load_wn(ck: &VoxCheckpoint, prefix: &str, bias: bool) -> Result<(Tensor, Option<Tensor>), VoxError> {
    let g = ck.vae(&format!("{prefix}.weight_g"))?;
    let v = ck.vae(&format!("{prefix}.weight_v"))?;
    let d = v.dims().to_vec();
    let (d0, d1, k) = (d[0], d[1], d[2]);
    let norm = v
        .reshape((d0, d1 * k))?
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .reshape((d0, 1usize, 1usize))?;
    let scale = g.div(&norm)?;
    let w = v.broadcast_mul(&scale)?;
    let b = if bias { Some(ck.vae(&format!("{prefix}.bias"))?) } else { None };
    Ok((w, b))
}

fn dilate_weight(w: &Tensor, dilation: usize) -> Result<Tensor, VoxError> {
    if dilation <= 1 {
        return Ok(w.clone());
    }
    let d = w.dims().to_vec();
    let (c, one, k) = (d[0], d[1], d[2]);
    let gap = Tensor::zeros(vec![c, one, dilation - 1], w.dtype(), w.device())?;
    let mut parts: Vec<Tensor> = Vec::with_capacity(k * 2);
    for ki in 0..k {
        parts.push(w.narrow(2, ki, 1)?.contiguous()?);
        if ki + 1 < k {
            parts.push(gap.clone());
        }
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&refs, 2)?)
}

fn left_pad(x: &Tensor, pad: usize) -> Result<Tensor, VoxError> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let d = x.dims();
    let z = Tensor::zeros(vec![d[0], d[1], pad], x.dtype(), x.device())?;
    Ok(Tensor::cat(&[&z, x], 2)?)
}

struct Snake {
    alpha: Tensor,
}

impl Snake {
    fn load(ck: &VoxCheckpoint, prefix: &str) -> Result<Self, VoxError> {
        Ok(Self { alpha: ck.vae(&format!("{prefix}.alpha"))? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, VoxError> {
        let ax = x.broadcast_mul(&self.alpha)?;
        let s = ax.sin()?.sqr()?;
        let inv = self.alpha.affine(1.0, 1e-9)?.recip()?;
        Ok(x.add(&s.broadcast_mul(&inv)?)?)
    }
}

enum ConvMode {
    Full,
    Depthwise,
}

struct WnConv {
    w: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    left: usize,
    mode: ConvMode,
}

impl WnConv {
    fn forward(&self, x: &Tensor) -> Result<Tensor, VoxError> {
        let xp = left_pad(x, self.left)?;
        let y = match self.mode {
            ConvMode::Full => conv1d(&xp, &self.w, self.bias.as_ref(), self.stride, 0)?,
            ConvMode::Depthwise => {
                let c = xp.dims()[1];
                depthwise_conv(&xp, &self.w, self.bias.as_ref(), self.stride, 0, c)?
            }
        };
        Ok(y)
    }
}

struct WnConvT {
    w: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    trim: usize,
}

impl WnConvT {
    fn forward(&self, x: &Tensor) -> Result<Tensor, VoxError> {
        let y = conv_transpose1d(x, &self.w, self.bias.as_ref(), self.stride, 0, 0, 1, 1)?;
        let len = y.dims()[2];
        Ok(y.narrow(2, 0, len - self.trim)?.contiguous()?)
    }
}

fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

struct ResidualUnit {
    snake1: Snake,
    conv1: WnConv,
    snake2: Snake,
    conv2: WnConv,
}

impl ResidualUnit {
    fn load(ck: &VoxCheckpoint, prefix: &str, dim: usize, dilation: usize) -> Result<Self, VoxError> {
        let (w1, b1) = load_wn(ck, &format!("{prefix}.block.1"), true)?;
        let w1 = dilate_weight(&w1, dilation)?;
        let (w2, b2) = load_wn(ck, &format!("{prefix}.block.3"), true)?;
        let _ = dim;
        Ok(Self {
            snake1: Snake::load(ck, &format!("{prefix}.block.0"))?,
            conv1: WnConv { w: w1, bias: b1, stride: 1, left: 6 * dilation, mode: ConvMode::Depthwise },
            snake2: Snake::load(ck, &format!("{prefix}.block.2"))?,
            conv2: WnConv { w: w2, bias: b2, stride: 1, left: 0, mode: ConvMode::Full },
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, VoxError> {
        let y = self.snake1.forward(x)?;
        let y = self.conv1.forward(&y)?;
        let y = self.snake2.forward(&y)?;
        let y = self.conv2.forward(&y)?;
        Ok(x.add(&y)?)
    }
}

struct EncoderBlock {
    res: Vec<ResidualUnit>,
    snake: Snake,
    down: WnConv,
}

impl EncoderBlock {
    fn load(ck: &VoxCheckpoint, prefix: &str, input_dim: usize, stride: usize) -> Result<Self, VoxError> {
        let mut res = Vec::with_capacity(3);
        for (j, dil) in [1usize, 3, 9].into_iter().enumerate() {
            res.push(ResidualUnit::load(ck, &format!("{prefix}.block.{j}"), input_dim, dil)?);
        }
        let (w, b) = load_wn(ck, &format!("{prefix}.block.4"), true)?;
        let left = 2 * ceil_div(stride, 2) - stride % 2;
        Ok(Self {
            res,
            snake: Snake::load(ck, &format!("{prefix}.block.3"))?,
            down: WnConv { w, bias: b, stride, left, mode: ConvMode::Full },
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, VoxError> {
        let mut h = x.clone();
        for r in &self.res {
            h = r.forward(&h)?;
        }
        h = self.snake.forward(&h)?;
        self.down.forward(&h)
    }
}

struct Encoder {
    first: WnConv,
    blocks: Vec<EncoderBlock>,
    fc_mu: WnConv,
}

impl Encoder {
    fn load(ck: &VoxCheckpoint) -> Result<Self, VoxError> {
        let cfg = ck.config.audio_vae_config.clone();
        let (w0, b0) = load_wn(ck, "encoder.block.0", true)?;
        let first = WnConv { w: w0, bias: b0, stride: 1, left: 6, mode: ConvMode::Full };

        let mut blocks = Vec::new();
        let mut d_model = cfg.encoder_dim;
        for (i, &stride) in cfg.encoder_rates.iter().enumerate() {
            let input_dim = d_model;
            d_model *= 2;
            blocks.push(EncoderBlock::load(
                ck,
                &format!("encoder.block.{}", i + 1),
                input_dim,
                stride,
            )?);
            let _ = input_dim;
        }
        let (wmu, bmu) = load_wn(ck, "encoder.fc_mu", true)?;
        let fc_mu = WnConv { w: wmu, bias: bmu, stride: 1, left: 2, mode: ConvMode::Full };
        Ok(Self { first, blocks, fc_mu })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, VoxError> {
        let mut h = self.first.forward(x)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        self.fc_mu.forward(&h)
    }
}

struct SrCond {
    scale: Tensor,
    bias: Tensor,
}

impl SrCond {
    fn load(ck: &VoxCheckpoint, prefix: &str, dim: usize, idx: usize) -> Result<Self, VoxError> {
        let se = ck.vae(&format!("{prefix}.scale_embed.weight"))?;
        let be = ck.vae(&format!("{prefix}.bias_embed.weight"))?;
        let scale = se.narrow(0, idx, 1)?.contiguous()?.reshape((1usize, dim, 1usize))?;
        let bias = be.narrow(0, idx, 1)?.contiguous()?.reshape((1usize, dim, 1usize))?;
        Ok(Self { scale, bias })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, VoxError> {
        Ok(x.broadcast_mul(&self.scale)?.broadcast_add(&self.bias)?)
    }
}

struct DecoderBlock {
    snake: Snake,
    up: WnConvT,
    res: Vec<ResidualUnit>,
}

impl DecoderBlock {
    fn load(ck: &VoxCheckpoint, prefix: &str, output_dim: usize, stride: usize) -> Result<Self, VoxError> {
        let (w, b) = load_wn(ck, &format!("{prefix}.block.1"), true)?;
        let trim = 2 * ceil_div(stride, 2) - stride % 2;
        let mut res = Vec::with_capacity(3);
        for (j, dil) in [1usize, 3, 9].into_iter().enumerate() {
            res.push(ResidualUnit::load(ck, &format!("{prefix}.block.{}", j + 2), output_dim, dil)?);
        }
        Ok(Self {
            snake: Snake::load(ck, &format!("{prefix}.block.0"))?,
            up: WnConvT { w, bias: b, stride, trim },
            res,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, VoxError> {
        let mut h = self.snake.forward(x)?;
        h = self.up.forward(&h)?;
        for r in &self.res {
            h = r.forward(&h)?;
        }
        Ok(h)
    }
}

struct Decoder {
    conv0: WnConv,
    conv1: WnConv,
    blocks: Vec<(SrCond, DecoderBlock)>,
    snake_final: Snake,
    conv_final: WnConv,
}

impl Decoder {
    fn load(ck: &VoxCheckpoint) -> Result<Self, VoxError> {
        let cfg = ck.config.audio_vae_config.clone();
        let idx = cfg.sr_bucket(cfg.out_sample_rate);

        let (w0, b0) = load_wn(ck, "decoder.model.0", true)?;
        let conv0 = WnConv { w: w0, bias: b0, stride: 1, left: 6, mode: ConvMode::Depthwise };
        let (w1, b1) = load_wn(ck, "decoder.model.1", true)?;
        let conv1 = WnConv { w: w1, bias: b1, stride: 1, left: 0, mode: ConvMode::Full };

        let channels = cfg.decoder_dim;
        let mut blocks = Vec::new();
        let mut model_idx = 2usize;
        let mut output_dim = channels;
        for (i, &stride) in cfg.decoder_rates.iter().enumerate() {
            let input_dim = channels >> i;
            output_dim = channels >> (i + 1);
            let sr = SrCond::load(ck, &format!("decoder.sr_cond_model.{model_idx}"), input_dim, idx)?;
            let blk = DecoderBlock::load(ck, &format!("decoder.model.{model_idx}"), output_dim, stride)?;
            blocks.push((sr, blk));
            model_idx += 1;
        }
        let snake_idx = model_idx;
        let conv_idx = model_idx + 1;
        let snake_final = Snake::load(ck, &format!("decoder.model.{snake_idx}"))?;
        let (wf, bf) = load_wn(ck, &format!("decoder.model.{conv_idx}"), true)?;
        let conv_final = WnConv { w: wf, bias: bf, stride: 1, left: 6, mode: ConvMode::Full };
        let _ = output_dim;
        Ok(Self { conv0, conv1, blocks, snake_final, conv_final })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor, VoxError> {
        let mut h = self.conv0.forward(x)?;
        h = self.conv1.forward(&h)?;
        for (sr, blk) in &self.blocks {
            h = sr.forward(&h)?;
            h = blk.forward(&h)?;
        }
        h = self.snake_final.forward(&h)?;
        h = self.conv_final.forward(&h)?;
        Ok(h.tanh()?)
    }
}

pub struct AudioVae {
    encoder: Encoder,
    decoder: Decoder,
    hop_length: usize,
}

impl AudioVae {
    pub fn load(ck: &VoxCheckpoint) -> Result<Self, VoxError> {
        Ok(Self {
            encoder: Encoder::load(ck)?,
            decoder: Decoder::load(ck)?,
            hop_length: ck.config.audio_vae_config.hop_length(),
        })
    }

    pub fn encode(&self, audio: &Tensor) -> Result<Tensor, VoxError> {
        let x = if audio.rank() == 2 {
            audio.unsqueeze(1)?
        } else {
            audio.clone()
        };
        let len = x.dims()[2];
        let pad_to = self.hop_length;
        let right = ceil_div(len, pad_to) * pad_to - len;
        let x = if right > 0 {
            let d = x.dims();
            let z = Tensor::zeros(vec![d[0], d[1], right], x.dtype(), x.device())?;
            Tensor::cat(&[&x, &z], 2)?
        } else {
            x
        };
        self.encoder.forward(&x.to_dtype(DType::F32)?)
    }

    pub fn decode(&self, z: &Tensor) -> Result<Tensor, VoxError> {
        self.decoder.forward(&z.to_dtype(DType::F32)?)
    }
}
