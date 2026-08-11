use std::f64::consts::PI;

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::conv1d::conv1d_dilated;

use crate::config::AudioVaeConfig;
use crate::loader::ComponentLoader;
use crate::H3Error;

type R<T> = Result<T, SynaptixError>;

const FILTER_KERNEL: usize = 12;
const RESAMPLE_RATIO: usize = 2;
const SNAKE_EPS: f32 = 1e-9;

fn kaiser_sinc_filter1d(cutoff: f64, half_width: f64, kernel_size: usize) -> Vec<f32> {
    let even = kernel_size % 2 == 0;
    let half_size = kernel_size / 2;
    let delta_f = 4.0 * half_width;
    let a = 2.285 * (half_size as f64 - 1.0) * PI * delta_f + 7.95;
    let beta = if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    };
    let win = kaiser_window(kernel_size, beta);
    let mut f = vec![0f64; kernel_size];
    let mut sum = 0f64;
    for i in 0..kernel_size {
        let time = if even {
            (i as f64) - (half_size as f64) + 0.5
        } else {
            (i as f64) - (half_size as f64)
        };
        let arg = 2.0 * cutoff * time;
        let sinc = if arg.abs() < 1e-12 { 1.0 } else { (PI * arg).sin() / (PI * arg) };
        f[i] = 2.0 * cutoff * win[i] * sinc;
        sum += f[i];
    }
    f.iter().map(|v| (v / sum) as f32).collect()
}

fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    let denom = bessel_i0(beta);
    (0..n)
        .map(|i| {
            let r = 2.0 * i as f64 / (n as f64 - 1.0) - 1.0;
            bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denom
        })
        .collect()
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let hx = x / 2.0;
    for k in 1..64 {
        term *= (hx / k as f64) * (hx / k as f64);
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    sum
}

fn pad_replicate(x: &Tensor, left: usize, right: usize) -> R<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let l = x.dims()[2];
    let first = x.narrow(2, 0, 1)?.contiguous()?;
    let last = x.narrow(2, l - 1, 1)?.contiguous()?;
    let lefts: Vec<Tensor> = (0..left).map(|_| first.clone()).collect();
    let rights: Vec<Tensor> = (0..right).map(|_| last.clone()).collect();
    let xc = x.contiguous()?;
    let mut parts: Vec<&Tensor> = Vec::with_capacity(left + 1 + right);
    parts.extend(lefts.iter());
    parts.push(&xc);
    parts.extend(rights.iter());
    Tensor::cat(&parts, 2)
}

struct Resampler {
    up_filter: Tensor,
    down_filter: Tensor,
    up_pad: usize,
    up_pad_left: usize,
    up_pad_right: usize,
    down_pad_left: usize,
    down_pad_right: usize,
}

impl Resampler {
    fn new(channels: usize, device: Device, dtype: DType) -> R<Self> {
        let k = FILTER_KERNEL;
        let r = RESAMPLE_RATIO;
        let up = kaiser_sinc_filter1d(0.5 / r as f64, 0.6 / r as f64, k);
        let down = kaiser_sinc_filter1d(0.5 / r as f64, 0.6 / r as f64, k);
        let expand = |f: Vec<f32>| -> R<Tensor> {
            let mut v = Vec::with_capacity(channels * k);
            for _ in 0..channels {
                v.extend_from_slice(&f);
            }
            Tensor::from_vec(v, vec![channels, 1, k], device)?.to_dtype(dtype)
        };
        let pad = k / r - 1;
        Ok(Self {
            up_filter: expand(up)?,
            down_filter: expand(down)?,
            up_pad: pad,
            up_pad_left: pad * r + (k - r) / 2,
            up_pad_right: pad * r + (k - r + 1) / 2,
            down_pad_left: k / 2 - usize::from(k % 2 == 0),
            down_pad_right: k / 2,
        })
    }

    fn upsample(&self, x: &Tensor) -> R<Tensor> {
        let x = pad_replicate(x, self.up_pad, self.up_pad)?;
        let c = x.dims()[1];
        let y = match x.dwconv1d(&self.up_filter, None, RESAMPLE_RATIO, 0, true) {
            Ok(v) => v,
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                synaptix_ops::conv::conv_transpose1d::conv_transpose1d(
                    &x,
                    &self.up_filter,
                    None,
                    RESAMPLE_RATIO,
                    0,
                    0,
                    c,
                    1,
                )?
            }
            Err(e) => return Err(e),
        };
        let y = y.mul_scalar(RESAMPLE_RATIO as f32)?;
        let n = y.dims()[2];
        let keep = n - self.up_pad_left - self.up_pad_right;
        y.narrow(2, self.up_pad_left, keep)?.contiguous()
    }

    fn downsample(&self, x: &Tensor) -> R<Tensor> {
        let x = pad_replicate(x, self.down_pad_left, self.down_pad_right)?;
        let c = x.dims()[1];
        synaptix_ops::conv::depthwise::depthwise_conv(
            &x,
            &self.down_filter,
            None,
            RESAMPLE_RATIO,
            0,
            c,
        )
    }
}

struct SnakeBetaAct {
    alpha: Tensor,
    beta: Tensor,
    resampler: Resampler,
}

impl SnakeBetaAct {
    fn load(
        w: &ComponentLoader,
        prefix: &str,
        device: Device,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let alpha = w.get_as(&format!("{prefix}.act.alpha"), DType::F32)?;
        let beta = w.get_as(&format!("{prefix}.act.beta"), DType::F32)?;
        let channels = alpha.dims().iter().product();
        Ok(Self {
            alpha,
            beta,
            resampler: Resampler::new(channels, device, dtype)?,
        })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let up = self.resampler.upsample(x)?;
        let act = match up.snake(&self.alpha, &self.beta, SNAKE_EPS) {
            Ok(y) => y,
            Err(_) => snake_decomposed_log(&up, &self.alpha, &self.beta)?,
        };
        self.resampler.downsample(&act)
    }
}

fn snake_decomposed_log(x: &Tensor, alpha_log: &Tensor, beta_log: &Tensor) -> R<Tensor> {
    let c = x.dims()[1];
    let a = alpha_log.to_dtype(DType::F32)?.exp()?.reshape(vec![1, c, 1])?.to_dtype(x.dtype())?;
    let b = beta_log.to_dtype(DType::F32)?.exp()?.reshape(vec![1, c, 1])?.to_dtype(x.dtype())?;
    snake_raw(x, &a, &b)
}

fn snake_raw(x: &Tensor, alpha: &Tensor, beta: &Tensor) -> R<Tensor> {
    let s = x.broadcast_mul(alpha)?.sin()?;
    let sq = s.mul(&s)?;
    let inv = beta.add_scalar(SNAKE_EPS)?.recip()?;
    x.add(&sq.broadcast_mul(&inv)?)
}

struct Snake1d {
    alpha: Tensor,
}

impl Snake1d {
    fn load(w: &ComponentLoader, prefix: &str, dtype: DType) -> Result<Self, H3Error> {
        let a = w.get_as(&format!("{prefix}.alpha"), DType::F32)?;
        let c: usize = a.dims().iter().product();
        Ok(Self { alpha: a.reshape(vec![1, c, 1])?.to_dtype(dtype)? })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        snake_raw(x, &self.alpha, &self.alpha)
    }
}

fn load_weight_norm(
    w: &ComponentLoader,
    prefix: &str,
    bias: bool,
    dtype: DType,
) -> Result<(Tensor, Option<Tensor>), H3Error> {
    let g = w.get_as(&format!("{prefix}.weight_g"), DType::F32)?;
    let v = w.get_as(&format!("{prefix}.weight_v"), DType::F32)?;
    let d = v.dims().to_vec();
    let norm = v
        .reshape(vec![d[0], d[1] * d[2]])?
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .reshape(vec![d[0], 1, 1])?;
    let scale = g.broadcast_mul(&norm.recip()?)?;
    let weight = v.broadcast_mul(&scale)?.to_dtype(dtype)?;
    let b = if bias {
        Some(w.get_as(&format!("{prefix}.bias"), dtype)?)
    } else {
        None
    };
    Ok((weight, b))
}

struct Conv1dW {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv1dW {
    fn load_plain(
        w: &ComponentLoader,
        prefix: &str,
        bias: bool,
        stride: usize,
        padding: usize,
        dilation: usize,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        Ok(Self {
            weight: w.get_as(&format!("{prefix}.weight"), dtype)?,
            bias: if bias { Some(w.get_as(&format!("{prefix}.bias"), dtype)?) } else { None },
            stride,
            padding,
            dilation,
        })
    }

    fn load_wn(
        w: &ComponentLoader,
        prefix: &str,
        bias: bool,
        stride: usize,
        padding: usize,
        dilation: usize,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let (weight, b) = load_weight_norm(w, prefix, bias, dtype)?;
        Ok(Self { weight, bias: b, stride, padding, dilation })
    }

    fn load_any(
        w: &ComponentLoader,
        prefix: &str,
        bias: bool,
        stride: usize,
        padding: usize,
        dilation: usize,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        if w.contains(&format!("{prefix}.weight_v")) {
            Self::load_wn(w, prefix, bias, stride, padding, dilation, dtype)
        } else {
            Self::load_plain(w, prefix, bias, stride, padding, dilation, dtype)
        }
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        conv1d_dilated(
            x,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}

struct ConvT1dW {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl ConvT1dW {
    fn load(
        w: &ComponentLoader,
        prefix: &str,
        stride: usize,
        padding: usize,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let (weight, bias) = load_weight_norm(w, prefix, true, dtype)?;
        Ok(Self { weight, bias, stride, padding })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        synaptix_ops::conv::conv_transpose1d::conv_transpose1d(
            x,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            0,
            1,
            1,
        )
    }
}

struct ResidualUnit {
    snake1: Snake1d,
    conv1: Conv1dW,
    snake2: Snake1d,
    conv2: Conv1dW,
}

impl ResidualUnit {
    fn load(
        w: &ComponentLoader,
        prefix: &str,
        dilation: usize,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let pad = ((7 - 1) * dilation) / 2;
        Ok(Self {
            snake1: Snake1d::load(w, &format!("{prefix}.block.0"), dtype)?,
            conv1: Conv1dW::load_any(w, &format!("{prefix}.block.1"), true, 1, pad, dilation, dtype)?,
            snake2: Snake1d::load(w, &format!("{prefix}.block.2"), dtype)?,
            conv2: Conv1dW::load_any(w, &format!("{prefix}.block.3"), true, 1, 0, 1, dtype)?,
        })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let y = self.conv1.forward(&self.snake1.forward(x)?)?;
        let y = self.conv2.forward(&self.snake2.forward(&y)?)?;
        let lx = x.dims()[2];
        let ly = y.dims()[2];
        let pad = (lx.saturating_sub(ly)) / 2;
        let xs = if pad > 0 { x.narrow(2, pad, ly)?.contiguous()? } else { x.clone() };
        y.add(&xs)
    }
}

struct EncoderBlock {
    units: Vec<ResidualUnit>,
    snake: Snake1d,
    conv: Conv1dW,
}

impl EncoderBlock {
    fn load(
        w: &ComponentLoader,
        prefix: &str,
        stride: usize,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let mut units = Vec::with_capacity(3);
        for (i, d) in [1usize, 3, 9].iter().enumerate() {
            units.push(ResidualUnit::load(w, &format!("{prefix}.block.{i}"), *d, dtype)?);
        }
        Ok(Self {
            units,
            snake: Snake1d::load(w, &format!("{prefix}.block.3"), dtype)?,
            conv: Conv1dW::load_any(
                w,
                &format!("{prefix}.block.4"),
                true,
                stride,
                stride.div_ceil(2),
                1,
                dtype,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut h = x.clone();
        for u in &self.units {
            h = u.forward(&h)?;
        }
        self.conv.forward(&self.snake.forward(&h)?)
    }
}

pub struct DacEncoder {
    conv_in: Conv1dW,
    blocks: Vec<EncoderBlock>,
    snake_out: Snake1d,
    conv_out: Conv1dW,
}

impl DacEncoder {
    pub fn load(
        w: &ComponentLoader,
        cfg: &AudioVaeConfig,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let n = cfg.encoder_rates.len();
        let mut blocks = Vec::with_capacity(n);
        for (i, s) in cfg.encoder_rates.iter().enumerate() {
            blocks.push(EncoderBlock::load(w, &format!("encoder.block.{}", i + 1), *s, dtype)?);
        }
        Ok(Self {
            conv_in: Conv1dW::load_any(w, "encoder.block.0", true, 1, 3, 1, dtype)?,
            blocks,
            snake_out: Snake1d::load(w, &format!("encoder.block.{}", n + 1), dtype)?,
            conv_out: Conv1dW::load_any(
                w,
                &format!("encoder.block.{}", n + 2),
                true,
                1,
                1,
                1,
                dtype,
            )?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut h = self.conv_in.forward(x)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        self.conv_out.forward(&self.snake_out.forward(&h)?)
    }
}

struct LayerNormW {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNormW {
    fn load(w: &ComponentLoader, prefix: &str, dtype: DType) -> Result<Self, H3Error> {
        Ok(Self {
            weight: w.get_as(&format!("{prefix}.weight"), dtype)?,
            bias: w.get_as(&format!("{prefix}.bias"), dtype)?,
            eps: 1e-5,
        })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        if let Ok(y) = x.layer_norm_fused(&self.weight, Some(&self.bias), self.eps) {
            return Ok(y);
        }
        synaptix_ops::norm::layer_norm::layer_norm(x, Some(&self.weight), Some(&self.bias), self.eps)
    }
}

struct LinearW {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl LinearW {
    fn load(w: &ComponentLoader, prefix: &str, bias: bool, dtype: DType) -> Result<Self, H3Error> {
        Ok(Self {
            weight: w.get_as(&format!("{prefix}.weight"), dtype)?,
            bias: if bias { Some(w.get_as(&format!("{prefix}.bias"), dtype)?) } else { None },
        })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let y = x.matmul(&self.weight.transpose(0, 1)?.contiguous()?)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

struct CausalAttention {
    qkv: LinearW,
    qkv_bias: Tensor,
    proj: LinearW,
    heads: usize,
    head_dim: usize,
    out_dim: usize,
    scale: f32,
}

impl CausalAttention {
    fn load(
        w: &ComponentLoader,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        heads: usize,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let q_bias = w.get_as(&format!("{prefix}.q_bias"), dtype)?;
        let k_bias = w.get_as(&format!("{prefix}.zero_k_bias"), dtype)?;
        let v_bias = w.get_as(&format!("{prefix}.v_bias"), dtype)?;
        let qkv_bias = Tensor::cat(&[&q_bias, &k_bias, &v_bias], 0)?;
        let head_dim = in_dim / heads;
        Ok(Self {
            qkv: LinearW::load(w, &format!("{prefix}.qkv"), false, dtype)?,
            qkv_bias,
            proj: LinearW::load(w, &format!("{prefix}.proj"), true, dtype)?,
            heads,
            head_dim,
            out_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let n = x.dims()[0];
        let qkv = self.qkv.forward(x)?.broadcast_add(&self.qkv_bias)?;
        let qkv = qkv.reshape(vec![n, 3, self.heads, self.head_dim])?;
        let take = |i: usize| -> R<Tensor> {
            qkv.narrow(1, i, 1)?
                .reshape(vec![n, self.heads, self.head_dim])?
                .transpose(0, 1)?
                .contiguous()?
                .reshape(vec![1, self.heads, n, self.head_dim])
        };
        let q = take(0)?;
        let k = take(1)?;
        let v = take(2)?;
        let attn = match q.dtype() {
            DType::BF16 | DType::F16 => q
                .flash_attention(&k, &v, self.scale, true)
                .or_else(|_| scaled_dot_attention(&q, &k, &v, self.scale, Some(&causal_mask(n, q.dtype(), q.device())?)))?,
            _ => scaled_dot_attention(
                &q,
                &k,
                &v,
                self.scale,
                Some(&causal_mask(n, q.dtype(), q.device())?),
            )?,
        };
        let mean = attn
            .reshape(vec![self.heads, n, self.head_dim])?
            .mean_keepdim(0)?
            .reshape(vec![n, self.head_dim])?;
        let pooled = adaptive_avg_pool_last(&mean, self.out_dim)?;
        self.proj.forward(&pooled)
    }
}

fn causal_mask(n: usize, dtype: DType, device: Device) -> R<Tensor> {
    let mut v = vec![0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            v[i * n + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(v, vec![1, 1, n, n], device)?.to_dtype(dtype)
}

fn adaptive_avg_pool_last(x: &Tensor, out_dim: usize) -> R<Tensor> {
    let d = x.dims().to_vec();
    let n = d[0];
    let in_dim = d[1];
    if in_dim == out_dim {
        return Ok(x.clone());
    }
    if in_dim % out_dim != 0 {
        return Err(SynaptixError::Unsupported("adaptive_avg_pool: in_dim % out_dim != 0"));
    }
    let group = in_dim / out_dim;
    x.reshape(vec![n, out_dim, group])?.mean_keepdim(2)?.reshape(vec![n, out_dim])
}

struct GeGluMlp {
    norm: LayerNormW,
    w0: LinearW,
    w1: LinearW,
    w2: LinearW,
}

impl GeGluMlp {
    fn load(w: &ComponentLoader, prefix: &str, dtype: DType) -> Result<Self, H3Error> {
        Ok(Self {
            norm: LayerNormW::load(w, &format!("{prefix}.norm"), dtype)?,
            w0: LinearW::load(w, &format!("{prefix}.w0"), true, dtype)?,
            w1: LinearW::load(w, &format!("{prefix}.w1"), true, dtype)?,
            w2: LinearW::load(w, &format!("{prefix}.w2"), true, dtype)?,
        })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let h = self.norm.forward(x)?;
        let a = self.w0.forward(&h)?.gelu_tanh()?;
        let b = self.w1.forward(&h)?;
        self.w2.forward(&a.mul(&b)?)
    }
}

pub struct AttnProjection {
    norm1: LayerNormW,
    norm2: LayerNormW,
    norm3: LayerNormW,
    attn: CausalAttention,
    proj: LinearW,
    mlp: GeGluMlp,
}

impl AttnProjection {
    pub fn load(
        w: &ComponentLoader,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        heads: usize,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        Ok(Self {
            norm1: LayerNormW::load(w, &format!("{prefix}.norm1"), dtype)?,
            norm2: LayerNormW::load(w, &format!("{prefix}.norm2"), dtype)?,
            norm3: LayerNormW::load(w, &format!("{prefix}.norm3"), dtype)?,
            attn: CausalAttention::load(w, &format!("{prefix}.attn"), in_dim, out_dim, heads, dtype)?,
            proj: LinearW::load(w, &format!("{prefix}.proj"), true, dtype)?,
            mlp: GeGluMlp::load(w, &format!("{prefix}.mlp"), dtype)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> R<Tensor> {
        let a = self.attn.forward(&self.norm1.forward(x)?)?;
        let p = self.proj.forward(&self.norm3.forward(x)?)?;
        let y = p.add(&a)?;
        let m = self.mlp.forward(&self.norm2.forward(&y)?)?;
        y.add(&m)
    }
}

struct AmpBlock {
    convs1: Vec<Conv1dW>,
    convs2: Vec<Conv1dW>,
    acts: Vec<SnakeBetaAct>,
}

impl AmpBlock {
    fn load(
        w: &ComponentLoader,
        prefix: &str,
        kernel: usize,
        dilations: &[usize],
        device: Device,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let mut convs1 = Vec::with_capacity(dilations.len());
        let mut convs2 = Vec::with_capacity(dilations.len());
        for (i, d) in dilations.iter().enumerate() {
            convs1.push(Conv1dW::load_wn(
                w,
                &format!("{prefix}.convs1.{i}"),
                true,
                1,
                (kernel * d - d) / 2,
                *d,
                dtype,
            )?);
            convs2.push(Conv1dW::load_wn(
                w,
                &format!("{prefix}.convs2.{i}"),
                true,
                1,
                (kernel - 1) / 2,
                1,
                dtype,
            )?);
        }
        let mut acts = Vec::with_capacity(dilations.len() * 2);
        for i in 0..dilations.len() * 2 {
            acts.push(SnakeBetaAct::load(
                w,
                &format!("{prefix}.activations.{i}"),
                device,
                dtype,
            )?);
        }
        Ok(Self { convs1, convs2, acts })
    }

    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut h = x.clone();
        for i in 0..self.convs1.len() {
            let a1 = &self.acts[i * 2];
            let a2 = &self.acts[i * 2 + 1];
            let t = self.convs1[i].forward(&a1.forward(&h)?)?;
            let t = self.convs2[i].forward(&a2.forward(&t)?)?;
            h = t.add(&h)?;
        }
        Ok(h)
    }
}

pub struct BigVgan {
    conv_pre: Conv1dW,
    ups: Vec<ConvT1dW>,
    resblocks: Vec<AmpBlock>,
    act_post: SnakeBetaAct,
    conv_post: Conv1dW,
    num_kernels: usize,
}

impl BigVgan {
    pub fn load(
        w: &ComponentLoader,
        cfg: &AudioVaeConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let mut ups = Vec::with_capacity(cfg.decoder_rates.len());
        for (i, (u, k)) in cfg
            .decoder_rates
            .iter()
            .zip(cfg.decoder_kernel_sizes.iter())
            .enumerate()
        {
            ups.push(ConvT1dW::load(
                w,
                &format!("decoder.ups.{i}.0"),
                *u,
                (k - u) / 2,
                dtype,
            )?);
        }
        let num_kernels = cfg.resblock_kernel_sizes.len();
        let mut resblocks = Vec::with_capacity(ups.len() * num_kernels);
        for i in 0..ups.len() {
            for (j, k) in cfg.resblock_kernel_sizes.iter().enumerate() {
                resblocks.push(AmpBlock::load(
                    w,
                    &format!("decoder.resblocks.{}", i * num_kernels + j),
                    *k,
                    &cfg.resblock_dilation_sizes[j],
                    device,
                    dtype,
                )?);
            }
        }
        Ok(Self {
            conv_pre: Conv1dW::load_wn(w, "decoder.conv_pre", true, 1, 3, 1, dtype)?,
            ups,
            resblocks,
            act_post: SnakeBetaAct::load(w, "decoder.activation_post", device, dtype)?,
            conv_post: Conv1dW::load_wn(w, "decoder.conv_post", false, 1, 3, 1, dtype)?,
            num_kernels,
        })
    }

    pub fn forward(&self, x: &Tensor) -> R<Tensor> {
        let mut h = self.conv_pre.forward(x)?;
        for i in 0..self.ups.len() {
            h = self.ups[i].forward(&h)?;
            let mut acc: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let y = self.resblocks[i * self.num_kernels + j].forward(&h)?;
                acc = Some(match acc {
                    Some(a) => a.add(&y)?,
                    None => y,
                });
            }
            h = acc.unwrap().mul_scalar(1.0 / self.num_kernels as f32)?;
        }
        let h = self.act_post.forward(&h)?;
        self.conv_post.forward(&h)?.clamp(-1.0, 1.0)
    }
}

pub struct AudioVae {
    pub cfg: AudioVaeConfig,
    encoder: Option<DacEncoder>,
    pre_block: Option<AttnProjection>,
    mean_proj: Option<Conv1dW>,
    dec_in_proj: Conv1dW,
    decoder: BigVgan,
    latents_mean: Tensor,
    latents_std: Tensor,
    dtype: DType,
}

impl AudioVae {
    pub fn load_decoder(
        w: &ComponentLoader,
        cfg: AudioVaeConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let c = cfg.vae_latent_channels;
        let latents_mean = Tensor::from_vec(cfg.latents_mean.clone(), vec![c], device)?;
        let latents_std = Tensor::from_vec(cfg.latents_std.clone(), vec![c], device)?;
        Ok(Self {
            dec_in_proj: Conv1dW::load_any(w, "dec_in_proj", true, 1, 0, 1, dtype)?,
            decoder: BigVgan::load(w, &cfg, device, dtype)?,
            encoder: None,
            pre_block: None,
            mean_proj: None,
            latents_mean,
            latents_std,
            cfg,
            dtype,
        })
    }

    pub fn load_full(
        w: &ComponentLoader,
        cfg: AudioVaeConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self, H3Error> {
        let mut me = Self::load_decoder(w, cfg, device, dtype)?;
        me.encoder = Some(DacEncoder::load(w, &me.cfg, dtype)?);
        me.pre_block = Some(AttnProjection::load(
            w,
            "pre_block",
            me.cfg.latent_dim,
            me.cfg.vae_latent_channels,
            me.cfg.num_attention_heads,
            dtype,
        )?);
        me.mean_proj = Some(Conv1dW::load_any(w, "mean_proj", true, 1, 0, 1, dtype)?);
        Ok(me)
    }

    pub fn sample_rate(&self) -> usize {
        self.cfg.sampling_rate
    }

    pub fn samples_per_latent(&self) -> usize {
        self.cfg.hop_length()
    }

    fn stats(&self, dtype: DType) -> R<(Tensor, Tensor)> {
        let c = self.cfg.vae_latent_channels;
        Ok((
            self.latents_mean.reshape(vec![1, c, 1])?.to_dtype(dtype)?,
            self.latents_std.reshape(vec![1, c, 1])?.to_dtype(dtype)?,
        ))
    }

    pub fn decode(&self, latent: &Tensor) -> Result<Tensor, H3Error> {
        let d = latent.dims().to_vec();
        if d.len() != 4 {
            return Err(H3Error::Layout("audio decode: ожидался [B,C,S,T]".into()));
        }
        let (b, c, s, t) = (d[0], d[1], d[2], d[3]);
        let z = latent
            .permute([0, 2, 1, 3])?
            .contiguous()?
            .reshape(vec![b * s, c, t])?
            .to_dtype(self.dtype)?;
        let (m, sd) = self.stats(self.dtype)?;
        let z = z.broadcast_mul(&sd)?.broadcast_add(&m)?;
        let x = self.dec_in_proj.forward(&z)?;
        let wave = self.decoder.forward(&x)?;
        let l = wave.dims()[2];
        Ok(wave.reshape(vec![b, s, l])?.to_dtype(DType::F32)?)
    }

    pub fn encode(&self, waveform: &Tensor) -> Result<Tensor, H3Error> {
        let enc = self
            .encoder
            .as_ref()
            .ok_or_else(|| H3Error::Load("audio VAE загружен без энкодера".into()))?;
        let pre = self.pre_block.as_ref().unwrap();
        let mean_proj = self.mean_proj.as_ref().unwrap();

        let d = waveform.dims().to_vec();
        if d.len() != 3 {
            return Err(H3Error::Layout("audio encode: ожидался [B,S,L]".into()));
        }
        let (b, s, l) = (d[0], d[1], d[2]);
        let hop = self.cfg.hop_length();
        let pad = l.div_ceil(hop) * hop - l;
        let x = if pad > 0 {
            let z = Tensor::zeros(vec![b, s, pad], waveform.dtype(), waveform.device())?;
            Tensor::cat(&[waveform, &z], 2)?
        } else {
            waveform.clone()
        };
        let lp = x.dims()[2];
        let x = x.reshape(vec![b * s, 1, lp])?.to_dtype(self.dtype)?;
        let h = enc.forward(&x)?;
        let t = h.dims()[2];
        let seq = h.reshape(vec![b * s, self.cfg.latent_dim, t])?;
        let mut outs = Vec::with_capacity(b * s);
        for i in 0..b * s {
            let row = seq
                .narrow(0, i, 1)?
                .reshape(vec![self.cfg.latent_dim, t])?
                .transpose(0, 1)?
                .contiguous()?;
            let y = pre.forward(&row)?;
            outs.push(
                y.transpose(0, 1)?
                    .contiguous()?
                    .reshape(vec![1, self.cfg.vae_latent_channels, t])?,
            );
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        let pooled = Tensor::cat(&refs, 0)?;
        let z = mean_proj.forward(&pooled)?;
        let (m, sd) = self.stats(z.dtype())?;
        let z = z.broadcast_sub(&m)?.broadcast_div(&sd)?;
        Ok(z
            .reshape(vec![b, s, self.cfg.vae_latent_channels, t])?
            .permute([0, 2, 1, 3])?
            .contiguous()?)
    }
}

pub fn interleave_stereo(wave: &Tensor) -> Result<Vec<f32>, H3Error> {
    let d = wave.dims().to_vec();
    let (channels, len) = (d[1], d[2]);
    let host = wave.to_device(Device::Cpu)?.to_dtype(DType::F32)?;
    let flat = host.reshape(vec![d[0] * channels * len])?.to_vec1::<f32>()?;
    let mut out = vec![0f32; channels * len];
    for c in 0..channels {
        for i in 0..len {
            out[i * channels + c] = flat[c * len + i];
        }
    }
    Ok(out)
}
