//! Настоящий `AutoencoderKL` (diffusers-совместимый, conv2d): encoder +
//! decoder. Заменил линейные болванки `[B,T,C]`, удалённые из этого модуля.
//!
//! Структура полностью определяется конфигом (`block_out_channels`,
//! `layers_per_block`, `norm_num_groups`) — никакого хардкода под конкретную
//! модель. Совместим с SD-1.x / SDXL / SD3 VAE (отличаются только числами).
//!
//! Декодер: `post_quant_conv` → `conv_in` → mid (resnet, attn, resnet) →
//! N up-блоков (`layers_per_block+1` resnet + опц. nearest-2× upsample) →
//! `GroupNorm`+SiLU → `conv_out`. Все нормы — `GroupNorm` с `eps=1e-6`,
//! активация SiLU, как в `diffusers`.

use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::conv2d;
use synaptix_ops::norm::group_norm;

use crate::linear::Linear;
use crate::module::Module;

/// Конфиг `AutoencoderKL`. Все размеры — поля, ничего не зашито под модель.
#[derive(Debug, Clone)]
pub struct AutoencoderKlConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub norm_eps: f32,
    pub scaling_factor: f32,
    /// Сдвиг латента (`latents = z/scaling + shift` на decode). `None` = 0 (SDXL).
    /// FLUX: 0.1159.
    pub shift_factor: Option<f32>,
    /// Есть ли `post_quant_conv`/`quant_conv`. SDXL — true; FLUX — false (этих
    /// ключей нет в state_dict, латент идёт прямо в `conv_in`).
    pub use_quant_conv: bool,
}

impl AutoencoderKlConfig {
    /// SDXL VAE (`stabilityai/stable-diffusion-xl-base-1.0/vae`).
    pub fn sdxl() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            norm_eps: 1e-6,
            scaling_factor: 0.13025,
            shift_factor: None,
            use_quant_conv: true,
        }
    }

    /// FLUX.1 VAE (`black-forest-labs/FLUX.1-dev/vae`): 16 латентных каналов,
    /// БЕЗ quant_conv/post_quant_conv, scaling 0.3611 / shift 0.1159.
    pub fn flux() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 16,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            norm_eps: 1e-6,
            scaling_factor: 0.3611,
            shift_factor: Some(0.1159),
            use_quant_conv: false,
        }
    }
}

/// Conv2d-слой с фиксированными stride/padding (dilation=1).
struct Conv2dLayer {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
}

impl Conv2dLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        conv2d(x, &self.weight, self.bias.as_ref(), self.stride, self.padding, (1, 1))
    }

    /// `conv2d(x) + residual` в один fused-эпилог (убирает финальный binary add).
    /// CPU/неподдержка fused-пути → ops conv2d (CPU-способный) + binary add.
    fn forward_add(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        match x.conv2d_add(&self.weight, self.bias.as_ref(), self.stride, self.padding, residual) {
            Ok(t) => Ok(t),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                conv2d(x, &self.weight, self.bias.as_ref(), self.stride, self.padding, (1, 1))?.add(residual)
            }
            Err(e) => Err(e),
        }
    }

    fn load<F>(get: &F, prefix: &str, padding: (usize, usize)) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        Self::load_strided(get, prefix, (1, 1), padding)
    }

    fn load_strided<F>(get: &F, prefix: &str, stride: (usize, usize), padding: (usize, usize)) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        Ok(Self {
            weight: get(&format!("{prefix}.weight"))?,
            bias: Some(get(&format!("{prefix}.bias"))?),
            stride,
            padding,
        })
    }
}

struct GroupNormLayer {
    weight: Tensor,
    bias: Tensor,
    num_groups: usize,
    eps: f32,
}

impl GroupNormLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        group_norm(x, Some(&self.weight), Some(&self.bias), self.num_groups, self.eps)
    }

    /// Fused GN + SiLU (один kernel-launch вместо `norm.forward(x).silu()`).
    fn forward_silu(&self, x: &Tensor) -> Result<Tensor> {
        synaptix_ops::norm::group_norm_silu(
            x, Some(&self.weight), Some(&self.bias), self.num_groups, self.eps,
        )
    }

    fn load<F>(get: &F, prefix: &str, num_groups: usize, eps: f32) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        Ok(Self {
            weight: get(&format!("{prefix}.weight"))?,
            bias: get(&format!("{prefix}.bias"))?,
            num_groups,
            eps,
        })
    }
}

/// `ResnetBlock2D`: (GN→SiLU→conv3×3)×2 + residual (с conv_shortcut 1×1
/// если меняются каналы). `output_scale_factor=1` (без деления), как в VAE.
struct ResnetBlock2D {
    norm1: GroupNormLayer,
    conv1: Conv2dLayer,
    norm2: GroupNormLayer,
    conv2: Conv2dLayer,
    shortcut: Option<Conv2dLayer>,
}

impl ResnetBlock2D {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward_silu(x)?;
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward_silu(&h)?;
        let res = match &self.shortcut {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        // conv2 + residual fused в одном эпилоге.
        self.conv2.forward_add(&h, &res)
    }

    fn load<F>(get: &F, prefix: &str, in_ch: usize, out_ch: usize, ng: usize, eps: f32) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        let shortcut = if in_ch != out_ch {
            Some(Conv2dLayer::load(get, &format!("{prefix}.conv_shortcut"), (0, 0))?)
        } else {
            None
        };
        Ok(Self {
            norm1: GroupNormLayer::load(get, &format!("{prefix}.norm1"), ng, eps)?,
            conv1: Conv2dLayer::load(get, &format!("{prefix}.conv1"), (1, 1))?,
            norm2: GroupNormLayer::load(get, &format!("{prefix}.norm2"), ng, eps)?,
            conv2: Conv2dLayer::load(get, &format!("{prefix}.conv2"), (1, 1))?,
            shortcut,
        })
    }
}

/// Пространственный self-attention (single-head) из mid-блока VAE.
/// `diffusers`: GN → [B,HW,C] → q/k/v → SDPA(scale=C^-0.5) → to_out → +residual.
struct VaeAttention {
    group_norm: GroupNormLayer,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    channels: usize,
}

impl VaeAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.dims();
        let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
        let hw = h * w;
        let gh = self.group_norm.forward(x)?;
        // [B,C,H,W] -> [B,C,HW] -> [B,HW,C]
        let seq = gh.reshape(vec![b, c, hw])?.permute(vec![0, 2, 1])?.contiguous()?;
        let q = self.to_q.forward(&seq)?.reshape(vec![b, 1, hw, c])?;
        let k = self.to_k.forward(&seq)?.reshape(vec![b, 1, hw, c])?;
        let v = self.to_v.forward(&seq)?.reshape(vec![b, 1, hw, c])?;
        let scale = 1.0 / (self.channels as f32).sqrt();
        let attn = scaled_dot_attention(&q, &k, &v, scale, None)?;
        let attn = attn.reshape(vec![b, hw, c])?;
        let out = self.to_out.forward(&attn)?;
        // [B,HW,C] -> [B,C,HW] -> [B,C,H,W]
        let out = out.permute(vec![0, 2, 1])?.contiguous()?.reshape(vec![b, c, h, w])?;
        out.add(x)
    }

    fn load<F>(get: &F, prefix: &str, channels: usize, ng: usize, eps: f32) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        let lin = |name: &str| -> Result<Linear> {
            Linear::new(get(&format!("{prefix}.{name}.weight"))?, Some(get(&format!("{prefix}.{name}.bias"))?))
        };
        Ok(Self {
            group_norm: GroupNormLayer::load(get, &format!("{prefix}.group_norm"), ng, eps)?,
            to_q: lin("to_q")?,
            to_k: lin("to_k")?,
            to_v: lin("to_v")?,
            to_out: lin("to_out.0")?,
            channels,
        })
    }
}

struct UpBlock {
    resnets: Vec<ResnetBlock2D>,
    upsampler: Option<Conv2dLayer>,
}

/// Nearest-neighbour 2× upsample по H и W (точная дупликация пикселей,
/// bit-exact к `F.interpolate(scale_factor=2, mode="nearest")` для целого ×2).
fn upsample_nearest2x(x: &Tensor) -> Result<Tensor> {
    // Быстрый путь: выделенное backend-ядро (CUDA — один launch). На CPU /
    // неподдержке падаем в cat-based reshape (медленный на CUDA, см. SDXL-перф).
    match x.upsample_nearest2x() {
        Ok(out) => return Ok(out),
        Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
        Err(e) => return Err(e),
    }
    let d = x.dims();
    let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
    let xw = x.reshape(vec![b, c, h, w, 1])?;
    let xw = Tensor::cat(&[&xw, &xw], 4)?.contiguous()?.reshape(vec![b, c, h, w * 2])?;
    let xh = xw.reshape(vec![b, c, h, 1, w * 2])?;
    let xh = Tensor::cat(&[&xh, &xh], 3)?.contiguous()?.reshape(vec![b, c, h * 2, w * 2])?;
    xh.contiguous()
}

/// Асимметричный zero-pad справа и снизу на 1 (`F.pad(x,(0,1,0,1))`) — как в
/// `diffusers` `Downsample2D` с `padding=0` перед stride-2 conv.
fn pad_bottom_right(x: &Tensor) -> Result<Tensor> {
    let d = x.dims();
    let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
    let right = Tensor::zeros(vec![b, c, h, 1], x.dtype(), x.device())?;
    let x = Tensor::cat(&[x, &right], 3)?;
    let bottom = Tensor::zeros(vec![b, c, 1, w + 1], x.dtype(), x.device())?;
    Tensor::cat(&[&x, &bottom], 2)?.contiguous()
}

/// Декодер `AutoencoderKL` (latent → image), conv2d, config-driven.
pub struct AutoencoderKlDecoder {
    post_quant_conv: Option<Conv2dLayer>,
    conv_in: Conv2dLayer,
    mid_resnet1: ResnetBlock2D,
    mid_attn: VaeAttention,
    mid_resnet2: ResnetBlock2D,
    up_blocks: Vec<UpBlock>,
    conv_norm_out: GroupNormLayer,
    conv_out: Conv2dLayer,
    config: AutoencoderKlConfig,
}

impl AutoencoderKlDecoder {
    pub fn config(&self) -> &AutoencoderKlConfig {
        &self.config
    }

    /// Загрузка из произвольного источника весов по HF-именам (`diffusers`
    /// `AutoencoderKL`): `post_quant_conv.*`, `decoder.*`.
    pub fn load<F>(cfg: &AutoencoderKlConfig, get: &F) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        let ng = cfg.norm_num_groups;
        let eps = cfg.norm_eps;
        // reversed: SDXL [512,512,256,128]
        let rev: Vec<usize> = cfg.block_out_channels.iter().rev().copied().collect();
        let base = rev[0];

        let post_quant_conv = if cfg.use_quant_conv {
            Some(Conv2dLayer::load(get, "post_quant_conv", (0, 0))?)
        } else {
            None
        };
        let conv_in = Conv2dLayer::load(get, "decoder.conv_in", (1, 1))?;

        let mid_resnet1 =
            ResnetBlock2D::load(get, "decoder.mid_block.resnets.0", base, base, ng, eps)?;
        let mid_attn =
            VaeAttention::load(get, "decoder.mid_block.attentions.0", base, ng, eps)?;
        let mid_resnet2 =
            ResnetBlock2D::load(get, "decoder.mid_block.resnets.1", base, base, ng, eps)?;

        let n = rev.len();
        let n_resnets = cfg.layers_per_block + 1;
        let mut up_blocks = Vec::with_capacity(n);
        let mut prev = base;
        for i in 0..n {
            let out_ch = rev[i];
            let bp = format!("decoder.up_blocks.{i}");
            let mut resnets = Vec::with_capacity(n_resnets);
            for r in 0..n_resnets {
                let in_ch = if r == 0 { prev } else { out_ch };
                resnets.push(ResnetBlock2D::load(get, &format!("{bp}.resnets.{r}"), in_ch, out_ch, ng, eps)?);
            }
            let upsampler = if i != n - 1 {
                Some(Conv2dLayer::load(get, &format!("{bp}.upsamplers.0.conv"), (1, 1))?)
            } else {
                None
            };
            up_blocks.push(UpBlock { resnets, upsampler });
            prev = out_ch;
        }

        let conv_norm_out = GroupNormLayer::load(get, "decoder.conv_norm_out", ng, eps)?;
        let conv_out = Conv2dLayer::load(get, "decoder.conv_out", (1, 1))?;

        Ok(Self {
            post_quant_conv,
            conv_in,
            mid_resnet1,
            mid_attn,
            mid_resnet2,
            up_blocks,
            conv_norm_out,
            conv_out,
            config: cfg.clone(),
        })
    }

    /// `z: [B, latent_channels, H, W]` → image `[B, out_channels, H·2^k, W·2^k]`
    /// (raw `decoder` output, как `AutoencoderKL.decode(z).sample` — без
    /// деления на `scaling_factor`, это делает пайплайн).
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let z = match &self.post_quant_conv {
            Some(pq) => pq.forward(z)?,
            None => z.clone(),
        };
        let mut h = self.conv_in.forward(&z)?;
        h = self.mid_resnet1.forward(&h)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_resnet2.forward(&h)?;
        for ub in self.up_blocks.iter() {
            for r in &ub.resnets {
                h = r.forward(&h)?;
            }
            if let Some(up) = &ub.upsampler {
                h = up.forward(&upsample_nearest2x(&h)?)?;
            }
        }
        let h = self.conv_norm_out.forward_silu(&h)?;
        let out = self.conv_out.forward(&h)?;
        Ok(out)
    }
}

struct DownBlock {
    resnets: Vec<ResnetBlock2D>,
    downsampler: Option<Conv2dLayer>,
}

/// Энкодер `AutoencoderKL` (image → moments), conv2d, config-driven.
/// `encode` включает `quant_conv` и возвращает moments `[B, 2·latent, h, w]`
/// (== `vae.encode(x).latent_dist.parameters`), которые делятся на (mean, logvar).
pub struct AutoencoderKlEncoder {
    conv_in: Conv2dLayer,
    down_blocks: Vec<DownBlock>,
    mid_resnet1: ResnetBlock2D,
    mid_attn: VaeAttention,
    mid_resnet2: ResnetBlock2D,
    conv_norm_out: GroupNormLayer,
    conv_out: Conv2dLayer,
    quant_conv: Conv2dLayer,
    latent_channels: usize,
}

impl AutoencoderKlEncoder {
    pub fn load<F>(cfg: &AutoencoderKlConfig, get: &F) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        let ng = cfg.norm_num_groups;
        let eps = cfg.norm_eps;
        let bo = &cfg.block_out_channels;
        let n = bo.len();
        let last = bo[n - 1];

        let conv_in = Conv2dLayer::load(get, "encoder.conv_in", (1, 1))?;

        let n_resnets = cfg.layers_per_block;
        let mut down_blocks = Vec::with_capacity(n);
        let mut prev = bo[0];
        for i in 0..n {
            let out_ch = bo[i];
            let bp = format!("encoder.down_blocks.{i}");
            let mut resnets = Vec::with_capacity(n_resnets);
            for r in 0..n_resnets {
                let in_ch = if r == 0 { prev } else { out_ch };
                resnets.push(ResnetBlock2D::load(get, &format!("{bp}.resnets.{r}"), in_ch, out_ch, ng, eps)?);
            }
            let downsampler = if i != n - 1 {
                Some(Conv2dLayer::load_strided(get, &format!("{bp}.downsamplers.0.conv"), (2, 2), (0, 0))?)
            } else {
                None
            };
            down_blocks.push(DownBlock { resnets, downsampler });
            prev = out_ch;
        }

        let mid_resnet1 = ResnetBlock2D::load(get, "encoder.mid_block.resnets.0", last, last, ng, eps)?;
        let mid_attn = VaeAttention::load(get, "encoder.mid_block.attentions.0", last, ng, eps)?;
        let mid_resnet2 = ResnetBlock2D::load(get, "encoder.mid_block.resnets.1", last, last, ng, eps)?;

        let conv_norm_out = GroupNormLayer::load(get, "encoder.conv_norm_out", ng, eps)?;
        let conv_out = Conv2dLayer::load(get, "encoder.conv_out", (1, 1))?;
        let quant_conv = Conv2dLayer::load(get, "quant_conv", (0, 0))?;

        Ok(Self {
            conv_in,
            down_blocks,
            mid_resnet1,
            mid_attn,
            mid_resnet2,
            conv_norm_out,
            conv_out,
            quant_conv,
            latent_channels: cfg.latent_channels,
        })
    }

    /// `x: [B, in_channels, H, W]` → moments `[B, 2·latent_channels, h, w]`
    /// (после `quant_conv`), как `vae.encode(x).latent_dist.parameters`.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;
        for db in &self.down_blocks {
            for r in &db.resnets {
                h = r.forward(&h)?;
            }
            if let Some(ds) = &db.downsampler {
                h = ds.forward(&pad_bottom_right(&h)?)?;
            }
        }
        h = self.mid_resnet1.forward(&h)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_resnet2.forward(&h)?;
        let h = self.conv_norm_out.forward_silu(&h)?;
        let h = self.conv_out.forward(&h)?;
        self.quant_conv.forward(&h)
    }

    /// Разбить moments на `(mean, logvar)` по каналам.
    pub fn split_moments(&self, moments: &Tensor) -> Result<(Tensor, Tensor)> {
        let lc = self.latent_channels;
        let mean = moments.narrow(1, 0, lc)?.contiguous()?;
        let logvar = moments.narrow(1, lc, lc)?.contiguous()?;
        Ok((mean, logvar))
    }
}
