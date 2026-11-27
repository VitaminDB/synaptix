//! BigVGAN v2 (NVIDIA): мел-спектрограмма `[B, n_mels, T]` → волна
//! `[B, 1, T·hop]` в [−1, 1].
//!
//! Генератор: `conv_pre` (k7) → на каждый апсэмпл `ConvTranspose1d` и среднее
//! по AMP-блокам (ядра `resblock_kernel_sizes`, три пары свёрток с дилатациями)
//! → SnakeBeta → `conv_post` (k7). Активации — SnakeBeta с лог-параметрами в
//! сглаженной обёртке `Activation1d`: апсэмпл ×2 (Kaiser-sinc, k12) → snake →
//! даунсэмпл ÷2; фильтры считаются здесь, из чекпойнта не берутся.
//!
//! Имена весов — как в чекпойнтах NVIDIA (`bigvgan_generator.pt`, ключ
//! `generator`): `conv_pre`, `ups.{i}.0`, `resblocks.{n}.convs1|convs2.{j}`,
//! `resblocks.{n}.activations.{k}.act.alpha|beta`, `activation_post.act.*`,
//! `conv_post`; свёртки — с weight-norm (`weight_g`/`weight_v`) либо уже
//! свёрнутые (`weight`). Источник весов — замыкание, чтобы модуль не зависел
//! от формата хранения.

use std::f64::consts::PI;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;
use synaptix_ops::conv::conv1d::conv1d_dilated;
use synaptix_ops::conv::conv_transpose1d::conv_transpose1d;
use synaptix_ops::conv::depthwise::depthwise_conv;

const FILTER_KERNEL: usize = 12;
const RATIO: usize = 2;
const SNAKE_EPS: f32 = 1e-9;

#[derive(Debug, Clone)]
pub struct BigVganConfig {
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    /// `use_bias_at_final`.
    pub bias_at_final: bool,
    /// `use_tanh_at_final`; иначе выход обрезается в [−1, 1].
    pub tanh_at_final: bool,
}

impl BigVganConfig {
    /// `nvidia/bigvgan_v2_22khz_80band_256x` (80 мел, hop 256, 22,05 кГц).
    pub fn v2_22khz_80band_256x() -> Self {
        Self {
            upsample_rates: vec![4, 4, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![8, 8, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5]; 3],
            bias_at_final: false,
            tanh_at_final: false,
        }
    }

    pub fn hop(&self) -> usize {
        self.upsample_rates.iter().product()
    }
}

/// Источник весов: тензор по имени (F32 на нужном устройстве) и проверка наличия.
pub struct Weights<'a> {
    pub get: &'a dyn Fn(&str) -> Result<Tensor>,
    pub has: &'a dyn Fn(&str) -> bool,
}

fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
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

/// `kaiser_sinc_filter1d` из BigVGAN (`alias_free_activation/torch/filter.py`).
pub fn kaiser_sinc_filter1d(cutoff: f64, half_width: f64, kernel_size: usize) -> Vec<f32> {
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
    let denom = bessel_i0(beta);
    let win: Vec<f64> = (0..kernel_size)
        .map(|i| {
            if kernel_size == 1 {
                return 1.0;
            }
            let r = 2.0 * i as f64 / (kernel_size as f64 - 1.0) - 1.0;
            bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denom
        })
        .collect();
    let mut f = vec![0f64; kernel_size];
    let mut sum = 0f64;
    for i in 0..kernel_size {
        let time = if even { i as f64 - half_size as f64 + 0.5 } else { i as f64 - half_size as f64 };
        let arg = 2.0 * cutoff * time;
        let sinc = if arg.abs() < 1e-12 { 1.0 } else { (PI * arg).sin() / (PI * arg) };
        f[i] = 2.0 * cutoff * win[i] * sinc;
        sum += f[i];
    }
    f.iter().map(|v| (v / sum) as f32).collect()
}

fn pad_replicate(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let (b, c, l) = (x.dims()[0], x.dims()[1], x.dims()[2]);
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if left > 0 {
        parts.push(x.narrow(2, 0, 1)?.broadcast_as(vec![b, c, left])?.contiguous()?);
    }
    parts.push(x.contiguous()?);
    if right > 0 {
        parts.push(x.narrow(2, l - 1, 1)?.broadcast_as(vec![b, c, right])?.contiguous()?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 2)
}

/// `Activation1d(SnakeBeta)`: апсэмпл ×2 → `x + sin²(x·e^α)/(e^β + ε)` → даунсэмпл ÷2.
struct Act {
    alpha: Tensor,
    beta: Tensor,
    /// `e^α`, `1/(e^β + ε)` формы `[1, C, 1]` — запасной путь без fused-ядра.
    a_exp: Tensor,
    b_inv: Tensor,
    filter: Tensor,
}

impl Act {
    fn load(w: &Weights<'_>, prefix: &str, device: Device, dtype: DType) -> Result<Self> {
        let alpha = (w.get)(&format!("{prefix}.act.alpha"))?.to_dtype(DType::F32)?;
        let beta = (w.get)(&format!("{prefix}.act.beta"))?.to_dtype(DType::F32)?;
        let c: usize = alpha.dims().iter().product();
        let f = kaiser_sinc_filter1d(0.5 / RATIO as f64, 0.6 / RATIO as f64, FILTER_KERNEL);
        let mut v = Vec::with_capacity(c * FILTER_KERNEL);
        for _ in 0..c {
            v.extend_from_slice(&f);
        }
        Ok(Self {
            a_exp: alpha.exp()?.reshape(vec![1, c, 1])?.to_dtype(dtype)?,
            b_inv: beta.exp()?.add_scalar(SNAKE_EPS)?.recip()?.reshape(vec![1, c, 1])?.to_dtype(dtype)?,
            alpha,
            beta,
            filter: Tensor::from_vec(v, vec![c, 1, FILTER_KERNEL], device)?.to_dtype(dtype)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (k, r) = (FILTER_KERNEL, RATIO);
        let c = x.dims()[1];
        // Апсэмпл: replicate-pad 5/5 → транспонированная depthwise-свёртка шагом 2 → ×2 → кроп 15/15.
        let pad = k / r - 1;
        let xp = pad_replicate(x, pad, pad)?;
        let up = match xp.dwconv1d(&self.filter, None, r, 0, true) {
            Ok(v) => v,
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => conv_transpose1d(&xp, &self.filter, None, r, 0, 0, c, 1)?,
            Err(e) => return Err(e),
        };
        let (left, right) = (pad * r + (k - r) / 2, pad * r + (k - r + 1) / 2);
        let n = up.dims()[2];
        let up = up.mul_scalar(r as f32)?.narrow(2, left, n - left - right)?.contiguous()?;
        let act = match up.snake(&self.alpha, &self.beta, SNAKE_EPS) {
            Ok(y) => y,
            Err(_) => {
                let s = up.broadcast_mul(&self.a_exp)?.sin()?.sqr()?;
                up.add(&s.broadcast_mul(&self.b_inv)?)?
            }
        };
        // Даунсэмпл: replicate-pad 5/6 → depthwise-свёртка шагом 2.
        let xp = pad_replicate(&act, k / 2 - usize::from(k % 2 == 0), k / 2)?;
        depthwise_conv(&xp, &self.filter, None, r, 0, c)
    }
}

struct Conv {
    w: Tensor,
    b: Option<Tensor>,
    padding: usize,
    dilation: usize,
}

/// Вес свёртки: свёрнутый `weight` либо `weight_g · weight_v / ‖weight_v‖`
/// (норма по всем осям, кроме нулевой).
fn conv_weight(w: &Weights<'_>, prefix: &str, dtype: DType) -> Result<Tensor> {
    let plain = format!("{prefix}.weight");
    if (w.has)(&plain) {
        return (w.get)(&plain)?.to_dtype(dtype);
    }
    let g = (w.get)(&format!("{prefix}.weight_g"))?.to_dtype(DType::F32)?;
    let v = (w.get)(&format!("{prefix}.weight_v"))?.to_dtype(DType::F32)?;
    let d = v.dims().to_vec();
    let norm = v.sqr()?.reshape(vec![d[0], d[1] * d[2]])?.sum_keepdim(1)?.sqrt()?.reshape(vec![d[0], 1, 1])?;
    v.broadcast_div(&norm)?.broadcast_mul(&g)?.contiguous()?.to_dtype(dtype)
}

impl Conv {
    fn load(w: &Weights<'_>, prefix: &str, bias: bool, padding: usize, dilation: usize, dtype: DType) -> Result<Self> {
        Ok(Self {
            w: conv_weight(w, prefix, dtype)?,
            b: if bias { Some((w.get)(&format!("{prefix}.bias"))?.to_dtype(dtype)?) } else { None },
            padding,
            dilation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        conv1d_dilated(x, &self.w, self.b.as_ref(), 1, self.padding, self.dilation)
    }
}

struct AmpBlock {
    convs1: Vec<Conv>,
    convs2: Vec<Conv>,
    acts: Vec<Act>,
}

impl AmpBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for i in 0..self.convs1.len() {
            let t = self.convs1[i].forward(&self.acts[2 * i].forward(&h)?)?;
            let t = self.convs2[i].forward(&self.acts[2 * i + 1].forward(&t)?)?;
            h = t.add(&h)?;
        }
        Ok(h)
    }
}

struct Up {
    w: Tensor,
    b: Tensor,
    stride: usize,
    padding: usize,
}

pub struct BigVgan {
    conv_pre: Conv,
    ups: Vec<Up>,
    blocks: Vec<AmpBlock>,
    act_post: Act,
    conv_post: Conv,
    kernels: usize,
    tanh: bool,
    dtype: DType,
}

impl BigVgan {
    /// `prefix` — приставка имён с точкой на конце (`""`, `"decoder."`).
    pub fn load(w: &Weights<'_>, cfg: &BigVganConfig, prefix: &str, device: Device, dtype: DType) -> Result<Self> {
        let mut ups = Vec::with_capacity(cfg.upsample_rates.len());
        for (i, (&u, &k)) in cfg.upsample_rates.iter().zip(&cfg.upsample_kernel_sizes).enumerate() {
            let p = format!("{prefix}ups.{i}.0");
            ups.push(Up { w: conv_weight(w, &p, dtype)?, b: (w.get)(&format!("{p}.bias"))?.to_dtype(dtype)?, stride: u, padding: (k - u) / 2 });
        }
        let kernels = cfg.resblock_kernel_sizes.len();
        let mut blocks = Vec::with_capacity(ups.len() * kernels);
        for i in 0..ups.len() {
            for (j, &k) in cfg.resblock_kernel_sizes.iter().enumerate() {
                let p = format!("{prefix}resblocks.{}", i * kernels + j);
                let dil = &cfg.resblock_dilation_sizes[j];
                let (mut convs1, mut convs2, mut acts) = (Vec::new(), Vec::new(), Vec::new());
                for (n, &d) in dil.iter().enumerate() {
                    convs1.push(Conv::load(w, &format!("{p}.convs1.{n}"), true, (k * d - d) / 2, d, dtype)?);
                    convs2.push(Conv::load(w, &format!("{p}.convs2.{n}"), true, (k - 1) / 2, 1, dtype)?);
                }
                for n in 0..dil.len() * 2 {
                    acts.push(Act::load(w, &format!("{p}.activations.{n}"), device, dtype)?);
                }
                blocks.push(AmpBlock { convs1, convs2, acts });
            }
        }
        Ok(Self {
            conv_pre: Conv::load(w, &format!("{prefix}conv_pre"), true, 3, 1, dtype)?,
            ups,
            blocks,
            act_post: Act::load(w, &format!("{prefix}activation_post"), device, dtype)?,
            conv_post: Conv::load(w, &format!("{prefix}conv_post"), cfg.bias_at_final, 3, 1, dtype)?,
            kernels,
            tanh: cfg.tanh_at_final,
            dtype,
        })
    }

    /// Мел `[B, n_mels, T]` → волна `[B, 1, T·hop]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_pre.forward(&mel.to_dtype(self.dtype)?.contiguous()?)?;
        for (i, up) in self.ups.iter().enumerate() {
            h = conv_transpose1d(&h, &up.w, Some(&up.b), up.stride, up.padding, 0, 1, 1)?;
            let mut acc: Option<Tensor> = None;
            for j in 0..self.kernels {
                let y = self.blocks[i * self.kernels + j].forward(&h)?;
                acc = Some(match acc {
                    Some(a) => a.add(&y)?,
                    None => y,
                });
            }
            h = acc.expect("resblock_kernel_sizes не пуст").mul_scalar(1.0 / self.kernels as f32)?;
        }
        let y = self.conv_post.forward(&self.act_post.forward(&h)?)?;
        if self.tanh {
            y.tanh()
        } else {
            y.clamp(-1.0, 1.0)
        }
    }
}
