//! Настоящий `UNet2DConditionModel` (diffusers-совместимый, conv2d) для
//! SD/SDXL — заменяет линейную болванку `unet_2d.rs`.
//!
//! Config-driven (block types / channels / heads / transformer-глубина), а
//! каналы resnet'ов выводятся прямо из форм весов (надёжно к skip-concat
//! арифметике up-блоков). SDXL: time-embedding + `added_cond_kwargs`
//! (text_time: pooled text-embeds + time_ids → add-embedding), cross-attention
//! к prompt_embeds `[B,77,2048]`, Transformer2D с linear-проекциями
//! (self-attn + cross-attn + GEGLU FF).

use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::activation::gelu_exact;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::conv2d;
use synaptix_ops::norm::{group_norm, layer_norm};

use crate::linear::Linear;
use crate::module::Module;
use crate::quant_linear::QuantLinear;

use synaptix_core::dtype::DType;

// Точность весов UNet (как в LLM/FLUX): загрузчик SDXL задаёт перед UNet::load.
// (BF16, BF16) = dense-дефолт (бит-в-бит прежнее поведение). NVFP4/MXFP8 квантуют
// param-тяжёлые линейки трансформера (attn to_q/k/v/out, GEGLU proj/out, proj_in/out);
// resnet/time-emb/conv остаются dense.
thread_local! {
    static UNET_PREC: std::cell::Cell<(DType, DType)> =
        const { std::cell::Cell::new((DType::BF16, DType::BF16)) };
}
/// Задаёт (quant, compute) для последующих `UNet2DConditionModel::load`.
pub fn set_unet_precision(quant: DType, compute: DType) {
    UNET_PREC.with(|c| c.set((quant, compute)));
}
fn unet_prec() -> (DType, DType) {
    UNET_PREC.with(|c| c.get())
}
/// QuantLinear из загруженного веса (+опц. bias) по текущей точности UNet.
/// Dense-путь СОХРАНЯЕТ dtype веса (бит-в-бит как `Linear::new` — без каста, чтобы
/// не ломать F32/BF16-загрузку); каст в compute только при квантовании.
fn qlin(weight: Tensor, bias: Option<Tensor>) -> Result<QuantLinear> {
    let (q, comp) = unet_prec();
    if q.is_quantized() {
        QuantLinear::build(weight, bias, q, comp)
    } else {
        QuantLinear::dense(weight, bias)
    }
}

/// Тип down/up-блока: с cross-attention (Transformer2D) или без.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    Plain,
    CrossAttn,
}

#[derive(Debug, Clone)]
pub struct UNet2DConditionConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub down_block_types: Vec<BlockKind>,
    pub up_block_types: Vec<BlockKind>,
    /// Число голов на разрешении (SDXL `attention_head_dim`=[5,10,20]).
    pub num_attention_heads: Vec<usize>,
    pub transformer_layers_per_block: Vec<usize>,
    pub cross_attention_dim: usize,
    pub norm_num_groups: usize,
    pub norm_eps: f32,
    pub addition_time_embed_dim: usize,
    pub freq_shift: f32,
    pub max_period: f32,
}

impl UNet2DConditionConfig {
    pub fn sdxl() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            block_out_channels: vec![320, 640, 1280],
            layers_per_block: 2,
            down_block_types: vec![BlockKind::Plain, BlockKind::CrossAttn, BlockKind::CrossAttn],
            up_block_types: vec![BlockKind::CrossAttn, BlockKind::CrossAttn, BlockKind::Plain],
            num_attention_heads: vec![5, 10, 20],
            transformer_layers_per_block: vec![1, 2, 10],
            cross_attention_dim: 2048,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            addition_time_embed_dim: 256,
            freq_shift: 0.0,
            max_period: 10000.0,
        }
    }
}

/// HF diffusers `get_timestep_embedding` (flip_sin_to_cos=true): `[N] → [N, dim]`,
/// cat([cos, sin]). `downscale_freq_shift` параметризован (SDXL = 0).
pub fn get_timestep_embedding(
    timesteps: &Tensor,
    dim: usize,
    downscale_freq_shift: f32,
    max_period: f32,
) -> Result<Tensor> {
    if dim % 2 != 0 {
        return Err(SynaptixError::Unsupported("get_timestep_embedding: dim must be even"));
    }
    let device = timesteps.device();
    let half = dim / 2;
    let denom = (half as f32) - downscale_freq_shift;
    let log_max = max_period.ln();
    let freqs: Vec<f32> = (0..half).map(|i| (-log_max * i as f32 / denom).exp()).collect();
    let freqs_t = Tensor::from_vec(freqs, (1, half), device)?;
    let n = timesteps.dims()[0];
    let t_col = timesteps.to_dtype(synaptix_core::dtype::DType::F32)?.reshape(vec![n, 1])?;
    let args = t_col.broadcast_mul(&freqs_t)?;
    let cos = args.cos()?.contiguous()?;
    let sin = args.sin()?.contiguous()?;
    Tensor::cat(&[&cos, &sin], 1)
}

/// `linear_1 → SiLU → linear_2` (time / add embedding MLP).
struct TimestepMlp {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear_1.forward(x)?.silu()?;
        self.linear_2.forward(&h)
    }

    fn load<F>(get: &F, prefix: &str) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        let lin = |name: &str| -> Result<Linear> {
            Linear::new(get(&format!("{prefix}.{name}.weight"))?, Some(get(&format!("{prefix}.{name}.bias"))?))
        };
        Ok(Self { linear_1: lin("linear_1")?, linear_2: lin("linear_2")? })
    }
}

struct Conv2dLayer {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
}

#[allow(dead_code)]
impl Conv2dLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        conv2d(x, &self.weight, self.bias.as_ref(), self.stride, self.padding, (1, 1))
    }

    /// `conv2d(x) + residual` в один fused-эпилог (убирает финальный binary add).
    fn forward_add(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        x.conv2d_add(&self.weight, self.bias.as_ref(), self.stride, self.padding, residual)
    }

    /// `conv2d(x) + temb[:,:,None,None]` fused-эпилог (убирает broadcast_add).
    fn forward_temb(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        x.conv2d_temb(&self.weight, self.bias.as_ref(), self.stride, self.padding, temb)
    }

    fn out_hw_nhwc(&self, x: &Tensor) -> (usize, usize) {
        let d = x.dims();
        let (h, w) = (d[1], d[2]);
        let (kh, kw) = (self.weight.dims()[2], self.weight.dims()[3]);
        let oh = (h + 2 * self.padding.0 - kh) / self.stride.0 + 1;
        let ow = (w + 2 * self.padding.1 - kw) / self.stride.1 + 1;
        (oh, ow)
    }

    fn nhwc_fallback(
        &self,
        x: &Tensor,
        residual: Option<&Tensor>,
        temb: Option<&Tensor>,
    ) -> Result<Tensor> {
        let x_nchw = x.permute(vec![0, 3, 1, 2])?.contiguous()?;
        let mut out = conv2d(&x_nchw, &self.weight, self.bias.as_ref(), self.stride, self.padding, (1, 1))?;
        if let Some(t) = temb {
            let od = out.dims();
            let t4 = t.reshape(vec![od[0], od[1], 1, 1])?;
            out = out.broadcast_add(&t4)?;
        }
        if let Some(r) = residual {
            let r_nchw = r.permute(vec![0, 3, 1, 2])?.contiguous()?;
            out = out.add(&r_nchw)?;
        }
        out.permute(vec![0, 2, 3, 1])?.contiguous()
    }

    fn forward_nhwc(&self, x: &Tensor) -> Result<Tensor> {
        let (oh, ow) = self.out_hw_nhwc(x);
        match x.conv2d_nhwc_io(&self.weight, self.bias.as_ref(), None, None, self.stride, self.padding, oh, ow) {
            Ok(t) => Ok(t),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => self.nhwc_fallback(x, None, None),
            Err(e) => Err(e),
        }
    }

    fn forward_add_nhwc(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let (oh, ow) = self.out_hw_nhwc(x);
        match x.conv2d_nhwc_io(&self.weight, self.bias.as_ref(), Some(residual), None, self.stride, self.padding, oh, ow) {
            Ok(t) => Ok(t),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => self.nhwc_fallback(x, Some(residual), None),
            Err(e) => Err(e),
        }
    }

    fn forward_temb_nhwc(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let (oh, ow) = self.out_hw_nhwc(x);
        match x.conv2d_nhwc_io(&self.weight, self.bias.as_ref(), None, Some(temb), self.stride, self.padding, oh, ow) {
            Ok(t) => Ok(t),
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => self.nhwc_fallback(x, None, Some(temb)),
            Err(e) => Err(e),
        }
    }

    /// conv_in: NCHW-вход → NHWC-выход (граница). Cin=4 → implicit-conv не годится,
    /// обычный NCHW conv + транспоз результата.
    fn forward_nchw_to_nhwc(&self, x: &Tensor) -> Result<Tensor> {
        let out = conv2d(x, &self.weight, self.bias.as_ref(), self.stride, self.padding, (1, 1))?;
        out.permute(vec![0, 2, 3, 1])?.contiguous()
    }

    /// conv_out: NHWC-вход → NCHW-выход (граница). Cout=4 → обычный NCHW conv.
    fn forward_nhwc_to_nchw(&self, x: &Tensor) -> Result<Tensor> {
        let x_nchw = x.permute(vec![0, 3, 1, 2])?.contiguous()?;
        conv2d(&x_nchw, &self.weight, self.bias.as_ref(), self.stride, self.padding, (1, 1))
    }

    fn load<F>(get: &F, prefix: &str, stride: (usize, usize), padding: (usize, usize)) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        Ok(Self {
            weight: get(&format!("{prefix}.weight"))?,
            bias: get(&format!("{prefix}.bias")).ok(),
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

#[allow(dead_code)]
impl GroupNormLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        group_norm(x, Some(&self.weight), Some(&self.bias), self.num_groups, self.eps)
    }

    /// Fused GN + SiLU (один kernel-launch).
    fn forward_silu(&self, x: &Tensor) -> Result<Tensor> {
        synaptix_ops::norm::group_norm_silu(
            x, Some(&self.weight), Some(&self.bias), self.num_groups, self.eps,
        )
    }

    fn forward_nhwc(&self, x: &Tensor) -> Result<Tensor> {
        synaptix_ops::norm::group_norm_nhwc(
            x, Some(&self.weight), Some(&self.bias), self.num_groups, self.eps, false,
        )
    }

    fn forward_silu_nhwc(&self, x: &Tensor) -> Result<Tensor> {
        synaptix_ops::norm::group_norm_nhwc(
            x, Some(&self.weight), Some(&self.bias), self.num_groups, self.eps, true,
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

/// UNet ResnetBlock2D (`resnet_time_scale_shift="default"`): инъекция temb
/// после conv1. Каналы выводятся из форм весов.
struct UnetResnet2D {
    norm1: GroupNormLayer,
    conv1: Conv2dLayer,
    time_emb_proj: Linear,
    norm2: GroupNormLayer,
    conv2: Conv2dLayer,
    shortcut: Option<Conv2dLayer>,
}

impl UnetResnet2D {
    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward_silu_nhwc(x)?;
        let t = self.time_emb_proj.forward(&temb.silu()?)?;
        let h = self.conv1.forward_temb_nhwc(&h, &t)?;
        let h = self.norm2.forward_silu_nhwc(&h)?;
        let res = match &self.shortcut {
            Some(c) => c.forward_nhwc(x)?,
            None => x.clone(),
        };
        self.conv2.forward_add_nhwc(&h, &res)
    }

    fn load<F>(get: &F, prefix: &str, ng: usize, eps: f32) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        let conv1 = Conv2dLayer::load(get, &format!("{prefix}.conv1"), (1, 1), (1, 1))?;
        let in_ch = conv1.weight.dims()[1];
        let out_ch = conv1.weight.dims()[0];
        let _ = (in_ch, out_ch);
        let shortcut = get(&format!("{prefix}.conv_shortcut.weight"))
            .ok()
            .map(|_| Conv2dLayer::load(get, &format!("{prefix}.conv_shortcut"), (1, 1), (0, 0)))
            .transpose()?;
        Ok(Self {
            norm1: GroupNormLayer::load(get, &format!("{prefix}.norm1"), ng, eps)?,
            conv1,
            time_emb_proj: Linear::new(
                get(&format!("{prefix}.time_emb_proj.weight"))?,
                Some(get(&format!("{prefix}.time_emb_proj.bias"))?),
            )?,
            norm2: GroupNormLayer::load(get, &format!("{prefix}.norm2"), ng, eps)?,
            conv2: Conv2dLayer::load(get, &format!("{prefix}.conv2"), (1, 1), (1, 1))?,
            shortcut,
        })
    }
}

/// Multi-head attention (self или cross). q/k/v без bias, to_out.0 с bias.
struct Attention {
    to_q: QuantLinear,
    to_k: QuantLinear,
    to_v: QuantLinear,
    to_out: QuantLinear,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn split(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s) = (x.dims()[0], x.dims()[1]);
        x.reshape(vec![b, s, self.num_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?
            .contiguous()
    }

    #[allow(dead_code)]
    fn forward(&self, hidden: &Tensor, context: &Tensor) -> Result<Tensor> {
        self.forward_opt(hidden, context, None)
    }

    fn forward_add(&self, hidden: &Tensor, context: &Tensor, residual: &Tensor) -> Result<Tensor> {
        self.forward_opt(hidden, context, Some(residual))
    }

    fn forward_opt(&self, hidden: &Tensor, context: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
        let (b, sq) = (hidden.dims()[0], hidden.dims()[1]);
        let skv = context.dims()[1];
        let (nh, hd) = (self.num_heads, self.head_dim);
        let scale = 1.0 / (hd as f32).sqrt();
        let qp = self.to_q.forward(hidden)?;
        let kp = self.to_k.forward(context)?;
        let vp = self.to_v.forward(context)?;
        // BSHD-путь: q/k/v как [B,S,H,D] (reshape без permute+contiguous) → flash
        // без транспоза → reshape назад. Fallback на [B,H,S,D] (split-транспоз).
        let attn = {
            let qb = qp.reshape(vec![b, sq, nh, hd])?;
            let kb = kp.reshape(vec![b, skv, nh, hd])?;
            let vb = vp.reshape(vec![b, skv, nh, hd])?;
            match qb.flash_attention_bshd(&kb, &vb, scale, false) {
                Ok(a) => a.reshape(vec![b, sq, nh * hd])?,
                Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                    let q = self.split(&qp)?;
                    let k = self.split(&kp)?;
                    let v = self.split(&vp)?;
                    let a = match q.flash_attention(&k, &v, scale, false) {
                        Ok(a) => a,
                        Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                            scaled_dot_attention(&q, &k, &v, scale, None)?
                        }
                        Err(e) => return Err(e),
                    };
                    a.permute(vec![0, 2, 1, 3])?.contiguous()?.reshape(vec![b, sq, nh * hd])?
                }
                Err(e) => return Err(e),
            }
        };
        match residual {
            Some(r) => self.to_out.forward_add(&attn, r),
            None => self.to_out.forward(&attn),
        }
    }

    fn load<F>(get: &F, prefix: &str, num_heads: usize, head_dim: usize) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        let nobias = |name: &str| -> Result<QuantLinear> {
            qlin(get(&format!("{prefix}.{name}.weight"))?, None)
        };
        Ok(Self {
            to_q: nobias("to_q")?,
            to_k: nobias("to_k")?,
            to_v: nobias("to_v")?,
            to_out: qlin(
                get(&format!("{prefix}.to_out.0.weight"))?,
                Some(get(&format!("{prefix}.to_out.0.bias"))?),
            )?,
            num_heads,
            head_dim,
        })
    }
}

/// GEGLU FeedForward: `proj(dim → 2·inner)`, split → `x · gelu(gate)`, `out(inner → dim)`.
struct GeGlu {
    proj: QuantLinear,
    out: QuantLinear,
}

impl GeGlu {
    #[allow(dead_code)]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_opt(x, None)
    }

    fn forward_add(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        self.forward_opt(x, Some(residual))
    }

    fn forward_opt(&self, x: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
        let p = self.proj.forward(x)?;
        let h = match p.geglu_split() {
            Ok(h) => h,
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                let inner = p.dims()[p.rank() - 1] / 2;
                let xs = p.narrow(p.rank() - 1, 0, inner)?.contiguous()?;
                let gate = p.narrow(p.rank() - 1, inner, inner)?.contiguous()?;
                xs.mul(&gelu_exact(&gate)?)?
            }
            Err(e) => return Err(e),
        };
        match residual {
            Some(r) => self.out.forward_add(&h, r),
            None => self.out.forward(&h),
        }
    }

    fn load<F>(get: &F, prefix: &str) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        Ok(Self {
            proj: qlin(
                get(&format!("{prefix}.net.0.proj.weight"))?,
                Some(get(&format!("{prefix}.net.0.proj.bias"))?),
            )?,
            out: qlin(
                get(&format!("{prefix}.net.2.weight"))?,
                Some(get(&format!("{prefix}.net.2.bias"))?),
            )?,
        })
    }
}

struct LayerNormLayer {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNormLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        layer_norm(x, Some(&self.weight), Some(&self.bias), self.eps)
    }

    fn load<F>(get: &F, prefix: &str, eps: f32) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        Ok(Self {
            weight: get(&format!("{prefix}.weight"))?,
            bias: get(&format!("{prefix}.bias"))?,
            eps,
        })
    }
}

/// BasicTransformerBlock: (LN→self-attn)+, (LN→cross-attn)+, (LN→GEGLU FF)+.
struct BasicTransformerBlock {
    norm1: LayerNormLayer,
    attn1: Attention,
    norm2: LayerNormLayer,
    attn2: Attention,
    norm3: LayerNormLayer,
    ff: GeGlu,
}

impl BasicTransformerBlock {
    fn forward(&self, hidden: &Tensor, context: &Tensor) -> Result<Tensor> {
        let n1 = self.norm1.forward(hidden)?;
        let hidden = self.attn1.forward_add(&n1, &n1, hidden)?;
        let n2 = self.norm2.forward(&hidden)?;
        let hidden = self.attn2.forward_add(&n2, context, &hidden)?;
        let n3 = self.norm3.forward(&hidden)?;
        self.ff.forward_add(&n3, &hidden)
    }

    fn load<F>(get: &F, prefix: &str, num_heads: usize, head_dim: usize, eps: f32) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        Ok(Self {
            norm1: LayerNormLayer::load(get, &format!("{prefix}.norm1"), eps)?,
            attn1: Attention::load(get, &format!("{prefix}.attn1"), num_heads, head_dim)?,
            norm2: LayerNormLayer::load(get, &format!("{prefix}.norm2"), eps)?,
            attn2: Attention::load(get, &format!("{prefix}.attn2"), num_heads, head_dim)?,
            norm3: LayerNormLayer::load(get, &format!("{prefix}.norm3"), eps)?,
            ff: GeGlu::load(get, &format!("{prefix}.ff"))?,
        })
    }
}

/// Transformer2DModel (use_linear_projection=true): GN → proj_in → N блоков → proj_out → +residual.
struct Transformer2D {
    norm: GroupNormLayer,
    proj_in: Linear,
    blocks: Vec<BasicTransformerBlock>,
    proj_out: Linear,
}

impl Transformer2D {
    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let d = x.dims();
        let (b, h, w, c) = (d[0], d[1], d[2], d[3]);
        let res = x;
        let hidden = self.norm.forward_nhwc(x)?;
        let hidden = hidden.reshape(vec![b, h * w, c])?;
        let mut hidden = self.proj_in.forward(&hidden)?;
        for blk in &self.blocks {
            hidden = blk.forward(&hidden, context)?;
        }
        let hidden = self.proj_out.forward(&hidden)?;
        let hidden = hidden.reshape(vec![b, h, w, c])?;
        hidden.add(res)
    }

    fn load<F>(get: &F, prefix: &str, num_blocks: usize, num_heads: usize, head_dim: usize, ng: usize) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            blocks.push(BasicTransformerBlock::load(
                get,
                &format!("{prefix}.transformer_blocks.{i}"),
                num_heads,
                head_dim,
                1e-5,
            )?);
        }
        Ok(Self {
            // Transformer2D input-norm в diffusers = GroupNorm eps=1e-6.
            norm: GroupNormLayer::load(get, &format!("{prefix}.norm"), ng, 1e-6)?,
            proj_in: Linear::new(
                get(&format!("{prefix}.proj_in.weight"))?,
                Some(get(&format!("{prefix}.proj_in.bias"))?),
            )?,
            blocks,
            proj_out: Linear::new(
                get(&format!("{prefix}.proj_out.weight"))?,
                Some(get(&format!("{prefix}.proj_out.bias"))?),
            )?,
        })
    }
}

/// nearest-2x для NHWC `[B,H,W,C]`. Round-trip через быстрое NCHW-ядро (апсемпл
/// всего 3× за шаг — транспоз пренебрежим vs ~60 устранённых в Transformer2D).
fn upsample_nearest2x_nhwc(x: &Tensor) -> Result<Tensor> {
    let x_nchw = x.permute(vec![0, 3, 1, 2])?.contiguous()?;
    let up = upsample_nearest2x(&x_nchw)?;
    up.permute(vec![0, 2, 3, 1])?.contiguous()
}

fn upsample_nearest2x(x: &Tensor) -> Result<Tensor> {
    // Быстрый backend-путь (CUDA — один launch); fallback cat-based на CPU.
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

struct DownBlock {
    resnets: Vec<UnetResnet2D>,
    attentions: Vec<Transformer2D>,
    downsampler: Option<Conv2dLayer>,
}

impl DownBlock {
    /// Возвращает обновлённый `h` и residual-сэмплы (в порядке записи).
    fn forward(&self, mut h: Tensor, temb: &Tensor, context: &Tensor, res: &mut Vec<Tensor>) -> Result<Tensor> {
        for (i, rn) in self.resnets.iter().enumerate() {
            h = rn.forward(&h, temb)?;
            if let Some(attn) = self.attentions.get(i) {
                h = attn.forward(&h, context)?;
            }
            res.push(h.clone());
        }
        if let Some(ds) = &self.downsampler {
            h = ds.forward_nhwc(&h)?;
            res.push(h.clone());
        }
        Ok(h)
    }
}

struct UpBlock {
    resnets: Vec<UnetResnet2D>,
    attentions: Vec<Transformer2D>,
    upsampler: Option<Conv2dLayer>,
}

impl UpBlock {
    fn forward(&self, mut h: Tensor, temb: &Tensor, context: &Tensor, res: &mut Vec<Tensor>) -> Result<Tensor> {
        for (i, rn) in self.resnets.iter().enumerate() {
            let skip = res.pop().ok_or(SynaptixError::Unsupported("unet up: res_samples exhausted"))?;
            let cat = Tensor::cat(&[&h, &skip], 3)?.contiguous()?;
            h = rn.forward(&cat, temb)?;
            if let Some(attn) = self.attentions.get(i) {
                h = attn.forward(&h, context)?;
            }
        }
        if let Some(up) = &self.upsampler {
            h = up.forward_nhwc(&upsample_nearest2x_nhwc(&h)?)?;
        }
        Ok(h)
    }
}

pub struct UNet2DConditionModel {
    conv_in: Conv2dLayer,
    time_embedding: TimestepMlp,
    add_embedding: TimestepMlp,
    down_blocks: Vec<DownBlock>,
    mid_resnet1: UnetResnet2D,
    mid_attn: Transformer2D,
    mid_resnet2: UnetResnet2D,
    up_blocks: Vec<UpBlock>,
    conv_norm_out: GroupNormLayer,
    conv_out: Conv2dLayer,
    config: UNet2DConditionConfig,
}

impl UNet2DConditionModel {
    pub fn config(&self) -> &UNet2DConditionConfig {
        &self.config
    }

    /// Сколько `transformer_blocks` лежит под `{prefix}.attentions.0` (проба весов).
    fn count_tblocks<F>(get: &F, attn_prefix: &str) -> usize
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        let mut n = 0;
        while get(&format!("{attn_prefix}.transformer_blocks.{n}.norm1.weight")).is_ok() {
            n += 1;
        }
        n
    }

    pub fn load<F>(cfg: &UNet2DConditionConfig, get: &F) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        let ng = cfg.norm_num_groups;
        let eps = cfg.norm_eps;
        let n_res = cfg.layers_per_block;
        let bo = &cfg.block_out_channels;
        let n = bo.len();

        let conv_in = Conv2dLayer::load(get, "conv_in", (1, 1), (1, 1))?;
        let time_embedding = TimestepMlp::load(get, "time_embedding")?;
        let add_embedding = TimestepMlp::load(get, "add_embedding")?;

        // ── down blocks ──
        let mut down_blocks = Vec::with_capacity(n);
        for i in 0..n {
            let bp = format!("down_blocks.{i}");
            let kind = cfg.down_block_types[i];
            let num_heads = cfg.num_attention_heads[i];
            let head_dim = bo[i] / num_heads;
            let mut resnets = Vec::with_capacity(n_res);
            let mut attentions = Vec::new();
            for r in 0..n_res {
                resnets.push(UnetResnet2D::load(get, &format!("{bp}.resnets.{r}"), ng, eps)?);
                if kind == BlockKind::CrossAttn {
                    let ap = format!("{bp}.attentions.{r}");
                    let tb = Self::count_tblocks(get, &ap);
                    attentions.push(Transformer2D::load(get, &ap, tb, num_heads, head_dim, ng)?);
                }
            }
            let downsampler = if get(&format!("{bp}.downsamplers.0.conv.weight")).is_ok() {
                Some(Conv2dLayer::load(get, &format!("{bp}.downsamplers.0.conv"), (2, 2), (1, 1))?)
            } else {
                None
            };
            down_blocks.push(DownBlock { resnets, attentions, downsampler });
        }

        // ── mid block ──
        let mid_heads = cfg.num_attention_heads[n - 1];
        let mid_head_dim = bo[n - 1] / mid_heads;
        let mid_tb = Self::count_tblocks(get, "mid_block.attentions.0");
        let mid_resnet1 = UnetResnet2D::load(get, "mid_block.resnets.0", ng, eps)?;
        let mid_attn = Transformer2D::load(get, "mid_block.attentions.0", mid_tb, mid_heads, mid_head_dim, ng)?;
        let mid_resnet2 = UnetResnet2D::load(get, "mid_block.resnets.1", ng, eps)?;

        // ── up blocks (reversed channels/heads) ──
        let mut up_blocks = Vec::with_capacity(n);
        for i in 0..n {
            let bp = format!("up_blocks.{i}");
            let kind = cfg.up_block_types[i];
            let ri = n - 1 - i; // обратный индекс разрешения
            let num_heads = cfg.num_attention_heads[ri];
            let head_dim = bo[ri] / num_heads;
            let mut resnets = Vec::with_capacity(n_res + 1);
            let mut attentions = Vec::new();
            for r in 0..(n_res + 1) {
                resnets.push(UnetResnet2D::load(get, &format!("{bp}.resnets.{r}"), ng, eps)?);
                if kind == BlockKind::CrossAttn {
                    let ap = format!("{bp}.attentions.{r}");
                    let tb = Self::count_tblocks(get, &ap);
                    attentions.push(Transformer2D::load(get, &ap, tb, num_heads, head_dim, ng)?);
                }
            }
            let upsampler = if get(&format!("{bp}.upsamplers.0.conv.weight")).is_ok() {
                Some(Conv2dLayer::load(get, &format!("{bp}.upsamplers.0.conv"), (1, 1), (1, 1))?)
            } else {
                None
            };
            up_blocks.push(UpBlock { resnets, attentions, upsampler });
        }

        let conv_norm_out = GroupNormLayer::load(get, "conv_norm_out", ng, eps)?;
        let conv_out = Conv2dLayer::load(get, "conv_out", (1, 1), (1, 1))?;

        Ok(Self {
            conv_in,
            time_embedding,
            add_embedding,
            down_blocks,
            mid_resnet1,
            mid_attn,
            mid_resnet2,
            up_blocks,
            conv_norm_out,
            conv_out,
            config: cfg.clone(),
        })
    }

    /// `sample: [B,4,H,W]`, `timesteps: [B]`, `encoder_hidden_states: [B,77,2048]`,
    /// `text_embeds: [B,1280]` (pooled bigG), `time_ids: [B,6]` → `[B,4,H,W]` (eps-предсказание).
    pub fn forward(
        &self,
        sample: &Tensor,
        timesteps: &Tensor,
        encoder_hidden_states: &Tensor,
        text_embeds: &Tensor,
        time_ids: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        // time embedding
        let t_emb = get_timestep_embedding(timesteps, cfg.block_out_channels[0], cfg.freq_shift, cfg.max_period)?
            .to_dtype(sample.dtype())?;
        let emb = self.time_embedding.forward(&t_emb)?;
        // added cond (text_time): time_ids -> sinusoidal -> concat pooled text -> add_embedding
        let b = time_ids.dims()[0];
        let n_ids = time_ids.dims()[1];
        let ids_flat = time_ids.reshape(vec![b * n_ids])?;
        let time_proj = get_timestep_embedding(&ids_flat, cfg.addition_time_embed_dim, cfg.freq_shift, cfg.max_period)?
            .reshape(vec![b, n_ids * cfg.addition_time_embed_dim])?
            .to_dtype(sample.dtype())?;
        let add_in = Tensor::cat(&[text_embeds, &time_proj], 1)?.contiguous()?;
        let aug = self.add_embedding.forward(&add_in)?;
        let emb = emb.add(&aug)?;

        let mut h = self.conv_in.forward_nchw_to_nhwc(sample)?;
        let mut res_samples: Vec<Tensor> = vec![h.clone()];
        for db in self.down_blocks.iter() {
            h = db.forward(h, &emb, encoder_hidden_states, &mut res_samples)?;
        }

        h = self.mid_resnet1.forward(&h, &emb)?;
        h = self.mid_attn.forward(&h, encoder_hidden_states)?;
        h = self.mid_resnet2.forward(&h, &emb)?;

        for ub in self.up_blocks.iter() {
            h = ub.forward(h, &emb, encoder_hidden_states, &mut res_samples)?;
        }

        let h = self.conv_norm_out.forward_silu_nhwc(&h)?;
        let out = self.conv_out.forward_nhwc_to_nchw(&h)?;
        Ok(out)
    }
}
