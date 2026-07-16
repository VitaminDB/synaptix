//! Нативный GigaAM-v3-e2e-CTC: StridingSubsampling + 16 Conformer-слоёв (Macaron
//! FFN, RoPE-attention, ConformerConvolution с GLU+depthwise+LayerNorm) + CTC-head.
//!
//! Источник истины: `~/Temp/GigaAM/gigaam/encoder.py` + `decoder.py`. Прогон —
//! batch=1 (одна реплика), поэтому attention/pad-маски не нужны (все кадры
//! валидны): `att_mask=None`, `pad_mask=None` в upstream при `shape[0]==1`.

use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_ops::activation::glu;
use synaptix_ops::attention::log_softmax_dim;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::conv::conv1d;
use synaptix_ops::norm::layer_norm;
use synaptix_ops::pos::rope::{apply_rope_range, RopeLayout};
use synaptix_ops::pos::rope_cache::RopeCache;

use crate::loader::{enc_layer, GigaAmWeights};
use crate::{GigaAmError, Result};

struct Ln {
    weight: Tensor,
    bias: Tensor,
}

impl Ln {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(layer_norm(x, Some(&self.weight), Some(&self.bias), 1e-5)?)
    }
}

/// Conv1d-веса (вес `[Cout,Cin,K]`, опц. bias `[Cout]`) + stride/padding/groups.
struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    groups: usize,
}

impl Conv1d {
    /// `[B,Cin,L] -> [B,Cout,Lout]`. groups>1 (depthwise) выполняется как набор
    /// независимых conv по группам каналов.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.groups == 1 {
            return Ok(conv1d(
                x,
                &self.weight,
                self.bias.as_ref(),
                self.stride,
                self.padding,
            )?);
        }
        // Depthwise: groups == Cin == Cout, вес [C,1,K]. conv1d ждёт
        // weight[Cout,Cin/groups,K]; здесь Cin/groups==1, поэтому считаем по
        // каждому каналу свой 1-канальный conv и склеиваем по оси каналов.
        let c = x.dims()[1];
        let mut chans: Vec<Tensor> = Vec::with_capacity(c);
        for ci in 0..c {
            let x_c = x.narrow(1, ci, 1)?.contiguous()?; // [B,1,L]
            let w_c = self.weight.narrow(0, ci, 1)?.contiguous()?; // [1,1,K]
            let b_c = match &self.bias {
                Some(b) => Some(b.narrow(0, ci, 1)?.contiguous()?),
                None => None,
            };
            let o = conv1d(&x_c, &w_c, b_c.as_ref(), self.stride, self.padding)?; // [B,1,Lout]
            chans.push(o);
        }
        let refs: Vec<&Tensor> = chans.iter().collect();
        Ok(Tensor::cat(&refs, 1)?)
    }
}

/// StridingSubsampling (conv1d, factor 4 = две стадии stride-2, kernel 5, pad 2,
/// ReLU между/после стадий). Вход mel `[B, feat_in, T]` → `[B, T', d_model]`.
struct Subsampling {
    conv0: Conv1d,
    conv2: Conv1d,
}

impl Subsampling {
    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = self.conv0.forward(mel)?.relu()?;
        let x = self.conv2.forward(&x)?.relu()?;
        // [B, d_model, T'] -> [B, T', d_model]
        Ok(x.transpose(1, 2)?.contiguous()?)
    }
}

struct FeedForward {
    linear1: (Tensor, Tensor),
    linear2: (Tensor, Tensor),
}

impl FeedForward {
    /// linear2(SiLU(linear1(x))).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.linear(&self.linear1.0)?.broadcast_add(&self.linear1.1)?;
        let h = h.silu()?;
        Ok(h.linear(&self.linear2.0)?.broadcast_add(&self.linear2.1)?)
    }
}

struct Attn {
    q: (Tensor, Tensor),
    k: (Tensor, Tensor),
    v: (Tensor, Tensor),
    out: (Tensor, Tensor),
    n_heads: usize,
    head_dim: usize,
}

impl Attn {
    fn proj(x: &Tensor, w: &(Tensor, Tensor)) -> Result<Tensor> {
        Ok(x.linear(&w.0)?.broadcast_add(&w.1)?)
    }

    /// `[B,S,D] -> [B,H,S,Dh]`.
    fn heads(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.dims();
        let (b, s) = (d[0], d[1]);
        Ok(x.reshape(vec![b, s, self.n_heads, self.head_dim])?
            .permute(vec![0, 2, 1, 3])?
            .contiguous()?)
    }

    /// RoPE-attention (batch=1, маски нет). ВАЖНО: upstream
    /// (`RotaryPositionMultiHeadAttention.forward`) применяет RoPE к ВХОДУ `x`,
    /// разбитому на головы, ДО линейных q/k/v-проекций. q и k берут одну и ту же
    /// rotated-версию `x` (query==key==x), v — НЕ ротированный `x`.
    fn forward(&self, x: &Tensor, rope: &RopeCache) -> Result<Tensor> {
        let d = x.dims();
        let (b, s) = (d[0], d[1]);

        // RoPE на входе, по головам: [B,S,D] -> [B,H,S,Dh] -> rope -> [B,S,D].
        let x_heads = self.heads(x)?; // [B,H,S,Dh]
        let x_rot = apply_rope_range(&x_heads, rope, 0, s, RopeLayout::Split)?;
        let x_rot = x_rot
            .permute(vec![0, 2, 1, 3])?
            .contiguous()?
            .reshape(vec![b, s, self.n_heads * self.head_dim])?; // [B,S,D]

        let q = self.heads(&Self::proj(&x_rot, &self.q)?)?;
        let k = self.heads(&Self::proj(&x_rot, &self.k)?)?;
        let v = self.heads(&Self::proj(x, &self.v)?)?;

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let attn = scaled_dot_attention(&q, &k, &v, scale, None)?; // [B,H,S,Dh]
        let merged = attn
            .permute(vec![0, 2, 1, 3])?
            .contiguous()?
            .reshape(vec![b, s, self.n_heads * self.head_dim])?;
        Self::proj(&merged, &self.out)
    }
}

/// ConformerConvolution: pointwise_conv1 (→2d) → GLU(dim=канал) → depthwise_conv
/// (k5, pad2, groups=d) → LayerNorm (по каналу) → SiLU → pointwise_conv2.
struct ConformerConv {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    norm: Ln,
    pointwise_conv2: Conv1d,
}

impl ConformerConv {
    /// Вход `[B,S,D]` → выход `[B,S,D]` (внутри работает по `[B,D,S]`).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?.contiguous()?; // [B,D,S]
        let x = self.pointwise_conv1.forward(&x)?; // [B,2D,S]
        let x = glu(&x, 1)?; // [B,D,S]
        let x = self.depthwise_conv.forward(&x)?; // [B,D,S]
        // LayerNorm по каналу: norm(x.transpose(1,2)).transpose(1,2).
        let x = x.transpose(1, 2)?.contiguous()?; // [B,S,D]
        let x = self.norm.forward(&x)?;
        let x = x.transpose(1, 2)?.contiguous()?; // [B,D,S]
        let x = x.silu()?;
        let x = self.pointwise_conv2.forward(&x)?; // [B,D,S]
        Ok(x.transpose(1, 2)?.contiguous()?) // [B,S,D]
    }
}

/// Conformer-слой (Macaron): 0.5·FF1 → self-attn → conv → 0.5·FF2 → norm_out.
struct ConformerLayer {
    norm_ff1: Ln,
    ff1: FeedForward,
    norm_self_att: Ln,
    attn: Attn,
    norm_conv: Ln,
    conv: ConformerConv,
    norm_ff2: Ln,
    ff2: FeedForward,
    norm_out: Ln,
}

impl ConformerLayer {
    fn forward(&self, x: &Tensor, rope: &RopeCache) -> Result<Tensor> {
        let residual = x.clone();
        let h = self.norm_ff1.forward(x)?;
        let h = self.ff1.forward(&h)?;
        let residual = residual.add(&h.affine(0.5, 0.0)?)?;

        let h = self.norm_self_att.forward(&residual)?;
        let h = self.attn.forward(&h, rope)?;
        let residual = residual.add(&h)?;

        let h = self.norm_conv.forward(&residual)?;
        let h = self.conv.forward(&h)?;
        let residual = residual.add(&h)?;

        let h = self.norm_ff2.forward(&residual)?;
        let h = self.ff2.forward(&h)?;
        let residual = residual.add(&h.affine(0.5, 0.0)?)?;

        self.norm_out.forward(&residual)
    }
}

pub struct GigaAmModel {
    subsampling: Subsampling,
    layers: Vec<ConformerLayer>,
    head_w: Tensor,
    head_b: Tensor,
    rope: RopeCache,
    device: Device,
}

impl GigaAmModel {
    /// mel `[1, feat_in, T]` → энкодер `[1, d_model, T']` (как upstream-хук).
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        let mut h = self.subsampling.forward(mel)?; // [1, T', d_model]
        for layer in &self.layers {
            h = layer.forward(&h, &self.rope)?;
        }
        // upstream encoder.forward возвращает audio_signal.transpose(1,2) -> [B,D,T'].
        Ok(h.transpose(1, 2)?.contiguous()?)
    }

    /// CTC-head: conv1d 1×1 (d_model→num_classes) → log_softmax. Вход энкодер
    /// `[1, d_model, T']`, выход log-probs `[1, T', num_classes]`.
    pub fn head_logits(&self, encoded: &Tensor) -> Result<Tensor> {
        let logits = conv1d(encoded, &self.head_w, Some(&self.head_b), 1, 0)?; // [1,C,T']
        let logits = logits.transpose(1, 2)?.contiguous()?; // [1,T',C]
        Ok(log_softmax_dim(&logits, 2)?)
    }

    pub fn load(w: &GigaAmWeights) -> Result<Self> {
        let cfg = &w.config;
        let enc = &cfg.encoder;
        let pad = (enc.subs_kernel_size - 1) / 2;

        let conv = |prefix: &str, stride: usize, padding: usize, groups: usize| -> Result<Conv1d> {
            Ok(Conv1d {
                weight: w.get(&format!("{prefix}.weight"))?,
                bias: Some(w.get(&format!("{prefix}.bias"))?),
                stride,
                padding,
                groups,
            })
        };

        let subsampling = Subsampling {
            conv0: conv("encoder.pre_encode.conv.0", 2, pad, 1)?,
            conv2: conv("encoder.pre_encode.conv.2", 2, pad, 1)?,
        };

        let lin = |prefix: &str| -> Result<(Tensor, Tensor)> {
            Ok((w.get(&format!("{prefix}.weight"))?, w.get(&format!("{prefix}.bias"))?))
        };
        let ln = |prefix: &str| -> Result<Ln> {
            Ok(Ln {
                weight: w.get(&format!("{prefix}.weight"))?,
                bias: w.get(&format!("{prefix}.bias"))?,
            })
        };

        let conv_pad = (enc.conv_kernel_size - 1) / 2;
        let mut layers = Vec::with_capacity(enc.n_layers);
        for i in 0..enc.n_layers {
            let p = |s: &str| enc_layer(i, s);
            layers.push(ConformerLayer {
                norm_ff1: ln(&p("norm_feed_forward1"))?,
                ff1: FeedForward {
                    linear1: lin(&p("feed_forward1.linear1"))?,
                    linear2: lin(&p("feed_forward1.linear2"))?,
                },
                norm_self_att: ln(&p("norm_self_att"))?,
                attn: Attn {
                    q: lin(&p("self_attn.linear_q"))?,
                    k: lin(&p("self_attn.linear_k"))?,
                    v: lin(&p("self_attn.linear_v"))?,
                    out: lin(&p("self_attn.linear_out"))?,
                    n_heads: enc.n_heads,
                    head_dim: cfg.head_dim(),
                },
                norm_conv: ln(&p("norm_conv"))?,
                conv: ConformerConv {
                    pointwise_conv1: Conv1d {
                        weight: w.get(&p("conv.pointwise_conv1.weight"))?,
                        bias: Some(w.get(&p("conv.pointwise_conv1.bias"))?),
                        stride: 1,
                        padding: 0,
                        groups: 1,
                    },
                    depthwise_conv: Conv1d {
                        weight: w.get(&p("conv.depthwise_conv.weight"))?,
                        bias: Some(w.get(&p("conv.depthwise_conv.bias"))?),
                        stride: 1,
                        padding: conv_pad,
                        groups: enc.d_model,
                    },
                    norm: ln(&p("conv.batch_norm"))?,
                    pointwise_conv2: Conv1d {
                        weight: w.get(&p("conv.pointwise_conv2.weight"))?,
                        bias: Some(w.get(&p("conv.pointwise_conv2.bias"))?),
                        stride: 1,
                        padding: 0,
                        groups: 1,
                    },
                },
                norm_ff2: ln(&p("norm_feed_forward2"))?,
                ff2: FeedForward {
                    linear1: lin(&p("feed_forward2.linear1"))?,
                    linear2: lin(&p("feed_forward2.linear2"))?,
                },
                norm_out: ln(&p("norm_out"))?,
            });
        }

        // RoPE: dim = head_dim, base = pos_emb_max_len (upstream
        // RotaryPositionalEmbedding(d_model//n_heads, pos_emb_max_len)).
        let head_dim = cfg.head_dim();
        let rope = RopeCache::new(
            head_dim,
            enc.pos_emb_max_len,
            enc.pos_emb_max_len as f32,
            w.device,
        )
        .map_err(|e| GigaAmError::Tensor(e))?;

        Ok(Self {
            subsampling,
            layers,
            head_w: w.get("head.decoder_layers.0.weight")?,
            head_b: w.get("head.decoder_layers.0.bias")?,
            rope,
            device: w.device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}
