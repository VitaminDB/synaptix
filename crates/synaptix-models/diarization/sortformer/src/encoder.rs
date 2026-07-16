//! FastConformer-энкодер (NeMo `ConformerEncoder`, self_attention_model="rel_pos").
//!
//! Источник истины: NeMo `conformer_encoder.py` + `parts/submodules/{subsampling,
//! conformer_modules, multi_head_attention}.py`. Прогон batch=1 (все кадры валидны →
//! att/pad-маски не нужны). Цепочка:
//!   mel (1,128,T) → DwStriding8x subsampling (conv2d) → (1,T',512) → xscale(√512)
//!   → 17× ConformerLayer (Macaron ½FFN-SiLU → RelPosMHSA(rel_shift) → ConvModule
//!     (GLU+depthwise+BatchNorm1d+SiLU) → ½FFN → LN) → (1,T',512).
//! pos_emb (1,2T'−1,512) — интерливинг sin/cos, считается один раз на всю длину.

use synaptix_core::tensor::Tensor;
use synaptix_ops::activation::glu;
use synaptix_ops::attention::softmax_dim;
use synaptix_ops::conv::{conv1d, conv2d};
use synaptix_ops::norm::layer_norm;

use crate::config::FastConformerConfig;
use crate::loader::{enc_layer, SortformerWeights};
use crate::Result;

struct Ln {
    weight: Tensor,
    bias: Tensor,
}
impl Ln {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(layer_norm(x, Some(&self.weight), Some(&self.bias), 1e-5)?)
    }
}

/// BatchNorm1d (inference, frozen running stats): y = x·scale + shift по каналу,
/// где scale = weight/√(var+eps), shift = bias − mean·scale. scale/shift = (1,C,1).
struct BatchNorm1d {
    scale: Tensor,
    shift: Tensor,
}
impl BatchNorm1d {
    /// Вход/выход `[B,C,T]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.broadcast_mul(&self.scale)?.broadcast_add(&self.shift)?)
    }
}

/// FFN: linear2(SiLU(linear1(x))). Веса nn.Linear (out,in) → `x.linear(w)` = x·Wᵀ.
struct FeedForward {
    l1: (Tensor, Tensor),
    l2: (Tensor, Tensor),
}
impl FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.linear(&self.l1.0)?.broadcast_add(&self.l1.1)?.silu()?;
        Ok(h.linear(&self.l2.0)?.broadcast_add(&self.l2.1)?)
    }
}

/// RelPosMHSA (Transformer-XL / T5-XL). scores = (matrix_ac + rel_shift(matrix_bd))/√d_k.
struct RelAttn {
    q: (Tensor, Tensor),
    k: (Tensor, Tensor),
    v: (Tensor, Tensor),
    out: (Tensor, Tensor),
    linear_pos: Tensor, // (n_feat,n_feat) bias=None
    pos_bias_u: Tensor, // (1,1,H,d_k)
    pos_bias_v: Tensor, // (1,1,H,d_k)
    n_heads: usize,
    d_k: usize,
}
impl RelAttn {
    fn proj(x: &Tensor, w: &(Tensor, Tensor)) -> Result<Tensor> {
        Ok(x.linear(&w.0)?.broadcast_add(&w.1)?)
    }
    /// `[B,S,D] -> [B,H,S,d_k]`.
    fn heads(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.dims();
        let (b, s) = (d[0], d[1]);
        Ok(x.reshape(vec![b, s, self.n_heads, self.d_k])?.permute(vec![0, 2, 1, 3])?.contiguous()?)
    }

    /// NeMo `rel_shift`: x (B,H,L,P=2L−1) → сдвинутая (B,H,L,P).
    fn rel_shift(x: &Tensor) -> Result<Tensor> {
        let d = x.dims();
        let (b, h, l, p) = (d[0], d[1], d[2], d[3]);
        let zeros = x.narrow(3, 0, 1)?.zeros_like()?; // (B,H,L,1)
        let x = Tensor::cat(&[&zeros, x], 3)?; // pad left 1 → (B,H,L,P+1)
        let x = x.reshape(vec![b, h, p + 1, l])?; // view (B,H,P+1,L)
        let x = x.narrow(2, 1, p)?.contiguous()?; // drop row 0 → (B,H,P,L)
        Ok(x.reshape(vec![b, h, l, p])?) // view (B,H,L,P)
    }

    /// `x` (1,L,D), `pos_emb` (1,2L−1,D) → (1,L,D).
    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let d = x.dims();
        let (b, l) = (d[0], d[1]);
        let q = self.heads(&Self::proj(x, &self.q)?)?; // (B,H,L,d_k)
        let k = self.heads(&Self::proj(x, &self.k)?)?;
        let v = self.heads(&Self::proj(x, &self.v)?)?;

        // p = linear_pos(pos_emb) → (1,P,H,d_k) → (1,H,P,d_k).
        let pe = pos_emb.dims().to_vec();
        let p = pos_emb
            .linear(&self.linear_pos)?
            .reshape(vec![pe[0], pe[1], self.n_heads, self.d_k])?
            .permute(vec![0, 2, 1, 3])?
            .contiguous()?;

        // q_t (B,L,H,d_k) + pos_bias → transpose(1,2) → (B,H,L,d_k).
        let q_t = q.permute(vec![0, 2, 1, 3])?.contiguous()?; // (B,L,H,d_k)
        let q_u = q_t.broadcast_add(&self.pos_bias_u)?.permute(vec![0, 2, 1, 3])?.contiguous()?;
        let q_v = q_t.broadcast_add(&self.pos_bias_v)?.permute(vec![0, 2, 1, 3])?.contiguous()?;

        let matrix_ac = q_u.matmul(&k.transpose(2, 3)?.contiguous()?)?; // (B,H,L,L)
        let matrix_bd = q_v.matmul(&p.transpose(2, 3)?.contiguous()?)?; // (B,H,L,P)
        let matrix_bd = Self::rel_shift(&matrix_bd)?; // (B,H,L,P)
        let matrix_bd = matrix_bd.narrow(3, 0, l)?.contiguous()?; // (B,H,L,L)

        let scale = 1.0f32 / (self.d_k as f32).sqrt();
        let scores = matrix_ac.add(&matrix_bd)?.affine(scale, 0.0)?;
        let attn = softmax_dim(&scores, 3)?; // batch=1 → без маски
        let ctx = attn.matmul(&v)?; // (B,H,L,d_k)
        let merged = ctx.permute(vec![0, 2, 1, 3])?.contiguous()?.reshape(vec![b, l, self.n_heads * self.d_k])?;
        Self::proj(&merged, &self.out)
    }
}

/// ConformerConvolution: pointwise1(→2D)→GLU→depthwise(k9,pad4)→BatchNorm1d→SiLU→pointwise2.
struct ConvModule {
    pw1: (Tensor, Tensor), // conv1d 1×1 weight (2D,D,1)
    dw: (Tensor, Tensor),  // depthwise conv1d (D,1,K)
    bn: BatchNorm1d,
    pw2: (Tensor, Tensor),
    dw_pad: usize,
    d_model: usize,
}
impl ConvModule {
    /// `[B,S,D]` → `[B,S,D]` (внутри `[B,D,S]`).
    fn depthwise(&self, x: &Tensor) -> Result<Tensor> {
        let c = x.dims()[1];
        let mut chans: Vec<Tensor> = Vec::with_capacity(c);
        for ci in 0..c {
            let x_c = x.narrow(1, ci, 1)?.contiguous()?; // [B,1,S]
            let w_c = self.dw.0.narrow(0, ci, 1)?.contiguous()?; // [1,1,K]
            let b_c = self.dw.1.narrow(0, ci, 1)?.contiguous()?;
            chans.push(conv1d(&x_c, &w_c, Some(&b_c), 1, self.dw_pad)?);
        }
        let refs: Vec<&Tensor> = chans.iter().collect();
        Ok(Tensor::cat(&refs, 1)?)
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?.contiguous()?; // [B,D,S]
        let x = conv1d(&x, &self.pw1.0, Some(&self.pw1.1), 1, 0)?; // [B,2D,S]
        let x = glu(&x, 1)?; // [B,D,S]
        let _ = self.d_model;
        let x = self.depthwise(&x)?; // [B,D,S]
        let x = self.bn.forward(&x)?; // BatchNorm1d по [B,D,S]
        let x = x.silu()?;
        let x = conv1d(&x, &self.pw2.0, Some(&self.pw2.1), 1, 0)?; // [B,D,S]
        Ok(x.transpose(1, 2)?.contiguous()?) // [B,S,D]
    }
}

/// FastConformer-слой (Macaron): ½FF1 → RelPosMHSA → Conv → ½FF2 → norm_out.
struct ConformerLayer {
    norm_ff1: Ln,
    ff1: FeedForward,
    norm_attn: Ln,
    attn: RelAttn,
    norm_conv: Ln,
    conv: ConvModule,
    norm_ff2: Ln,
    ff2: FeedForward,
    norm_out: Ln,
}
impl ConformerLayer {
    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let residual = x.add(&self.ff1.forward(&self.norm_ff1.forward(x)?)?.affine(0.5, 0.0)?)?;
        let residual = residual.add(&self.attn.forward(&self.norm_attn.forward(&residual)?, pos_emb)?)?;
        let residual = residual.add(&self.conv.forward(&self.norm_conv.forward(&residual)?)?)?;
        let residual = residual.add(&self.ff2.forward(&self.norm_ff2.forward(&residual)?)?.affine(0.5, 0.0)?)?;
        self.norm_out.forward(&residual)
    }
}

/// DwStriding8x subsampling (3 стадии stride-2, conv2d): mel (1,128,T) → (1,T',512).
struct Subsampling {
    conv0: (Tensor, Tensor),   // (256,1,3,3)
    dw1: (Tensor, Tensor),     // (256,1,3,3) groups=256
    pw1: (Tensor, Tensor),     // (256,256,1,1)
    dw2: (Tensor, Tensor),
    pw2: (Tensor, Tensor),
    out: (Tensor, Tensor),     // Linear(256*16 → 512)
}
impl Subsampling {
    fn depthwise2d(w: &Tensor, b: &Tensor, x: &Tensor) -> Result<Tensor> {
        let c = x.dims()[1];
        let mut chans: Vec<Tensor> = Vec::with_capacity(c);
        for ci in 0..c {
            let x_c = x.narrow(1, ci, 1)?.contiguous()?; // (1,1,H,W)
            let w_c = w.narrow(0, ci, 1)?.contiguous()?; // (1,1,3,3)
            let b_c = b.narrow(0, ci, 1)?.contiguous()?;
            chans.push(conv2d(&x_c, &w_c, Some(&b_c), (2, 2), (1, 1), (1, 1))?);
        }
        let refs: Vec<&Tensor> = chans.iter().collect();
        Ok(Tensor::cat(&refs, 1)?)
    }
    /// mel `[1,128,T]` → `[1,T',512]`.
    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // (1,128,T) → (1,T,128) → (1,1,T,128).
        let x = mel.transpose(1, 2)?.contiguous()?.unsqueeze(1)?;
        // conv0 (regular, stride2 pad1) + ReLU.
        let x = conv2d(&x, &self.conv0.0, Some(&self.conv0.1), (2, 2), (1, 1), (1, 1))?.relu()?;
        // dw1 → pw1(k1) → ReLU.
        let x = Self::depthwise2d(&self.dw1.0, &self.dw1.1, &x)?;
        let x = conv2d(&x, &self.pw1.0, Some(&self.pw1.1), (1, 1), (0, 0), (1, 1))?.relu()?;
        // dw2 → pw2(k1) → ReLU.
        let x = Self::depthwise2d(&self.dw2.0, &self.dw2.1, &x)?;
        let x = conv2d(&x, &self.pw2.0, Some(&self.pw2.1), (1, 1), (0, 0), (1, 1))?.relu()?;
        // (1,256,T',16) → transpose(1,2) → (1,T',256,16) → reshape (1,T',4096) → out.
        let d = x.dims().to_vec();
        let (t, c, f) = (d[2], d[1], d[3]);
        let x = x.transpose(1, 2)?.contiguous()?.reshape(vec![1, t, c * f])?;
        Ok(x.linear(&self.out.0)?.broadcast_add(&self.out.1)?)
    }
}

pub struct FastConformer {
    subsampling: Subsampling,
    layers: Vec<ConformerLayer>,
    xscale: f32,
    d_model: usize,
    n_heads: usize,
}

impl FastConformer {
    /// pre_encode (subsampling): mel `[1,128,T]` → `[1,T',512]` (NeMo `encoder.pre_encode`).
    pub fn pre_encode(&self, mel: &Tensor) -> Result<Tensor> {
        self.subsampling.forward(mel)
    }

    /// bypass_pre_encode: эмбеддинги `[1,L,512]` → encoder states `[1,512,L]`
    /// (xscale + pos_emb + 17 слоёв + transpose). Для streaming (spkcache+fifo+chunk).
    pub fn encode_bypass(&self, emb: &Tensor) -> Result<Tensor> {
        let mut x = emb.affine(self.xscale, 0.0)?; // xscaling внутри pos_enc
        let l = x.dims()[1];
        let pos_emb = self.pos_emb(l, &x)?; // (1,2L−1,512)
        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb)?;
        }
        Ok(x.transpose(1, 2)?.contiguous()?) // (1,512,L)
    }

    /// mel `[1,128,T]` → encoder states `[1,512,T']` (как NeMo `encoder.forward`).
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        self.encode_bypass(&self.pre_encode(mel)?)
    }

    /// Постадийная отладка: (preenc pre-xscale, layer0, layer8, layer16, final), все (1,T',512).
    pub fn forward_debug(&self, mel: &Tensor) -> Result<Vec<(String, Tensor)>> {
        let preenc = self.subsampling.forward(mel)?; // pre-xscale
        let mut x = preenc.affine(self.xscale, 0.0)?;
        let l = x.dims()[1];
        let pos_emb = self.pos_emb(l, &x)?;
        let mut out = vec![("preenc".to_string(), preenc)];
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &pos_emb)?;
            if i == 0 || i == 8 || i == 16 {
                out.push((format!("enc_l{i}"), x.clone()));
            }
        }
        out.push(("final".to_string(), x.transpose(1, 2)?.contiguous()?));
        Ok(out)
    }

    /// RelPositionalEncoding: позиции [L−1 … −(L−1)], pe[:,0::2]=sin, pe[:,1::2]=cos,
    /// div_term = exp(arange(0,D,2)·−ln(10000)/D). Возвращает (1,2L−1,D).
    fn pos_emb(&self, l: usize, like: &Tensor) -> Result<Tensor> {
        let d = self.d_model;
        let p = 2 * l - 1;
        let inf = 10000.0f32;
        let half = d / 2;
        let div: Vec<f32> = (0..half).map(|i| (-(inf.ln()) / d as f32 * (2 * i) as f32).exp()).collect();
        let mut pe = vec![0.0f32; p * d];
        for (row, pos) in (0..p).map(|r| (r, (l as i64 - 1) - r as i64)).take(p) {
            let pf = pos as f32;
            for i in 0..half {
                let phase = pf * div[i];
                pe[row * d + 2 * i] = phase.sin();
                pe[row * d + 2 * i + 1] = phase.cos();
            }
        }
        Ok(Tensor::from_vec(pe, (1, p, d), like.device())?.to_dtype(like.dtype())?)
    }

    pub fn load(w: &SortformerWeights) -> Result<Self> {
        let cfg = &w.config.encoder;
        let lin = |name: &str| -> Result<(Tensor, Tensor)> {
            Ok((w.get(&format!("{name}.weight"))?, w.get(&format!("{name}.bias"))?))
        };
        let ln = |name: &str| -> Result<Ln> {
            Ok(Ln { weight: w.get(&format!("{name}.weight"))?, bias: w.get(&format!("{name}.bias"))? })
        };

        let subsampling = Subsampling {
            conv0: lin("encoder.pre_encode.conv.0")?,
            dw1: lin("encoder.pre_encode.conv.2")?,
            pw1: lin("encoder.pre_encode.conv.3")?,
            dw2: lin("encoder.pre_encode.conv.5")?,
            pw2: lin("encoder.pre_encode.conv.6")?,
            out: lin("encoder.pre_encode.out")?,
        };

        let d_k = cfg.d_model / cfg.n_heads;
        let dw_pad = (cfg.conv_kernel_size - 1) / 2;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| enc_layer(i, s);
            // BatchNorm1d: precompute scale/shift в F32.
            let bn = Self::batch_norm(w, &p("conv.batch_norm"), cfg)?;
            // pos_bias_u/v (H,d_k) → (1,1,H,d_k).
            let pbu = w.get(&p("self_attn.pos_bias_u"))?.reshape(vec![1, 1, cfg.n_heads, d_k])?;
            let pbv = w.get(&p("self_attn.pos_bias_v"))?.reshape(vec![1, 1, cfg.n_heads, d_k])?;
            layers.push(ConformerLayer {
                norm_ff1: ln(&p("norm_feed_forward1"))?,
                ff1: FeedForward { l1: lin(&p("feed_forward1.linear1"))?, l2: lin(&p("feed_forward1.linear2"))? },
                norm_attn: ln(&p("norm_self_att"))?,
                attn: RelAttn {
                    q: lin(&p("self_attn.linear_q"))?,
                    k: lin(&p("self_attn.linear_k"))?,
                    v: lin(&p("self_attn.linear_v"))?,
                    out: lin(&p("self_attn.linear_out"))?,
                    linear_pos: w.get(&p("self_attn.linear_pos.weight"))?,
                    pos_bias_u: pbu,
                    pos_bias_v: pbv,
                    n_heads: cfg.n_heads,
                    d_k,
                },
                norm_conv: ln(&p("norm_conv"))?,
                conv: ConvModule {
                    pw1: lin(&p("conv.pointwise_conv1"))?,
                    dw: lin(&p("conv.depthwise_conv"))?,
                    bn,
                    pw2: lin(&p("conv.pointwise_conv2"))?,
                    dw_pad,
                    d_model: cfg.d_model,
                },
                norm_ff2: ln(&p("norm_feed_forward2"))?,
                ff2: FeedForward { l1: lin(&p("feed_forward2.linear1"))?, l2: lin(&p("feed_forward2.linear2"))? },
                norm_out: ln(&p("norm_out"))?,
            });
        }

        let xscale = if cfg.xscaling { (cfg.d_model as f32).sqrt() } else { 1.0 };
        Ok(Self { subsampling, layers, xscale, d_model: cfg.d_model, n_heads: cfg.n_heads })
    }

    /// scale = weight/√(var+eps), shift = bias − mean·scale; → (1,C,1) тензоры.
    fn batch_norm(w: &SortformerWeights, name: &str, cfg: &FastConformerConfig) -> Result<BatchNorm1d> {
        use synaptix_core::dtype::DType;
        let weight: Vec<f32> = w.get_dtype(&format!("{name}.weight"), DType::F32)?.to_vec1::<f32>()?;
        let bias: Vec<f32> = w.get_dtype(&format!("{name}.bias"), DType::F32)?.to_vec1::<f32>()?;
        let mean: Vec<f32> = w.get_dtype(&format!("{name}.running_mean"), DType::F32)?.to_vec1::<f32>()?;
        let var: Vec<f32> = w.get_dtype(&format!("{name}.running_var"), DType::F32)?.to_vec1::<f32>()?;
        let eps = 1e-5f32;
        let c = cfg.d_model;
        let mut scale = vec![0.0f32; c];
        let mut shift = vec![0.0f32; c];
        for i in 0..c {
            let s = weight[i] / (var[i] + eps).sqrt();
            scale[i] = s;
            shift[i] = bias[i] - mean[i] * s;
        }
        let dev = w.device;
        // привести к compute-dtype через from_vec(f32)→to_dtype.
        let scale_t = Tensor::from_vec(scale, (1, c, 1), dev)?.to_dtype(w.dtype)?;
        let shift_t = Tensor::from_vec(shift, (1, c, 1), dev)?.to_dtype(w.dtype)?;
        Ok(BatchNorm1d { scale: scale_t, shift: shift_t })
    }

    pub fn n_heads(&self) -> usize {
        self.n_heads
    }
}
