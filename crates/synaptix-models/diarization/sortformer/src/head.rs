//! Sortformer-head: encoder_proj (512→192) → 18× POST-LN transformer-слой (NeMo
//! `TransformerEncoder`, pre_ln=False, без финального LN) → sigmoid-голова.
//!
//! Источник истины: NeMo `modules/transformer/{transformer_encoders,transformer_modules}.py`
//! + `sortformer_modules.forward_speaker_sigmoids`. Прогон batch=1 (маска не нужна).
//!   layer (post-LN): a=MHA(x); a+=x; a=LN1(a); f=FF(a); f+=a; out=LN2(f).
//!   MHA: q,k = q/k_net(x) / (head_size^0.25); scores=q·kᵀ; softmax; ·v; out_proj.
//!   FF (PositionWiseFF): dense_out(ReLU(dense_in(x))).
//!   sigmoid-голова: ReLU → hidden_proj(192→192) → ReLU → classifier(192→4) → sigmoid.

use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::softmax_dim;
use synaptix_ops::norm::layer_norm;

use crate::config::SortformerHeadConfig;
use crate::loader::{head_layer, SortformerWeights};
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

fn proj(x: &Tensor, w: &(Tensor, Tensor)) -> Result<Tensor> {
    Ok(x.linear(&w.0)?.broadcast_add(&w.1)?)
}

struct Mha {
    q: (Tensor, Tensor),
    k: (Tensor, Tensor),
    v: (Tensor, Tensor),
    out: (Tensor, Tensor),
    n_heads: usize,
    head_size: usize,
}
impl Mha {
    fn heads(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.dims();
        let (b, s) = (d[0], d[1]);
        Ok(x.reshape(vec![b, s, self.n_heads, self.head_size])?.permute(vec![0, 2, 1, 3])?.contiguous()?)
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let d = x.dims();
        let (b, s) = (d[0], d[1]);
        let inv = 1.0f32 / (self.head_size as f32).sqrt().sqrt(); // 1/head_size^0.25
        let q = self.heads(&proj(x, &self.q)?)?.affine(inv, 0.0)?;
        let k = self.heads(&proj(x, &self.k)?)?.affine(inv, 0.0)?;
        let v = self.heads(&proj(x, &self.v)?)?;
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?; // (B,H,S,S)
        let attn = softmax_dim(&scores, 3)?;
        let ctx = attn.matmul(&v)?; // (B,H,S,hd)
        let merged =
            ctx.permute(vec![0, 2, 1, 3])?.contiguous()?.reshape(vec![b, s, self.n_heads * self.head_size])?;
        proj(&merged, &self.out)
    }
}

struct Ff {
    dense_in: (Tensor, Tensor),
    dense_out: (Tensor, Tensor),
}
impl Ff {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = proj(x, &self.dense_in)?.relu()?;
        proj(&h, &self.dense_out)
    }
}

struct HeadLayer {
    mha: Mha,
    ln1: Ln,
    ff: Ff,
    ln2: Ln,
}
impl HeadLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = self.mha.forward(x)?.add(x)?;
        let a = self.ln1.forward(&a)?;
        let f = self.ff.forward(&a)?.add(&a)?;
        self.ln2.forward(&f)
    }
}

pub struct SortformerHead {
    encoder_proj: (Tensor, Tensor),
    layers: Vec<HeadLayer>,
    hidden_proj: (Tensor, Tensor),
    classifier: (Tensor, Tensor),
}

impl SortformerHead {
    /// encoder_proj: `[1,T',512]` → `[1,T',192]` (NeMo `sortformer_modules.encoder_proj`).
    pub fn project(&self, emb: &Tensor) -> Result<Tensor> {
        proj(emb, &self.encoder_proj)
    }

    /// 18× post-LN transformer-слой: `[1,T',192]` → `[1,T',192]`.
    pub fn transformer(&self, emb_seq: &Tensor) -> Result<Tensor> {
        let mut x = emb_seq.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    /// forward_speaker_sigmoids: ReLU → hidden_proj → ReLU → classifier → sigmoid.
    pub fn sigmoids(&self, trans: &Tensor) -> Result<Tensor> {
        let h = trans.relu()?;
        let h = proj(&h, &self.hidden_proj)?.relu()?;
        let logits = proj(&h, &self.classifier)?;
        Ok(logits.sigmoid()?)
    }

    /// encoder states `[1,T',512]` → per-speaker probs `[1,T',n_spk]`.
    pub fn forward(&self, emb: &Tensor) -> Result<Tensor> {
        let emb_seq = self.project(emb)?;
        let trans = self.transformer(&emb_seq)?;
        self.sigmoids(&trans)
    }

    pub fn load(w: &SortformerWeights) -> Result<Self> {
        let cfg: &SortformerHeadConfig = &w.config.head;
        let lin = |name: &str| -> Result<(Tensor, Tensor)> {
            Ok((w.get(&format!("{name}.weight"))?, w.get(&format!("{name}.bias"))?))
        };
        let ln = |name: &str| -> Result<Ln> {
            Ok(Ln { weight: w.get(&format!("{name}.weight"))?, bias: w.get(&format!("{name}.bias"))? })
        };

        let head_size = cfg.d_model / cfg.n_heads;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| head_layer(i, s);
            layers.push(HeadLayer {
                mha: Mha {
                    q: lin(&p("self_attn.linear_q"))?,
                    k: lin(&p("self_attn.linear_k"))?,
                    v: lin(&p("self_attn.linear_v"))?,
                    out: lin(&p("self_attn.linear_out"))?,
                    n_heads: cfg.n_heads,
                    head_size,
                },
                ln1: ln(&p("norm1"))?,
                ff: Ff {
                    dense_in: lin(&p("feed_forward.linear1"))?,
                    dense_out: lin(&p("feed_forward.linear2"))?,
                },
                ln2: ln(&p("norm2"))?,
            });
        }

        Ok(Self {
            encoder_proj: lin("head.encoder_proj")?,
            layers,
            hidden_proj: lin("head.hidden_proj")?,
            classifier: lin("head.classifier")?,
        })
    }
}
