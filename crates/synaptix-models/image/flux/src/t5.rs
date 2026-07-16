//! T5-XXL encoder (google/t5-v1_1-xxl) — FLUX `text_encoder_2`.
//!
//! Encoder-only, bit-exact к HF `T5EncoderModel`. Особенности T5:
//! - T5LayerNorm = RMSNorm (variance-only, f32-upcast, `*weight`, без `1+w`/bias).
//! - attention БЕЗ масштаба `1/sqrt(d_kv)` (масштаб впитан в веса).
//! - relative position bias: считается один раз (block 0), шарится во все слои.
//! - gated-gelu FFN: `gelu_new(wi_0·x) * (wi_1·x) → wo`.
//! - НЕТ масштаба эмбеддингов, НЕТ BOS, FLUX не маскирует padding.

use synaptix_core::{
    device::Device,
    error::{Result, SynaptixError},
    tensor::Tensor,
};
use synaptix_nn::linear::Linear;
use synaptix_nn::module::Module;
use synaptix_ops::activation::gelu_tanh;
use synaptix_ops::attention::softmax_dim;
use synaptix_ops::norm::rms_norm;

#[derive(Debug, Clone)]
pub struct T5Config {
    pub d_model: usize,
    pub d_ff: usize,
    pub d_kv: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub num_buckets: usize,
    pub max_distance: usize,
    pub eps: f32,
}

impl T5Config {
    /// google/t5-v1_1-xxl (FLUX text_encoder_2).
    pub fn xxl() -> Self {
        Self {
            d_model: 4096,
            d_ff: 10240,
            d_kv: 64,
            num_heads: 64,
            num_layers: 24,
            num_buckets: 32,
            max_distance: 128,
            eps: 1e-6,
        }
    }
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.d_kv
    }
}

/// T5 relative-position bucket (bidirectional encoder). Точный порт
/// `_relative_position_bucket`: num_buckets//2=16, max_exact=8; малые `|rp|<8`
/// → напрямую, большие → `8 + trunc(log(|rp|/8)/log(16)*8)` clamp до 15; +16 если rp>0.
fn relative_position_bucket(rp: i64, num_buckets: i64, max_distance: i64) -> i64 {
    let mut ret = 0i64;
    let nb = num_buckets / 2; // 16
    if rp > 0 {
        ret += nb;
    }
    let n = rp.abs();
    let max_exact = nb / 2; // 8
    if n < max_exact {
        ret + n
    } else {
        let large = ((n as f64 / max_exact as f64).ln()
            / (max_distance as f64 / max_exact as f64).ln()
            * (nb - max_exact) as f64) as i64; // trunc к нулю = .to(long)
        let rp_if_large = (max_exact + large).min(nb - 1);
        ret + rp_if_large
    }
}

/// position_bias `[1, H, S, S]` = lookup `rel_bias[bucket(k-q)]`, permute в head-major.
fn build_position_bias(
    rel_bias_weight: &Tensor, // [num_buckets, H]
    s: usize,
    num_heads: usize,
    cfg: &T5Config,
    device: Device,
) -> Result<Tensor> {
    let mut buckets = vec![0u32; s * s];
    for q in 0..s {
        for k in 0..s {
            let b = relative_position_bucket(
                (k as i64) - (q as i64),
                cfg.num_buckets as i64,
                cfg.max_distance as i64,
            );
            buckets[q * s + k] = b as u32;
        }
    }
    let idx = Tensor::from_vec(buckets, (s * s,), device)?;
    let vals = rel_bias_weight.index_select(0, &idx)?; // [s*s, H]
    // [s*s,H] -> [s,s,H] -> [H,s,s] -> [1,H,s,s]
    vals.reshape((s, s, num_heads))?
        .permute([2, 0, 1])?
        .contiguous()?
        .unsqueeze(0)
}

struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    num_heads: usize,
    d_kv: usize,
}

impl T5Attention {
    fn forward(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let d = x.dims();
        let (b, s) = (d[0], d[1]);
        let (h, dh) = (self.num_heads, self.d_kv);
        // [b,s,inner] -> [b,h,s,dh]
        let to_heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, s, h, dh))?.transpose(1, 2)?.contiguous()
        };
        let q = to_heads(self.q.forward(x)?)?;
        let k = to_heads(self.k.forward(x)?)?;
        let v = to_heads(self.v.forward(x)?)?;
        // scores = q @ k^T (БЕЗ масштаба) + bias -> softmax -> @ v
        let scores = q.matmul(&k.transpose(2, 3)?)?; // [b,h,s,s]
        let scores = scores.broadcast_add(bias)?; // bias [1,h,s,s]
        let attn = softmax_dim(&scores, 3)?;
        let out = attn.matmul(&v)?; // [b,h,s,dh]
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, s, h * dh))?;
        self.o.forward(&out)
    }
}

struct T5Block {
    ln0: Tensor,
    attn: T5Attention,
    ln1: Tensor,
    wi_0: Linear,
    wi_1: Linear,
    wo: Linear,
    eps: f32,
}

impl T5Block {
    fn forward(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let n = rms_norm(x, &self.ln0, self.eps)?;
        let a = self.attn.forward(&n, bias)?;
        let x = x.add(&a)?;
        let n2 = rms_norm(&x, &self.ln1, self.eps)?;
        let g = gelu_tanh(&self.wi_0.forward(&n2)?)?;
        let l = self.wi_1.forward(&n2)?;
        let ff = self.wo.forward(&g.mul(&l)?)?;
        x.add(&ff)
    }
}

pub struct T5Encoder {
    embed: Tensor, // shared.weight [vocab, d_model]
    rel_bias: Tensor,
    blocks: Vec<T5Block>,
    final_ln: Tensor,
    config: T5Config,
}

impl T5Encoder {
    pub fn config(&self) -> &T5Config {
        &self.config
    }

    pub fn load<F>(cfg: &T5Config, get: &F) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        let lin = |name: &str| -> Result<Linear> { Linear::new(get(name)?, None) };
        let embed = get("shared.weight")?;
        let rel_bias =
            get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("encoder.block.{i}");
            blocks.push(T5Block {
                ln0: get(&format!("{p}.layer.0.layer_norm.weight"))?,
                attn: T5Attention {
                    q: lin(&format!("{p}.layer.0.SelfAttention.q.weight"))?,
                    k: lin(&format!("{p}.layer.0.SelfAttention.k.weight"))?,
                    v: lin(&format!("{p}.layer.0.SelfAttention.v.weight"))?,
                    o: lin(&format!("{p}.layer.0.SelfAttention.o.weight"))?,
                    num_heads: cfg.num_heads,
                    d_kv: cfg.d_kv,
                },
                ln1: get(&format!("{p}.layer.1.layer_norm.weight"))?,
                wi_0: lin(&format!("{p}.layer.1.DenseReluDense.wi_0.weight"))?,
                wi_1: lin(&format!("{p}.layer.1.DenseReluDense.wi_1.weight"))?,
                wo: lin(&format!("{p}.layer.1.DenseReluDense.wo.weight"))?,
                eps: cfg.eps,
            });
        }
        let final_ln = get("encoder.final_layer_norm.weight")?;
        Ok(Self { embed, rel_bias, blocks, final_ln, config: cfg.clone() })
    }

    /// `input_ids: [B, S]` (U32) → `last_hidden_state: [B, S, d_model]`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let d = input_ids.dims();
        let (b, s) = (d[0], d[1]);
        if input_ids.device() != self.embed.device() {
            return Err(SynaptixError::device_mismatch(
                input_ids.device(),
                self.embed.device(),
            ));
        }
        // embedding lookup (без масштаба): [b*s] -> [b*s, d] -> [b,s,d]
        let ids_flat = input_ids.reshape((b * s,))?;
        let h = self
            .embed
            .index_select(0, &ids_flat)?
            .reshape((b, s, self.config.d_model))?;
        let bias = build_position_bias(
            &self.rel_bias,
            s,
            self.config.num_heads,
            &self.config,
            input_ids.device(),
        )?;
        let mut h = h;
        for blk in &self.blocks {
            h = blk.forward(&h, &bias)?;
        }
        rms_norm(&h, &self.final_ln, self.config.eps)
    }
}
