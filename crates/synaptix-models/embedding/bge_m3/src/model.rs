//! XLM-RoBERTa encoder BGE-M3 (post-LN BERT-стек, абсолютные позиции, exact-gelu).
//!
//! Источник истины — HF `XLMRobertaModel` (transformers). Dense-эмбеддинг BGE-M3 =
//! `L2-normalize(last_hidden[:,0,:])` (CLS; пулера НЕТ — `pooler.*` отсутствует).
//!
//! Forward:
//!   position_ids (RoBERTa-сдвиг): `mask=(ids!=pad)`, `pos = cumsum(mask)*mask + pad`
//!     (первый реальный токен → позиция pad+1). embeddings = word[ids] +
//!     position[pos] + token_type[0]; LayerNorm(eps).
//!   24 POST-LN слоя: self-attn (Q/K/V Linear+bias, scale 1/√hd, softmax по ключам
//!     с −inf на pad-ключах) → output.dense(+bias)+residual → output.LayerNorm;
//!     intermediate.dense(+bias)→gelu_exact→output.dense(+bias)+residual→output.LayerNorm.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::layer_norm::layer_norm;

use crate::config::BgeConfig;
use crate::loader::{layer_key, BgeWeights};
use crate::BgeError;

fn err<E: std::fmt::Display>(e: E) -> BgeError {
    BgeError::Inference(e.to_string())
}

struct SelfAttn {
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    out_ln_w: Tensor,
    out_ln_b: Tensor,
}

struct Ffn {
    inter_w: Tensor,
    inter_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    out_ln_w: Tensor,
    out_ln_b: Tensor,
}

struct Layer {
    attn: SelfAttn,
    ffn: Ffn,
}

/// XLM-RoBERTa encoder (BGE-M3 dense backbone).
pub struct BgeEncoder {
    word_embeddings: Tensor,
    position_embeddings: Tensor,
    token_type_embeddings: Tensor,
    emb_ln_w: Tensor,
    emb_ln_b: Tensor,
    layers: Vec<Layer>,
    cfg: BgeConfig,
    device: Device,
    dtype: DType,
}

impl BgeEncoder {
    pub fn build(cfg: &BgeConfig, weights: &BgeWeights) -> Result<Self, BgeError> {
        let device = weights.device;
        let dtype = weights.dtype;
        let get = |name: &str| -> Result<Tensor, BgeError> { weights.get(name).map(|t| t.clone()) };

        let word_embeddings = get("embeddings.word_embeddings.weight")?;
        let position_embeddings = get("embeddings.position_embeddings.weight")?;
        let token_type_embeddings = get("embeddings.token_type_embeddings.weight")?;
        let emb_ln_w = get("embeddings.LayerNorm.weight")?;
        let emb_ln_b = get("embeddings.LayerNorm.bias")?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let k = |s: &str| layer_key(i, s);
            layers.push(Layer {
                attn: SelfAttn {
                    q_w: get(&k("attention.self.query.weight"))?,
                    q_b: get(&k("attention.self.query.bias"))?,
                    k_w: get(&k("attention.self.key.weight"))?,
                    k_b: get(&k("attention.self.key.bias"))?,
                    v_w: get(&k("attention.self.value.weight"))?,
                    v_b: get(&k("attention.self.value.bias"))?,
                    out_w: get(&k("attention.output.dense.weight"))?,
                    out_b: get(&k("attention.output.dense.bias"))?,
                    out_ln_w: get(&k("attention.output.LayerNorm.weight"))?,
                    out_ln_b: get(&k("attention.output.LayerNorm.bias"))?,
                },
                ffn: Ffn {
                    inter_w: get(&k("intermediate.dense.weight"))?,
                    inter_b: get(&k("intermediate.dense.bias"))?,
                    out_w: get(&k("output.dense.weight"))?,
                    out_b: get(&k("output.dense.bias"))?,
                    out_ln_w: get(&k("output.LayerNorm.weight"))?,
                    out_ln_b: get(&k("output.LayerNorm.bias"))?,
                },
            });
        }

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            emb_ln_w,
            emb_ln_b,
            layers,
            cfg: cfg.clone(),
            device,
            dtype,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn config(&self) -> &BgeConfig {
        &self.cfg
    }

    fn ln(&self, x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor, BgeError> {
        layer_norm(x, Some(w), Some(b), self.cfg.layer_norm_eps as f32).map_err(err)
    }

    fn linear_bias(&self, x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor, BgeError> {
        x.linear(w).and_then(|t| t.broadcast_add(b)).map_err(err)
    }

    /// RoBERTa position_ids: `mask=(ids!=pad)`, `pos = cumsum(mask)*mask + pad`.
    /// `ids` — host I64 [B*S]. Возвращает host I64 [B*S].
    fn position_ids(&self, ids: &[i64], bsz: usize, s: usize) -> Vec<i64> {
        let pad = self.cfg.pad_token_id;
        let mut pos = vec![0i64; bsz * s];
        for bi in 0..bsz {
            let mut cum = 0i64;
            for si in 0..s {
                let id = ids[bi * s + si];
                let m = (id != pad) as i64;
                cum += m;
                pos[bi * s + si] = cum * m + pad;
            }
        }
        pos
    }

    /// `input_ids` [B,S] (I64), `attention_mask` [B,S] (1=real, 0=pad). Возвращает
    /// last_hidden_state [B,S,hidden].
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor, BgeError> {
        let dims = input_ids.dims().to_vec();
        let (bsz, s) = (dims[0], dims[1]);
        let hidden = self.cfg.hidden_size;
        let nh = self.cfg.num_attention_heads;
        let hd = self.cfg.head_dim();
        let scale = 1.0f32 / (hd as f32).sqrt();

        let ids = input_ids
            .to_dtype(DType::I64)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<i64>())
            .map_err(err)?;
        let mask_vals = attention_mask
            .to_dtype(DType::I64)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<i64>())
            .map_err(err)?;

        // word + position + token_type(0) embeddings.
        let ids_t = Tensor::from_vec(ids.clone(), vec![bsz * s], self.device).map_err(err)?;
        let word = self
            .word_embeddings
            .index_select(0, &ids_t)
            .and_then(|t| t.reshape(vec![bsz, s, hidden]))
            .map_err(err)?;

        let pos_ids = self.position_ids(&ids, bsz, s);
        let pos_t = Tensor::from_vec(pos_ids, vec![bsz * s], self.device).map_err(err)?;
        let pos = self
            .position_embeddings
            .index_select(0, &pos_t)
            .and_then(|t| t.reshape(vec![bsz, s, hidden]))
            .map_err(err)?;

        // token_type_ids = 0 → token_type_embeddings[0] (broadcast row).
        let tt = self
            .token_type_embeddings
            .narrow(0, 0, 1)
            .and_then(|t| t.reshape(vec![1, 1, hidden]))
            .map_err(err)?;

        let mut h = word
            .add(&pos)
            .and_then(|t| t.broadcast_add(&tt))
            .map_err(err)?;
        h = self.ln(&h, &self.emb_ln_w, &self.emb_ln_b)?;

        // Аддитивная маска ключей [B,1,1,S]: 0 на real, −inf (большой минус) на pad.
        let neg = -1.0e9f32;
        let mask_add: Vec<f32> = mask_vals
            .iter()
            .map(|&m| if m != 0 { 0.0f32 } else { neg })
            .collect();
        let attn_mask =
            Tensor::from_vec(mask_add, vec![bsz, 1, 1, s], self.device).map_err(err)?;
        let attn_mask = if self.dtype == DType::F32 {
            attn_mask
        } else {
            attn_mask.to_dtype(self.dtype).map_err(err)?
        };

        for layer in &self.layers {
            h = self.layer_forward(layer, &h, &attn_mask, bsz, s, nh, hd, scale)?;
        }

        Ok(h)
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_forward(
        &self,
        layer: &Layer,
        h: &Tensor,
        attn_mask: &Tensor,
        bsz: usize,
        s: usize,
        nh: usize,
        hd: usize,
        scale: f32,
    ) -> Result<Tensor, BgeError> {
        let a = &layer.attn;

        let q = self
            .linear_bias(h, &a.q_w, &a.q_b)?
            .reshape(vec![bsz, s, nh, hd])
            .and_then(|t| t.permute(vec![0, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)?;
        let k = self
            .linear_bias(h, &a.k_w, &a.k_b)?
            .reshape(vec![bsz, s, nh, hd])
            .and_then(|t| t.permute(vec![0, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)?;
        let v = self
            .linear_bias(h, &a.v_w, &a.v_b)?
            .reshape(vec![bsz, s, nh, hd])
            .and_then(|t| t.permute(vec![0, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)?;

        let attn = scaled_dot_attention(&q, &k, &v, scale, Some(attn_mask)).map_err(err)?;
        let attn = attn
            .permute(vec![0, 2, 1, 3])
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![bsz, s, nh * hd]))
            .map_err(err)?;

        // attention.output: dense(+bias) → +residual(h) → LayerNorm.
        let attn_out = self.linear_bias(&attn, &a.out_w, &a.out_b)?;
        let attn_res = attn_out.add(h).map_err(err)?;
        let attn_res = self.ln(&attn_res, &a.out_ln_w, &a.out_ln_b)?;

        // FFN: intermediate.dense(+bias) → gelu_exact → output.dense(+bias) →
        //      +residual(attn_res) → LayerNorm.
        let f = &layer.ffn;
        let inter = self
            .linear_bias(&attn_res, &f.inter_w, &f.inter_b)?
            .gelu_exact()
            .map_err(err)?;
        let ffn_out = self.linear_bias(&inter, &f.out_w, &f.out_b)?;
        let ffn_res = ffn_out.add(&attn_res).map_err(err)?;
        self.ln(&ffn_res, &f.out_ln_w, &f.out_ln_b)
    }

    /// CLS-pool + L2-normalize: last_hidden[:,0,:] → L2-norm → [B,hidden].
    pub fn dense_embed(&self, last_hidden: &Tensor) -> Result<Tensor, BgeError> {
        let dims = last_hidden.dims().to_vec();
        let (bsz, hidden) = (dims[0], dims[2]);
        // CLS = позиция 0 по оси S.
        // narrow(1,0,1) → non-contiguous при bsz>1 (срез по оси S); contiguous ДО reshape.
        let cls = last_hidden
            .narrow(1, 0, 1)
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![bsz, hidden]))
            .map_err(err)?;
        l2_normalize(&cls)
    }
}

/// L2-normalize по последней оси (eps в знаменателе для устойчивости).
pub fn l2_normalize(x: &Tensor) -> Result<Tensor, BgeError> {
    let x32 = x.to_dtype(DType::F32).map_err(err)?;
    let last = x32.rank() - 1;
    let norm = x32
        .sqr()
        .and_then(|t| t.sum_keepdim(last))
        .and_then(|t| t.sqrt())
        .and_then(|t| t.add_scalar(1e-12))
        .map_err(err)?;
    x32.broadcast_div(&norm).map_err(err)
}
