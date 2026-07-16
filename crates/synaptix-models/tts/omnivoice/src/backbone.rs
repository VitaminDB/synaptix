//! Qwen3-бэкбон OmniVoice (двунаправленный: full-attention, БЕЗ causal, БЕЗ
//! KV-cache) + audio-embed merge + audio-heads.
//!
//! Слой-логика (RMSNorm, GQA-attention с qk-norm, RoPE θ1e6 split-layout,
//! SwiGLU-MLP) повторяет `synaptix-llm-common` `FullAttn`/`Mlp`, но прогон
//! bidirectional (mask = None во всех (q,k)-парах) и без KV-cache: каждый forward
//! считает полную последовательность. Источник истины:
//! `~/Temp/OmniVoice/omnivoice/models/omnivoice.py` (`_prepare_embed_inputs`,
//! `forward`) + Qwen3 в `llm/qwen3`/`llm/common`. SPEC.md «Критичные места» п.1-2.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::rms_norm::rms_norm;
use synaptix_ops::pos::rope::{apply_rope_range, RopeLayout};
use synaptix_ops::pos::rope_cache::RopeCache;

use crate::config::OmniVoiceConfig;
use crate::loader::OmniVoiceLmWeights;
use crate::{OmniVoiceError, Result};

fn err<E: std::fmt::Display>(e: E) -> OmniVoiceError {
    OmniVoiceError::Inference(e.to_string())
}

struct Attn {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
}

struct Mlp {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

struct Block {
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
    attn: Attn,
    mlp: Mlp,
}

/// Двунаправленный Qwen3-бэкбон OmniVoice + audio-embed + audio-heads.
pub struct Backbone {
    embed_tokens: Tensor,
    final_norm: Tensor,
    blocks: Vec<Block>,
    audio_embeddings: Tensor,
    audio_heads: Tensor,
    codebook_layer_offsets: Vec<i64>,
    rope: RopeCache,
    cfg: OmniVoiceConfig,
    device: Device,
}

impl Backbone {
    /// Собрать бэкбон из `lm`-весов + конфига. `rope_capacity` — макс. длина
    /// последовательности (RoPE-таблица). compute = `weights.dtype` (F32 для гейта).
    pub fn build(
        cfg: &OmniVoiceConfig,
        weights: &OmniVoiceLmWeights,
        rope_capacity: usize,
    ) -> Result<Self> {
        let device = weights.device;
        let dtype = weights.dtype;
        let b = &cfg.backbone;

        let get = |name: &str| -> Result<Tensor> {
            let t = weights.get(name)?;
            if t.dtype() == dtype {
                Ok(t.clone())
            } else {
                t.to_dtype(dtype).map_err(err)
            }
        };

        let embed_tokens = get("llm.embed_tokens.weight")?;
        let final_norm = get("llm.norm.weight")?;
        let audio_embeddings = get("audio_embeddings.weight")?;
        let audio_heads = get("audio_heads.weight")?;

        let cbo = weights.get("codebook_layer_offsets")?;
        let codebook_layer_offsets = cbo
            .to_dtype(DType::I64)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<i64>())
            .map_err(err)?;

        let mut blocks = Vec::with_capacity(b.num_hidden_layers);
        for l in 0..b.num_hidden_layers {
            let k = |s: &str| format!("llm.layers.{l}.{s}");
            blocks.push(Block {
                input_layernorm: get(&k("input_layernorm.weight"))?,
                post_attention_layernorm: get(&k("post_attention_layernorm.weight"))?,
                attn: Attn {
                    q_proj: get(&k("self_attn.q_proj.weight"))?,
                    k_proj: get(&k("self_attn.k_proj.weight"))?,
                    v_proj: get(&k("self_attn.v_proj.weight"))?,
                    o_proj: get(&k("self_attn.o_proj.weight"))?,
                    q_norm: get(&k("self_attn.q_norm.weight"))?,
                    k_norm: get(&k("self_attn.k_norm.weight"))?,
                },
                mlp: Mlp {
                    gate_proj: get(&k("mlp.gate_proj.weight"))?,
                    up_proj: get(&k("mlp.up_proj.weight"))?,
                    down_proj: get(&k("mlp.down_proj.weight"))?,
                },
            });
        }

        // RoPE θ=rope_theta, rotary_dim = head_dim (Qwen3 full rope), split-layout.
        let rope = RopeCache::new(b.head_dim, rope_capacity.max(1), cfg.rope_theta as f32, device)
            .map_err(err)?;

        Ok(Self {
            embed_tokens,
            final_norm,
            blocks,
            audio_embeddings,
            audio_heads,
            codebook_layer_offsets,
            rope,
            cfg: cfg.clone(),
            device,
        })
    }

    /// `audio_mask_id` (MASK-токен), для masked-decode.
    pub fn audio_mask_id(&self) -> i64 {
        self.cfg.audio_mask_id as i64
    }

    /// Размер аудио-словаря (последняя ось логитов).
    pub fn audio_vocab_size(&self) -> usize {
        self.cfg.audio_vocab_size
    }

    /// Устройство модели (для размещения тензоров-входов forward на нём же).
    pub fn device(&self) -> Device {
        self.device
    }

    /// `input_ids` [B,8,S] (I64), `audio_mask` [B,S] (bool как U8) →
    /// `inputs_embeds` [B,S,hidden].
    fn prepare_embed_inputs(&self, input_ids: &Tensor, audio_mask: &Tensor) -> Result<Tensor> {
        let dims = input_ids.dims();
        let (bsz, n_cb, s) = (dims[0], dims[1], dims[2]);
        let hidden = self.cfg.backbone.hidden_size;

        // input_ids/audio_mask → host (I64/U8). Индекс-тензоры строим контигуозно
        // через from_vec: narrow/squeeze дают non-contiguous, а contiguous() на
        // int-тензоре на CUDA идёт через float-only copy-kernel → "cuda unary: dtype".
        let ids = input_ids
            .to_dtype(DType::I64)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<i64>())
            .map_err(err)?;
        let mask_u8 = audio_mask
            .to_dtype(DType::U8)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<u8>())
            .map_err(err)?;

        // text_embeds = embed_tokens(input_ids[:,0,:]) → [B,S,hidden]
        let mut text_flat_vec = vec![0i64; bsz * s];
        for bi in 0..bsz {
            for si in 0..s {
                text_flat_vec[bi * s + si] = ids[(bi * n_cb) * s + si];
            }
        }
        let text_flat = Tensor::from_vec(text_flat_vec, vec![bsz * s], self.device).map_err(err)?;
        let text_embeds = self
            .embed_tokens
            .index_select(0, &text_flat)
            .map_err(err)?
            .reshape(vec![bsz, s, hidden])
            .map_err(err)?;

        // shifted = input_ids * audio_mask[:,None,:] + codebook_layer_offsets[None,:,None]

        let mut shifted = vec![0i64; bsz * n_cb * s];
        for bi in 0..bsz {
            for c in 0..n_cb {
                let off = self.codebook_layer_offsets[c];
                for si in 0..s {
                    let m = mask_u8[bi * s + si] as i64;
                    let id = ids[(bi * n_cb + c) * s + si];
                    shifted[(bi * n_cb + c) * s + si] = id * m + off;
                }
            }
        }

        // audio_embeds = audio_embeddings(shifted).sum(dim=1) → [B,S,hidden]
        let shifted_t = Tensor::from_vec(shifted, vec![bsz * n_cb * s], self.device).map_err(err)?;
        let ae = self
            .audio_embeddings
            .index_select(0, &shifted_t)
            .map_err(err)?
            .reshape(vec![bsz, n_cb, s, hidden])
            .map_err(err)?;
        let audio_embeds = ae.sum([1usize]).map_err(err)?; // [B,S,hidden]

        // out = where(audio_mask[...,None], audio_embeds, text_embeds).
        // where_cond — CPU-only в synaptix; для mask∈{0,1} бит-идентично арифметике
        // a·mask + b·(1−mask) (broadcast_mul/affine/add — есть на CUDA). mask_f строим
        // из host-вектора mask_u8 через from_vec (U8→F32 cast на CUDA не поддержан).
        let mask_f_vec: Vec<f32> = mask_u8.iter().map(|&m| m as f32).collect();
        let mask_f = Tensor::from_vec(mask_f_vec, vec![bsz, s, 1], self.device).map_err(err)?;
        let inv = mask_f.affine(-1.0, 1.0).map_err(err)?;
        let a = audio_embeds.broadcast_mul(&mask_f).map_err(err)?;
        let b = text_embeds.broadcast_mul(&inv).map_err(err)?;
        a.broadcast_add(&b).map_err(err)
    }

    fn attn_forward(&self, blk: &Block, h: &Tensor, bsz: usize, s: usize) -> Result<Tensor> {
        let b = &self.cfg.backbone;
        let (nh, nkv, hd) = (b.num_attention_heads, b.num_key_value_heads, b.head_dim);
        let eps = b.rms_norm_eps as f32;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let a = &blk.attn;

        let q = h
            .linear(&a.q_proj)
            .map_err(err)?
            .reshape(vec![bsz, s, nh, hd])
            .and_then(|t| t.permute(vec![0, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)?;
        let k = h
            .linear(&a.k_proj)
            .map_err(err)?
            .reshape(vec![bsz, s, nkv, hd])
            .and_then(|t| t.permute(vec![0, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)?;
        let v = h
            .linear(&a.v_proj)
            .map_err(err)?
            .reshape(vec![bsz, s, nkv, hd])
            .and_then(|t| t.permute(vec![0, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)?;

        // qk-norm: RMSNorm по head_dim (последняя ось [B,H,S,hd]) ПОСЛЕ reshape.
        let q = rms_norm(&q, &a.q_norm, eps).map_err(err)?;
        let k = rms_norm(&k, &a.k_norm, eps).map_err(err)?;

        // RoPE θ1e6, full rotary (rotary_dim = head_dim), split-layout, позиции 0..S.
        let q = apply_rope_range(&q, &self.rope, 0, s, RopeLayout::Split).map_err(err)?;
        let k = apply_rope_range(&k, &self.rope, 0, s, RopeLayout::Split).map_err(err)?;

        // GQA: повторить kv-головы до nh.
        let group = nh / nkv;
        let k = repeat_kv(&k, group)?;
        let v = repeat_kv(&v, group)?;

        // Bidirectional full attention: mask = None (все True).
        let attn = scaled_dot_attention(&q, &k, &v, scale, None).map_err(err)?;
        let attn = attn
            .permute(vec![0, 2, 1, 3])
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![bsz, s, nh * hd]))
            .map_err(err)?;
        attn.linear(&a.o_proj).map_err(err)
    }

    fn mlp_forward(&self, mlp: &Mlp, h: &Tensor) -> Result<Tensor> {
        let gate = h.linear(&mlp.gate_proj).map_err(err)?;
        let up = h.linear(&mlp.up_proj).map_err(err)?;
        let gated = gate.silu().and_then(|g| g.mul(&up)).map_err(err)?;
        gated.linear(&mlp.down_proj).map_err(err)
    }

    /// Полный bidirectional forward.
    /// `input_ids` [B,8,S] (I64), `audio_mask` [B,S] (bool/U8) → audio_logits
    /// [B,8,S,1025] (= num_audio_codebook × audio_vocab_size).
    pub fn forward(&self, input_ids: &Tensor, audio_mask: &Tensor) -> Result<Tensor> {
        let dims = input_ids.dims();
        let (bsz, s) = (dims[0], dims[2]);
        let b = &self.cfg.backbone;
        let eps = b.rms_norm_eps as f32;

        let mut hidden = self.prepare_embed_inputs(input_ids, audio_mask)?;

        for blk in &self.blocks {
            let residual = hidden.clone();
            let h = rms_norm(&hidden, &blk.input_layernorm, eps).map_err(err)?;
            let mixed = self.attn_forward(blk, &h, bsz, s)?;
            hidden = residual.add(&mixed).map_err(err)?;

            let residual2 = hidden.clone();
            let h = rms_norm(&hidden, &blk.post_attention_layernorm, eps).map_err(err)?;
            let mlp_out = self.mlp_forward(&blk.mlp, &h)?;
            hidden = residual2.add(&mlp_out).map_err(err)?;
        }

        let hidden = rms_norm(&hidden, &self.final_norm, eps).map_err(err)?;

        // audio_heads: Linear(hidden → 8·1025); reshape [B,S,8,1025] → [B,8,S,1025].
        let n_cb = self.cfg.num_audio_codebook;
        let av = self.cfg.audio_vocab_size;
        let logits_flat = hidden.linear(&self.audio_heads).map_err(err)?; // [B,S,8·1025]
        logits_flat
            .reshape(vec![bsz, s, n_cb, av])
            .and_then(|t| t.permute(vec![0, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)
    }
}

/// Повторить kv-головы (GQA): [B,nkv,S,hd] → [B,nkv*group,S,hd]. group==1 → копия.
fn repeat_kv(x: &Tensor, group: usize) -> Result<Tensor> {
    if group == 1 {
        return Ok(x.clone());
    }
    let d = x.dims();
    let (bsz, nkv, s, hd) = (d[0], d[1], d[2], d[3]);
    x.unsqueeze(2)
        .and_then(|t| t.broadcast_as(vec![bsz, nkv, group, s, hd]))
        .and_then(|t| t.contiguous())
        .and_then(|t| t.reshape(vec![bsz, nkv * group, s, hd]))
        .map_err(err)
}
