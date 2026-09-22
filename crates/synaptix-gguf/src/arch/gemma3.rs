//! Gemma-3 (`gemma3`): сэндвич-нормы, локальный/глобальный RoPE, скользящее
//! окно с шагом 6. Нормы в GGUF лежат как `1 + w` (конвертер прибавляет 1),
//! движок (`NormGain::OnePlus`) прибавляет единицу сам — вычитаем при чтении.

use serde_json::json;

use super::common::{check_sources, direct, direct_sub1, plan, standard_files, Keys};
use crate::error::Result;
use crate::plan::{ConversionPlan, MappedTensor};
use crate::reader::GgufFile;
use crate::tokenizer::GgufVocab;

pub struct Shape {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling_factor: Option<f32>,
    pub sliding_window: usize,
    pub final_logit_softcapping: Option<f32>,
    pub context_length: usize,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
}

impl Shape {
    pub fn read(f: &GgufFile) -> Result<Self> {
        let k = Keys::new(f)?;
        let num_attention_heads = k.opt_usize("attention.head_count").unwrap_or(8);
        let hidden_size = k.usize("embedding_length")?;
        Ok(Self {
            hidden_size,
            intermediate_size: k.usize("feed_forward_length")?,
            num_hidden_layers: k.usize("block_count")?,
            num_attention_heads,
            num_key_value_heads: k.opt_usize("attention.head_count_kv").unwrap_or(4),
            head_dim: k.opt_usize("attention.key_length").unwrap_or(256),
            rms_norm_eps: k.opt_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            rope_theta: k.opt_f32("rope.freq_base").unwrap_or(1_000_000.0),
            rope_scaling_factor: match k.opt_str("rope.scaling.type") {
                Some("linear") => k.opt_f32("rope.scaling.factor"),
                _ => None,
            },
            sliding_window: k.opt_usize("attention.sliding_window").unwrap_or(1024),
            final_logit_softcapping: k.opt_f32("final_logit_softcapping"),
            context_length: k.opt_usize("context_length").unwrap_or(131_072),
            vocab_size: k.embed_rows().unwrap_or_else(|| k.vocab_size()),
            tie_word_embeddings: k.tied(),
        })
    }

    /// `query_pre_attn_scalar`: у 27B — `hidden/heads`, у остальных `head_dim`
    /// (в GGUF не хранится; llama.cpp различает по размеру модели).
    pub fn query_pre_attn_scalar(&self) -> usize {
        if self.hidden_size == 5376 {
            self.hidden_size / self.num_attention_heads
        } else {
            self.head_dim
        }
    }
}

pub fn tensors(f: &GgufFile, s: &Shape) -> Result<Vec<MappedTensor>> {
    let mut out = vec![
        direct("model.embed_tokens.weight", "token_embd.weight"),
        direct_sub1("model.norm.weight", "output_norm.weight"),
    ];
    if !s.tie_word_embeddings {
        out.push(direct("lm_head.weight", "output.weight"));
    }
    for i in 0..s.num_hidden_layers {
        let p = format!("model.layers.{i}");
        out.push(direct_sub1(format!("{p}.input_layernorm.weight"), format!("blk.{i}.attn_norm.weight")));
        out.push(direct_sub1(format!("{p}.post_attention_layernorm.weight"), format!("blk.{i}.post_attention_norm.weight")));
        out.push(direct_sub1(format!("{p}.pre_feedforward_layernorm.weight"), format!("blk.{i}.ffn_norm.weight")));
        out.push(direct_sub1(format!("{p}.post_feedforward_layernorm.weight"), format!("blk.{i}.post_ffw_norm.weight")));
        for (hf, g) in [("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")] {
            out.push(direct(format!("{p}.self_attn.{hf}.weight"), format!("blk.{i}.{g}.weight")));
        }
        out.push(direct_sub1(format!("{p}.self_attn.q_norm.weight"), format!("blk.{i}.attn_q_norm.weight")));
        out.push(direct_sub1(format!("{p}.self_attn.k_norm.weight"), format!("blk.{i}.attn_k_norm.weight")));
        for (hf, g) in [("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down")] {
            out.push(direct(format!("{p}.mlp.{hf}.weight"), format!("blk.{i}.{g}.weight")));
        }
    }
    check_sources(f, &out)?;
    Ok(out)
}

pub fn config_json(s: &Shape, vocab: &GgufVocab) -> Result<Vec<u8>> {
    let mut doc = json!({
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3_text",
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": vocab.bos,
        "eos_token_id": vocab.eos_ids(),
        "dtype": "bfloat16",
        "final_logit_softcapping": s.final_logit_softcapping,
        "head_dim": s.head_dim,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": s.hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": s.intermediate_size,
        "max_position_embeddings": s.context_length,
        "num_attention_heads": s.num_attention_heads,
        "num_hidden_layers": s.num_hidden_layers,
        "num_key_value_heads": s.num_key_value_heads,
        "query_pre_attn_scalar": s.query_pre_attn_scalar(),
        "rms_norm_eps": s.rms_norm_eps,
        "rope_local_base_freq": 10000.0,
        "rope_theta": s.rope_theta,
        "sliding_window": s.sliding_window,
        "sliding_window_pattern": 6,
        "tie_word_embeddings": s.tie_word_embeddings,
        "use_cache": true,
        "vocab_size": s.vocab_size,
    });
    if let Some(f) = s.rope_scaling_factor {
        doc["rope_scaling"] = json!({"factor": f, "rope_type": "linear"});
    }
    Ok(serde_json::to_vec_pretty(&doc)?)
}

pub fn build_plan(model: &GgufFile, bundle_id: &str) -> Result<ConversionPlan> {
    let shape = Shape::read(model)?;
    let vocab = GgufVocab::read(model)?;
    let tensors = tensors(model, &shape)?;
    let files = standard_files(model, &vocab, config_json(&shape, &vocab)?)?;
    Ok(plan(bundle_id, "gemma3_text", tensors, files))
}
