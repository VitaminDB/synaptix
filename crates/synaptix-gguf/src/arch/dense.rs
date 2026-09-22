//! Плотные декодеры семейства Llama: `llama`, `qwen2`, `qwen3`, а также
//! `qwen3moe` (те же блоки + стопки экспертов). Раскладка в GGUF одна и та же
//! с небольшими отличиями: у Qwen2 — bias у q/k/v, у Qwen3 — q/k-нормы, у
//! MoE — `ffn_gate_inp` и `ffn_{gate,up,down}_exps`. Нормы хранятся как есть
//! (движок: `NormGain::Plain`).

use serde_json::{json, Value as J};

use super::common::{check_sources, direct, llama3_rope_factors, plan, rope_freqs, stack_concat, standard_files, Keys};
use crate::error::{GgufError, Result};
use crate::plan::{ConversionPlan, MappedTensor, Producer, Transform};
use crate::reader::GgufFile;
use crate::tokenizer::GgufVocab;

pub struct Shape {
    pub arch: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub context_length: usize,
    pub vocab_size: usize,
    pub attention_bias: bool,
    pub qk_norm: bool,
    pub tie_word_embeddings: bool,
    /// `rope_scaling` для config.json (`None` — без масштабирования).
    pub rope_scaling: Option<J>,
    pub moe: Option<Moe>,
}

pub struct Moe {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    /// Отдельные `ffn_gate_exps`/`ffn_up_exps` (склеиваем) или готовый
    /// `ffn_gate_up_exps`.
    pub fused_gate_up: bool,
}

impl Shape {
    pub fn read(f: &GgufFile) -> Result<Self> {
        let k = Keys::new(f)?;
        let num_hidden_layers = k.usize("block_count")?;
        let num_attention_heads = k.usize("attention.head_count")?;
        let hidden_size = k.usize("embedding_length")?;
        let head_dim = k.opt_usize("attention.key_length").unwrap_or(hidden_size / num_attention_heads.max(1));
        let rope_theta = k.opt_f32("rope.freq_base").unwrap_or(10_000.0);
        let rotary_dim = k.opt_usize("rope.dimension_count").unwrap_or(head_dim);
        let rope_scaling = rope_scaling_json(f, &k, rope_theta, rotary_dim)?;
        let moe = match k.opt_usize("expert_count") {
            Some(e) if e > 0 => Some(Moe {
                num_experts: e,
                num_experts_per_tok: k.opt_usize("expert_used_count").unwrap_or(2),
                moe_intermediate_size: k.opt_usize("expert_feed_forward_length").unwrap_or(k.usize("feed_forward_length")?),
                fused_gate_up: k.has_tensor("blk.0.ffn_gate_up_exps.weight"),
            }),
            _ => None,
        };
        Ok(Self {
            arch: k.arch.clone(),
            hidden_size,
            intermediate_size: k.usize("feed_forward_length")?,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: k.opt_usize("attention.head_count_kv").unwrap_or(num_attention_heads),
            head_dim,
            rms_norm_eps: k.opt_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            rope_theta,
            context_length: k.opt_usize("context_length").unwrap_or(32_768),
            vocab_size: k.embed_rows().unwrap_or_else(|| k.vocab_size()),
            attention_bias: k.has_tensor("blk.0.attn_q.bias"),
            qk_norm: k.has_tensor("blk.0.attn_q_norm.weight"),
            tie_word_embeddings: k.tied(),
            rope_scaling,
            moe,
        })
    }

    pub fn model_type(&self) -> &'static str {
        match self.arch.as_str() {
            "llama" => "llama",
            "qwen2" => "qwen2",
            "qwen3" => "qwen3",
            "qwen3moe" => "qwen3_moe",
            _ => "llama",
        }
    }
}

/// `rope_scaling` из ключей `rope.scaling.*` или из `rope_freqs.weight`
/// (Llama-3: конвертер пишет только множители частот — подбираем профиль).
fn rope_scaling_json(f: &GgufFile, k: &Keys, theta: f32, dim: usize) -> Result<Option<J>> {
    if let Some(factors) = rope_freqs(f)? {
        if factors.iter().all(|x| (*x - 1.0).abs() < 1e-6) {
            return Ok(None);
        }
        let old_ctx = k.opt_usize("rope.scaling.original_context_length").unwrap_or(8192) as f32;
        let fmax = factors.iter().cloned().fold(0f32, f32::max);
        for (factor, low, high) in [(fmax, 1.0f32, 4.0f32), (8.0, 1.0, 4.0), (32.0, 1.0, 4.0)] {
            let want = llama3_rope_factors(theta, dim, factor, low, high, old_ctx);
            if want.len() == factors.len()
                && want.iter().zip(&factors).all(|(a, b)| (a - b).abs() <= 1e-3 * a.abs().max(1.0))
            {
                return Ok(Some(json!({
                    "rope_type": "llama3",
                    "factor": factor,
                    "low_freq_factor": low,
                    "high_freq_factor": high,
                    "original_max_position_embeddings": old_ctx as usize,
                })));
            }
        }
        return Err(GgufError::UnsupportedArch(
            "rope_freqs.weight не совпадает ни с одним известным профилем Llama-3 (factor 8/32, low 1, high 4)".into(),
        ));
    }
    match (k.opt_str("rope.scaling.type"), k.opt_f32("rope.scaling.factor")) {
        (Some("linear"), Some(factor)) => Ok(Some(json!({"rope_type": "linear", "factor": factor}))),
        (Some("yarn"), Some(factor)) => Ok(Some(json!({
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": k.opt_usize("rope.scaling.original_context_length").unwrap_or(32_768),
        }))),
        _ => Ok(None),
    }
}

/// Обратная перестановка строк q/k после `permute` конвертера llama.cpp:
/// строка HF `h·hd + j·hd/2 + i` лежит в GGUF по индексу `h·hd + 2i + j`.
fn unpermute_rows(heads: usize, head_dim: usize) -> Vec<u32> {
    let half = head_dim / 2;
    (0..heads * head_dim)
        .map(|r| {
            let (h, w) = (r / head_dim, r % head_dim);
            let (j, i) = (w / half, w % half);
            (h * head_dim + 2 * i + j) as u32
        })
        .collect()
}

pub fn tensors(f: &GgufFile, s: &Shape) -> Result<Vec<MappedTensor>> {
    let mut out = Vec::new();
    out.push(direct("model.embed_tokens.weight", "token_embd.weight"));
    out.push(direct("model.norm.weight", "output_norm.weight"));
    if !s.tie_word_embeddings {
        out.push(direct("lm_head.weight", "output.weight"));
    }
    for i in 0..s.num_hidden_layers {
        let p = format!("model.layers.{i}");
        out.push(direct(format!("{p}.input_layernorm.weight"), format!("blk.{i}.attn_norm.weight")));
        out.push(direct(format!("{p}.post_attention_layernorm.weight"), format!("blk.{i}.ffn_norm.weight")));
        for (hf, g) in [("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")] {
            // Конвертер llama.cpp у арха `llama` переставляет строки q/k
            // под чередующийся RoPE (`permute` в convert_hf_to_gguf.py:
            // (heads, 2, hd/2) → (heads, hd/2, 2)); движок считает RoPE по
            // половинам, как HF, — возвращаем исходный порядок строк.
            let heads = match (s.arch.as_str(), hf) {
                ("llama", "q_proj") => Some(s.num_attention_heads),
                ("llama", "k_proj") => Some(s.num_key_value_heads),
                _ => None,
            };
            match heads {
                Some(h) => out.push(MappedTensor {
                    hf_name: format!("{p}.self_attn.{hf}.weight"),
                    producer: Producer::PermuteRows {
                        src: format!("blk.{i}.{g}.weight"),
                        row_elems: s.hidden_size,
                        map: unpermute_rows(h, s.head_dim),
                    },
                    shape: Some(vec![h * s.head_dim, s.hidden_size]),
                    transform: Transform::None,
                }),
                None => out.push(direct(format!("{p}.self_attn.{hf}.weight"), format!("blk.{i}.{g}.weight"))),
            }
            if s.attention_bias && f.tensor(&format!("blk.{i}.{g}.bias")).is_some() {
                out.push(direct(format!("{p}.self_attn.{hf}.bias"), format!("blk.{i}.{g}.bias")));
            }
        }
        if s.qk_norm {
            out.push(direct(format!("{p}.self_attn.q_norm.weight"), format!("blk.{i}.attn_q_norm.weight")));
            out.push(direct(format!("{p}.self_attn.k_norm.weight"), format!("blk.{i}.attn_k_norm.weight")));
        }
        match &s.moe {
            None => {
                for (hf, g) in [("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down")] {
                    out.push(direct(format!("{p}.mlp.{hf}.weight"), format!("blk.{i}.{g}.weight")));
                }
            }
            Some(m) => {
                out.push(direct(format!("{p}.mlp.gate.weight"), format!("blk.{i}.ffn_gate_inp.weight")));
                if m.fused_gate_up {
                    out.push(direct(format!("{p}.mlp.experts.gate_up_proj"), format!("blk.{i}.ffn_gate_up_exps.weight")));
                } else {
                    out.push(stack_concat(
                        format!("{p}.mlp.experts.gate_up_proj"),
                        vec![format!("blk.{i}.ffn_gate_exps.weight"), format!("blk.{i}.ffn_up_exps.weight")],
                    ));
                }
                out.push(direct(format!("{p}.mlp.experts.down_proj"), format!("blk.{i}.ffn_down_exps.weight")));
                // Плотный MLP рядом с экспертами (shared expert у Qwen2-MoE) — если есть.
                if f.tensor(&format!("blk.{i}.ffn_gate_shexp.weight")).is_some() {
                    for (hf, g) in [("gate_proj", "ffn_gate_shexp"), ("up_proj", "ffn_up_shexp"), ("down_proj", "ffn_down_shexp")] {
                        out.push(direct(format!("{p}.mlp.shared_expert.{hf}.weight"), format!("blk.{i}.{g}.weight")));
                    }
                    if f.tensor(&format!("blk.{i}.ffn_gate_inp_shexp.weight")).is_some() {
                        out.push(direct(format!("{p}.mlp.shared_expert_gate.weight"), format!("blk.{i}.ffn_gate_inp_shexp.weight")));
                    }
                }
            }
        }
    }
    check_sources(f, &out)?;
    Ok(out)
}

pub fn config_json(s: &Shape, vocab: &GgufVocab) -> Result<Vec<u8>> {
    let arch_name = match s.model_type() {
        "llama" => "LlamaForCausalLM",
        "qwen2" => "Qwen2ForCausalLM",
        "qwen3" => "Qwen3ForCausalLM",
        _ => "Qwen3MoeForCausalLM",
    };
    let mut doc = json!({
        "architectures": [arch_name],
        "model_type": s.model_type(),
        "attention_bias": s.attention_bias,
        "attention_dropout": 0.0,
        "bos_token_id": vocab.bos,
        // Скаляр: Qwen3Config ждёт `u32`; полный список eos/eot — в
        // generation_config.json, его читает фасад.
        "eos_token_id": vocab.eos,
        "dtype": "bfloat16",
        "head_dim": s.head_dim,
        "hidden_act": "silu",
        "hidden_size": s.hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": s.intermediate_size,
        "max_position_embeddings": s.context_length,
        "num_attention_heads": s.num_attention_heads,
        "num_hidden_layers": s.num_hidden_layers,
        "num_key_value_heads": s.num_key_value_heads,
        "rms_norm_eps": s.rms_norm_eps,
        "rope_theta": s.rope_theta,
        "tie_word_embeddings": s.tie_word_embeddings,
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": s.vocab_size,
    });
    if let Some(rs) = &s.rope_scaling {
        doc["rope_scaling"] = rs.clone();
    }
    if let Some(m) = &s.moe {
        doc["num_experts"] = J::from(m.num_experts);
        doc["num_experts_per_tok"] = J::from(m.num_experts_per_tok);
        doc["moe_intermediate_size"] = J::from(m.moe_intermediate_size);
        doc["norm_topk_prob"] = J::from(true);
        doc["decoder_sparse_step"] = J::from(1);
    }
    Ok(serde_json::to_vec_pretty(&doc)?)
}

pub fn build_plan(model: &GgufFile, bundle_id: &str) -> Result<ConversionPlan> {
    let shape = Shape::read(model)?;
    let vocab = GgufVocab::read(model)?;
    let tensors = tensors(model, &shape)?;
    let files = standard_files(model, &vocab, config_json(&shape, &vocab)?)?;
    Ok(plan(bundle_id, shape.model_type(), tensors, files))
}
