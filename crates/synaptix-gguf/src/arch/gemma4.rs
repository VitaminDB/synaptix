//! Gemma-4 (`gemma4`): скользящие/полные слои с разными размерами голов,
//! K = V на полных слоях, пропорциональный RoPE, MoE-ветка рядом с плотным
//! MLP (26B-A4B). Нормы хранятся как есть (`NormGain::Plain`). PLE (E2B/E4B) и
//! общие KV-слои движок не поддерживает — конфиг это скажет при загрузке.

use serde_json::{json, Value as J};

use super::common::{check_sources, direct, plan, rope_freqs, standard_files, Keys};
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
    pub num_global_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub layer_types: Vec<&'static str>,
    pub attention_k_eq_v: bool,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_theta_swa: f32,
    pub partial_rotary_factor_full: f32,
    pub sliding_window: usize,
    pub final_logit_softcapping: Option<f32>,
    pub context_length: usize,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub moe_intermediate_size: usize,
    pub hidden_size_per_layer_input: usize,
    pub num_kv_shared_layers: usize,
    pub has_layer_scalar: bool,
}

impl Shape {
    pub fn read(f: &GgufFile) -> Result<Self> {
        let k = Keys::new(f)?;
        let n = k.usize("block_count")?;
        let swa = k.opt_bool_vec("attention.sliding_window_pattern").unwrap_or_else(|| (0..n).map(|i| (i + 1) % 6 != 0).collect());
        let layer_types: Vec<&'static str> = (0..n).map(|i| if swa.get(i).copied().unwrap_or(true) { "sliding_attention" } else { "full_attention" }).collect();
        let first_full = layer_types.iter().position(|t| *t == "full_attention");
        let first_swa = layer_types.iter().position(|t| *t == "sliding_attention");
        let kv = k.usize_per_layer("attention.head_count_kv", n).unwrap_or_else(|| vec![1; n]);
        let ff = k.usize_per_layer("feed_forward_length", n).unwrap_or_else(|| vec![0; n]);
        let global_head_dim = k.opt_usize("attention.key_length").unwrap_or(256);
        let head_dim = k.opt_usize("attention.key_length_swa").unwrap_or(global_head_dim);
        // Доля вращаемых измерений полных слоёв — из rope_freqs (1.0 = вращается).
        let partial_rotary_factor_full = match rope_freqs(f)? {
            Some(v) => {
                let rot = v.iter().filter(|x| (**x - 1.0).abs() < 1e-6).count();
                (2 * rot) as f32 / global_head_dim as f32
            }
            None => k.opt_usize("rope.dimension_count").map(|d| d as f32 / global_head_dim as f32).unwrap_or(1.0),
        };
        let attention_k_eq_v = first_full.is_some_and(|i| f.tensor(&format!("blk.{i}.attn_v.weight")).is_none());
        Ok(Self {
            hidden_size: k.usize("embedding_length")?,
            intermediate_size: ff.first().copied().filter(|x| *x > 0).unwrap_or(0),
            num_hidden_layers: n,
            num_attention_heads: k.usize("attention.head_count")?,
            num_key_value_heads: first_swa.and_then(|i| kv.get(i).copied()).unwrap_or(kv[0]),
            num_global_key_value_heads: first_full.and_then(|i| kv.get(i).copied()).unwrap_or(kv[0]),
            head_dim,
            global_head_dim,
            layer_types,
            attention_k_eq_v,
            rms_norm_eps: k.opt_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            rope_theta: k.opt_f32("rope.freq_base").unwrap_or(1_000_000.0),
            rope_theta_swa: k.opt_f32("rope.freq_base_swa").unwrap_or(10_000.0),
            partial_rotary_factor_full,
            sliding_window: k.opt_usize("attention.sliding_window").unwrap_or(1024),
            final_logit_softcapping: k.opt_f32("final_logit_softcapping"),
            context_length: k.opt_usize("context_length").unwrap_or(262_144),
            vocab_size: k.embed_rows().unwrap_or_else(|| k.vocab_size()),
            tie_word_embeddings: k.tied(),
            num_experts: k.opt_usize("expert_count").unwrap_or(0),
            top_k_experts: k.opt_usize("expert_used_count").unwrap_or(0),
            moe_intermediate_size: k.opt_usize("expert_feed_forward_length").unwrap_or(0),
            hidden_size_per_layer_input: k.opt_usize("embedding_length_per_layer_input").unwrap_or(0),
            num_kv_shared_layers: k.opt_usize("shared_kv_layers").unwrap_or(0),
            has_layer_scalar: k.has_tensor("blk.0.layer_output_scale.weight"),
        })
    }
}

pub fn tensors(f: &GgufFile, s: &Shape) -> Result<Vec<MappedTensor>> {
    let mut out = vec![direct("model.embed_tokens.weight", "token_embd.weight"), direct("model.norm.weight", "output_norm.weight")];
    if !s.tie_word_embeddings {
        out.push(direct("lm_head.weight", "output.weight"));
    }
    let moe = s.num_experts > 0;
    for i in 0..s.num_hidden_layers {
        let p = format!("model.layers.{i}");
        let full = s.layer_types[i] == "full_attention";
        out.push(direct(format!("{p}.input_layernorm.weight"), format!("blk.{i}.attn_norm.weight")));
        out.push(direct(format!("{p}.post_attention_layernorm.weight"), format!("blk.{i}.post_attention_norm.weight")));
        out.push(direct(format!("{p}.pre_feedforward_layernorm.weight"), format!("blk.{i}.ffn_norm.weight")));
        out.push(direct(format!("{p}.post_feedforward_layernorm.weight"), format!("blk.{i}.post_ffw_norm.weight")));
        out.push(direct(format!("{p}.self_attn.q_proj.weight"), format!("blk.{i}.attn_q.weight")));
        out.push(direct(format!("{p}.self_attn.k_proj.weight"), format!("blk.{i}.attn_k.weight")));
        if !(full && s.attention_k_eq_v) {
            out.push(direct(format!("{p}.self_attn.v_proj.weight"), format!("blk.{i}.attn_v.weight")));
        }
        out.push(direct(format!("{p}.self_attn.o_proj.weight"), format!("blk.{i}.attn_output.weight")));
        out.push(direct(format!("{p}.self_attn.q_norm.weight"), format!("blk.{i}.attn_q_norm.weight")));
        out.push(direct(format!("{p}.self_attn.k_norm.weight"), format!("blk.{i}.attn_k_norm.weight")));
        for (hf, g) in [("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down")] {
            out.push(direct(format!("{p}.mlp.{hf}.weight"), format!("blk.{i}.{g}.weight")));
        }
        if moe {
            out.push(direct(format!("{p}.router.proj.weight"), format!("blk.{i}.ffn_gate_inp.weight")));
            out.push(direct(format!("{p}.router.scale"), format!("blk.{i}.ffn_gate_inp.scale")));
            out.push(direct(format!("{p}.router.per_expert_scale"), format!("blk.{i}.ffn_down_exps.scale")));
            out.push(direct(format!("{p}.experts.gate_up_proj"), format!("blk.{i}.ffn_gate_up_exps.weight")));
            out.push(direct(format!("{p}.experts.down_proj"), format!("blk.{i}.ffn_down_exps.weight")));
            out.push(direct(format!("{p}.pre_feedforward_layernorm_2.weight"), format!("blk.{i}.pre_ffw_norm_2.weight")));
            out.push(direct(format!("{p}.post_feedforward_layernorm_1.weight"), format!("blk.{i}.post_ffw_norm_1.weight")));
            out.push(direct(format!("{p}.post_feedforward_layernorm_2.weight"), format!("blk.{i}.post_ffw_norm_2.weight")));
        }
        if s.has_layer_scalar {
            out.push(direct(format!("{p}.layer_scalar"), format!("blk.{i}.layer_output_scale.weight")));
        }
    }
    check_sources(f, &out)?;
    Ok(out)
}

pub fn config_json(s: &Shape, vocab: &GgufVocab) -> Result<Vec<u8>> {
    let doc = json!({
        "architectures": ["Gemma4ForCausalLM"],
        "model_type": "gemma4_text",
        "attention_bias": false,
        "attention_k_eq_v": s.attention_k_eq_v,
        "bos_token_id": vocab.bos,
        "eos_token_id": vocab.eos_ids(),
        "dtype": "bfloat16",
        "enable_moe_block": s.num_experts > 0,
        "final_logit_softcapping": s.final_logit_softcapping,
        "global_head_dim": s.global_head_dim,
        "head_dim": s.head_dim,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": s.hidden_size,
        "hidden_size_per_layer_input": s.hidden_size_per_layer_input,
        "intermediate_size": s.intermediate_size,
        "layer_types": s.layer_types,
        "max_position_embeddings": s.context_length,
        "moe_intermediate_size": s.moe_intermediate_size,
        "num_attention_heads": s.num_attention_heads,
        "num_experts": s.num_experts,
        "num_global_key_value_heads": s.num_global_key_value_heads,
        "num_hidden_layers": s.num_hidden_layers,
        "num_key_value_heads": s.num_key_value_heads,
        "num_kv_shared_layers": s.num_kv_shared_layers,
        "rms_norm_eps": s.rms_norm_eps,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": s.partial_rotary_factor_full, "rope_theta": s.rope_theta, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": s.rope_theta_swa, "rope_type": "default"}
        },
        "sliding_window": s.sliding_window,
        "tie_word_embeddings": s.tie_word_embeddings,
        "top_k_experts": s.top_k_experts,
        "use_cache": true,
        "vocab_size": s.vocab_size,
    });
    let _ = J::Null;
    Ok(serde_json::to_vec_pretty(&doc)?)
}

pub fn build_plan(model: &GgufFile, bundle_id: &str) -> Result<ConversionPlan> {
    let shape = Shape::read(model)?;
    let vocab = GgufVocab::read(model)?;
    let tensors = tensors(model, &shape)?;
    let files = standard_files(model, &vocab, config_json(&shape, &vocab)?)?;
    Ok(plan(bundle_id, "gemma4_text", tensors, files))
}
