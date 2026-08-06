use serde_json::{json, Value as J};

use crate::error::{GgufError, Result};
use crate::plan::{
    value_head_map, Component, ConversionPlan, MappedFile, MappedTensor, Producer, Transform,
};
use crate::reader::GgufFile;
use crate::tokenizer::GgufVocab;

pub const LM: &str = "model.language_model";
pub const VIS: &str = "model.visual";

pub struct TextShape {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub mtp_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub mrope_section: Vec<i64>,
    pub full_attention_interval: usize,
    pub conv_kernel: usize,
    pub state_size: usize,
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub context_length: usize,
    pub vocab_size: usize,
    pub attn_output_gate: bool,
    pub tie_word_embeddings: bool,
}

impl TextShape {
    pub fn read(f: &GgufFile) -> Result<Self> {
        let k = |s: &str| format!("qwen35.{s}");
        let block_count = f.usize_of(&k("block_count"))?;
        let mtp_layers = f.opt_usize(&k("nextn_predict_layers")).unwrap_or(0);
        let num_hidden_layers = block_count
            .checked_sub(mtp_layers)
            .ok_or_else(|| GgufError::UnsupportedArch(format!(
                "block_count={block_count} меньше nextn_predict_layers={mtp_layers}"
            )))?;
        let head_dim = f
            .opt_usize(&k("attention.key_length"))
            .unwrap_or_else(|| f.usize_of(&k("embedding_length")).unwrap_or(0) / f.usize_of(&k("attention.head_count")).unwrap_or(1));
        let num_attention_heads = f.usize_of(&k("attention.head_count"))?;
        let vocab_size = f
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let q = f.tensor("blk.0.attn_q.weight").or_else(|| {
            (0..block_count).find_map(|i| f.tensor(&format!("blk.{i}.attn_q.weight")))
        });
        let attn_output_gate = match q {
            Some(t) => {
                let out = *t.dims.last().unwrap_or(&0) as usize;
                out == num_attention_heads * head_dim * 2
            }
            None => false,
        };
        let mrope_section = f
            .get(&k("rope.dimension_sections"))
            .and_then(|v| v.as_array())
            .and_then(|a| a.as_i64_vec())
            .map(|v| v.into_iter().filter(|x| *x > 0).collect::<Vec<_>>())
            .unwrap_or_default();

        Ok(Self {
            hidden_size: f.usize_of(&k("embedding_length"))?,
            intermediate_size: f.usize_of(&k("feed_forward_length"))?,
            num_hidden_layers,
            mtp_layers,
            num_attention_heads,
            num_key_value_heads: f.usize_of(&k("attention.head_count_kv"))?,
            head_dim,
            rotary_dim: f.opt_usize(&k("rope.dimension_count")).unwrap_or(head_dim),
            rms_norm_eps: f.opt_f32(&k("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6),
            rope_theta: f.opt_f32(&k("rope.freq_base")).unwrap_or(10_000_000.0),
            mrope_section,
            full_attention_interval: f.opt_usize(&k("full_attention_interval")).unwrap_or(4),
            conv_kernel: f.opt_usize(&k("ssm.conv_kernel")).unwrap_or(4),
            state_size: f.opt_usize(&k("ssm.state_size")).unwrap_or(128),
            num_key_heads: f.opt_usize(&k("ssm.group_count")).unwrap_or(16),
            num_value_heads: f.opt_usize(&k("ssm.time_step_rank")).unwrap_or(48),
            context_length: f.opt_usize(&k("context_length")).unwrap_or(262_144),
            vocab_size,
            attn_output_gate,
            tie_word_embeddings: f.tensor("output.weight").is_none(),
        })
    }

    pub fn is_full_attention(&self, layer: usize) -> bool {
        self.full_attention_interval <= 1 || (layer + 1) % self.full_attention_interval == 0
    }

    pub fn layer_types(&self) -> Vec<&'static str> {
        (0..self.num_hidden_layers)
            .map(|i| {
                if self.is_full_attention(i) {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect()
    }

    pub fn partial_rotary_factor(&self) -> f32 {
        self.rotary_dim as f32 / self.head_dim as f32
    }
}

pub struct VisionShape {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    pub deepstack_indexes: Vec<usize>,
    pub layer_norm_eps: f32,
}

impl VisionShape {
    pub fn read(f: &GgufFile) -> Result<Self> {
        let deepstack_indexes = f
            .get("clip.vision.is_deepstack_layers")
            .and_then(|v| v.as_array())
            .and_then(|a| a.as_i64_vec())
            .map(|v| {
                v.into_iter()
                    .enumerate()
                    .filter(|(_, x)| *x != 0)
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let pos = f
            .tensor("v.position_embd.weight")
            .map(|t| t.hf_shape()[0])
            .unwrap_or(0);
        let temporal = (0..8)
            .take_while(|i| {
                let n = if *i == 0 {
                    "v.patch_embd.weight".to_string()
                } else {
                    format!("v.patch_embd.weight.{i}")
                };
                f.tensor(&n).is_some()
            })
            .count()
            .max(1);
        let farr = |key: &str| {
            f.get(key)
                .and_then(|v| v.as_array())
                .map(|a| match a {
                    crate::reader::Array::F32(v) => v.clone(),
                    other => other
                        .as_i64_vec()
                        .map(|v| v.into_iter().map(|x| x as f32).collect())
                        .unwrap_or_default(),
                })
                .unwrap_or_else(|| vec![0.5, 0.5, 0.5])
        };
        Ok(Self {
            depth: f.usize_of("clip.vision.block_count")?,
            hidden_size: f.usize_of("clip.vision.embedding_length")?,
            intermediate_size: f.usize_of("clip.vision.feed_forward_length")?,
            num_heads: f.usize_of("clip.vision.attention.head_count")?,
            patch_size: f.usize_of("clip.vision.patch_size")?,
            spatial_merge_size: f.opt_usize("clip.vision.spatial_merge_size").unwrap_or(2),
            temporal_patch_size: temporal,
            out_hidden_size: f.opt_usize("clip.vision.projection_dim").unwrap_or(0),
            num_position_embeddings: pos,
            image_mean: farr("clip.vision.image_mean"),
            image_std: farr("clip.vision.image_std"),
            deepstack_indexes,
            layer_norm_eps: f
                .opt_f32("clip.vision.attention.layer_norm_epsilon")
                .unwrap_or(1e-6),
        })
    }
}

fn text_tensors(f: &GgufFile, s: &TextShape) -> Result<Vec<MappedTensor>> {
    let mut out = Vec::new();
    let a_log_transform = a_log_transform(f, s)?;

    out.push(MappedTensor::direct(
        format!("{LM}.embed_tokens.weight"),
        "token_embd.weight",
    ));
    out.push(
        MappedTensor::direct(format!("{LM}.norm.weight"), "output_norm.weight")
            .with_transform(Transform::SubOne),
    );
    if !s.tie_word_embeddings {
        out.push(MappedTensor::direct("lm_head.weight", "output.weight"));
    }

    for i in 0..s.num_hidden_layers {
        let p = format!("{LM}.layers.{i}");
        out.extend(block_common(&p, i, s));
        if s.is_full_attention(i) {
            out.extend(full_attn(&format!("{p}.self_attn"), i));
        } else {
            out.extend(linear_attn(&format!("{p}.linear_attn"), i, s, a_log_transform));
        }
    }

    for j in 0..s.mtp_layers {
        let blk = s.num_hidden_layers + j;
        out.push(MappedTensor::direct(
            "mtp.fc.weight",
            format!("blk.{blk}.nextn.eh_proj.weight"),
        ));
        out.push(
            MappedTensor::direct(
                "mtp.pre_fc_norm_embedding.weight",
                format!("blk.{blk}.nextn.enorm.weight"),
            )
            .with_transform(Transform::SubOne),
        );
        out.push(
            MappedTensor::direct(
                "mtp.pre_fc_norm_hidden.weight",
                format!("blk.{blk}.nextn.hnorm.weight"),
            )
            .with_transform(Transform::SubOne),
        );
        out.push(
            MappedTensor::direct(
                "mtp.norm.weight",
                format!("blk.{blk}.nextn.shared_head_norm.weight"),
            )
            .with_transform(Transform::SubOne),
        );
        let p = format!("mtp.layers.{j}");
        out.extend(block_common(&p, blk, s));
        out.extend(full_attn(&format!("{p}.self_attn"), blk));
    }

    let missing: Vec<String> = out
        .iter()
        .flat_map(|m| m.producer.sources().iter().cloned())
        .filter(|n| f.tensor(n).is_none())
        .collect();
    if !missing.is_empty() {
        return Err(GgufError::BadTensor {
            name: missing[0].clone(),
            reason: format!("не найден в GGUF (всего пропущено {})", missing.len()),
        });
    }
    Ok(out)
}

fn block_common(prefix: &str, blk: usize, s: &TextShape) -> Vec<MappedTensor> {
    let _ = s;
    vec![
        MappedTensor::direct(
            format!("{prefix}.input_layernorm.weight"),
            format!("blk.{blk}.attn_norm.weight"),
        )
        .with_transform(Transform::SubOne),
        MappedTensor::direct(
            format!("{prefix}.post_attention_layernorm.weight"),
            format!("blk.{blk}.post_attention_norm.weight"),
        )
        .with_transform(Transform::SubOne),
        MappedTensor::direct(
            format!("{prefix}.mlp.gate_proj.weight"),
            format!("blk.{blk}.ffn_gate.weight"),
        ),
        MappedTensor::direct(
            format!("{prefix}.mlp.up_proj.weight"),
            format!("blk.{blk}.ffn_up.weight"),
        ),
        MappedTensor::direct(
            format!("{prefix}.mlp.down_proj.weight"),
            format!("blk.{blk}.ffn_down.weight"),
        ),
    ]
}

fn full_attn(prefix: &str, blk: usize) -> Vec<MappedTensor> {
    vec![
        MappedTensor::direct(
            format!("{prefix}.q_proj.weight"),
            format!("blk.{blk}.attn_q.weight"),
        ),
        MappedTensor::direct(
            format!("{prefix}.k_proj.weight"),
            format!("blk.{blk}.attn_k.weight"),
        ),
        MappedTensor::direct(
            format!("{prefix}.v_proj.weight"),
            format!("blk.{blk}.attn_v.weight"),
        ),
        MappedTensor::direct(
            format!("{prefix}.o_proj.weight"),
            format!("blk.{blk}.attn_output.weight"),
        ),
        MappedTensor::direct(
            format!("{prefix}.q_norm.weight"),
            format!("blk.{blk}.attn_q_norm.weight"),
        )
        .with_transform(Transform::SubOne),
        MappedTensor::direct(
            format!("{prefix}.k_norm.weight"),
            format!("blk.{blk}.attn_k_norm.weight"),
        )
        .with_transform(Transform::SubOne),
    ]
}

fn linear_attn(prefix: &str, blk: usize, s: &TextShape, a_log: Transform) -> Vec<MappedTensor> {
    let vmap = value_head_map(s.num_value_heads, s.num_key_heads);
    let hd = s.state_size;
    let key_dim = s.num_key_heads * hd;
    let value_dim = s.num_value_heads * hd;
    let conv_dim = key_dim * 2 + value_dim;
    let head_rows = |off: usize| -> Vec<u32> {
        let mut m: Vec<u32> = (0..off as u32).collect();
        for h in &vmap {
            let base = off + *h as usize * hd;
            m.extend((base..base + hd).map(|r| r as u32));
        }
        m
    };
    vec![
        MappedTensor {
            hf_name: format!("{prefix}.in_proj_qkv.weight"),
            producer: Producer::PermuteRows {
                src: format!("blk.{blk}.attn_qkv.weight"),
                row_elems: s.hidden_size,
                map: head_rows(key_dim * 2),
            },
            shape: Some(vec![conv_dim, s.hidden_size]),
            transform: Transform::None,
        },
        MappedTensor {
            hf_name: format!("{prefix}.in_proj_z.weight"),
            producer: Producer::PermuteRows {
                src: format!("blk.{blk}.attn_gate.weight"),
                row_elems: s.hidden_size,
                map: head_rows(0),
            },
            shape: Some(vec![value_dim, s.hidden_size]),
            transform: Transform::None,
        },
        MappedTensor {
            hf_name: format!("{prefix}.in_proj_a.weight"),
            producer: Producer::PermuteRows {
                src: format!("blk.{blk}.ssm_alpha.weight"),
                row_elems: s.hidden_size,
                map: vmap.clone(),
            },
            shape: Some(vec![s.num_value_heads, s.hidden_size]),
            transform: Transform::None,
        },
        MappedTensor {
            hf_name: format!("{prefix}.in_proj_b.weight"),
            producer: Producer::PermuteRows {
                src: format!("blk.{blk}.ssm_beta.weight"),
                row_elems: s.hidden_size,
                map: vmap.clone(),
            },
            shape: Some(vec![s.num_value_heads, s.hidden_size]),
            transform: Transform::None,
        },
        MappedTensor {
            hf_name: format!("{prefix}.A_log"),
            producer: Producer::PermuteRows {
                src: format!("blk.{blk}.ssm_a"),
                row_elems: 1,
                map: vmap.clone(),
            },
            shape: Some(vec![s.num_value_heads]),
            transform: a_log,
        },
        MappedTensor {
            hf_name: format!("{prefix}.dt_bias"),
            producer: Producer::PermuteRows {
                src: format!("blk.{blk}.ssm_dt.bias"),
                row_elems: 1,
                map: vmap.clone(),
            },
            shape: Some(vec![s.num_value_heads]),
            transform: Transform::None,
        },
        MappedTensor {
            hf_name: format!("{prefix}.conv1d.weight"),
            producer: Producer::PermuteRows {
                src: format!("blk.{blk}.ssm_conv1d.weight"),
                row_elems: s.conv_kernel,
                map: head_rows(key_dim * 2),
            },
            shape: Some(vec![conv_dim, 1, s.conv_kernel]),
            transform: Transform::None,
        },
        MappedTensor::direct(
            format!("{prefix}.norm.weight"),
            format!("blk.{blk}.ssm_norm.weight"),
        ),
        MappedTensor {
            hf_name: format!("{prefix}.out_proj.weight"),
            producer: Producer::PermuteCols {
                src: format!("blk.{blk}.ssm_out.weight"),
                row_elems: value_dim,
                block: hd,
                map: vmap,
            },
            shape: Some(vec![s.hidden_size, value_dim]),
            transform: Transform::None,
        },
    ]
}

fn a_log_transform(f: &GgufFile, s: &TextShape) -> Result<Transform> {
    let name = (0..s.num_hidden_layers)
        .map(|i| format!("blk.{i}.ssm_a"))
        .find(|n| f.tensor(n).is_some());
    let Some(name) = name else {
        return Ok(Transform::None);
    };
    let info = f.tensor(&name).unwrap();
    let n = info.elem_count();
    let mut buf = vec![0f32; n];
    crate::dequant::dequantize(info.ty, f.tensor_bytes(info)?, n, &mut buf)?;
    let all_negative = buf.iter().all(|x| *x < 0.0);
    if all_negative {
        tracing::info!(
            tensor = %name,
            "ssm_a хранится как -exp(A_log); восстанавливаю A_log через ln(-x)"
        );
        Ok(Transform::LogNeg)
    } else {
        Ok(Transform::None)
    }
}

fn vision_tensors(f: &GgufFile, v: &VisionShape) -> Result<Vec<MappedTensor>> {
    let mut out = Vec::new();
    let mut parts = vec!["v.patch_embd.weight".to_string()];
    for i in 1..v.temporal_patch_size {
        parts.push(format!("v.patch_embd.weight.{i}"));
    }
    let block = v.patch_size * v.patch_size;
    out.push(MappedTensor {
        hf_name: format!("{VIS}.patch_embed.proj.weight"),
        producer: if parts.len() == 1 {
            Producer::Direct(parts[0].clone())
        } else {
            Producer::Interleave { parts, block }
        },
        shape: Some(vec![
            v.hidden_size,
            3,
            v.temporal_patch_size,
            v.patch_size,
            v.patch_size,
        ]),
        transform: Transform::None,
    });
    out.push(MappedTensor::direct(
        format!("{VIS}.patch_embed.proj.bias"),
        "v.patch_embd.bias",
    ));
    out.push(MappedTensor::direct(
        format!("{VIS}.pos_embed.weight"),
        "v.position_embd.weight",
    ));

    for i in 0..v.depth {
        let p = format!("{VIS}.blocks.{i}");
        for (hf, gg) in [
            ("norm1.weight", "ln1.weight"),
            ("norm1.bias", "ln1.bias"),
            ("norm2.weight", "ln2.weight"),
            ("norm2.bias", "ln2.bias"),
            ("attn.qkv.weight", "attn_qkv.weight"),
            ("attn.qkv.bias", "attn_qkv.bias"),
            ("attn.proj.weight", "attn_out.weight"),
            ("attn.proj.bias", "attn_out.bias"),
            ("mlp.linear_fc1.weight", "ffn_up.weight"),
            ("mlp.linear_fc1.bias", "ffn_up.bias"),
            ("mlp.linear_fc2.weight", "ffn_down.weight"),
            ("mlp.linear_fc2.bias", "ffn_down.bias"),
        ] {
            out.push(MappedTensor::direct(
                format!("{p}.{hf}"),
                format!("v.blk.{i}.{gg}"),
            ));
        }
    }

    for (hf, gg) in [
        ("merger.norm.weight", "v.post_ln.weight"),
        ("merger.norm.bias", "v.post_ln.bias"),
        ("merger.linear_fc1.weight", "mm.0.weight"),
        ("merger.linear_fc1.bias", "mm.0.bias"),
        ("merger.linear_fc2.weight", "mm.2.weight"),
        ("merger.linear_fc2.bias", "mm.2.bias"),
    ] {
        out.push(MappedTensor::direct(format!("{VIS}.{hf}"), gg));
    }

    let missing: Vec<String> = out
        .iter()
        .flat_map(|m| m.producer.sources().iter().cloned())
        .filter(|n| f.tensor(n).is_none())
        .collect();
    if !missing.is_empty() {
        return Err(GgufError::BadTensor {
            name: missing[0].clone(),
            reason: format!("не найден в mmproj (всего пропущено {})", missing.len()),
        });
    }
    Ok(out)
}

fn token_id(vocab: &GgufVocab, name: &str) -> Option<u32> {
    vocab.tokens.iter().position(|t| t == name).map(|i| i as u32)
}

pub fn config_json(
    s: &TextShape,
    v: Option<&VisionShape>,
    vocab: &GgufVocab,
) -> Result<Vec<u8>> {
    let mut text = json!({
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attn_output_gate": s.attn_output_gate,
        "bos_token_id": vocab.bos,
        "dtype": "bfloat16",
        "eos_token_id": vocab.eos_ids(),
        "full_attention_interval": s.full_attention_interval,
        "head_dim": s.head_dim,
        "hidden_act": "silu",
        "hidden_size": s.hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": s.intermediate_size,
        "layer_types": s.layer_types(),
        "linear_conv_kernel_dim": s.conv_kernel,
        "linear_key_head_dim": s.state_size,
        "linear_num_key_heads": s.num_key_heads,
        "linear_num_value_heads": s.num_value_heads,
        "linear_value_head_dim": s.state_size,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": s.context_length,
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": s.mtp_layers,
        "mtp_use_dedicated_embeddings": false,
        "num_attention_heads": s.num_attention_heads,
        "num_hidden_layers": s.num_hidden_layers,
        "num_key_value_heads": s.num_key_value_heads,
        "output_gate_type": "swish",
        "partial_rotary_factor": s.partial_rotary_factor(),
        "rms_norm_eps": s.rms_norm_eps,
        "tie_word_embeddings": s.tie_word_embeddings,
        "use_cache": false,
        "vocab_size": s.vocab_size,
    });
    let mut rope = json!({
        "partial_rotary_factor": s.partial_rotary_factor(),
        "rope_theta": s.rope_theta,
        "rope_type": "default",
    });
    if !s.mrope_section.is_empty() {
        rope["mrope_interleaved"] = J::from(true);
        rope["mrope_section"] = J::from(s.mrope_section.clone());
    }
    text["rope_parameters"] = rope;

    let mut root = json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "dtype": "bfloat16",
        "model_type": "qwen3_5",
        "pad_token_id": vocab.pad,
        "text_config": text,
        "tie_word_embeddings": s.tie_word_embeddings,
    });

    if let Some(v) = v {
        root["language_model_only"] = J::from(false);
        root["vision_config"] = json!({
            "deepstack_visual_indexes": v.deepstack_indexes,
            "depth": v.depth,
            "dtype": "bfloat16",
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": v.hidden_size,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": v.intermediate_size,
            "layer_norm_eps": v.layer_norm_eps,
            "model_type": "qwen3_5",
            "num_heads": v.num_heads,
            "num_position_embeddings": v.num_position_embeddings,
            "out_hidden_size": if v.out_hidden_size > 0 { v.out_hidden_size } else { s.hidden_size },
            "patch_size": v.patch_size,
            "spatial_merge_size": v.spatial_merge_size,
            "temporal_patch_size": v.temporal_patch_size,
        });
        for (key, tok) in [
            ("image_token_id", "<|image_pad|>"),
            ("video_token_id", "<|video_pad|>"),
            ("vision_start_token_id", "<|vision_start|>"),
            ("vision_end_token_id", "<|vision_end|>"),
        ] {
            if let Some(id) = token_id(vocab, tok) {
                root[key] = J::from(id);
            }
        }
    } else {
        root["language_model_only"] = J::from(true);
    }

    Ok(serde_json::to_vec_pretty(&root)?)
}

fn generation_config_json(f: &GgufFile, vocab: &GgufVocab) -> Result<Vec<u8>> {
    let mut doc = json!({
        "do_sample": true,
        "eos_token_id": vocab.eos_ids(),
    });
    if let Some(b) = vocab.bos {
        doc["bos_token_id"] = J::from(b);
    }
    if let Some(p) = vocab.pad {
        doc["pad_token_id"] = J::from(p);
    }
    if let Some(t) = f.opt_f32("general.sampling.temp") {
        doc["temperature"] = J::from(t);
    }
    if let Some(t) = f.opt_f32("general.sampling.top_p") {
        doc["top_p"] = J::from(t);
    }
    if let Some(t) = f.opt_usize("general.sampling.top_k") {
        doc["top_k"] = J::from(t);
    }
    Ok(serde_json::to_vec_pretty(&doc)?)
}

fn preprocessor_config_json(v: &VisionShape) -> Result<Vec<u8>> {
    let doc = json!({
        "image_mean": v.image_mean,
        "image_std": v.image_std,
        "merge_size": v.spatial_merge_size,
        "patch_size": v.patch_size,
        "temporal_patch_size": v.temporal_patch_size,
        "image_processor_type": "Qwen2VLImageProcessorFast",
        "processor_class": "Qwen3VLProcessor",
    });
    Ok(serde_json::to_vec_pretty(&doc)?)
}

pub fn build_plan(
    model: &GgufFile,
    mmproj: Option<&GgufFile>,
    bundle_id: &str,
) -> Result<ConversionPlan> {
    let shape = TextShape::read(model)?;
    let vocab = GgufVocab::read(model)?;
    let vision = match mmproj {
        Some(f) => Some(VisionShape::read(f)?),
        None => None,
    };

    let mut components = vec![Component {
        name: "main".into(),
        tensors: text_tensors(model, &shape)?,
    }];
    if let (Some(f), Some(v)) = (mmproj, vision.as_ref()) {
        components.push(Component {
            name: "vision".into(),
            tensors: vision_tensors(f, v)?,
        });
    }

    let mut files = vec![
        MappedFile {
            path: "config.json".into(),
            bytes: config_json(&shape, vision.as_ref(), &vocab)?,
        },
        MappedFile {
            path: "tokenizer.json".into(),
            bytes: vocab.to_tokenizer_json()?,
        },
        MappedFile {
            path: "tokenizer_config.json".into(),
            bytes: vocab.to_tokenizer_config_json()?,
        },
        MappedFile {
            path: "generation_config.json".into(),
            bytes: generation_config_json(model, &vocab)?,
        },
    ];
    if let Some(t) = &vocab.chat_template {
        files.push(MappedFile {
            path: "chat_template.jinja".into(),
            bytes: t.clone().into_bytes(),
        });
    }
    if let Some(v) = vision.as_ref() {
        files.push(MappedFile {
            path: "preprocessor_config.json".into(),
            bytes: preprocessor_config_json(v)?,
        });
    }

    Ok(ConversionPlan {
        bundle_id: bundle_id.to_string(),
        arch: "qwen3_5".into(),
        components,
        files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> TextShape {
        TextShape {
            hidden_size: 5120,
            intermediate_size: 17408,
            num_hidden_layers: 64,
            mtp_layers: 1,
            num_attention_heads: 24,
            num_key_value_heads: 4,
            head_dim: 256,
            rotary_dim: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 1e7,
            mrope_section: vec![11, 11, 10],
            full_attention_interval: 4,
            conv_kernel: 4,
            state_size: 128,
            num_key_heads: 16,
            num_value_heads: 48,
            context_length: 262_144,
            vocab_size: 248_320,
            attn_output_gate: true,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn layer_types_match_3_to_1_pattern() {
        let s = shape();
        let lt = s.layer_types();
        assert_eq!(lt.len(), 64);
        assert_eq!(lt[0], "linear_attention");
        assert_eq!(lt[2], "linear_attention");
        assert_eq!(lt[3], "full_attention");
        assert_eq!(lt[63], "full_attention");
        assert_eq!(lt.iter().filter(|x| **x == "full_attention").count(), 16);
    }

    #[test]
    fn conv1d_gets_hf_three_dim_shape() {
        let s = shape();
        let t = linear_attn("p", 0, &s, Transform::None);
        let conv = t.iter().find(|m| m.hf_name.ends_with("conv1d.weight")).unwrap();
        assert_eq!(conv.shape, Some(vec![10240, 1, 4]));
    }

    #[test]
    fn value_head_permutation_matches_upstream_layout() {
        let m = crate::plan::value_head_map(48, 16);
        assert_eq!(m[0], 0);
        assert_eq!(m[1], 16);
        assert_eq!(m[2], 32);
        assert_eq!(m[3], 1);
        assert_eq!(m[16], 21);
        assert_eq!(m[17], 37);
        assert_eq!(m.len(), 48);
        let mut sorted = m.clone();
        sorted.sort();
        assert_eq!(sorted, (0..48u32).collect::<Vec<_>>());
    }

    #[test]
    fn qkv_permutation_keeps_qk_identity_and_moves_v_heads() {
        let s = shape();
        let t = linear_attn("p", 0, &s, Transform::None);
        let qkv = t.iter().find(|m| m.hf_name.ends_with("in_proj_qkv.weight")).unwrap();
        let crate::plan::Producer::PermuteRows { map, row_elems, .. } = &qkv.producer else {
            panic!("ожидалась перестановка строк");
        };
        assert_eq!(*row_elems, 5120);
        assert_eq!(map.len(), 10240);
        assert_eq!(map[0], 0);
        assert_eq!(map[2048], 2048);
        assert_eq!(map[4096], 4096);
        assert_eq!(map[4096 + 128], 4096 + 16 * 128);
        assert_eq!(map[4096 + 2048], 4096 + 21 * 128);
    }

    #[test]
    fn norms_are_shifted_by_minus_one_except_ssm_norm() {
        let s = shape();
        let blk = block_common("p", 0, &s);
        for m in &blk {
            if m.hf_name.ends_with("layernorm.weight") {
                assert_eq!(m.transform, Transform::SubOne, "{}", m.hf_name);
            }
        }
        let lin = linear_attn("p", 0, &s, Transform::None);
        let ssm = lin.iter().find(|m| m.hf_name.ends_with("linear_attn.norm.weight") || m.hf_name == "p.norm.weight").unwrap();
        assert_eq!(ssm.transform, Transform::None);
        let attn = full_attn("p", 3);
        for m in &attn {
            if m.hf_name.ends_with("_norm.weight") {
                assert_eq!(m.transform, Transform::SubOne, "{}", m.hf_name);
            }
        }
    }

    #[test]
    fn config_json_parses_into_hybrid_layout() {
        let s = shape();
        let vocab = GgufVocab {
            tokens: vec!["a".into()],
            types: vec![1],
            merges: vec![],
            model: "gpt2".into(),
            pre: None,
            bos: Some(248044),
            eos: Some(248046),
            pad: Some(248044),
            unk: None,
            eot: None,
            add_bos: None,
            add_eos: None,
            chat_template: None,
        };
        let bytes = config_json(&s, None, &vocab).unwrap();
        let d: J = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(d["text_config"]["hidden_size"], 5120);
        assert_eq!(d["text_config"]["partial_rotary_factor"], 0.25);
        assert_eq!(d["text_config"]["rope_parameters"]["rope_theta"], 1e7);
        assert_eq!(d["text_config"]["mtp_num_hidden_layers"], 1);
        assert_eq!(d["text_config"]["layer_types"].as_array().unwrap().len(), 64);
        assert_eq!(d["language_model_only"], true);
    }

    #[test]
    fn mtp_block_maps_to_dedicated_prefix() {
        let s = shape();
        let names: Vec<String> = block_common("mtp.layers.0", 64, &s)
            .iter()
            .map(|m| m.hf_name.clone())
            .collect();
        assert!(names.contains(&"mtp.layers.0.input_layernorm.weight".to_string()));
    }
}
