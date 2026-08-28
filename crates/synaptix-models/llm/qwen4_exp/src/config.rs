use serde_json::Value;

use synaptix_llm_common::config::LinearAttnConfig;
use synaptix_llm_common::moe::MoeConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Linear,
    Qsa,
}

#[derive(Debug, Clone)]
pub struct RopeConfig {
    pub theta: f32,
    pub rotary_dim: usize,
    pub mrope_section: Option<[usize; 3]>,
    pub mrope_interleaved: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct IndexerConfig {
    pub n_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub budget: usize,
    pub compress_ratio: usize,
}

impl IndexerConfig {
    /// Сколько блоков оставляет индексатор. `SYN_QWEN4EXP_QSA_TOPK` урезает
    /// бюджет — так разреженный путь включается на коротком контексте, где его
    /// можно сверить с полным вниманием.
    pub fn block_topk(&self) -> usize {
        let full = self.budget / self.compress_ratio;
        match std::env::var("SYN_QWEN4EXP_QSA_TOPK").ok().and_then(|v| v.trim().parse().ok()) {
            Some(n) if n > 0 && n < full => n,
            _ => full,
        }
    }
    pub fn qk_dim(&self) -> usize {
        (self.n_heads + self.kv_heads) * self.head_dim
    }
}

#[derive(Debug, Clone)]
pub struct PleConfig {
    pub layer_ids: Vec<usize>,
    pub embed_dim: usize,
    pub conv_kernel_size: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ngram_vocab_size_base: u64,
    pub make_vocab_divisible_by: usize,
    pub seed: u64,
    pub split_parts: usize,
}

impl PleConfig {
    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.ngram_heads()
    }
    pub fn conv_state_len(&self) -> usize {
        (self.conv_kernel_size - 1) * self.ngram_size
    }
    pub fn context_len(&self) -> usize {
        self.ngram_size - 1
    }
    pub fn index_of(&self, layer: usize) -> Option<usize> {
        self.layer_ids.iter().position(|l| *l == layer)
    }
}

#[derive(Debug, Clone)]
pub struct MtpConfig {
    pub num_hidden_layers: usize,
    pub rope_theta: f32,
    pub hybrid: bool,
}

#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
}

#[derive(Debug, Clone)]
pub struct Qwen4ExpConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,

    pub layer_types: Vec<LayerType>,
    pub linear: LinearAttnConfig,
    pub moe: MoeConfig,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub ple: Option<PleConfig>,
    pub indexer: IndexerConfig,
    pub rope: RopeConfig,
    pub output_gate_sigmoid: bool,

    pub tie_word_embeddings: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: Option<u32>,

    pub image_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
    pub vision_start_token_id: Option<u32>,
    pub vision_end_token_id: Option<u32>,

    pub mtp: Option<MtpConfig>,
    pub vision: Option<VisionConfig>,
}

fn usize_at(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
}

fn f32_at(v: &Value, key: &str) -> Option<f32> {
    v.get(key).and_then(|x| x.as_f64()).map(|x| x as f32)
}

fn u32_list(v: &Value, key: &str) -> Vec<u32> {
    match v.get(key) {
        Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect(),
        _ => Vec::new(),
    }
}

impl Qwen4ExpConfig {
    pub fn from_hf_bytes(bytes: &[u8]) -> Result<Self, ConfigError> {
        let root: Value = serde_json::from_slice(bytes).map_err(|e| ConfigError(e.to_string()))?;
        Self::from_value(&root)
    }

    pub fn from_value(root: &Value) -> Result<Self, ConfigError> {
        let text = root.get("text_config").unwrap_or(root);
        let need = |key: &str| -> Result<usize, ConfigError> {
            usize_at(text, key).ok_or_else(|| ConfigError(format!("нет поля {key}")))
        };

        let hidden_size = need("hidden_size")?;
        let num_hidden_layers = need("num_hidden_layers")?;
        let head_dim = usize_at(text, "head_dim").unwrap_or_else(|| {
            hidden_size / usize_at(text, "num_attention_heads").unwrap_or(1)
        });

        let layer_types = match text.get("layer_types").and_then(|x| x.as_array()) {
            Some(arr) => arr
                .iter()
                .map(|x| match x.as_str().unwrap_or("") {
                    "linear_attention" => Ok(LayerType::Linear),
                    "full_attention" | "qwen_sparse_attention" => Ok(LayerType::Qsa),
                    other => Err(ConfigError(format!("неизвестный layer_type: {other}"))),
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => {
                let interval = usize_at(text, "full_attention_interval").unwrap_or(4);
                (0..num_hidden_layers)
                    .map(|i| {
                        if (i + 1) % interval == 0 {
                            LayerType::Qsa
                        } else {
                            LayerType::Linear
                        }
                    })
                    .collect()
            }
        };
        if layer_types.len() != num_hidden_layers {
            return Err(ConfigError(format!(
                "layer_types: {} записей при num_hidden_layers={num_hidden_layers}",
                layer_types.len()
            )));
        }

        let linear = LinearAttnConfig {
            num_key_heads: usize_at(text, "linear_num_key_heads").unwrap_or(16),
            num_value_heads: usize_at(text, "linear_num_value_heads").unwrap_or(32),
            key_head_dim: usize_at(text, "linear_key_head_dim").unwrap_or(128),
            value_head_dim: usize_at(text, "linear_value_head_dim").unwrap_or(128),
            conv_kernel: usize_at(text, "linear_conv_kernel_dim").unwrap_or(4),
        };

        let moe = MoeConfig {
            hidden_size,
            moe_intermediate_size: usize_at(text, "moe_intermediate_size").unwrap_or(640),
            num_experts: usize_at(text, "num_experts").unwrap_or(512),
            num_experts_per_tok: usize_at(text, "num_experts_per_tok").unwrap_or(10),
            shared_intermediate_size: usize_at(text, "shared_expert_intermediate_size").unwrap_or(640),
            norm_topk_prob: text
                .get("norm_topk_prob")
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
            chunk: 512,
            skip_below: std::env::var("SYN_MOE_SKIP_BELOW")
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .unwrap_or(0.0),
        };

        let ple_layer_ids: Vec<usize> = text
            .get("ple_layer_ids")
            .and_then(|x| x.as_array())
            .map(|a| {
                let mut v: Vec<usize> = a
                    .iter()
                    .filter_map(|x| x.as_u64())
                    .filter(|x| *x >= 1)
                    .map(|x| x as usize - 1)
                    .collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .unwrap_or_default();
        let ple = if ple_layer_ids.is_empty() {
            None
        } else {
            let cfg = PleConfig {
                layer_ids: ple_layer_ids,
                embed_dim: usize_at(text, "ple_embed_dim").unwrap_or(hidden_size),
                conv_kernel_size: usize_at(text, "ple_conv_kernel_size").unwrap_or(4),
                ngram_size: usize_at(text, "ngram_size").unwrap_or(3),
                heads_per_ngram: usize_at(text, "heads_per_ngram").unwrap_or(8),
                ngram_vocab_size_base: text
                    .get("ngram_vocab_size_base")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(20_000_000),
                make_vocab_divisible_by: usize_at(text, "make_ngram_vocab_size_divisible_by")
                    .unwrap_or(128),
                seed: text.get("seed").and_then(|x| x.as_u64()).unwrap_or(1234),
                split_parts: usize_at(text, "split_ngram_parts").unwrap_or(512),
            };
            if cfg.ngram_heads() == 0 || cfg.embed_dim % cfg.ngram_heads() != 0 {
                return Err(ConfigError(format!(
                    "ple_embed_dim {} не делится на {} n-gram-голов",
                    cfg.embed_dim,
                    cfg.ngram_heads()
                )));
            }
            Some(cfg)
        };

        let partial = f32_at(text, "partial_rotary_factor")
            .or_else(|| {
                text.get("rope_parameters")
                    .and_then(|r| f32_at(r, "partial_rotary_factor"))
            })
            .unwrap_or(1.0);
        let rope_params = text.get("rope_parameters");
        let theta = rope_params
            .and_then(|r| f32_at(r, "rope_theta"))
            .or_else(|| f32_at(text, "rope_theta"))
            .unwrap_or(10_000.0);
        let mrope_section = rope_params
            .and_then(|r| r.get("mrope_section"))
            .and_then(|x| x.as_array())
            .and_then(|a| {
                let v: Vec<usize> = a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect();
                (v.len() == 3).then(|| [v[0], v[1], v[2]])
            });
        let rope = RopeConfig {
            theta,
            rotary_dim: ((head_dim as f32 * partial) as usize) / 2 * 2,
            mrope_section,
            mrope_interleaved: rope_params
                .and_then(|r| r.get("mrope_interleaved"))
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
        };

        let indexer = IndexerConfig {
            n_heads: usize_at(text, "indexer_n_heads").unwrap_or(0),
            kv_heads: usize_at(text, "indexer_kv_heads").unwrap_or(1),
            head_dim: usize_at(text, "indexer_head_dim").unwrap_or(0),
            budget: usize_at(text, "indexer_budget").unwrap_or(0),
            compress_ratio: usize_at(text, "indexer_compress_ratio").unwrap_or(1),
        };
        let qsa_used = layer_types.iter().any(|t| *t == LayerType::Qsa);
        if qsa_used {
            if indexer.n_heads == 0 || indexer.head_dim == 0 || indexer.budget == 0 {
                return Err(ConfigError("QSA-слои есть, а indexer_* не заданы".into()));
            }
            if indexer.kv_heads != 1 {
                return Err(ConfigError("QSA требует indexer_kv_heads = 1".into()));
            }
            if indexer.budget % indexer.compress_ratio != 0 {
                return Err(ConfigError("indexer_budget не делится на indexer_compress_ratio".into()));
            }
            if rope.rotary_dim > indexer.head_dim {
                return Err(ConfigError(format!(
                    "rotary_dim {} больше indexer_head_dim {}",
                    rope.rotary_dim, indexer.head_dim
                )));
            }
        }

        let mtp = text.get("mtp").map(|m| MtpConfig {
            num_hidden_layers: usize_at(m, "num_hidden_layers")
                .or_else(|| usize_at(text, "mtp_num_hidden_layers"))
                .unwrap_or(1),
            rope_theta: f32_at(m, "rope_theta").unwrap_or(theta),
            hybrid: m.get("hybrid").and_then(|x| x.as_bool()).unwrap_or(false),
        });

        let vision = root.get("vision_config").map(|v| VisionConfig {
            depth: usize_at(v, "depth").unwrap_or(0),
            hidden_size: usize_at(v, "hidden_size").unwrap_or(0),
            intermediate_size: usize_at(v, "intermediate_size").unwrap_or(0),
            num_heads: usize_at(v, "num_heads").unwrap_or(0),
            patch_size: usize_at(v, "patch_size").unwrap_or(16),
            temporal_patch_size: usize_at(v, "temporal_patch_size").unwrap_or(2),
            spatial_merge_size: usize_at(v, "spatial_merge_size").unwrap_or(2),
            out_hidden_size: usize_at(v, "out_hidden_size").unwrap_or(hidden_size),
            num_position_embeddings: usize_at(v, "num_position_embeddings").unwrap_or(0),
        });

        let output_gate_sigmoid = text
            .get("output_gate_type")
            .and_then(|x| x.as_str())
            .map(|s| s == "sigmoid")
            .unwrap_or(false);

        let token = |key: &str| -> Option<u32> {
            root.get(key)
                .or_else(|| text.get(key))
                .and_then(|x| x.as_u64())
                .map(|x| x as u32)
        };

        Ok(Self {
            vocab_size: need("vocab_size")?,
            hidden_size,
            num_hidden_layers,
            num_attention_heads: need("num_attention_heads")?,
            num_key_value_heads: need("num_key_value_heads")?,
            head_dim,
            rms_norm_eps: f32_at(text, "rms_norm_eps").unwrap_or(1e-6),
            max_position_embeddings: usize_at(text, "max_position_embeddings").unwrap_or(32768),
            layer_types,
            linear,
            moe,
            hc_count: usize_at(text, "hc_count").unwrap_or(4),
            hc_lowrank: usize_at(text, "hc_lowrank").unwrap_or(320),
            ple,
            indexer,
            rope,
            output_gate_sigmoid,
            tie_word_embeddings: text
                .get("tie_word_embeddings")
                .or_else(|| root.get("tie_word_embeddings"))
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            bos_token_id: text.get("bos_token_id").and_then(|x| x.as_u64()).map(|x| x as u32),
            eos_token_ids: u32_list(text, "eos_token_id"),
            pad_token_id: text.get("pad_token_id").and_then(|x| x.as_u64()).map(|x| x as u32),
            image_token_id: token("image_token_id"),
            video_token_id: token("video_token_id"),
            vision_start_token_id: token("vision_start_token_id"),
            vision_end_token_id: token("vision_end_token_id"),
            mtp,
            vision,
        })
    }

    pub fn hc_hidden(&self) -> usize {
        self.hc_count * self.hidden_size
    }

    pub fn layer_type(&self, idx: usize) -> LayerType {
        self.layer_types.get(idx).copied().unwrap_or(LayerType::Qsa)
    }

    pub fn q_total_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_total_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    pub fn ple_at(&self, layer: usize) -> Option<usize> {
        self.ple.as_ref().and_then(|p| p.index_of(layer))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("config: {0}")]
pub struct ConfigError(pub String);

#[cfg(test)]
mod tests {
    use super::*;

    const REAL: &str = r#"{
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen4_exp",
        "text_config": {
            "full_attention_interval": 4, "hc_count": 4, "hc_lowrank": 320,
            "head_dim": 256, "heads_per_ngram": 8, "hidden_act": "silu",
            "hidden_size": 2560, "indexer_budget": 2048, "indexer_compress_ratio": 4,
            "indexer_head_dim": 128, "indexer_kv_heads": 1, "indexer_n_heads": 4,
            "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"],
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "linear_value_head_dim": 128, "make_ngram_vocab_size_divisible_by": 128,
            "max_position_embeddings": 262144, "moe_intermediate_size": 640,
            "mtp": {"hybrid": true, "num_hidden_layers": 1, "rope_theta": 10000000},
            "ngram_size": 3, "ngram_vocab_size_base": 20000000,
            "num_attention_heads": 24, "num_experts": 512, "num_experts_per_tok": 10,
            "num_hidden_layers": 4, "num_key_value_heads": 2, "output_gate_type": "sigmoid",
            "partial_rotary_factor": 0.25, "ple_conv_kernel_size": 4, "ple_embed_dim": 2560,
            "ple_layer_ids": [2], "rms_norm_eps": 1e-06,
            "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11,11,10],
                                "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                                "rope_type": "default"},
            "shared_expert_intermediate_size": 640, "split_ngram_parts": 128,
            "tie_word_embeddings": false, "vocab_size": 248320, "eos_token_id": 248044
        },
        "vision_config": {"depth": 27, "hidden_size": 1152, "intermediate_size": 4304,
                          "num_heads": 16, "out_hidden_size": 2560, "patch_size": 16,
                          "num_position_embeddings": 2304, "spatial_merge_size": 2,
                          "temporal_patch_size": 2}
    }"#;

    #[test]
    fn parses_real_config() {
        let c = Qwen4ExpConfig::from_hf_bytes(REAL.as_bytes()).unwrap();
        assert_eq!(c.hidden_size, 2560);
        assert_eq!(c.hc_hidden(), 10240);
        assert_eq!(c.layer_types, vec![
            LayerType::Linear,
            LayerType::Linear,
            LayerType::Linear,
            LayerType::Qsa
        ]);
        assert_eq!(c.rope.rotary_dim, 64);
        assert_eq!(c.rope.mrope_section, Some([11, 11, 10]));
        assert!(c.output_gate_sigmoid);
        assert_eq!(c.indexer.block_topk(), 512);
        assert_eq!(c.indexer.qk_dim(), 640);
        assert_eq!(c.linear.conv_dim(), 10240);
        assert_eq!(c.linear.value_dim(), 6144);
        assert_eq!(c.moe.num_experts, 512);
        assert!(c.moe.norm_topk_prob);
        let ple = c.ple.as_ref().unwrap();
        assert_eq!(ple.layer_ids, vec![1]);
        assert_eq!(ple.ngram_heads(), 16);
        assert_eq!(ple.head_dim(), 160);
        assert_eq!(ple.conv_state_len(), 9);
        assert_eq!(ple.split_parts, 128);
        assert_eq!(c.ple_at(1), Some(0));
        assert_eq!(c.ple_at(0), None);
        assert_eq!(c.mtp.as_ref().unwrap().num_hidden_layers, 1);
        assert_eq!(c.vision.as_ref().unwrap().depth, 27);
        assert_eq!(c.image_token_id, Some(248056));
        assert_eq!(c.eos_token_ids, vec![248044]);
    }

    #[test]
    fn layer_types_from_interval() {
        let json = r#"{"hidden_size": 64, "num_hidden_layers": 8, "vocab_size": 100,
                       "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
                       "full_attention_interval": 4, "indexer_n_heads": 2, "indexer_head_dim": 32,
                       "indexer_budget": 8, "indexer_compress_ratio": 4, "indexer_kv_heads": 1}"#;
        let c = Qwen4ExpConfig::from_hf_bytes(json.as_bytes()).unwrap();
        assert_eq!(c.layer_types[2], LayerType::Linear);
        assert_eq!(c.layer_types[3], LayerType::Qsa);
        assert_eq!(c.layer_types[7], LayerType::Qsa);
        assert!(c.ple.is_none());
    }
}
