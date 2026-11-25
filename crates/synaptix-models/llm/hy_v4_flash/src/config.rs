//! `config.json` Hy4-flash: всё нужное лежит в `text_config` (`model_type`
//! `hy_v4_flash`). Имена — как в чекпойнте tencent; соответствие эталону
//! transformers (PR #50903) — в комментариях полей.

use serde_json::Value;

/// Вид внимания слоя.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    /// `sliding_attention`: окно `sliding_window` (своя позиция и 511 прошлых), 64 головы.
    Sliding,
    /// `deepseek_sparse_attention` (= `indexed_attention`): плотно до
    /// `index_topk` ключей, дальше — группы, выбранные сжатым индексатором.
    Indexed,
}

/// Размеры MLA одного вида слоёв.
#[derive(Debug, Clone, Copy)]
pub struct MlaDims {
    pub heads: usize,
    pub qk_nope: usize,
    pub qk_rope: usize,
    pub v: usize,
    pub rope_theta: f64,
}

impl MlaDims {
    pub fn qk(&self) -> usize {
        self.qk_nope + self.qk_rope
    }
}

#[derive(Debug, Clone)]
pub struct EngramConfig {
    /// Индексы слоёв (с нуля), перед вниманием которых добавляется память:
    /// `engram_layer_ids` в конфиге — с единицы (слой `id − 1`).
    pub layers: Vec<usize>,
    /// Номер для хэша и размеров голов (`engram_layer_ids`, с единицы).
    pub layer_ids: Vec<usize>,
    pub max_ngram: usize,
    pub heads_per_ngram: usize,
    pub head_dim: usize,
    /// По слою: модули голов, n-граммы по порядку (2-граммы, затем 3-граммы).
    pub head_vocab: Vec<Vec<u64>>,
    pub pad_id: u32,
    pub seed: u64,
    pub fp8_embedding: bool,
}

impl EngramConfig {
    pub fn heads(&self) -> usize {
        (self.max_ngram - 1) * self.heads_per_ngram
    }
    pub fn width(&self) -> usize {
        self.heads() * self.head_dim
    }
    pub fn index_of(&self, layer: usize) -> Option<usize> {
        self.layers.iter().position(|&l| l == layer)
    }
}

#[derive(Debug, Clone)]
pub struct Hy4Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub attn: Vec<AttnKind>,
    pub indexed: MlaDims,
    pub sliding: MlaDims,
    pub sliding_window: usize,
    /// Плотные слои MLP (`mlp_layer_types` = `dense`).
    pub dense: Vec<bool>,
    pub intermediate_size: usize,
    /// Латентная ширина экспертов (`routed_expert_hidden_size`).
    pub moe_hidden: usize,
    pub moe_intermediate: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub n_shared: usize,
    pub routed_scale: f32,
    pub norm_topk: bool,
    pub swiglu_limit: f32,
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    /// Токенов в группе индексатора (`compressed_indexer_compress_ratio`).
    pub index_kpool: usize,
    pub hc_mult: usize,
    pub hc_magnitude: f32,
    pub hc_eps: f32,
    pub engram: Option<EngramConfig>,
    pub bos: u32,
    pub eos: Vec<u32>,
    pub pad: u32,
    pub max_position_embeddings: usize,
    pub image_token_id: Option<u32>,
}

fn u(v: &Value, k: &str) -> Result<usize, String> {
    v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| format!("config: нет {k}"))
}

fn f(v: &Value, k: &str) -> Result<f64, String> {
    v.get(k).and_then(|x| x.as_f64()).ok_or_else(|| format!("config: нет {k}"))
}

fn u_or(v: &Value, k: &str, d: usize) -> usize {
    v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(d)
}

impl Hy4Config {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let root: Value = serde_json::from_slice(bytes).map_err(|e| format!("config.json: {e}"))?;
        let t = root.get("text_config").unwrap_or(&root);
        let layers = u(t, "num_hidden_layers")?;
        let attn: Vec<AttnKind> = match t.get("layer_types").and_then(|x| x.as_array()) {
            Some(a) => a
                .iter()
                .map(|x| match x.as_str().unwrap_or("") {
                    "sliding_attention" => Ok(AttnKind::Sliding),
                    "deepseek_sparse_attention" | "indexed_attention" => Ok(AttnKind::Indexed),
                    o => Err(format!("config: вид внимания {o}")),
                })
                .collect::<Result<_, _>>()?,
            None => (0..layers).map(|l| if l < 2 || (l - 1) % 4 == 0 { AttnKind::Indexed } else { AttnKind::Sliding }).collect(),
        };
        if attn.len() != layers {
            return Err(format!("config: layer_types {} при {layers} слоях", attn.len()));
        }
        let dense: Vec<bool> = match t.get("mlp_layer_types").and_then(|x| x.as_array()) {
            Some(a) => a.iter().map(|x| x.as_str() == Some("dense")).collect(),
            None => (0..layers).map(|l| l == 0).collect(),
        };
        let rope = t.get("rope_parameters");
        let theta = |kind: &str, d: f64| -> f64 {
            rope.and_then(|r| r.get(kind)).and_then(|r| r.get("rope_theta")).and_then(|x| x.as_f64()).unwrap_or(d)
        };
        let heads = u(t, "num_attention_heads")?;
        let qk_nope = u(t, "qk_nope_head_dim")?;
        let qk_rope = u(t, "qk_rope_head_dim")?;
        let v = u(t, "v_head_dim")?;
        let indexed = MlaDims {
            heads,
            qk_nope,
            qk_rope,
            v,
            rope_theta: theta("deepseek_sparse_attention", theta("indexed_attention", 1e7)),
        };
        let sliding = MlaDims {
            heads: u_or(t, "swa_num_attention_heads", 64),
            qk_nope: u_or(t, "swa_qk_nope_head_dim", qk_nope),
            qk_rope: u_or(t, "swa_qk_rope_head_dim", qk_rope),
            v: u_or(t, "swa_v_head_dim", v),
            rope_theta: theta("sliding_attention", 1e4),
        };
        let engram = match t.get("engram_layer_ids").and_then(|x| x.as_array()) {
            Some(ids) if !ids.is_empty() => {
                let ids: Vec<usize> = ids.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect();
                let max_ngram = u_or(t, "engram_max_ngram_size", 3);
                let heads_per_ngram = u_or(t, "engram_n_head_per_ngram", 8);
                let per = u_or(t, "engram_n_embed_per_ngram", 768);
                let sizes = t.get("engram_layer_head_vocab_sizes").ok_or("config: нет engram_layer_head_vocab_sizes")?;
                let mut head_vocab = Vec::new();
                for id in &ids {
                    let orders = sizes.get(id.to_string()).and_then(|x| x.as_array()).ok_or_else(|| format!("config: нет размеров engram слоя {id}"))?;
                    let flat: Vec<u64> = orders.iter().flat_map(|o| o.as_array().into_iter().flatten().filter_map(|x| x.as_u64())).collect();
                    if flat.len() != (max_ngram - 1) * heads_per_ngram {
                        return Err(format!("config: engram слоя {id}: {} голов", flat.len()));
                    }
                    head_vocab.push(flat);
                }
                Some(EngramConfig {
                    layers: ids.iter().map(|i| i - 1).collect(),
                    layer_ids: ids,
                    max_ngram,
                    heads_per_ngram,
                    head_dim: per / heads_per_ngram,
                    head_vocab,
                    pad_id: u_or(t, "engram_pad_id", 120002) as u32,
                    seed: u_or(t, "engram_seed", 0) as u64,
                    fp8_embedding: t.get("engram_fp8_embedding").and_then(|x| x.as_bool()).unwrap_or(false),
                })
            }
            _ => None,
        };
        let eos: Vec<u32> = match t.get("eos_token_id").or_else(|| root.get("eos_token_id")) {
            Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect(),
            Some(x) => x.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            None => Vec::new(),
        };
        Ok(Self {
            hidden_size: u(t, "hidden_size")?,
            num_hidden_layers: layers,
            vocab_size: u(t, "vocab_size")?,
            rms_norm_eps: f(t, "rms_norm_eps").unwrap_or(1e-5) as f32,
            q_lora_rank: u(t, "q_lora_rank")?,
            kv_lora_rank: u(t, "kv_lora_rank")?,
            attn,
            indexed,
            sliding,
            sliding_window: u_or(t, "sliding_window", 512),
            dense,
            intermediate_size: u(t, "intermediate_size")?,
            moe_hidden: t.get("routed_expert_hidden_size").or_else(|| t.get("moe_hidden_size")).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(1536),
            moe_intermediate: u(t, "moe_intermediate_size")?,
            n_experts: u(t, "n_routed_experts")?,
            top_k: u(t, "num_experts_per_tok")?,
            n_shared: u_or(t, "n_shared_experts", 1),
            routed_scale: f(t, "routed_scaling_factor").unwrap_or(1.0) as f32,
            norm_topk: t.get("norm_topk_prob").and_then(|x| x.as_bool()).unwrap_or(true),
            swiglu_limit: f(t, "swiglu_limit").unwrap_or(0.0) as f32,
            index_heads: u_or(t, "index_n_heads", 16),
            index_head_dim: u_or(t, "index_head_dim", 128),
            index_topk: u_or(t, "index_topk", 2048),
            index_kpool: t.get("compressed_indexer_compress_ratio").or_else(|| t.get("index_kpool")).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(4),
            hc_mult: u_or(t, "hc_mult", 4),
            hc_magnitude: f(t, "hc_magnitude").unwrap_or(2.0) as f32,
            hc_eps: f(t, "hc_eps").unwrap_or(1e-6) as f32,
            engram,
            bos: u_or(t, "bos_token_id", 120000) as u32,
            eos,
            pad: u_or(t, "pad_token_id", 120002) as u32,
            max_position_embeddings: u_or(t, "max_position_embeddings", 1 << 20),
            image_token_id: root.get("image_token_id").and_then(|x| x.as_u64()).map(|x| x as u32),
        })
    }

    pub fn dims(&self, layer: usize) -> &MlaDims {
        match self.attn[layer] {
            AttnKind::Sliding => &self.sliding,
            AttnKind::Indexed => &self.indexed,
        }
    }

    /// Группы индексатора, которые выбирает запрос сверх своей.
    pub fn index_groups(&self) -> usize {
        self.index_topk / self.index_kpool - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_released_config() {
        let Ok(p) = std::env::var("SYN_HY4_DIR") else { return };
        let c = Hy4Config::from_bytes(&std::fs::read(format!("{p}/config.json")).unwrap()).unwrap();
        assert_eq!(c.num_hidden_layers, 42);
        assert_eq!(c.attn.iter().filter(|a| **a == AttnKind::Indexed).count(), 12);
        assert_eq!(c.attn[0], AttnKind::Indexed);
        assert_eq!(c.attn[2], AttnKind::Sliding);
        assert_eq!(c.sliding.heads, 64);
        assert_eq!(c.indexed.heads, 32);
        assert_eq!(c.sliding.rope_theta, 1e4);
        assert_eq!(c.indexed.rope_theta, 1e7);
        let e = c.engram.clone().unwrap();
        assert_eq!(e.layers, vec![1, 11, 21, 31]);
        assert_eq!(e.heads(), 16);
        assert_eq!(e.head_dim, 96);
        assert_eq!(c.index_kpool, 4);
        assert_eq!(c.moe_hidden, 1536);
        assert_eq!(c.index_groups(), 511);
    }
}
