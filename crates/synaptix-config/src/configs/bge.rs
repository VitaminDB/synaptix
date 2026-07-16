use serde::{Deserialize, Serialize};

fn default_vocab_size() -> usize { 30522 }
fn default_hidden_size() -> usize { 768 }
fn default_num_hidden_layers() -> usize { 12 }
fn default_num_attention_heads() -> usize { 12 }
fn default_intermediate_size() -> usize { 3072 }
fn default_max_position_embeddings() -> usize { 512 }
fn default_layer_norm_eps() -> f64 { 1e-12 }
fn default_hidden_dropout_prob() -> f64 { 0.1 }
fn default_attention_probs_dropout_prob() -> f64 { 0.1 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BgeConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_hidden_dropout_prob")]
    pub hidden_dropout_prob: f64,
    #[serde(default = "default_attention_probs_dropout_prob")]
    pub attention_probs_dropout_prob: f64,
}

impl Default for BgeConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            hidden_size: default_hidden_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            intermediate_size: default_intermediate_size(),
            max_position_embeddings: default_max_position_embeddings(),
            layer_norm_eps: default_layer_norm_eps(),
            hidden_dropout_prob: default_hidden_dropout_prob(),
            attention_probs_dropout_prob: default_attention_probs_dropout_prob(),
        }
    }
}
