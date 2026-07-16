use serde::{Deserialize, Serialize};

fn default_d_model() -> usize { 192 }
fn default_n_heads() -> usize { 8 }
fn default_n_layers() -> usize { 6 }
fn default_ffn_dim() -> usize { 768 }
fn default_max_speakers() -> usize { 4 }
fn default_max_seq_len() -> usize { 50 }
fn default_dropout() -> f64 { 0.1 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortformerConfig {
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_n_heads")]
    pub n_heads: usize,
    #[serde(default = "default_n_layers")]
    pub n_layers: usize,
    #[serde(default = "default_ffn_dim")]
    pub ffn_dim: usize,
    #[serde(default = "default_max_speakers")]
    pub max_speakers: usize,
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
}

impl Default for SortformerConfig {
    fn default() -> Self {
        Self {
            d_model: default_d_model(),
            n_heads: default_n_heads(),
            n_layers: default_n_layers(),
            ffn_dim: default_ffn_dim(),
            max_speakers: default_max_speakers(),
            max_seq_len: default_max_seq_len(),
            dropout: default_dropout(),
        }
    }
}
