//! Конфиг BGE-M3 (XLM-RoBERTa backbone). Парсится из `config.json` HF-снапшота
//! или из `.syn`-бандла (`config.json` file-чанк).

use serde::Deserialize;

use synaptix_bundle::Bundle;

use crate::BgeError;

#[derive(Debug, Clone, Deserialize)]
pub struct BgeConfig {
    #[serde(default = "d_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_inter")]
    pub intermediate_size: usize,
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "d_type_vocab")]
    pub type_vocab_size: usize,
    #[serde(default = "d_ln_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "d_pad")]
    pub pad_token_id: i64,
    #[serde(default = "d_act")]
    pub hidden_act: String,
}

fn d_hidden() -> usize { 1024 }
fn d_layers() -> usize { 24 }
fn d_heads() -> usize { 16 }
fn d_inter() -> usize { 4096 }
fn d_vocab() -> usize { 250002 }
fn d_max_pos() -> usize { 8194 }
fn d_type_vocab() -> usize { 1 }
fn d_ln_eps() -> f64 { 1e-5 }
fn d_pad() -> i64 { 1 }
fn d_act() -> String { "gelu".to_string() }

impl Default for BgeConfig {
    fn default() -> Self {
        Self {
            hidden_size: d_hidden(),
            num_hidden_layers: d_layers(),
            num_attention_heads: d_heads(),
            intermediate_size: d_inter(),
            vocab_size: d_vocab(),
            max_position_embeddings: d_max_pos(),
            type_vocab_size: d_type_vocab(),
            layer_norm_eps: d_ln_eps(),
            pad_token_id: d_pad(),
            hidden_act: d_act(),
        }
    }
}

impl BgeConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, BgeError> {
        serde_json::from_slice(bytes).map_err(|e| BgeError::Config(e.to_string()))
    }

    pub fn from_bundle(bundle: &Bundle) -> Result<Self, BgeError> {
        let bytes = bundle
            .read_file("config.json")
            .map_err(|e| BgeError::Bundle(e.to_string()))?;
        Self::from_json_bytes(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNPACK: &str = "tmp/bge_unpack";

    #[test]
    fn parse_real_bge_config() {
        let p = format!("{UNPACK}/config.json");
        let Ok(bytes) = std::fs::read(&p) else {
            return;
        };
        let cfg = BgeConfig::from_json_bytes(&bytes).expect("parse config.json");
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.intermediate_size, 4096);
        assert_eq!(cfg.vocab_size, 250002);
        assert_eq!(cfg.max_position_embeddings, 8194);
        assert_eq!(cfg.type_vocab_size, 1);
        assert_eq!(cfg.layer_norm_eps, 1e-5);
        assert_eq!(cfg.pad_token_id, 1);
        assert_eq!(cfg.hidden_act, "gelu");
    }
}
