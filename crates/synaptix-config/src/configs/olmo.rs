use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OlmoConfig {
    #[serde(default = "OlmoConfig::default_d_model")]
    pub d_model: usize,
    #[serde(default = "OlmoConfig::default_n_heads")]
    pub n_heads: usize,
    #[serde(default)]
    pub n_kv_heads: Option<usize>,
    #[serde(default = "OlmoConfig::default_n_layers")]
    pub n_layers: usize,
    #[serde(default)]
    pub mlp_hidden_size: Option<usize>,
    #[serde(default = "OlmoConfig::default_max_sequence_length")]
    pub max_sequence_length: usize,
    #[serde(default = "OlmoConfig::default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default)]
    pub weight_tying: bool,
    #[serde(default)]
    pub rope: bool,
    #[serde(default = "OlmoConfig::default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "OlmoConfig::default_layer_norm_type")]
    pub layer_norm_type: String,
    #[serde(default = "OlmoConfig::default_norm_eps")]
    pub norm_eps: f64,
}

impl OlmoConfig {
    fn default_d_model() -> usize { 4096 }
    fn default_n_heads() -> usize { 32 }
    fn default_n_layers() -> usize { 32 }
    fn default_max_sequence_length() -> usize { 2048 }
    fn default_vocab_size() -> usize { 50280 }
    fn default_rope_theta() -> f64 { 10000.0 }
    fn default_layer_norm_type() -> String { "low_precision".into() }
    fn default_norm_eps() -> f64 { 1e-5 }
}

impl Default for OlmoConfig {
    fn default() -> Self {
        Self {
            d_model: Self::default_d_model(),
            n_heads: Self::default_n_heads(),
            n_kv_heads: None,
            n_layers: Self::default_n_layers(),
            mlp_hidden_size: None,
            max_sequence_length: Self::default_max_sequence_length(),
            vocab_size: Self::default_vocab_size(),
            weight_tying: false,
            rope: false,
            rope_theta: Self::default_rope_theta(),
            layer_norm_type: Self::default_layer_norm_type(),
            norm_eps: Self::default_norm_eps(),
        }
    }
}
