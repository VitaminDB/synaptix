use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: f64,
    pub low_freq_factor: Option<f64>,
    pub high_freq_factor: Option<f64>,
    pub original_max_position_embeddings: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    #[serde(default = "LlamaConfig::default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "LlamaConfig::default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "LlamaConfig::default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "LlamaConfig::default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "LlamaConfig::default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "LlamaConfig::default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "LlamaConfig::default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "LlamaConfig::default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "LlamaConfig::default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "LlamaConfig::default_bos_token_id")]
    pub bos_token_id: u32,
    #[serde(default = "LlamaConfig::default_eos_token_id")]
    pub eos_token_id: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "LlamaConfig::default_hidden_act")]
    pub hidden_act: String,
}

impl LlamaConfig {
    fn default_vocab_size() -> usize { 32000 }
    fn default_hidden_size() -> usize { 4096 }
    fn default_intermediate_size() -> usize { 11008 }
    fn default_num_hidden_layers() -> usize { 32 }
    fn default_num_attention_heads() -> usize { 32 }
    fn default_num_key_value_heads() -> usize { 32 }
    fn default_max_position_embeddings() -> usize { 4096 }
    fn default_rms_norm_eps() -> f64 { 1e-5 }
    fn default_rope_theta() -> f64 { 10000.0 }
    fn default_bos_token_id() -> u32 { 1 }
    fn default_eos_token_id() -> u32 { 2 }
    fn default_hidden_act() -> String { "silu".into() }
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            vocab_size: Self::default_vocab_size(),
            hidden_size: Self::default_hidden_size(),
            intermediate_size: Self::default_intermediate_size(),
            num_hidden_layers: Self::default_num_hidden_layers(),
            num_attention_heads: Self::default_num_attention_heads(),
            num_key_value_heads: Self::default_num_key_value_heads(),
            max_position_embeddings: Self::default_max_position_embeddings(),
            rms_norm_eps: Self::default_rms_norm_eps(),
            rope_theta: Self::default_rope_theta(),
            rope_scaling: None,
            bos_token_id: Self::default_bos_token_id(),
            eos_token_id: Self::default_eos_token_id(),
            tie_word_embeddings: false,
            hidden_act: Self::default_hidden_act(),
        }
    }
}
