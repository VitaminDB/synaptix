use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalconConfig {
    #[serde(default = "FalconConfig::default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "FalconConfig::default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "FalconConfig::default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "FalconConfig::default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub bias: bool,
    #[serde(default = "FalconConfig::default_parallel_attn")]
    pub parallel_attn: bool,
    #[serde(default)]
    pub alibi: bool,
    #[serde(default)]
    pub new_decoder_architecture: bool,
    #[serde(default = "FalconConfig::default_multi_query")]
    pub multi_query: bool,
    #[serde(default)]
    pub num_kv_heads: Option<usize>,
    #[serde(default = "FalconConfig::default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "FalconConfig::default_rope_theta")]
    pub rope_theta: f64,
}

impl FalconConfig {
    fn default_vocab_size() -> usize { 65024 }
    fn default_hidden_size() -> usize { 4544 }
    fn default_num_hidden_layers() -> usize { 32 }
    fn default_num_attention_heads() -> usize { 71 }
    fn default_parallel_attn() -> bool { true }
    fn default_multi_query() -> bool { true }
    fn default_max_position_embeddings() -> usize { 2048 }
    fn default_rope_theta() -> f64 { 10000.0 }
}

impl Default for FalconConfig {
    fn default() -> Self {
        Self {
            vocab_size: Self::default_vocab_size(),
            hidden_size: Self::default_hidden_size(),
            num_hidden_layers: Self::default_num_hidden_layers(),
            num_attention_heads: Self::default_num_attention_heads(),
            bias: false,
            parallel_attn: Self::default_parallel_attn(),
            alibi: false,
            new_decoder_architecture: false,
            multi_query: Self::default_multi_query(),
            num_kv_heads: None,
            max_position_embeddings: Self::default_max_position_embeddings(),
            rope_theta: Self::default_rope_theta(),
        }
    }
}
