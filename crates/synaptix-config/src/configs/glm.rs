use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatGlmConfig {
    #[serde(default = "ChatGlmConfig::default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "ChatGlmConfig::default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "ChatGlmConfig::default_ffn_hidden_size")]
    pub ffn_hidden_size: usize,
    #[serde(default = "ChatGlmConfig::default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "ChatGlmConfig::default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "ChatGlmConfig::default_multi_query_attention")]
    pub multi_query_attention: bool,
    #[serde(default = "ChatGlmConfig::default_multi_query_group_num")]
    pub multi_query_group_num: usize,
    #[serde(default = "ChatGlmConfig::default_max_sequence_length")]
    pub max_sequence_length: usize,
    #[serde(default = "ChatGlmConfig::default_rmsnorm")]
    pub rmsnorm: bool,
    #[serde(default = "ChatGlmConfig::default_layernorm_epsilon")]
    pub layernorm_epsilon: f64,
    #[serde(default = "ChatGlmConfig::default_rope_ratio")]
    pub rope_ratio: f64,
    #[serde(default)]
    pub add_bias_linear: bool,
}

impl ChatGlmConfig {
    fn default_vocab_size() -> usize { 65024 }
    fn default_hidden_size() -> usize { 4096 }
    fn default_ffn_hidden_size() -> usize { 13696 }
    fn default_num_hidden_layers() -> usize { 28 }
    fn default_num_attention_heads() -> usize { 32 }
    fn default_multi_query_attention() -> bool { true }
    fn default_multi_query_group_num() -> usize { 2 }
    fn default_max_sequence_length() -> usize { 32768 }
    fn default_rmsnorm() -> bool { true }
    fn default_layernorm_epsilon() -> f64 { 1e-5 }
    fn default_rope_ratio() -> f64 { 1.0 }
}

impl Default for ChatGlmConfig {
    fn default() -> Self {
        Self {
            vocab_size: Self::default_vocab_size(),
            hidden_size: Self::default_hidden_size(),
            ffn_hidden_size: Self::default_ffn_hidden_size(),
            num_hidden_layers: Self::default_num_hidden_layers(),
            num_attention_heads: Self::default_num_attention_heads(),
            multi_query_attention: Self::default_multi_query_attention(),
            multi_query_group_num: Self::default_multi_query_group_num(),
            max_sequence_length: Self::default_max_sequence_length(),
            rmsnorm: Self::default_rmsnorm(),
            layernorm_epsilon: Self::default_layernorm_epsilon(),
            rope_ratio: Self::default_rope_ratio(),
            add_bias_linear: false,
        }
    }
}
