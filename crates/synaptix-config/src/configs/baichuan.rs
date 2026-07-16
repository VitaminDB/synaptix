use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaichuanConfig {
    #[serde(default = "BaichuanConfig::default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "BaichuanConfig::default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "BaichuanConfig::default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "BaichuanConfig::default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "BaichuanConfig::default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "BaichuanConfig::default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "BaichuanConfig::default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "BaichuanConfig::default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "BaichuanConfig::default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default)]
    pub eos_token_id: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "BaichuanConfig::default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub rope_scaling: Option<crate::configs::llama::RopeScaling>,
    #[serde(default = "BaichuanConfig::default_model_max_length")]
    pub model_max_length: usize,
    #[serde(default = "BaichuanConfig::default_user_token_id")]
    pub user_token_id: u32,
    #[serde(default = "BaichuanConfig::default_assistant_token_id")]
    pub assistant_token_id: u32,
}

impl BaichuanConfig {
    fn default_vocab_size() -> usize { 125696 }
    fn default_hidden_size() -> usize { 4096 }
    fn default_intermediate_size() -> usize { 11008 }
    fn default_num_hidden_layers() -> usize { 32 }
    fn default_num_attention_heads() -> usize { 32 }
    fn default_num_key_value_heads() -> usize { 32 }
    fn default_max_position_embeddings() -> usize { 4096 }
    fn default_rms_norm_eps() -> f64 { 1e-6 }
    fn default_rope_theta() -> f64 { 10000.0 }
    fn default_hidden_act() -> String { "silu".into() }
    fn default_model_max_length() -> usize { 4096 }
    fn default_user_token_id() -> u32 { 195 }
    fn default_assistant_token_id() -> u32 { 196 }
}

impl Default for BaichuanConfig {
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
            bos_token_id: 0,
            eos_token_id: 0,
            tie_word_embeddings: false,
            hidden_act: Self::default_hidden_act(),
            rope_scaling: None,
            model_max_length: Self::default_model_max_length(),
            user_token_id: Self::default_user_token_id(),
            assistant_token_id: Self::default_assistant_token_id(),
        }
    }
}
