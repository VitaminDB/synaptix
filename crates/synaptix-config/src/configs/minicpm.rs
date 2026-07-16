use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniCpmConfig {
    #[serde(default = "MiniCpmConfig::default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "MiniCpmConfig::default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "MiniCpmConfig::default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "MiniCpmConfig::default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "MiniCpmConfig::default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "MiniCpmConfig::default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "MiniCpmConfig::default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "MiniCpmConfig::default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "MiniCpmConfig::default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default)]
    pub eos_token_id: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "MiniCpmConfig::default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub rope_scaling: Option<crate::configs::llama::RopeScaling>,
    #[serde(default = "MiniCpmConfig::default_scale_depth")]
    pub scale_depth: f64,
    #[serde(default = "MiniCpmConfig::default_scale_emb")]
    pub scale_emb: f64,
    #[serde(default = "MiniCpmConfig::default_dim_model_base")]
    pub dim_model_base: usize,
}

impl MiniCpmConfig {
    fn default_vocab_size() -> usize { 122753 }
    fn default_hidden_size() -> usize { 2304 }
    fn default_intermediate_size() -> usize { 5760 }
    fn default_num_hidden_layers() -> usize { 40 }
    fn default_num_attention_heads() -> usize { 36 }
    fn default_num_key_value_heads() -> usize { 36 }
    fn default_max_position_embeddings() -> usize { 4096 }
    fn default_rms_norm_eps() -> f64 { 1e-5 }
    fn default_rope_theta() -> f64 { 10000.0 }
    fn default_hidden_act() -> String { "silu".into() }
    fn default_scale_depth() -> f64 { 1.4 }
    fn default_scale_emb() -> f64 { 12.0 }
    fn default_dim_model_base() -> usize { 256 }
}

impl Default for MiniCpmConfig {
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
            scale_depth: Self::default_scale_depth(),
            scale_emb: Self::default_scale_emb(),
            dim_model_base: Self::default_dim_model_base(),
        }
    }
}
