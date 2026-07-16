use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandRConfig {
    #[serde(default = "CommandRConfig::default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "CommandRConfig::default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "CommandRConfig::default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "CommandRConfig::default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "CommandRConfig::default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "CommandRConfig::default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "CommandRConfig::default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "CommandRConfig::default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "CommandRConfig::default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default)]
    pub eos_token_id: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "CommandRConfig::default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub rope_scaling: Option<crate::configs::llama::RopeScaling>,
    #[serde(default = "CommandRConfig::default_logit_scale")]
    pub logit_scale: f64,
    #[serde(default)]
    pub use_qk_norm: bool,
}

impl CommandRConfig {
    fn default_vocab_size() -> usize { 256000 }
    fn default_hidden_size() -> usize { 8192 }
    fn default_intermediate_size() -> usize { 32768 }
    fn default_num_hidden_layers() -> usize { 40 }
    fn default_num_attention_heads() -> usize { 64 }
    fn default_num_key_value_heads() -> usize { 8 }
    fn default_max_position_embeddings() -> usize { 131072 }
    fn default_rms_norm_eps() -> f64 { 1e-5 }
    fn default_rope_theta() -> f64 { 8000000.0 }
    fn default_hidden_act() -> String { "silu".into() }
    fn default_logit_scale() -> f64 { 0.0625 }
}

impl Default for CommandRConfig {
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
            logit_scale: Self::default_logit_scale(),
            use_qk_norm: false,
        }
    }
}
