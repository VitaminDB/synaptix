use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen36Config {
    #[serde(default = "Qwen36Config::default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "Qwen36Config::default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "Qwen36Config::default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "Qwen36Config::default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "Qwen36Config::default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "Qwen36Config::default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "Qwen36Config::default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "Qwen36Config::default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "Qwen36Config::default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "Qwen36Config::default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default)]
    pub eos_token_id: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<crate::configs::llama::RopeScaling>,
    #[serde(default = "Qwen36Config::default_head_dim")]
    pub head_dim: usize,
    #[serde(default)]
    pub use_mla: bool,
    #[serde(default)]
    pub qk_nope_head_dim: Option<usize>,
    #[serde(default)]
    pub qk_rope_head_dim: Option<usize>,
    #[serde(default)]
    pub kv_lora_rank: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub chunk_size: Option<usize>,
}

impl Qwen36Config {
    fn default_vocab_size() -> usize { 151936 }
    fn default_hidden_size() -> usize { 2048 }
    fn default_intermediate_size() -> usize { 6144 }
    fn default_num_hidden_layers() -> usize { 28 }
    fn default_num_attention_heads() -> usize { 16 }
    fn default_num_key_value_heads() -> usize { 8 }
    fn default_max_position_embeddings() -> usize { 131072 }
    fn default_rms_norm_eps() -> f64 { 1e-6 }
    fn default_rope_theta() -> f64 { 500000.0 }
    fn default_hidden_act() -> String { "silu".into() }
    fn default_head_dim() -> usize { 128 }
}

impl Default for Qwen36Config {
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
            hidden_act: Self::default_hidden_act(),
            bos_token_id: 0,
            eos_token_id: 0,
            tie_word_embeddings: false,
            rope_scaling: None,
            head_dim: Self::default_head_dim(),
            use_mla: false,
            qk_nope_head_dim: None,
            qk_rope_head_dim: None,
            kv_lora_rank: None,
            use_sliding_window: false,
            sliding_window: None,
            chunk_size: None,
        }
    }
}
