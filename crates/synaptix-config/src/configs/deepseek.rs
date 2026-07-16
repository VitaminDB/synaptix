use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepseekConfig {
    #[serde(default = "DeepseekConfig::default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "DeepseekConfig::default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "DeepseekConfig::default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "DeepseekConfig::default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "DeepseekConfig::default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "DeepseekConfig::default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "DeepseekConfig::default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "DeepseekConfig::default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "DeepseekConfig::default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default)]
    pub eos_token_id: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "DeepseekConfig::default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub rope_scaling: Option<crate::configs::llama::RopeScaling>,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub first_k_dense_replace: Option<usize>,
    #[serde(default = "DeepseekConfig::default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    #[serde(default)]
    pub kv_lora_rank: Option<usize>,
    #[serde(default)]
    pub qk_nope_head_dim: Option<usize>,
    #[serde(default)]
    pub qk_rope_head_dim: Option<usize>,
    #[serde(default)]
    pub v_head_dim: Option<usize>,
}

impl DeepseekConfig {
    fn default_vocab_size() -> usize { 102400 }
    fn default_hidden_size() -> usize { 5120 }
    fn default_intermediate_size() -> usize { 12288 }
    fn default_num_hidden_layers() -> usize { 60 }
    fn default_num_attention_heads() -> usize { 128 }
    fn default_num_key_value_heads() -> usize { 128 }
    fn default_max_position_embeddings() -> usize { 4096 }
    fn default_rms_norm_eps() -> f64 { 1e-5 }
    fn default_rope_theta() -> f64 { 10000.0 }
    fn default_hidden_act() -> String { "silu".into() }
    fn default_norm_topk_prob() -> bool { true }
}

impl Default for DeepseekConfig {
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
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            first_k_dense_replace: None,
            norm_topk_prob: Self::default_norm_topk_prob(),
            q_lora_rank: None,
            kv_lora_rank: None,
            qk_nope_head_dim: None,
            qk_rope_head_dim: None,
            v_head_dim: None,
        }
    }
}
