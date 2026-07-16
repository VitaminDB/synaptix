use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JambaConfig {
    #[serde(default = "JambaConfig::default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "JambaConfig::default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "JambaConfig::default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "JambaConfig::default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "JambaConfig::default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "JambaConfig::default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "JambaConfig::default_attn_layer_offset")]
    pub attn_layer_offset: usize,
    #[serde(default = "JambaConfig::default_attn_layer_period")]
    pub attn_layer_period: usize,
    #[serde(default = "JambaConfig::default_expert_layer_offset")]
    pub expert_layer_offset: usize,
    #[serde(default = "JambaConfig::default_expert_layer_period")]
    pub expert_layer_period: usize,
    #[serde(default = "JambaConfig::default_num_experts")]
    pub num_experts: usize,
    #[serde(default = "JambaConfig::default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "JambaConfig::default_mamba_d_state")]
    pub mamba_d_state: usize,
    #[serde(default = "JambaConfig::default_mamba_d_conv")]
    pub mamba_d_conv: usize,
    #[serde(default = "JambaConfig::default_mamba_expand")]
    pub mamba_expand: usize,
    #[serde(default = "JambaConfig::default_mamba_dt_rank")]
    pub mamba_dt_rank: String,
    #[serde(default = "JambaConfig::default_rms_norm_eps")]
    pub rms_norm_eps: f64,
}

impl JambaConfig {
    fn default_vocab_size() -> usize { 65536 }
    fn default_hidden_size() -> usize { 4096 }
    fn default_intermediate_size() -> usize { 14336 }
    fn default_num_hidden_layers() -> usize { 32 }
    fn default_num_attention_heads() -> usize { 32 }
    fn default_num_key_value_heads() -> usize { 8 }
    fn default_attn_layer_offset() -> usize { 4 }
    fn default_attn_layer_period() -> usize { 8 }
    fn default_expert_layer_offset() -> usize { 1 }
    fn default_expert_layer_period() -> usize { 2 }
    fn default_num_experts() -> usize { 16 }
    fn default_num_experts_per_tok() -> usize { 2 }
    fn default_mamba_d_state() -> usize { 16 }
    fn default_mamba_d_conv() -> usize { 4 }
    fn default_mamba_expand() -> usize { 2 }
    fn default_mamba_dt_rank() -> String { "auto".into() }
    fn default_rms_norm_eps() -> f64 { 1e-5 }
}

impl Default for JambaConfig {
    fn default() -> Self {
        Self {
            vocab_size: Self::default_vocab_size(),
            hidden_size: Self::default_hidden_size(),
            intermediate_size: Self::default_intermediate_size(),
            num_hidden_layers: Self::default_num_hidden_layers(),
            num_attention_heads: Self::default_num_attention_heads(),
            num_key_value_heads: Self::default_num_key_value_heads(),
            attn_layer_offset: Self::default_attn_layer_offset(),
            attn_layer_period: Self::default_attn_layer_period(),
            expert_layer_offset: Self::default_expert_layer_offset(),
            expert_layer_period: Self::default_expert_layer_period(),
            num_experts: Self::default_num_experts(),
            num_experts_per_tok: Self::default_num_experts_per_tok(),
            mamba_d_state: Self::default_mamba_d_state(),
            mamba_d_conv: Self::default_mamba_d_conv(),
            mamba_expand: Self::default_mamba_expand(),
            mamba_dt_rank: Self::default_mamba_dt_rank(),
            rms_norm_eps: Self::default_rms_norm_eps(),
        }
    }
}
