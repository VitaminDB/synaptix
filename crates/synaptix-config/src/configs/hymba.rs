use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HymbaConfig {
    #[serde(default = "HymbaConfig::default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "HymbaConfig::default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "HymbaConfig::default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "HymbaConfig::default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "HymbaConfig::default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "HymbaConfig::default_ssm_state_size")]
    pub ssm_state_size: usize,
    #[serde(default = "HymbaConfig::default_ssm_conv_size")]
    pub ssm_conv_size: usize,
    #[serde(default = "HymbaConfig::default_num_ssm_heads")]
    pub num_ssm_heads: usize,
    #[serde(default = "HymbaConfig::default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "HymbaConfig::default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "HymbaConfig::default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "HymbaConfig::default_rms_norm_eps")]
    pub rms_norm_eps: f64,
}

impl HymbaConfig {
    fn default_vocab_size() -> usize { 65536 }
    fn default_hidden_size() -> usize { 4096 }
    fn default_num_hidden_layers() -> usize { 32 }
    fn default_num_attention_heads() -> usize { 32 }
    fn default_num_key_value_heads() -> usize { 8 }
    fn default_ssm_state_size() -> usize { 8 }
    fn default_ssm_conv_size() -> usize { 4 }
    fn default_num_ssm_heads() -> usize { 4 }
    fn default_intermediate_size() -> usize { 11008 }
    fn default_head_dim() -> usize { 64 }
    fn default_max_position_embeddings() -> usize { 32768 }
    fn default_rms_norm_eps() -> f64 { 1e-5 }
}

impl Default for HymbaConfig {
    fn default() -> Self {
        Self {
            vocab_size: Self::default_vocab_size(),
            hidden_size: Self::default_hidden_size(),
            num_hidden_layers: Self::default_num_hidden_layers(),
            num_attention_heads: Self::default_num_attention_heads(),
            num_key_value_heads: Self::default_num_key_value_heads(),
            ssm_state_size: Self::default_ssm_state_size(),
            ssm_conv_size: Self::default_ssm_conv_size(),
            num_ssm_heads: Self::default_num_ssm_heads(),
            intermediate_size: Self::default_intermediate_size(),
            head_dim: Self::default_head_dim(),
            max_position_embeddings: Self::default_max_position_embeddings(),
            rms_norm_eps: Self::default_rms_norm_eps(),
        }
    }
}
