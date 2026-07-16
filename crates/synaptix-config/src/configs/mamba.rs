use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MambaConfig {
    #[serde(default = "MambaConfig::default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "MambaConfig::default_d_model")]
    pub d_model: usize,
    #[serde(default = "MambaConfig::default_n_layer")]
    pub n_layer: usize,
    #[serde(default = "MambaConfig::default_expand")]
    pub expand: usize,
    #[serde(default = "MambaConfig::default_dt_rank")]
    pub dt_rank: String,
    #[serde(default = "MambaConfig::default_d_state")]
    pub d_state: usize,
    #[serde(default = "MambaConfig::default_d_conv")]
    pub d_conv: usize,
    #[serde(default = "MambaConfig::default_pad_vocab_size_multiple")]
    pub pad_vocab_size_multiple: usize,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default = "MambaConfig::default_use_conv_bias")]
    pub use_conv_bias: bool,
    #[serde(default = "MambaConfig::default_hidden_act")]
    pub hidden_act: String,
}

impl MambaConfig {
    fn default_vocab_size() -> usize { 50280 }
    fn default_d_model() -> usize { 2560 }
    fn default_n_layer() -> usize { 64 }
    fn default_expand() -> usize { 2 }
    fn default_dt_rank() -> String { "auto".into() }
    fn default_d_state() -> usize { 16 }
    fn default_d_conv() -> usize { 4 }
    fn default_pad_vocab_size_multiple() -> usize { 8 }
    fn default_use_conv_bias() -> bool { true }
    fn default_hidden_act() -> String { "silu".into() }
}

impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            vocab_size: Self::default_vocab_size(),
            d_model: Self::default_d_model(),
            n_layer: Self::default_n_layer(),
            expand: Self::default_expand(),
            dt_rank: Self::default_dt_rank(),
            d_state: Self::default_d_state(),
            d_conv: Self::default_d_conv(),
            pad_vocab_size_multiple: Self::default_pad_vocab_size_multiple(),
            use_bias: false,
            use_conv_bias: Self::default_use_conv_bias(),
            hidden_act: Self::default_hidden_act(),
        }
    }
}
