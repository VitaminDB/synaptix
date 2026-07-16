use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RwkvConfig {
    #[serde(default = "RwkvConfig::default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "RwkvConfig::default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default = "RwkvConfig::default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "RwkvConfig::default_rescale_every")]
    pub rescale_every: usize,
    #[serde(default = "RwkvConfig::default_use_cache")]
    pub use_cache: bool,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default)]
    pub eos_token_id: u32,
    #[serde(default)]
    pub time_mixing_hidden_size: Option<usize>,
    #[serde(default)]
    pub time_decay_extra_dim: Option<usize>,
    #[serde(default)]
    pub attention_hidden_size: Option<usize>,
}

impl RwkvConfig {
    fn default_vocab_size() -> usize { 50277 }
    fn default_hidden_size() -> usize { 4096 }
    fn default_num_hidden_layers() -> usize { 32 }
    fn default_rescale_every() -> usize { 6 }
    fn default_use_cache() -> bool { true }
}

impl Default for RwkvConfig {
    fn default() -> Self {
        Self {
            vocab_size: Self::default_vocab_size(),
            hidden_size: Self::default_hidden_size(),
            intermediate_size: None,
            num_hidden_layers: Self::default_num_hidden_layers(),
            rescale_every: Self::default_rescale_every(),
            use_cache: Self::default_use_cache(),
            bos_token_id: 0,
            eos_token_id: 0,
            time_mixing_hidden_size: None,
            time_decay_extra_dim: None,
            attention_hidden_size: None,
        }
    }
}
