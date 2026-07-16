use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default = "RuntimeConfig::default_device")]
    pub device: String,
    #[serde(default = "RuntimeConfig::default_dtype")]
    pub dtype: String,
    #[serde(default = "RuntimeConfig::default_max_batch_size")]
    pub max_batch_size: usize,
    #[serde(default = "RuntimeConfig::default_max_seq_len")]
    pub max_seq_len: usize,
    #[serde(default = "RuntimeConfig::default_tensor_parallel")]
    pub tensor_parallel: usize,
    #[serde(default = "RuntimeConfig::default_pipeline_parallel")]
    pub pipeline_parallel: usize,
    #[serde(default = "RuntimeConfig::default_flash_attention")]
    pub flash_attention: bool,
    #[serde(default)]
    pub graph_decode: bool,
    #[serde(default)]
    pub kv_quant: bool,
    #[serde(default)]
    pub seed: u64,
}

impl RuntimeConfig {
    fn default_device() -> String { "cpu".into() }
    fn default_dtype() -> String { "bf16".into() }
    fn default_max_batch_size() -> usize { 8 }
    fn default_max_seq_len() -> usize { 4096 }
    fn default_tensor_parallel() -> usize { 1 }
    fn default_pipeline_parallel() -> usize { 1 }
    fn default_flash_attention() -> bool { true }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            device: Self::default_device(),
            dtype: Self::default_dtype(),
            max_batch_size: Self::default_max_batch_size(),
            max_seq_len: Self::default_max_seq_len(),
            tensor_parallel: Self::default_tensor_parallel(),
            pipeline_parallel: Self::default_pipeline_parallel(),
            flash_attention: Self::default_flash_attention(),
            graph_decode: false,
            kv_quant: false,
            seed: 0,
        }
    }
}
