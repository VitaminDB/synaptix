use synaptix_core::dtype::DType;

pub struct AutocastConfig { pub enabled: bool, pub dtype: DType }

impl Default for AutocastConfig {
    fn default() -> Self { Self { enabled: true, dtype: DType::BF16 } }
}
