use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
    pub layer_norm_eps: f32,
    pub deepstack_visual_indexes: Vec<usize>,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            depth: 27,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_heads: 16,
            in_channels: 3,
            patch_size: 16,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            out_hidden_size: 5120,
            num_position_embeddings: 2304,
            layer_norm_eps: 1e-6,
            deepstack_visual_indexes: Vec::new(),
        }
    }
}

impl VisionConfig {
    pub fn from_hf_bytes(bytes: &[u8]) -> Result<Self, ConfigError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| ConfigError::Parse(format!("config json: {e}")))?;
        let vc = root
            .get("vision_config")
            .cloned()
            .ok_or_else(|| ConfigError::Missing("vision_config".into()))?;
        let cfg: Self = serde_json::from_value(vc)
            .map_err(|e| ConfigError::Parse(format!("vision_config: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.hidden_size == 0 || self.num_heads == 0 {
            return Err(ConfigError::Invalid("hidden_size/num_heads == 0".into()));
        }
        if self.hidden_size % self.num_heads != 0 {
            return Err(ConfigError::Invalid(format!(
                "hidden_size {} не делится на num_heads {}",
                self.hidden_size, self.num_heads
            )));
        }
        if self.spatial_merge_size == 0 || self.patch_size == 0 {
            return Err(ConfigError::Invalid("patch/merge size == 0".into()));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn patch_features(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    pub fn merge_unit(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size
    }

    pub fn merged_dim(&self) -> usize {
        self.hidden_size * self.merge_unit()
    }

    pub fn size_factor(&self) -> usize {
        self.patch_size * self.spatial_merge_size
    }

    pub fn pos_grid(&self) -> usize {
        (self.num_position_embeddings as f64).sqrt().round() as usize
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("vision config parse: {0}")]
    Parse(String),
    #[error("vision config: нет секции `{0}`")]
    Missing(String),
    #[error("vision config invalid: {0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "vision_config": {
            "depth": 27, "hidden_size": 1152, "intermediate_size": 4304,
            "num_heads": 16, "in_channels": 3, "patch_size": 16,
            "spatial_merge_size": 2, "temporal_patch_size": 2,
            "out_hidden_size": 5120, "num_position_embeddings": 2304,
            "deepstack_visual_indexes": []
        }
    }"#;

    #[test]
    fn parses_and_derives() {
        let c = VisionConfig::from_hf_bytes(SAMPLE.as_bytes()).unwrap();
        assert_eq!(c.head_dim(), 72);
        assert_eq!(c.patch_features(), 1536);
        assert_eq!(c.merged_dim(), 4608);
        assert_eq!(c.size_factor(), 32);
        assert_eq!(c.pos_grid(), 48);
    }

    #[test]
    fn rejects_non_divisible_heads() {
        let bad = SAMPLE.replace("\"num_heads\": 16", "\"num_heads\": 5");
        assert!(VisionConfig::from_hf_bytes(bad.as_bytes()).is_err());
    }

    #[test]
    fn missing_section_errors() {
        assert!(VisionConfig::from_hf_bytes(b"{}").is_err());
    }
}
