//! Конфиг башни зрения и объединителя MiniCPM-V 4.7 из `config.json`.
//!
//! Читаются секция `vision_config` (SigLIP: 27 слоёв × 1152, патч 14,
//! сетка позиций 70×70), поля корня (`insert_layer_id`, `downsample_mode`,
//! `merge_kernel_size`, `merger_times`, id спецтокенов холста) и размер
//! скрытого слоя LLM из `text_config` — в него проецирует объединитель.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct VisionRaw {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_channels: usize,
    image_size: usize,
    patch_size: usize,
    layer_norm_eps: f32,
    hidden_act: String,
    window_kernel_size: Option<(usize, usize)>,
}

impl Default for VisionRaw {
    fn default() -> Self {
        Self {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            num_channels: 3,
            image_size: 980,
            patch_size: 14,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu_pytorch_tanh".into(),
            window_kernel_size: None,
        }
    }
}

/// Конфиг башни зрения + объединителя.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniCpmConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub layer_norm_eps: f32,
    pub hidden_act: String,
    /// После слоя с этим индексом стоит оконный объединитель (`vit_merger`).
    pub insert_layer_id: usize,
    /// `"16x"` — оконный объединитель 2×2 + объединитель 2×2 (один токен на
    /// 16 патчей); `"4x"` — только объединитель 2×2.
    pub downsample_mode: String,
    pub merge_kernel_size: (usize, usize),
    pub merger_times: usize,
    pub window_kernel_size: (usize, usize),
    /// `text_config.hidden_size` — размерность эмбеддингов LLM.
    pub llm_hidden_size: usize,
    pub image_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
    pub image_start_id: Option<u32>,
    pub image_end_id: Option<u32>,
    pub slice_start_id: Option<u32>,
    pub slice_end_id: Option<u32>,
    pub newline_id: Option<u32>,
}

impl MiniCpmConfig {
    pub fn from_hf_bytes(bytes: &[u8]) -> Result<Self, ConfigError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| ConfigError::Parse(format!("config json: {e}")))?;
        let vc = root
            .get("vision_config")
            .cloned()
            .ok_or_else(|| ConfigError::Missing("vision_config".into()))?;
        let raw: VisionRaw = serde_json::from_value(vc)
            .map_err(|e| ConfigError::Parse(format!("vision_config: {e}")))?;
        let u = |v: Option<&serde_json::Value>| v.and_then(|x| x.as_u64()).map(|x| x as usize);
        let id = |k: &str| root.get(k).and_then(|v| v.as_u64()).map(|v| v as u32);
        let pair = |v: Option<&serde_json::Value>| -> Option<(usize, usize)> {
            let a = v?.as_array()?;
            Some((a.first()?.as_u64()? as usize, a.get(1)?.as_u64()? as usize))
        };
        let llm_hidden_size = root
            .get("text_config")
            .and_then(|t| t.get("hidden_size"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| ConfigError::Missing("text_config.hidden_size".into()))?;
        let cfg = Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_channels: raw.num_channels,
            image_size: raw.image_size,
            patch_size: raw.patch_size,
            layer_norm_eps: raw.layer_norm_eps,
            hidden_act: raw.hidden_act,
            insert_layer_id: u(root.get("insert_layer_id")).unwrap_or(6),
            downsample_mode: root
                .get("downsample_mode")
                .and_then(|v| v.as_str())
                .unwrap_or("16x")
                .to_string(),
            merge_kernel_size: pair(root.get("merge_kernel_size")).unwrap_or((2, 2)),
            merger_times: u(root.get("merger_times")).unwrap_or(1),
            window_kernel_size: raw.window_kernel_size.unwrap_or((2, 2)),
            llm_hidden_size,
            image_token_id: id("image_token_id"),
            video_token_id: id("video_token_id"),
            image_start_id: id("image_start_id"),
            image_end_id: id("image_end_id"),
            slice_start_id: id("slice_start_id"),
            slice_end_id: id("slice_end_id"),
            newline_id: id("newline_id"),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.hidden_size == 0 || self.num_attention_heads == 0 {
            return Err(ConfigError::Invalid("hidden_size/num_attention_heads == 0".into()));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(ConfigError::Invalid(format!(
                "hidden_size {} не делится на num_attention_heads {}",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.patch_size == 0 || self.image_size % self.patch_size != 0 {
            return Err(ConfigError::Invalid(format!(
                "image_size {} не кратен patch_size {}",
                self.image_size, self.patch_size
            )));
        }
        if self.downsample_mode != "16x" && self.downsample_mode != "4x" {
            return Err(ConfigError::Invalid(format!("downsample_mode: {}", self.downsample_mode)));
        }
        if self.merge_kernel_size != (2, 2) || self.window_kernel_size != (2, 2) {
            return Err(ConfigError::Invalid(
                "поддержаны только merge_kernel_size и window_kernel_size 2×2".into(),
            ));
        }
        if self.merger_times != 1 {
            return Err(ConfigError::Invalid(format!("merger_times {} (поддержан 1)", self.merger_times)));
        }
        if self.hidden_act != "gelu_pytorch_tanh" {
            return Err(ConfigError::Invalid(format!("hidden_act: {}", self.hidden_act)));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Размер патча как вектора признаков: `C·p·p` (588).
    pub fn patch_features(&self) -> usize {
        self.num_channels * self.patch_size * self.patch_size
    }

    /// Сторона таблицы позиционных эмбеддингов (70).
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Есть ли оконный объединитель внутри энкодера (режим `16x`).
    pub fn use_vit_merger(&self) -> bool {
        self.downsample_mode != "4x"
    }

    /// Сколько патчей по стороне даёт один токен LLM: 4 у `16x`, 2 у `4x`.
    pub fn merge_factor(&self) -> usize {
        if self.use_vit_merger() {
            self.window_kernel_size.0 * self.merge_kernel_size.0
        } else {
            self.merge_kernel_size.0
        }
    }

    /// Токенов LLM на кроп `h×w` патчей.
    pub fn tokens_for(&self, h: usize, w: usize) -> usize {
        let f = self.merge_factor();
        (h * w) / (f * f)
    }

    /// Вход MLP оконного объединителя: `hidden · 2 · 2` (4608).
    pub fn window_hidden(&self) -> usize {
        self.hidden_size * self.window_kernel_size.0 * self.window_kernel_size.1
    }

    /// Промежуточный слой оконного объединителя: `intermediate · 2 · 2` (17216).
    pub fn window_intermediate(&self) -> usize {
        self.intermediate_size * self.window_kernel_size.0 * self.window_kernel_size.1
    }

    /// Вход объединителя 2×2: `hidden · 2 · 2` (4608).
    pub fn merged_dim(&self) -> usize {
        self.hidden_size * self.merge_kernel_size.0 * self.merge_kernel_size.1
    }

    /// Сторона патча в пикселях, на которую должна делиться картинка после
    /// ресайза: `patch · merge_factor` (56).
    pub fn size_factor(&self) -> usize {
        self.patch_size * self.merge_factor()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("vision config parse: {0}")]
    Parse(String),
    #[error("vision config: нет `{0}`")]
    Missing(String),
    #[error("vision config invalid: {0}")]
    Invalid(String),
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn sample() -> &'static str {
        r#"{
            "model_type": "minicpmv4_7",
            "text_config": {"hidden_size": 1024, "model_type": "qwen3_5_text"},
            "insert_layer_id": 6, "downsample_mode": "16x", "mrope_mode": "canvas",
            "vision_config": {"hidden_size": 1152, "image_size": 980, "intermediate_size": 4304,
                "layer_norm_eps": 1e-06, "model_type": "minicpmv4_7_vision",
                "num_attention_heads": 16, "num_channels": 3, "num_hidden_layers": 27,
                "patch_size": 14, "hidden_act": "gelu_pytorch_tanh"},
            "image_token_id": 248056, "video_token_id": 248057,
            "image_start_id": 248078, "image_end_id": 248079,
            "slice_start_id": 248088, "slice_end_id": 248089, "newline_id": 198
        }"#
    }

    #[test]
    fn parses_and_derives() {
        let c = MiniCpmConfig::from_hf_bytes(sample().as_bytes()).unwrap();
        assert_eq!(c.head_dim(), 72);
        assert_eq!(c.patch_features(), 588);
        assert_eq!(c.num_patches_per_side(), 70);
        assert_eq!(c.merge_factor(), 4);
        assert_eq!(c.tokens_for(8, 12), 6);
        assert_eq!(c.window_hidden(), 4608);
        assert_eq!(c.window_intermediate(), 17216);
        assert_eq!(c.merged_dim(), 4608);
        assert_eq!(c.size_factor(), 56);
        assert_eq!(c.llm_hidden_size, 1024);
        assert_eq!(c.image_token_id, Some(248056));
        assert_eq!(c.newline_id, Some(198));
        assert_eq!(c.window_kernel_size, (2, 2));
    }

    #[test]
    fn real_config_of_1b_if_present() {
        let p = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("Storage/Нейросети/Модели/_hf/MiniCPM-V-4.7-1B/config.json");
        let Ok(bytes) = std::fs::read(&p) else { return };
        let c = MiniCpmConfig::from_hf_bytes(&bytes).unwrap();
        assert_eq!(c.num_hidden_layers, 27);
        assert_eq!(c.insert_layer_id, 6);
        assert_eq!(c.llm_hidden_size, 1024);
    }

    #[test]
    fn four_x_mode() {
        let s = sample().replace("\"16x\"", "\"4x\"");
        let c = MiniCpmConfig::from_hf_bytes(s.as_bytes()).unwrap();
        assert!(!c.use_vit_merger());
        assert_eq!(c.merge_factor(), 2);
        assert_eq!(c.size_factor(), 28);
    }

    #[test]
    fn missing_sections() {
        assert!(MiniCpmConfig::from_hf_bytes(b"{}").is_err());
        let s = sample().replace("\"text_config\": {\"hidden_size\": 1024, \"model_type\": \"qwen3_5_text\"},", "");
        assert!(MiniCpmConfig::from_hf_bytes(s.as_bytes()).is_err());
    }
}
