//! Конфиги Qwen-Image из раскладки diffusers: трансформер, VAE, энкодер
//! Qwen2.5-VL (текст + башня зрения), процессор картинок, расписание и
//! вариант пайплайна (`model_index.json`).

use crate::QwenImageError;

fn parse(bytes: &[u8], what: &str) -> Result<serde_json::Value, QwenImageError> {
    serde_json::from_slice(bytes).map_err(|e| QwenImageError::Config(format!("{what}: {e}")))
}

fn num(v: &serde_json::Value, k: &str, def: usize) -> usize {
    v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(def)
}

fn flt(v: &serde_json::Value, k: &str, def: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(def)
}

fn flag(v: &serde_json::Value, k: &str) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(false)
}

fn usizes(v: &serde_json::Value, k: &str) -> Option<Vec<usize>> {
    v.get(k)
        .and_then(|a| a.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
}

fn floats(v: &serde_json::Value, k: &str) -> Option<Vec<f32>> {
    v.get(k)
        .and_then(|a| a.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect())
}

/// `transformer/config.json` (`QwenImageTransformer2DModel`).
#[derive(Debug, Clone, PartialEq)]
pub struct QwenImageConfig {
    /// Double-stream блоков (60 у всех выпусков).
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// Каналы упакованного латента: 16 каналов VAE × 2×2.
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    /// Ширина скрытых состояний энкодера (3584 у Qwen2.5-VL 7B).
    pub joint_attention_dim: usize,
    /// Оси RoPE (кадр, высота, ширина): 16 + 56 + 56 = голова.
    pub axes_dims: Vec<usize>,
    /// 2511: токены референсов модулируются временем 0, генерируемая
    /// картинка — текущим (`modulate_index` у diffusers).
    pub zero_cond_t: bool,
}

impl QwenImageConfig {
    pub fn from_json(bytes: &[u8]) -> Result<Self, QwenImageError> {
        let v = parse(bytes, "transformer/config.json")?;
        let cfg = Self {
            num_layers: num(&v, "num_layers", 60),
            num_heads: num(&v, "num_attention_heads", 24),
            head_dim: num(&v, "attention_head_dim", 128),
            in_channels: num(&v, "in_channels", 64),
            out_channels: num(&v, "out_channels", 16),
            patch_size: num(&v, "patch_size", 2),
            joint_attention_dim: num(&v, "joint_attention_dim", 3584),
            axes_dims: usizes(&v, "axes_dims_rope").unwrap_or_else(|| vec![16, 56, 56]),
            zero_cond_t: flag(&v, "zero_cond_t"),
        };
        if cfg.axes_dims.len() != 3 || cfg.axes_dims.iter().sum::<usize>() != cfg.head_dim {
            return Err(QwenImageError::Config(format!(
                "Qwen-Image: оси RoPE {:?} не складываются в голову {}",
                cfg.axes_dims, cfg.head_dim
            )));
        }
        if flag(&v, "guidance_embeds") {
            return Err(QwenImageError::Config("Qwen-Image: guidance_embeds не поддерживается".into()));
        }
        if flag(&v, "use_additional_t_cond") || flag(&v, "use_layer3d_rope") {
            return Err(QwenImageError::Config(
                "Qwen-Image-Layered (use_additional_t_cond / use_layer3d_rope) не поддерживается".into(),
            ));
        }
        if cfg.patch_size != 2 || cfg.in_channels != cfg.out_channels * 4 {
            return Err(QwenImageError::Config(format!(
                "Qwen-Image: ожидался патч 2×2 и in_channels = 4·out_channels, пришло patch {} in {} out {}",
                cfg.patch_size, cfg.in_channels, cfg.out_channels
            )));
        }
        Ok(cfg)
    }

    pub fn inner(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// FF: `4·inner` (GELU-tanh).
    pub fn mlp_hidden(&self) -> usize {
        4 * self.inner()
    }

    /// Параметров в одном блоке (для оценки памяти): внимание 8·d², два FF
    /// по 8·d², две модуляции по 6·d².
    pub fn block_params(&self) -> usize {
        let d = self.inner();
        8 * d * d + 2 * 2 * d * self.mlp_hidden() + 2 * 6 * d * d
    }
}

/// `vae/config.json` (`AutoencoderKLQwenImage`, VAE Wan 2.1 в одном кадре).
#[derive(Debug, Clone, PartialEq)]
pub struct QwenVaeConfig {
    pub base_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl QwenVaeConfig {
    pub fn from_json(bytes: &[u8]) -> Result<Self, QwenImageError> {
        let v = parse(bytes, "vae/config.json")?;
        let cfg = Self {
            base_dim: num(&v, "base_dim", 96),
            z_dim: num(&v, "z_dim", 16),
            dim_mult: usizes(&v, "dim_mult").unwrap_or_else(|| vec![1, 2, 4, 4]),
            num_res_blocks: num(&v, "num_res_blocks", 2),
            latents_mean: floats(&v, "latents_mean").unwrap_or_default(),
            latents_std: floats(&v, "latents_std").unwrap_or_default(),
        };
        if cfg.latents_mean.len() != cfg.z_dim || cfg.latents_std.len() != cfg.z_dim {
            return Err(QwenImageError::Config(format!(
                "VAE: latents_mean/std должны быть по {} значений",
                cfg.z_dim
            )));
        }
        if usizes(&v, "attn_scales").is_some_and(|a| !a.is_empty()) {
            return Err(QwenImageError::Config("VAE: attn_scales не поддерживаются".into()));
        }
        Ok(cfg)
    }
}

/// Башня зрения Qwen2.5-VL (`vision_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden: usize,
    pub num_heads: usize,
    pub intermediate: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub window_size: usize,
    pub fullatt_blocks: Vec<usize>,
    pub out_hidden: usize,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.num_heads
    }

    /// Сторона квадрата пикселей на один токен LLM (патч × слияние).
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// Окно внимания в токенах LLM по стороне (112 / 14 / 2 = 4).
    pub fn window_units(&self) -> usize {
        (self.window_size / self.merge_size / self.patch_size).max(1)
    }
}

/// `text_encoder/config.json` (`Qwen2_5_VLForConditionalGeneration`).
#[derive(Debug, Clone, PartialEq)]
pub struct TextEncoderConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
    pub rope_theta: f64,
    /// Разбиение частот RoPE по осям (t, h, w): 16 + 24 + 24 = head_dim/2.
    pub mrope_section: Vec<usize>,
    pub vocab: usize,
    pub image_token_id: u32,
    pub vision: VisionConfig,
}

impl TextEncoderConfig {
    pub fn from_json(bytes: &[u8]) -> Result<Self, QwenImageError> {
        let v = parse(bytes, "text_encoder/config.json")?;
        // transformers ≥ 4.57 кладёт текстовую часть ещё и в `text_config`;
        // значения те же, верхний уровень есть у обоих форматов.
        let t = if v.get("hidden_size").is_some() { &v } else { v.get("text_config").unwrap_or(&v) };
        let hidden = num(t, "hidden_size", 3584);
        let heads = num(t, "num_attention_heads", 28);
        let rope = t.get("rope_scaling").or_else(|| t.get("rope_parameters"));
        let mrope = rope.and_then(|r| usizes(r, "mrope_section")).unwrap_or_else(|| vec![16, 24, 24]);
        let theta = t
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .or_else(|| rope.and_then(|r| r.get("rope_theta")).and_then(|x| x.as_f64()))
            .unwrap_or(1_000_000.0);
        let vc = v
            .get("vision_config")
            .ok_or_else(|| QwenImageError::Config("text_encoder/config.json: нет vision_config".into()))?;
        let vision = VisionConfig {
            depth: num(vc, "depth", 32),
            hidden: num(vc, "hidden_size", 1280),
            num_heads: num(vc, "num_heads", 16),
            intermediate: num(vc, "intermediate_size", 3420),
            patch_size: num(vc, "patch_size", 14),
            temporal_patch_size: num(vc, "temporal_patch_size", 2),
            merge_size: num(vc, "spatial_merge_size", 2),
            window_size: num(vc, "window_size", 112),
            fullatt_blocks: usizes(vc, "fullatt_block_indexes").unwrap_or_else(|| vec![7, 15, 23, 31]),
            out_hidden: num(vc, "out_hidden_size", hidden),
        };
        let cfg = Self {
            hidden,
            intermediate: num(t, "intermediate_size", 18944),
            num_layers: num(t, "num_hidden_layers", 28),
            num_heads: heads,
            num_kv_heads: num(t, "num_key_value_heads", 4),
            head_dim: hidden / heads.max(1),
            rms_eps: flt(t, "rms_norm_eps", 1e-6) as f32,
            rope_theta: theta,
            mrope_section: mrope,
            vocab: num(t, "vocab_size", 152064),
            image_token_id: num(&v, "image_token_id", 151655) as u32,
            vision,
        };
        if cfg.mrope_section.iter().sum::<usize>() * 2 != cfg.head_dim {
            return Err(QwenImageError::Config(format!(
                "Qwen2.5-VL: mrope_section {:?} не покрывает голову {}",
                cfg.mrope_section, cfg.head_dim
            )));
        }
        if cfg.vision.out_hidden != cfg.hidden {
            return Err(QwenImageError::Config(format!(
                "Qwen2.5-VL: выход башни зрения {} ≠ hidden {}",
                cfg.vision.out_hidden, cfg.hidden
            )));
        }
        Ok(cfg)
    }

    /// Байт на слой LLM в BF16 (для раскладки по VRAM).
    pub fn layer_bytes(&self) -> usize {
        let (d, i) = (self.hidden, self.intermediate);
        let kv = self.num_kv_heads * self.head_dim;
        let params = d * d * 2 + 2 * d * kv + 3 * d * i;
        params * 2
    }
}

/// `processor/preprocessor_config.json` (`Qwen2VLImageProcessor`).
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessorConfig {
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            min_pixels: 56 * 56,
            max_pixels: 28 * 28 * 16384,
            mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            std: [0.268_629_54, 0.261_302_58, 0.275_777_1],
        }
    }
}

impl ProcessorConfig {
    pub fn from_json(bytes: Option<&[u8]>) -> Result<Self, QwenImageError> {
        let mut cfg = Self::default();
        let Some(bytes) = bytes else { return Ok(cfg) };
        let v = parse(bytes, "processor/preprocessor_config.json")?;
        let size = v.get("size");
        cfg.min_pixels = v
            .get("min_pixels")
            .and_then(|x| x.as_u64())
            .or_else(|| size.and_then(|s| s.get("shortest_edge")).and_then(|x| x.as_u64()))
            .map(|x| x as usize)
            .unwrap_or(cfg.min_pixels);
        cfg.max_pixels = v
            .get("max_pixels")
            .and_then(|x| x.as_u64())
            .or_else(|| size.and_then(|s| s.get("longest_edge")).and_then(|x| x.as_u64()))
            .map(|x| x as usize)
            .unwrap_or(cfg.max_pixels);
        if let Some(m) = floats(&v, "image_mean").filter(|m| m.len() == 3) {
            cfg.mean = [m[0], m[1], m[2]];
        }
        if let Some(s) = floats(&v, "image_std").filter(|s| s.len() == 3) {
            cfg.std = [s[0], s[1], s[2]];
        }
        Ok(cfg)
    }
}

/// `scheduler/scheduler_config.json` (`FlowMatchEulerDiscreteScheduler` с
/// динамическим экспоненциальным сдвигом и растяжкой к `shift_terminal`).
#[derive(Debug, Clone, PartialEq)]
pub struct SchedulerConfig {
    pub base_image_seq_len: usize,
    pub max_image_seq_len: usize,
    pub base_shift: f64,
    pub max_shift: f64,
    pub shift_terminal: Option<f64>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self { base_image_seq_len: 256, max_image_seq_len: 8192, base_shift: 0.5, max_shift: 0.9, shift_terminal: Some(0.02) }
    }
}

impl SchedulerConfig {
    pub fn from_json(bytes: Option<&[u8]>) -> Result<Self, QwenImageError> {
        let mut cfg = Self::default();
        let Some(bytes) = bytes else { return Ok(cfg) };
        let v = parse(bytes, "scheduler/scheduler_config.json")?;
        if v.get("use_dynamic_shifting").and_then(|x| x.as_bool()) == Some(false) {
            return Err(QwenImageError::Config("расписание без динамического сдвига не поддерживается".into()));
        }
        if v.get("time_shift_type").and_then(|x| x.as_str()).is_some_and(|t| t != "exponential") {
            return Err(QwenImageError::Config("поддерживается только экспоненциальный сдвиг расписания".into()));
        }
        cfg.base_image_seq_len = num(&v, "base_image_seq_len", cfg.base_image_seq_len);
        cfg.max_image_seq_len = num(&v, "max_image_seq_len", cfg.max_image_seq_len);
        cfg.base_shift = flt(&v, "base_shift", cfg.base_shift);
        cfg.max_shift = flt(&v, "max_shift", cfg.max_shift);
        cfg.shift_terminal = v.get("shift_terminal").and_then(|x| x.as_f64()).filter(|x| *x > 0.0);
        Ok(cfg)
    }

    /// `calculate_shift` пайплайна: линейно по длине латента.
    pub fn mu(&self, image_seq_len: usize) -> f64 {
        let m = (self.max_shift - self.base_shift) / (self.max_image_seq_len as f64 - self.base_image_seq_len as f64);
        let b = self.base_shift - m * self.base_image_seq_len as f64;
        image_seq_len as f64 * m + b
    }
}

/// Какой пайплайн описывает каталог.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenImageVariant {
    /// `QwenImagePipeline`: картинка по тексту.
    TextToImage,
    /// `QwenImageEditPipeline` (Qwen-Image-Edit, 08.2025): одна картинка,
    /// она же идёт в VL-энкодер (~1 Мп) и в VAE.
    Edit,
    /// `QwenImageEditPlusPipeline` (2509, 2511): до нескольких картинок,
    /// в VL-энкодер — уменьшенные до ~384², в VAE — до ~1 Мп.
    EditPlus,
}

impl QwenImageVariant {
    pub fn from_model_index(bytes: Option<&[u8]>) -> Self {
        let class = bytes
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok())
            .and_then(|v| v.get("_class_name").and_then(|c| c.as_str()).map(str::to_string))
            .unwrap_or_default();
        match class.as_str() {
            "QwenImageEditPlusPipeline" => Self::EditPlus,
            "QwenImageEditPipeline" | "QwenImageEditInpaintPipeline" => Self::Edit,
            _ => Self::TextToImage,
        }
    }

    pub fn is_edit(self) -> bool {
        !matches!(self, Self::TextToImage)
    }

    /// Сколько картинок принимает пайплайн.
    pub fn max_images(self) -> usize {
        match self {
            Self::TextToImage => 0,
            Self::Edit => 1,
            Self::EditPlus => 4,
        }
    }

    /// Шагов по умолчанию (карточки моделей: Edit 50, 2509/2511 40, Qwen-Image 50).
    pub fn default_steps(self) -> usize {
        match self {
            Self::EditPlus => 40,
            _ => 50,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::TextToImage => "Qwen-Image",
            Self::Edit => "Qwen-Image-Edit",
            Self::EditPlus => "Qwen-Image-Edit-Plus",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_matches_pipeline() {
        // calculate_shift(4096, 256, 8192, 0.5, 0.9)
        let c = SchedulerConfig::default();
        let m = (0.9 - 0.5) / (8192.0 - 256.0);
        let want = 4096.0 * m + (0.5 - m * 256.0);
        assert!((c.mu(4096) - want).abs() < 1e-12);
        assert!((c.mu(256) - 0.5).abs() < 1e-12);
        assert!((c.mu(8192) - 0.9).abs() < 1e-12);
    }

    #[test]
    fn variant_from_index() {
        let v = |s: &str| QwenImageVariant::from_model_index(Some(format!("{{\"_class_name\":\"{s}\"}}").as_bytes()));
        assert_eq!(v("QwenImageEditPlusPipeline"), QwenImageVariant::EditPlus);
        assert_eq!(v("QwenImageEditPipeline"), QwenImageVariant::Edit);
        assert_eq!(v("QwenImagePipeline"), QwenImageVariant::TextToImage);
        assert_eq!(QwenImageVariant::from_model_index(None), QwenImageVariant::TextToImage);
    }

    #[test]
    fn transformer_config_2511() {
        let j = br#"{"attention_head_dim":128,"axes_dims_rope":[16,56,56],"guidance_embeds":false,
            "in_channels":64,"joint_attention_dim":3584,"num_attention_heads":24,"num_layers":60,
            "out_channels":16,"patch_size":2,"zero_cond_t":true}"#;
        let c = QwenImageConfig::from_json(j).unwrap();
        assert!(c.zero_cond_t);
        assert_eq!(c.inner(), 3072);
        // ~20,4 млрд параметров в 60 блоках.
        let total = c.block_params() * c.num_layers;
        assert!((20_000_000_000..21_000_000_000).contains(&total), "{total}");
    }
}
