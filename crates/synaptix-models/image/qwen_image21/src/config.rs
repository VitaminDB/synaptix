//! Конфиги Qwen-Image 2.1 из раскладки diffusers: трансформер
//! (`QwenImage21Transformer2DModel`), VAE (`AutoencoderKLQwenImage21`),
//! энкодер Qwen3-VL (`text_config` + `vision_config`), процессор картинок
//! (`Qwen2VLImageProcessorFast`) и расписание (общее с Qwen-Image).

use crate::QwenImage21Error;

pub use synaptix_image_qwen::config::SchedulerConfig;

fn parse(bytes: &[u8], what: &str) -> Result<serde_json::Value, QwenImage21Error> {
    serde_json::from_slice(bytes).map_err(|e| QwenImage21Error::Config(format!("{what}: {e}")))
}

fn num(v: &serde_json::Value, k: &str, def: usize) -> usize {
    v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(def)
}

fn flt(v: &serde_json::Value, k: &str, def: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(def)
}

fn usizes(v: &serde_json::Value, k: &str) -> Option<Vec<usize>> {
    v.get(k).and_then(|a| a.as_array()).map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
}

fn bools(v: &serde_json::Value, k: &str) -> Option<Vec<bool>> {
    v.get(k).and_then(|a| a.as_array()).map(|a| a.iter().filter_map(|x| x.as_bool()).collect())
}

fn floats(v: &serde_json::Value, k: &str) -> Option<Vec<f32>> {
    v.get(k).and_then(|a| a.as_array()).map(|a| a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect())
}

/// `transformer/config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen21Config {
    /// Single-stream блоков (32).
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// Каналы латента (64 = z_dim VAE, без упаковки 2×2).
    pub in_channels: usize,
    pub out_channels: usize,
    /// Ширина скрытых состояний энкодера (4096 у Qwen3-VL 8B).
    pub context_in_dim: usize,
    /// FF: `mlp_ratio·inner` (SwiGLU).
    pub mlp_ratio: usize,
    /// Оси RoPE (кадр, высота, ширина): 16 + 56 + 56 = голова.
    pub axes_dims: Vec<usize>,
    pub eps: f32,
    /// Текст и референсы модулируются временем 0 (нужно для KV-кэша).
    pub causal_condition: bool,
}

impl Qwen21Config {
    pub fn from_json(bytes: &[u8]) -> Result<Self, QwenImage21Error> {
        let v = parse(bytes, "transformer/config.json")?;
        let class = v.get("_class_name").and_then(|c| c.as_str()).unwrap_or("");
        if !class.is_empty() && class != "QwenImage21Transformer2DModel" {
            return Err(QwenImage21Error::Config(format!(
                "transformer/config.json: ожидался QwenImage21Transformer2DModel, пришёл {class}"
            )));
        }
        let cfg = Self {
            num_layers: num(&v, "num_layers", 32),
            num_heads: num(&v, "num_attention_heads", 32),
            head_dim: num(&v, "attention_head_dim", 128),
            in_channels: num(&v, "in_channels", 64),
            out_channels: num(&v, "out_channels", 64),
            context_in_dim: num(&v, "context_in_dim", 4096),
            mlp_ratio: num(&v, "mlp_ratio", 3),
            axes_dims: usizes(&v, "axes_dims_rope").unwrap_or_else(|| vec![16, 56, 56]),
            eps: flt(&v, "eps", 1e-6) as f32,
            causal_condition: v.get("causal_condition").and_then(|x| x.as_bool()).unwrap_or(true),
        };
        if cfg.axes_dims.len() != 3 || cfg.axes_dims.iter().sum::<usize>() != cfg.head_dim {
            return Err(QwenImage21Error::Config(format!(
                "Qwen-Image 2.1: оси RoPE {:?} не складываются в голову {}",
                cfg.axes_dims, cfg.head_dim
            )));
        }
        if num(&v, "patch_size", 1) != 1 {
            return Err(QwenImage21Error::Config("Qwen-Image 2.1: поддерживается только patch_size 1".into()));
        }
        Ok(cfg)
    }

    pub fn inner(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub fn mlp_hidden(&self) -> usize {
        self.mlp_ratio * self.inner()
    }

    /// Параметров в одном блоке: внимание 4·d², SwiGLU 3·d·m.
    pub fn block_params(&self) -> usize {
        let d = self.inner();
        4 * d * d + 3 * d * self.mlp_hidden()
    }
}

/// `vae/config.json` (`AutoencoderKLQwenImage21`, резидуальный 3D-VAE в
/// одном кадре).
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen21VaeConfig {
    pub base_dim: usize,
    pub decoder_base_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    /// По стадиям энкодера: сжатие по времени (у одного кадра — половина
    /// каналов резидуального шортката нулевая).
    pub temporal_downsample: Vec<bool>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub scale_factor_spatial: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl Qwen21VaeConfig {
    pub fn from_json(bytes: &[u8]) -> Result<Self, QwenImage21Error> {
        let v = parse(bytes, "vae/config.json")?;
        let base_dim = num(&v, "base_dim", 96);
        let cfg = Self {
            base_dim,
            decoder_base_dim: num(&v, "decoder_base_dim", base_dim),
            z_dim: num(&v, "z_dim", 64),
            dim_mult: usizes(&v, "dim_mult").unwrap_or_else(|| vec![1, 2, 4, 8, 8]),
            num_res_blocks: num(&v, "num_res_blocks", 2),
            temporal_downsample: bools(&v, "temperal_downsample").unwrap_or_else(|| vec![false, true, true, true]),
            in_channels: num(&v, "in_channels", 4),
            out_channels: num(&v, "out_channels", 4),
            scale_factor_spatial: num(&v, "scale_factor_spatial", 16),
            latents_mean: floats(&v, "latents_mean").unwrap_or_default(),
            latents_std: floats(&v, "latents_std").unwrap_or_default(),
        };
        if cfg.latents_mean.len() != cfg.z_dim || cfg.latents_std.len() != cfg.z_dim {
            return Err(QwenImage21Error::Config(format!("VAE: latents_mean/std должны быть по {} значений", cfg.z_dim)));
        }
        if !v.get("is_residual").and_then(|x| x.as_bool()).unwrap_or(true) {
            return Err(QwenImage21Error::Config("VAE: ожидался резидуальный вариант (is_residual)".into()));
        }
        if v.get("patch_size").is_some_and(|p| !p.is_null()) {
            return Err(QwenImage21Error::Config("VAE: patch_size не поддерживается".into()));
        }
        if usizes(&v, "attn_scales").is_some_and(|a| !a.is_empty()) {
            return Err(QwenImage21Error::Config("VAE: attn_scales не поддерживаются".into()));
        }
        let expect = 1usize << (cfg.dim_mult.len() - 1);
        if cfg.scale_factor_spatial != expect {
            return Err(QwenImage21Error::Config(format!(
                "VAE: {} стадий дают сжатие {expect}, в конфиге {}",
                cfg.dim_mult.len(),
                cfg.scale_factor_spatial
            )));
        }
        Ok(cfg)
    }
}

/// Башня зрения Qwen3-VL (`vision_config`): SigLIP-подобный ViT с учёной
/// позиционной таблицей, 2D-RoPE и deepstack-отводами.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3VlVisionConfig {
    pub depth: usize,
    pub hidden: usize,
    pub num_heads: usize,
    pub intermediate: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub out_hidden: usize,
    pub num_position_embeddings: usize,
    pub deepstack_indexes: Vec<usize>,
    pub layer_norm_eps: f32,
}

impl Qwen3VlVisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.num_heads
    }

    /// Пикселей на сторону одного токена LLM (патч × слияние = 32).
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    pub fn merge_unit(&self) -> usize {
        self.merge_size * self.merge_size
    }

    pub fn patch_features(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// Сторона квадратной таблицы позиций (48).
    pub fn pos_grid(&self) -> usize {
        (self.num_position_embeddings as f64).sqrt().round() as usize
    }
}

/// `text_encoder/config.json` (`Qwen3VLForConditionalGeneration`).
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3VlConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
    pub rope_theta: f64,
    /// Разбиение частот по осям (t, h, w): у Qwen3-VL — чередованием
    /// (`mrope_interleaved`), 24 + 20 + 20 = head_dim/2.
    pub mrope_section: Vec<usize>,
    pub mrope_interleaved: bool,
    pub vocab: usize,
    pub image_token_id: u32,
    pub vision_start_id: u32,
    pub vision_end_id: u32,
    pub vision: Qwen3VlVisionConfig,
}

impl Qwen3VlConfig {
    pub fn from_json(bytes: &[u8]) -> Result<Self, QwenImage21Error> {
        let v = parse(bytes, "text_encoder/config.json")?;
        let t = v.get("text_config").unwrap_or(&v);
        let hidden = num(t, "hidden_size", 4096);
        let heads = num(t, "num_attention_heads", 32);
        let rope = t.get("rope_scaling").or_else(|| t.get("rope_parameters"));
        let mrope = rope.and_then(|r| usizes(r, "mrope_section")).unwrap_or_else(|| vec![24, 20, 20]);
        let interleaved = rope.and_then(|r| r.get("mrope_interleaved")).and_then(|x| x.as_bool()).unwrap_or(true);
        let theta = t
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .or_else(|| rope.and_then(|r| r.get("rope_theta")).and_then(|x| x.as_f64()))
            .unwrap_or(5_000_000.0);
        let vc = v
            .get("vision_config")
            .ok_or_else(|| QwenImage21Error::Config("text_encoder/config.json: нет vision_config".into()))?;
        let vhidden = num(vc, "hidden_size", 1152);
        let vision = Qwen3VlVisionConfig {
            depth: num(vc, "depth", 27),
            hidden: vhidden,
            num_heads: num(vc, "num_heads", 16),
            intermediate: num(vc, "intermediate_size", 4304),
            in_channels: num(vc, "in_channels", 3),
            patch_size: num(vc, "patch_size", 16),
            temporal_patch_size: num(vc, "temporal_patch_size", 2),
            merge_size: num(vc, "spatial_merge_size", 2),
            out_hidden: num(vc, "out_hidden_size", hidden),
            num_position_embeddings: num(vc, "num_position_embeddings", 2304),
            deepstack_indexes: usizes(vc, "deepstack_visual_indexes").unwrap_or_default(),
            layer_norm_eps: flt(vc, "layer_norm_eps", 1e-6) as f32,
        };
        let cfg = Self {
            hidden,
            intermediate: num(t, "intermediate_size", 12288),
            num_layers: num(t, "num_hidden_layers", 36),
            num_heads: heads,
            num_kv_heads: num(t, "num_key_value_heads", 8),
            head_dim: num(t, "head_dim", hidden / heads.max(1)),
            rms_eps: flt(t, "rms_norm_eps", 1e-6) as f32,
            rope_theta: theta,
            mrope_section: mrope,
            mrope_interleaved: interleaved,
            vocab: num(t, "vocab_size", 151936),
            image_token_id: num(&v, "image_token_id", 151655) as u32,
            vision_start_id: num(&v, "vision_start_token_id", 151652) as u32,
            vision_end_id: num(&v, "vision_end_token_id", 151653) as u32,
            vision,
        };
        if cfg.mrope_section.iter().sum::<usize>() * 2 != cfg.head_dim {
            return Err(QwenImage21Error::Config(format!(
                "Qwen3-VL: mrope_section {:?} не покрывает голову {}",
                cfg.mrope_section, cfg.head_dim
            )));
        }
        if cfg.vision.out_hidden != cfg.hidden {
            return Err(QwenImage21Error::Config(format!(
                "Qwen3-VL: выход башни зрения {} ≠ hidden {}",
                cfg.vision.out_hidden, cfg.hidden
            )));
        }
        if cfg.vision.hidden % cfg.vision.num_heads != 0 {
            return Err(QwenImage21Error::Config("Qwen3-VL: hidden башни не делится на головы".into()));
        }
        Ok(cfg)
    }

    /// Байт на слой LLM в `compute` (для раскладки по VRAM).
    pub fn layer_bytes(&self, bytes_per_param: usize) -> usize {
        let (d, i) = (self.hidden, self.intermediate);
        let q = self.num_heads * self.head_dim;
        let kv = self.num_kv_heads * self.head_dim;
        let params = d * q * 2 + 2 * d * kv + 3 * d * i;
        params * bytes_per_param
    }

    pub fn group_size(&self) -> usize {
        self.num_heads / self.num_kv_heads.max(1)
    }
}

/// `processor/preprocessor_config.json` (`Qwen2VLImageProcessorFast`):
/// границы площади (`size.shortest_edge` / `longest_edge` в пикселях),
/// нормировка. Патч/слияние — из конфига башни.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessorConfig {
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self { min_pixels: 65536, max_pixels: 16_777_216, mean: [0.5; 3], std: [0.5; 3] }
    }
}

impl ProcessorConfig {
    pub fn from_json(bytes: Option<&[u8]>) -> Result<Self, QwenImage21Error> {
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

/// Проверка, что `model_index.json` описывает именно 2.1.
pub fn check_model_index(bytes: Option<&[u8]>) -> Result<(), QwenImage21Error> {
    let Some(bytes) = bytes else { return Ok(()) };
    let class = serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()
        .and_then(|v| v.get("_class_name").and_then(|c| c.as_str()).map(str::to_string))
        .unwrap_or_default();
    if !class.is_empty() && class != "QwenImage21Pipeline" {
        return Err(QwenImage21Error::Config(format!(
            "model_index.json: это {class}, а не QwenImage21Pipeline — для Qwen-Image-Edit есть ноды Qwen-Image"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformer_config_21() {
        let j = br#"{"_class_name":"QwenImage21Transformer2DModel","attention_head_dim":128,"axes_dims_rope":[16,56,56],
            "context_in_dim":4096,"in_channels":64,"num_attention_heads":32,"num_layers":32,"out_channels":64,
            "patch_size":1,"mlp_ratio":3,"eps":1e-06,"causal_condition":true}"#;
        let c = Qwen21Config::from_json(j).unwrap();
        assert_eq!(c.inner(), 4096);
        assert_eq!(c.mlp_hidden(), 12288);
        // ~7 млрд параметров в 32 блоках.
        let total = c.block_params() * c.num_layers;
        assert!((6_900_000_000..7_100_000_000).contains(&total), "{total}");
        assert!(c.causal_condition);
        assert!(Qwen21Config::from_json(br#"{"_class_name":"QwenImageTransformer2DModel"}"#).is_err());
    }

    #[test]
    fn vae_config_checks() {
        let mean: Vec<String> = (0..64).map(|i| format!("{}", i as f32 * 0.1)).collect();
        let j = format!(
            r#"{{"base_dim":96,"decoder_base_dim":144,"z_dim":64,"dim_mult":[1,2,4,8,8],"num_res_blocks":2,
            "temperal_downsample":[false,true,true,true],"in_channels":4,"out_channels":4,"patch_size":null,
            "scale_factor_spatial":16,"is_residual":true,"latents_mean":[{}],"latents_std":[{}]}}"#,
            mean.join(","),
            mean.join(",")
        );
        let c = Qwen21VaeConfig::from_json(j.as_bytes()).unwrap();
        assert_eq!(c.decoder_base_dim, 144);
        assert_eq!(c.temporal_downsample, vec![false, true, true, true]);
    }

    #[test]
    fn text_encoder_config_qwen3vl() {
        let j = br#"{"image_token_id":151655,"model_type":"qwen3_vl","text_config":{"hidden_size":4096,
            "intermediate_size":12288,"num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,
            "head_dim":128,"rms_norm_eps":1e-06,"rope_scaling":{"mrope_interleaved":true,"mrope_section":[24,20,20]},
            "rope_theta":5000000,"vocab_size":151936},"vision_config":{"deepstack_visual_indexes":[8,16,24],
            "depth":27,"hidden_size":1152,"intermediate_size":4304,"num_heads":16,"num_position_embeddings":2304,
            "out_hidden_size":4096,"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2}}"#;
        let c = Qwen3VlConfig::from_json(j).unwrap();
        assert_eq!(c.group_size(), 4);
        assert!(c.mrope_interleaved);
        assert_eq!(c.vision.factor(), 32);
        assert_eq!(c.vision.pos_grid(), 48);
        assert_eq!(c.vision.head_dim(), 72);
        assert_eq!(c.vision.deepstack_indexes, vec![8, 16, 24]);
        assert_eq!(c.rope_theta, 5_000_000.0);
    }

    #[test]
    fn processor_from_size() {
        let j = br#"{"size":{"longest_edge":16777216,"shortest_edge":65536},"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}"#;
        let c = ProcessorConfig::from_json(Some(j)).unwrap();
        assert_eq!(c.min_pixels, 65536);
        assert_eq!(c.max_pixels, 16_777_216);
        assert!(check_model_index(Some(br#"{"_class_name":"QwenImage21Pipeline"}"#)).is_ok());
        assert!(check_model_index(Some(br#"{"_class_name":"QwenImageEditPlusPipeline"}"#)).is_err());
    }
}
