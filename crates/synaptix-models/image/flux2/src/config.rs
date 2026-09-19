//! Конфиги FLUX.2 из раскладки diffusers: трансформер, текстовый энкодер,
//! VAE и вариант пайплайна (`model_index.json`).

use crate::Flux2Error;

fn parse(bytes: &[u8], what: &str) -> Result<serde_json::Value, Flux2Error> {
    serde_json::from_slice(bytes).map_err(|e| Flux2Error::Config(format!("{what}: {e}")))
}

fn num(v: &serde_json::Value, k: &str, def: usize) -> usize {
    v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(def)
}

fn flt(v: &serde_json::Value, k: &str, def: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(def)
}

/// `transformer/config.json` (`Flux2Transformer2DModel`).
#[derive(Debug, Clone, PartialEq)]
pub struct Flux2Config {
    /// Double-stream блоков: dev 8, klein-4B 5, klein-9B 8.
    pub num_layers: usize,
    /// Single-stream блоков: dev 48, klein-4B 20, klein-9B 24.
    pub num_single_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// Каналы упакованного латента (32 канала VAE × 2×2).
    pub in_channels: usize,
    /// Ширина кондиционирования: 3 слоя × hidden энкодера.
    pub joint_attention_dim: usize,
    pub mlp_ratio: f64,
    pub axes_dims: Vec<usize>,
    pub rope_theta: f64,
    pub eps: f32,
    pub timestep_channels: usize,
    /// dev — guidance-эмбеддинг; у klein его нет.
    pub guidance_embeds: bool,
}

impl Flux2Config {
    pub fn from_json(bytes: &[u8]) -> Result<Self, Flux2Error> {
        let v = parse(bytes, "transformer/config.json")?;
        let axes: Vec<usize> = v
            .get("axes_dims_rope")
            .and_then(|a| a.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
            .unwrap_or_else(|| vec![32, 32, 32, 32]);
        let cfg = Self {
            num_layers: num(&v, "num_layers", 8),
            num_single_layers: num(&v, "num_single_layers", 48),
            num_heads: num(&v, "num_attention_heads", 48),
            head_dim: num(&v, "attention_head_dim", 128),
            in_channels: num(&v, "in_channels", 128),
            joint_attention_dim: num(&v, "joint_attention_dim", 15360),
            mlp_ratio: flt(&v, "mlp_ratio", 3.0),
            axes_dims: axes,
            rope_theta: flt(&v, "rope_theta", 2000.0),
            eps: flt(&v, "eps", 1e-6) as f32,
            timestep_channels: num(&v, "timestep_guidance_channels", 256),
            guidance_embeds: v.get("guidance_embeds").and_then(|x| x.as_bool()).unwrap_or(true),
        };
        if cfg.axes_dims.iter().sum::<usize>() != cfg.head_dim {
            return Err(Flux2Error::Config(format!(
                "FLUX.2: оси RoPE {:?} не складываются в голову {}",
                cfg.axes_dims, cfg.head_dim
            )));
        }
        if v.get("patch_size").and_then(|x| x.as_u64()).unwrap_or(1) != 1 {
            return Err(Flux2Error::Config("FLUX.2: patch_size ≠ 1 не поддерживается".into()));
        }
        Ok(cfg)
    }

    pub fn inner(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub fn mlp_hidden(&self) -> usize {
        (self.inner() as f64 * self.mlp_ratio) as usize
    }

    pub fn num_blocks(&self) -> usize {
        self.num_layers + self.num_single_layers
    }

    /// Параметров в double- и single-блоке (для оценки памяти).
    pub fn block_params(&self) -> (usize, usize) {
        let (d, m) = (self.inner(), self.mlp_hidden());
        let double = 8 * d * d + 2 * (d * 2 * m + m * d);
        let single = d * (3 * d + 2 * m) + (d + m) * d;
        (double, single)
    }
}

/// Архитектура LLM-энкодера текста.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextEncoderArch {
    /// `Mistral3ForConditionalGeneration` (FLUX.2-dev): Mistral-Small-3.1,
    /// ключи `language_model.model.*`, без qk-norm.
    Mistral3,
    /// `Qwen3ForCausalLM` (klein): ключи `model.*`, qk RMSNorm.
    Qwen3,
}

/// `text_encoder/config.json`: размеры декодера и где лежат его тензоры.
#[derive(Debug, Clone, PartialEq)]
pub struct TextEncoderConfig {
    pub arch: TextEncoderArch,
    pub hidden: usize,
    pub intermediate: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
    pub rope_theta: f64,
    pub vocab: usize,
}

impl TextEncoderConfig {
    pub fn from_json(bytes: &[u8]) -> Result<Self, Flux2Error> {
        let root = parse(bytes, "text_encoder/config.json")?;
        let model_type = root.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
        let (arch, v) = match model_type {
            "mistral3" => (
                TextEncoderArch::Mistral3,
                root.get("text_config").cloned().ok_or_else(|| {
                    Flux2Error::Config("text_encoder/config.json: у mistral3 нет text_config".into())
                })?,
            ),
            "qwen3" => (TextEncoderArch::Qwen3, root.clone()),
            other => {
                return Err(Flux2Error::Config(format!(
                    "текстовый энкодер `{other}` не поддерживается (ожидается mistral3 или qwen3)"
                )))
            }
        };
        let hidden = num(&v, "hidden_size", 0);
        let heads = num(&v, "num_attention_heads", 0);
        if hidden == 0 || heads == 0 {
            return Err(Flux2Error::Config("text_encoder/config.json: нет hidden_size/num_attention_heads".into()));
        }
        let head_dim = match num(&v, "head_dim", 0) {
            0 => hidden / heads,
            d => d,
        };
        Ok(Self {
            arch,
            hidden,
            intermediate: num(&v, "intermediate_size", 0),
            num_layers: num(&v, "num_hidden_layers", 0),
            num_heads: heads,
            num_kv_heads: num(&v, "num_key_value_heads", heads),
            head_dim,
            rms_eps: flt(&v, "rms_norm_eps", 1e-6) as f32,
            rope_theta: flt(&v, "rope_theta", 1e6),
            vocab: num(&v, "vocab_size", 0),
        })
    }

    /// Префикс тензоров декодера в чанке `text_encoder`.
    pub fn prefix(&self) -> &'static str {
        match self.arch {
            TextEncoderArch::Mistral3 => "language_model.model",
            TextEncoderArch::Qwen3 => "model",
        }
    }

    pub fn qk_norm(&self) -> bool {
        self.arch == TextEncoderArch::Qwen3
    }

    /// Байт на слой в BF16.
    pub fn layer_bytes(&self) -> usize {
        let h = self.hidden;
        let attn = h * self.head_dim * (2 * self.num_heads + 2 * self.num_kv_heads);
        (attn + 3 * h * self.intermediate) * 2
    }
}

/// Вариант пайплайна: определяет слои энкодера, шаблон промпта и
/// рекомендованные шаги/guidance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flux2Variant {
    /// `Flux2Pipeline`: guidance-эмбеддинг (не CFG), 50 шагов, guidance 4.
    Dev,
    /// `Flux2KleinPipeline` с `is_distilled`: 4 шага, без guidance.
    KleinDistilled,
    /// Недистиллированный klein (base): настоящий CFG с пустым негативом.
    KleinBase,
}

impl Flux2Variant {
    pub fn from_model_index(bytes: Option<&[u8]>, te: TextEncoderArch) -> Self {
        let v: Option<serde_json::Value> = bytes.and_then(|b| serde_json::from_slice(b).ok());
        let class = v
            .as_ref()
            .and_then(|v| v.get("_class_name"))
            .and_then(|x| x.as_str())
            .unwrap_or("");
        let klein = class.contains("Klein") || (class.is_empty() && te == TextEncoderArch::Qwen3);
        if !klein {
            return Flux2Variant::Dev;
        }
        let distilled = v
            .as_ref()
            .and_then(|v| v.get("is_distilled"))
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        if distilled {
            Flux2Variant::KleinDistilled
        } else {
            Flux2Variant::KleinBase
        }
    }

    /// Какие `hidden_states` энкодера склеиваются (индексы HF: 0 —
    /// эмбеддинги, k — выход k-го слоя).
    pub fn text_layers(self) -> [usize; 3] {
        match self {
            Flux2Variant::Dev => [10, 20, 30],
            _ => [9, 18, 27],
        }
    }

    pub fn default_steps(self) -> usize {
        match self {
            Flux2Variant::KleinDistilled => 4,
            _ => 50,
        }
    }

    pub fn default_guidance(self) -> f32 {
        match self {
            Flux2Variant::KleinDistilled => 1.0,
            _ => 4.0,
        }
    }

    /// Нужен ли второй проход с пустым промптом (классический CFG).
    pub fn uses_cfg(self, guidance: f32) -> bool {
        self == Flux2Variant::KleinBase && guidance > 1.0
    }

    pub fn is_klein(self) -> bool {
        self != Flux2Variant::Dev
    }
}

/// `vae/config.json` (`AutoencoderKLFlux2`).
#[derive(Debug, Clone, PartialEq)]
pub struct Flux2VaeConfig {
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub batch_norm_eps: f64,
    pub use_quant_conv: bool,
    pub use_post_quant_conv: bool,
}

impl Flux2VaeConfig {
    pub fn from_json(bytes: &[u8]) -> Result<Self, Flux2Error> {
        let v = parse(bytes, "vae/config.json")?;
        let patch: Vec<u64> = v
            .get("patch_size")
            .and_then(|a| a.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
            .unwrap_or_else(|| vec![2, 2]);
        if patch != [2, 2] {
            return Err(Flux2Error::Config(format!("VAE FLUX.2: patch_size {patch:?} не поддерживается")));
        }
        Ok(Self {
            latent_channels: num(&v, "latent_channels", 32),
            block_out_channels: v
                .get("block_out_channels")
                .and_then(|a| a.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
                .unwrap_or_else(|| vec![128, 256, 512, 512]),
            layers_per_block: num(&v, "layers_per_block", 2),
            norm_num_groups: num(&v, "norm_num_groups", 32),
            batch_norm_eps: flt(&v, "batch_norm_eps", 1e-4),
            use_quant_conv: v.get("use_quant_conv").and_then(|x| x.as_bool()).unwrap_or(true),
            use_post_quant_conv: v.get("use_post_quant_conv").and_then(|x| x.as_bool()).unwrap_or(true),
        })
    }

    /// Во сколько раз VAE ужимает сторону (8 при четырёх блоках).
    pub fn scale(&self) -> usize {
        1 << (self.block_out_channels.len().saturating_sub(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dev_and_klein_transformer_configs() {
        let dev = br#"{"attention_head_dim":128,"axes_dims_rope":[32,32,32,32],"eps":1e-06,
            "in_channels":128,"joint_attention_dim":15360,"mlp_ratio":3.0,"num_attention_heads":48,
            "num_layers":8,"num_single_layers":48,"patch_size":1,"rope_theta":2000,
            "timestep_guidance_channels":256}"#;
        let c = Flux2Config::from_json(dev).unwrap();
        assert_eq!((c.inner(), c.mlp_hidden(), c.num_blocks()), (6144, 18432, 56));
        assert!(c.guidance_embeds);
        let klein = br#"{"attention_head_dim":128,"guidance_embeds":false,"joint_attention_dim":7680,
            "num_attention_heads":24,"num_layers":5,"num_single_layers":20}"#;
        let k = Flux2Config::from_json(klein).unwrap();
        assert_eq!((k.inner(), k.mlp_hidden()), (3072, 9216));
        assert!(!k.guidance_embeds);
        // Параметры блока dev: double ≈ 981M, single ≈ 491M.
        let (d, s) = c.block_params();
        assert_eq!((d, s), (981_467_136, 490_733_568));
    }

    #[test]
    fn text_encoder_kinds() {
        let m = br#"{"model_type":"mistral3","text_config":{"hidden_size":5120,"intermediate_size":32768,
            "num_hidden_layers":40,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,
            "rms_norm_eps":1e-05,"rope_theta":1000000000.0,"vocab_size":131072}}"#;
        let c = TextEncoderConfig::from_json(m).unwrap();
        assert_eq!(c.arch, TextEncoderArch::Mistral3);
        assert_eq!((c.hidden, c.num_layers, c.head_dim, c.prefix()), (5120, 40, 128, "language_model.model"));
        assert!(!c.qk_norm());
        let q = br#"{"model_type":"qwen3","hidden_size":2560,"intermediate_size":9728,"num_hidden_layers":36,
            "num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"rope_theta":1000000}"#;
        let c = TextEncoderConfig::from_json(q).unwrap();
        assert_eq!((c.arch, c.prefix()), (TextEncoderArch::Qwen3, "model"));
        assert!(c.qk_norm());
    }

    #[test]
    fn variants_from_model_index() {
        let dev = br#"{"_class_name":"Flux2Pipeline"}"#;
        assert_eq!(Flux2Variant::from_model_index(Some(dev), TextEncoderArch::Mistral3), Flux2Variant::Dev);
        let kd = br#"{"_class_name":"Flux2KleinPipeline","is_distilled":true}"#;
        let v = Flux2Variant::from_model_index(Some(kd), TextEncoderArch::Qwen3);
        assert_eq!((v, v.default_steps(), v.text_layers()), (Flux2Variant::KleinDistilled, 4, [9, 18, 27]));
        let kb = br#"{"_class_name":"Flux2KleinPipeline"}"#;
        let v = Flux2Variant::from_model_index(Some(kb), TextEncoderArch::Qwen3);
        assert!(v.uses_cfg(4.0) && !v.uses_cfg(1.0));
    }
}
