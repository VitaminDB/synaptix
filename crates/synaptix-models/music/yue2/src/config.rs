//! Конфигурации YuE2 — читаются из `config.json` внутри бандла.

use serde_json::Value;

use crate::YueError;

fn usize_at(v: &Value, key: &str, default: usize) -> usize {
    v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(default)
}

fn f32_at(v: &Value, key: &str, default: f32) -> f32 {
    v.get(key).and_then(|x| x.as_f64()).map(|x| x as f32).unwrap_or(default)
}

/// Костяк AR–NAR Mixture-of-Transformers (`model_type: "yue2"`).
#[derive(Debug, Clone, PartialEq)]
pub struct Yue2Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    /// Размерность латента VAE (ширина NAR-головы).
    pub latent_dim: usize,
    /// Ёмкость таблицы позиционных эмбеддингов NAR-ветки.
    pub max_latent_frames: usize,
    /// Сдвиг шкалы времени flow matching (у релиза 1.0 — шкала без сдвига).
    pub timestep_shift: f32,
}

impl Default for Yue2Config {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 6144,
            vocab_size: 184704,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 24576,
            latent_dim: 64,
            max_latent_frames: 24576,
            timestep_shift: 1.0,
        }
    }
}

impl Yue2Config {
    pub fn from_json(bytes: &[u8]) -> Result<Self, YueError> {
        let v: Value = serde_json::from_slice(bytes)
            .map_err(|e| YueError::Config(format!("config.json: {e}")))?;
        let model_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
        if !model_type.is_empty() && model_type != "yue2" {
            return Err(YueError::Config(format!(
                "config.json: ожидался model_type=yue2, а там `{model_type}`"
            )));
        }
        let d = Self::default();
        let cfg = Self {
            hidden_size: usize_at(&v, "hidden_size", d.hidden_size),
            num_hidden_layers: usize_at(&v, "num_hidden_layers", d.num_hidden_layers),
            num_attention_heads: usize_at(&v, "num_attention_heads", d.num_attention_heads),
            num_key_value_heads: usize_at(&v, "num_key_value_heads", d.num_key_value_heads),
            head_dim: usize_at(&v, "head_dim", d.head_dim),
            intermediate_size: usize_at(&v, "intermediate_size", d.intermediate_size),
            vocab_size: usize_at(&v, "vocab_size", d.vocab_size),
            rms_norm_eps: f32_at(&v, "rms_norm_eps", d.rms_norm_eps),
            rope_theta: f32_at(&v, "rope_theta", d.rope_theta),
            max_position_embeddings: usize_at(&v, "max_position_embeddings", d.max_position_embeddings),
            latent_dim: usize_at(&v, "latent_dim", d.latent_dim),
            max_latent_frames: usize_at(&v, "max_latent_frames", d.max_latent_frames),
            timestep_shift: f32_at(&v, "timestep_shift", d.timestep_shift),
        };
        if let Some(t) = v.get("latent_type").and_then(|x| x.as_str()) {
            if t != "vae" {
                return Err(YueError::Config(format!("latent_type `{t}` не поддержан")));
            }
        }
        if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
            return Err(YueError::Config("num_attention_heads не кратно num_key_value_heads".into()));
        }
        Ok(cfg)
    }
}

/// Oobleck-VAE (`model_type: "yue2_vae"`): 48 кГц стерео, 1920 сэмплов на кадр.
#[derive(Debug, Clone, PartialEq)]
pub struct Yue2VaeConfig {
    pub sample_rate: u32,
    pub latent_dim: usize,
    /// Во сколько раз декодер поднимает частоту кадров (произведение страйдов).
    pub downsampling_ratio: usize,
    pub audio_channels: usize,
    /// Каналы первого уровня и множители по уровням.
    pub channels: usize,
    pub c_mults: Vec<usize>,
    pub strides: Vec<usize>,
    /// Активация: `true` — SnakeBeta, `false` — ELU.
    pub use_snake: bool,
    /// `tanh` на выходе декодера (у релиза выключен).
    pub final_tanh: bool,
    /// Выход энкодера: mean и scale по половине каналов.
    pub encoder_latent_dim: usize,
    pub decode_core_frames: usize,
    pub decode_halo_frames: usize,
    /// `standard` (для прослушивания) или `legacy` (протокол бенчмарка).
    pub release_variant: String,
}

impl Default for Yue2VaeConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            latent_dim: 64,
            downsampling_ratio: 1920,
            audio_channels: 2,
            channels: 64,
            c_mults: vec![1, 2, 4, 8, 16, 32],
            strides: vec![2, 2, 4, 4, 5, 6],
            use_snake: true,
            final_tanh: false,
            encoder_latent_dim: 128,
            decode_core_frames: 1024,
            decode_halo_frames: 16,
            release_variant: "standard".into(),
        }
    }
}

impl Yue2VaeConfig {
    pub fn from_json(bytes: &[u8]) -> Result<Self, YueError> {
        let v: Value = serde_json::from_slice(bytes)
            .map_err(|e| YueError::Config(format!("config.json VAE: {e}")))?;
        let model_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
        if !model_type.is_empty() && model_type != "yue2_vae" {
            return Err(YueError::Config(format!(
                "config.json: ожидался model_type=yue2_vae, а там `{model_type}`"
            )));
        }
        let d = Self::default();
        let dec = v.get("decoder_config");
        let enc = v.get("encoder_config");
        let list = |node: Option<&Value>, key: &str, fallback: &[usize]| -> Vec<usize> {
            node.and_then(|n| n.get(key))
                .and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect::<Vec<_>>())
                .filter(|v: &Vec<usize>| !v.is_empty())
                .unwrap_or_else(|| fallback.to_vec())
        };
        let cfg = Self {
            sample_rate: usize_at(&v, "sample_rate", d.sample_rate as usize) as u32,
            latent_dim: usize_at(&v, "latent_dim", d.latent_dim),
            downsampling_ratio: usize_at(&v, "downsampling_ratio", d.downsampling_ratio),
            audio_channels: usize_at(&v, "audio_channels", d.audio_channels),
            channels: dec.map(|n| usize_at(n, "channels", d.channels)).unwrap_or(d.channels),
            c_mults: list(dec, "c_mults", &d.c_mults),
            strides: list(dec, "strides", &d.strides),
            use_snake: dec
                .and_then(|n| n.get("use_snake"))
                .and_then(|x| x.as_bool())
                .unwrap_or(d.use_snake),
            final_tanh: dec
                .and_then(|n| n.get("final_tanh"))
                .and_then(|x| x.as_bool())
                .unwrap_or(d.final_tanh),
            encoder_latent_dim: enc
                .map(|n| usize_at(n, "latent_dim", d.encoder_latent_dim))
                .unwrap_or(d.encoder_latent_dim),
            decode_core_frames: usize_at(&v, "decode_core_frames", d.decode_core_frames),
            decode_halo_frames: usize_at(&v, "decode_halo_frames", d.decode_halo_frames),
            release_variant: v
                .get("release_variant")
                .and_then(|x| x.as_str())
                .unwrap_or(&d.release_variant)
                .to_string(),
        };
        let product: usize = cfg.strides.iter().product();
        if product != cfg.downsampling_ratio {
            return Err(YueError::Config(format!(
                "страйды декодера дают {product}, а downsampling_ratio = {}",
                cfg.downsampling_ratio
            )));
        }
        if cfg.c_mults.len() != cfg.strides.len() {
            return Err(YueError::Config("c_mults и strides разной длины".into()));
        }
        Ok(cfg)
    }

    /// Сэмплов на один латентный кадр.
    pub fn hop_length(&self) -> usize {
        self.downsampling_ratio
    }
}
