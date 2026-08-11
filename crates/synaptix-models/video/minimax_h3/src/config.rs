use std::path::Path;

use serde::Deserialize;

use crate::H3Error;

pub const DEFAULT_SIGMA_SHIFT_VIDEO: f32 = 12.0;
pub const DEFAULT_SIGMA_SHIFT_AUDIO: f32 = 3.0;
pub const VISUAL_COND_TIMESTEP: f32 = 0.999;
pub const AUDIO_COND_TIMESTEP: f32 = 1.0;
pub const FPS: f64 = 24.0;
pub const AUDIO_SAMPLE_RATE: usize = 32_000;
pub const AUDIO_SAMPLES_PER_LATENT: usize = 800;
pub const AUDIO_LATENTS_PER_SECOND: usize = 40;
pub const VAE_SPATIAL_RATIO: usize = 16;
pub const VAE_TEMPORAL_RATIO: usize = 4;
pub const FRAME_GRID_STEP: usize = 17;
pub const FRAME_GRID_BASE: usize = 5;
pub const MODALITY_VIDEO: usize = 0;
pub const MODALITY_TEXT: usize = 1;
pub const MODALITY_AUDIO: usize = 2;
pub const ADALN_MODALITIES: usize = 3;
pub const ADALN_CHUNKS: usize = 6;
pub const FINAL_ADALN_CHUNKS: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H3Variant {
    Fl2va,
    Ref2va,
}

impl H3Variant {
    pub fn dir_name(self) -> &'static str {
        match self {
            H3Variant::Fl2va => "FL2VA",
            H3Variant::Ref2va => "Ref2VA",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fl2va" | "t2va" => Some(H3Variant::Fl2va),
            "ref2va" => Some(H3Variant::Ref2va),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct H3Config {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub token_refiner_num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub ffn_hidden_size: usize,
    pub latents_dim: usize,
    pub audio_latents_dim: usize,
    pub patch_size: [usize; 3],
    pub text_dim: usize,
    pub timestep_input_dim: usize,
    pub time_embed_hidden_size: usize,
    pub time_embed_dim: usize,
    pub adaln_out_features: usize,
    pub final_adaln_out_features: usize,
    pub rope_inv_freq_len: usize,
    pub norm_eps: f32,
    pub qk_norm_eps: f32,
    pub final_norm_eps: f32,
    #[serde(skip)]
    pub sigma_shift_video: f32,
    #[serde(skip)]
    pub sigma_shift_audio: f32,
    #[serde(skip)]
    pub tasks: Vec<String>,
    #[serde(skip)]
    pub partition: String,
}

impl Default for H3Config {
    fn default() -> Self {
        Self {
            hidden_size: 5376,
            num_layers: 50,
            token_refiner_num_layers: 2,
            num_attention_heads: 56,
            attention_head_dim: 128,
            ffn_hidden_size: 14336,
            latents_dim: 24,
            audio_latents_dim: 32,
            patch_size: [1, 2, 2],
            text_dim: 5120,
            timestep_input_dim: 256,
            time_embed_hidden_size: 5376,
            time_embed_dim: 2688,
            adaln_out_features: 96768,
            final_adaln_out_features: 10752,
            rope_inv_freq_len: 16,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
            sigma_shift_video: DEFAULT_SIGMA_SHIFT_VIDEO,
            sigma_shift_audio: DEFAULT_SIGMA_SHIFT_AUDIO,
            tasks: Vec::new(),
            partition: String::new(),
        }
    }
}

impl H3Config {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, H3Error> {
        let dir = dir.as_ref();
        let cfg_path = dir.join("transformer").join("config.json");
        let bytes = std::fs::read(&cfg_path)
            .map_err(|e| H3Error::Config(format!("{}: {e}", cfg_path.display())))?;
        let mut cfg: Self = serde_json::from_slice(&bytes)
            .map_err(|e| H3Error::Config(format!("transformer/config.json: {e}")))?;
        cfg.apply_model_index(dir)?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn apply_model_index(&mut self, dir: &Path) -> Result<(), H3Error> {
        let path = dir.join("model_index.json");
        let Ok(bytes) = std::fs::read(&path) else {
            return Ok(());
        };
        let root: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| H3Error::Config(format!("model_index.json: {e}")))?;
        let Some(meta) = root.get("_minimax_h3") else {
            return Ok(());
        };
        if let Some(p) = meta.get("partition").and_then(|v| v.as_str()) {
            self.partition = p.to_string();
        }
        if let Some(t) = meta.get("tasks").and_then(|v| v.as_array()) {
            self.tasks = t.iter().filter_map(|v| v.as_str().map(String::from)).collect();
        }
        if let Some(s) = meta.get("sigma_shift_scales") {
            if let Some(v) = s.get("video").and_then(|v| v.as_f64()) {
                self.sigma_shift_video = v as f32;
            }
            if let Some(v) = s.get("audio").and_then(|v| v.as_f64()) {
                self.sigma_shift_audio = v as f32;
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), H3Error> {
        let expect_adaln = ADALN_CHUNKS * ADALN_MODALITIES * self.hidden_size;
        if self.adaln_out_features != expect_adaln {
            return Err(H3Error::Config(format!(
                "adaln_out_features {} != {expect_adaln}",
                self.adaln_out_features
            )));
        }
        let expect_final = FINAL_ADALN_CHUNKS * self.hidden_size;
        if self.final_adaln_out_features != expect_final {
            return Err(H3Error::Config(format!(
                "final_adaln_out_features {} != {expect_final}",
                self.final_adaln_out_features
            )));
        }
        if self.timestep_input_dim % 2 != 0 {
            return Err(H3Error::Config("timestep_input_dim нечётен".into()));
        }
        Ok(())
    }

    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn video_patch_dim(&self) -> usize {
        self.latents_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
    }

    pub fn rope_rot_dim(&self) -> usize {
        self.rope_inv_freq_len * 3 * 2
    }

    pub fn variant(&self) -> H3Variant {
        match self.partition.as_str() {
            "ref2va" => H3Variant::Ref2va,
            _ => H3Variant::Fl2va,
        }
    }

    pub fn supports_references(&self) -> bool {
        self.variant() == H3Variant::Ref2va
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VaeConfig {
    pub ch: usize,
    pub ch_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub z_channels: usize,
    pub embed_dim: usize,
    pub in_channels: usize,
    pub out_ch: usize,
    pub space_down: Vec<usize>,
    pub space_up: Vec<usize>,
    pub time_down: Vec<usize>,
    pub vae_ratio: usize,
    pub vae_ratio_t: usize,
    pub causal_encoder: bool,
    pub causal_decoder: bool,
    pub use_vit_decoder: bool,
    pub use_t_isolated_gn: bool,
    pub padding_mode: String,
    pub pixel_norm_type: String,
    pub vit_decoder_kwargs: VitDecoderConfig,
    #[serde(skip)]
    pub clip_length: usize,
    #[serde(skip)]
    pub token_drop: usize,
    #[serde(skip)]
    pub tile_size: usize,
    #[serde(skip)]
    pub tile_overlap_min: usize,
    #[serde(skip)]
    pub latents_mean: Vec<f32>,
    #[serde(skip)]
    pub latents_std: Vec<f32>,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            ch: 128,
            ch_mult: vec![1, 2, 2, 4, 4, 8],
            num_res_blocks: 2,
            z_channels: 24,
            embed_dim: 24,
            in_channels: 3,
            out_ch: 3,
            space_down: vec![2, 2, 2, 2, 1, 1],
            space_up: vec![1, 2, 2, 2, 2, 1],
            time_down: vec![1, 2, 2, 1, 1, 1],
            vae_ratio: 16,
            vae_ratio_t: 4,
            causal_encoder: true,
            causal_decoder: false,
            use_vit_decoder: true,
            use_t_isolated_gn: true,
            padding_mode: "reflect".into(),
            pixel_norm_type: "imagenet".into(),
            vit_decoder_kwargs: VitDecoderConfig::default(),
            clip_length: 17,
            token_drop: 3,
            tile_size: 256,
            tile_overlap_min: 64,
            latents_mean: Vec::new(),
            latents_std: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VitDecoderConfig {
    pub num_layers: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub rope_theta: f32,
    pub rope_dim_ratio: f32,
    pub norm_type: String,
    pub qk_norm_type: String,
    pub qk_norm_affine: bool,
    pub norm_affine: bool,
    pub ffn_activation_fn: String,
    pub ffn_use_gated: bool,
    #[serde(skip)]
    pub num_register_tokens: usize,
    #[serde(skip)]
    pub patch_size: usize,
    #[serde(skip)]
    pub patch_size_t: usize,
    #[serde(skip)]
    pub eps: f32,
}

impl Default for VitDecoderConfig {
    fn default() -> Self {
        Self {
            num_layers: 36,
            heads: 32,
            dim_head: 64,
            rope_theta: 100.0,
            rope_dim_ratio: 0.75,
            norm_type: "rms_norm".into(),
            qk_norm_type: "rms_norm".into(),
            qk_norm_affine: false,
            norm_affine: true,
            ffn_activation_fn: "silu".into(),
            ffn_use_gated: true,
            num_register_tokens: 4,
            patch_size: 16,
            patch_size_t: 4,
            eps: 1e-5,
        }
    }
}

impl VitDecoderConfig {
    pub fn dim(&self) -> usize {
        self.heads * self.dim_head
    }

    pub fn rope_dim(&self) -> usize {
        (self.dim_head as f32 * self.rope_dim_ratio) as usize
    }
}

impl VaeConfig {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, H3Error> {
        let dir = dir.as_ref().join("video_vae");
        let src = dir.join("source").join("config.json");
        let bytes = std::fs::read(&src)
            .map_err(|e| H3Error::Config(format!("{}: {e}", src.display())))?;
        let mut cfg: Self = serde_json::from_slice(&bytes)
            .map_err(|e| H3Error::Config(format!("video_vae/source/config.json: {e}")))?;

        let outer = dir.join("config.json");
        let bytes = std::fs::read(&outer)
            .map_err(|e| H3Error::Config(format!("{}: {e}", outer.display())))?;
        let root: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| H3Error::Config(format!("video_vae/config.json: {e}")))?;
        let num = |k: &str, d: usize| {
            root.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(d)
        };
        cfg.clip_length = num("vae_clip_length", 17);
        cfg.token_drop = num("vae_token_drop", 3);
        cfg.tile_size = num("vae_tile_size", 256);
        cfg.tile_overlap_min = num("vae_tile_overlap_min", 64);
        cfg.latents_mean = read_f32_array(&root, "latents_mean")?;
        cfg.latents_std = read_f32_array(&root, "latents_std")?;
        if cfg.latents_mean.len() != cfg.z_channels || cfg.latents_std.len() != cfg.z_channels {
            return Err(H3Error::Config("latents_mean/std: длина != z_channels".into()));
        }
        cfg.vit_decoder_kwargs.eps = 1e-5;
        Ok(cfg)
    }

    pub fn num_stages(&self) -> usize {
        self.ch_mult.len()
    }

    pub fn tokens_chunk_size(&self) -> usize {
        self.clip_length.div_ceil(self.vae_ratio_t)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AudioVaeConfig {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub decoder_kernel_sizes: Vec<usize>,
    pub vae_latent_channels: usize,
    pub num_attention_heads: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub sampling_rate: usize,
}

impl Default for AudioVaeConfig {
    fn default() -> Self {
        Self {
            encoder_dim: 64,
            encoder_rates: vec![2, 4, 4, 5, 5],
            latent_dim: 2048,
            decoder_dim: 1024,
            decoder_rates: vec![5, 5, 2, 2, 2, 2, 2],
            decoder_kernel_sizes: vec![9, 9, 4, 4, 4, 4, 4],
            vae_latent_channels: 32,
            num_attention_heads: 8,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            sampling_rate: AUDIO_SAMPLE_RATE,
        }
    }
}

impl AudioVaeConfig {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, H3Error> {
        let path = dir.as_ref().join("audio_vae").join("config.json");
        let bytes = std::fs::read(&path)
            .map_err(|e| H3Error::Config(format!("{}: {e}", path.display())))?;
        let mut cfg: Self = serde_json::from_slice(&bytes)
            .map_err(|e| H3Error::Config(format!("audio_vae/config.json: {e}")))?;
        if let Some(v) = read_opt_usize(&bytes, "vae_latent_channels")? {
            cfg.vae_latent_channels = v;
        } else if let Some(v) = read_opt_usize(&bytes, "latent_channels")? {
            cfg.vae_latent_channels = v;
        }
        Ok(cfg)
    }

    pub fn hop_length(&self) -> usize {
        self.encoder_rates.iter().product()
    }

    pub fn latents_per_second(&self) -> usize {
        self.sampling_rate / self.hop_length()
    }
}

fn read_opt_usize(bytes: &[u8], key: &str) -> Result<Option<usize>, H3Error> {
    let root: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| H3Error::Config(e.to_string()))?;
    Ok(root.get(key).and_then(|v| v.as_u64()).map(|v| v as usize))
}

fn read_f32_array(root: &serde_json::Value, key: &str) -> Result<Vec<f32>, H3Error> {
    let arr = root
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| H3Error::Config(format!("отсутствует {key}")))?;
    arr.iter()
        .map(|v| {
            v.as_f64()
                .map(|f| f as f32)
                .ok_or_else(|| H3Error::Config(format!("{key}: не число")))
        })
        .collect()
}

pub fn snap_frame_count(frames: usize) -> usize {
    let f = frames.max(FRAME_GRID_BASE);
    let k = (f - FRAME_GRID_BASE).div_ceil(FRAME_GRID_STEP);
    FRAME_GRID_STEP * k + FRAME_GRID_BASE
}

pub fn frames_for_duration(seconds: f64) -> usize {
    snap_frame_count((seconds * FPS).round() as usize)
}

pub fn latent_frames(frame_count: usize) -> usize {
    frame_count.div_ceil(VAE_TEMPORAL_RATIO)
}

pub fn audio_latent_frames(frame_count: usize) -> usize {
    let seconds = frame_count as f64 / FPS;
    (seconds * AUDIO_LATENTS_PER_SECOND as f64).round() as usize
}

pub fn latent_grid(width: usize, height: usize) -> (usize, usize) {
    (height.div_ceil(VAE_SPATIAL_RATIO), width.div_ceil(VAE_SPATIAL_RATIO))
}
