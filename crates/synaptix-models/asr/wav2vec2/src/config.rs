//! Конфиг Wav2Vec2 из HF `config.json` (поля `Wav2Vec2Config`).

use serde::Deserialize;

use crate::Wav2Vec2Error;

fn default_eps() -> f64 {
    1e-5
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct Wav2Vec2Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub conv_dim: Vec<usize>,
    pub conv_kernel: Vec<usize>,
    pub conv_stride: Vec<usize>,
    #[serde(default = "default_true")]
    pub conv_bias: bool,
    /// `"layer"` — LayerNorm после каждой свёртки; `"group"` — GroupNorm
    /// (num_groups = channels) только после первой.
    pub feat_extract_norm: String,
    #[serde(default)]
    pub feat_extract_activation: Option<String>,
    #[serde(default)]
    pub hidden_act: Option<String>,
    pub num_conv_pos_embeddings: usize,
    pub num_conv_pos_embedding_groups: usize,
    /// pre-LN-трансформер (XLS-R, large-lv60) vs post-LN (base).
    #[serde(default)]
    pub do_stable_layer_norm: bool,
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f64,
    pub vocab_size: usize,
    #[serde(default)]
    pub pad_token_id: Option<usize>,
    /// `wav2vec2` | `hubert` | `wavlm` (у WavLM — относительный позиционный
    /// bias с гейтом в каждом слое).
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default = "default_buckets")]
    pub num_buckets: usize,
    #[serde(default = "default_bucket_distance")]
    pub max_bucket_distance: usize,
}

fn default_buckets() -> usize {
    320
}

fn default_bucket_distance() -> usize {
    800
}

impl Wav2Vec2Config {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, Wav2Vec2Error> {
        let cfg: Self =
            serde_json::from_slice(bytes).map_err(|e| Wav2Vec2Error::Config(e.to_string()))?;
        let n = cfg.conv_dim.len();
        if cfg.conv_kernel.len() != n || cfg.conv_stride.len() != n || n == 0 {
            return Err(Wav2Vec2Error::Config(
                "conv_dim / conv_kernel / conv_stride разной длины".into(),
            ));
        }
        if !matches!(cfg.feat_extract_norm.as_str(), "layer" | "group") {
            return Err(Wav2Vec2Error::Config(format!(
                "feat_extract_norm={:?} (ожидается \"layer\" или \"group\")",
                cfg.feat_extract_norm
            )));
        }
        for act in [&cfg.feat_extract_activation, &cfg.hidden_act].into_iter().flatten() {
            if act != "gelu" {
                return Err(Wav2Vec2Error::Config(format!("активация {act:?} не поддержана")));
            }
        }
        if cfg.hidden_size % cfg.num_attention_heads != 0 {
            return Err(Wav2Vec2Error::Config("hidden_size % num_attention_heads != 0".into()));
        }
        Ok(cfg)
    }

    pub fn is_wavlm(&self) -> bool {
        self.model_type.as_deref() == Some("wavlm")
    }

    /// Префикс энкодера в именах весов HF (`wav2vec2.`, `wavlm.`, `hubert.`).
    pub fn weight_prefix(&self) -> &'static str {
        match self.model_type.as_deref() {
            Some("wavlm") => "wavlm.",
            Some("hubert") => "hubert.",
            _ => "wav2vec2.",
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// CTC blank — токен `<pad>` (HF: `pad_token_id`, по умолчанию 0).
    pub fn blank_id(&self) -> usize {
        self.pad_token_id.unwrap_or(0)
    }

    /// Число кадров энкодера для `n_samples` сэмплов (свёртки без паддинга).
    pub fn frames_for(&self, n_samples: usize) -> usize {
        let mut len = n_samples;
        for (&k, &s) in self.conv_kernel.iter().zip(&self.conv_stride) {
            if len < k {
                return 0;
            }
            len = (len - k) / s + 1;
        }
        len
    }

    /// Шаг кадра в сэмплах (произведение stride'ов; 320 = 20 мс при 16 кГц).
    pub fn hop_samples(&self) -> usize {
        self.conv_stride.iter().product()
    }
}
