//! Конфигурация Sortformer-streaming-4spk-v2.1.
//!
//! Defaults соответствуют NVIDIA NeMo config'у v2.1 (портировано дословно из
//! официального NVIDIA NeMo config v2.1). `.syn`-бандл может
//! не содержать `config.json` — тогда берутся hardcoded-defaults v2.1.

use serde::{Deserialize, Serialize};
use synaptix_bundle::Bundle;

use crate::SortformerError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortformerConfig {
    pub model_name: String,
    pub sample_rate: usize,
    pub max_speakers: usize,
    pub frame_rate_hz: f32,
    pub preprocessor: NemoPreprocessorConfig,
    pub encoder: FastConformerConfig,
    pub head: SortformerHeadConfig,
    #[serde(default)]
    pub streaming: StreamingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NemoPreprocessorConfig {
    pub sample_rate: usize,
    /// Длина окна STFT в сэмплах (NeMo: 400 при 25мс @ 16kHz).
    pub n_window_size: usize,
    /// Шаг между окнами (NeMo: 160 при 10мс).
    pub n_window_stride: usize,
    pub n_fft: usize,
    pub n_mels: usize,
    pub log: bool,
    pub dither: f32,
    pub preemph: f32,
    /// `"per_feature"` (стандарт NeMo) или `"NA"`/`"none"` (v2.1 — без нормализации).
    pub normalize: String,
    #[serde(default)]
    pub pad_to: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastConformerConfig {
    pub feat_in: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    /// Тип subsampling: `"dw_striding"` для FastConformer 8x.
    pub subsampling: String,
    pub subsampling_factor: usize,
    pub subs_kernel_size: usize,
    pub ff_expansion_factor: usize,
    /// `"rel_pos"` (T5-XL) или `"rotary"` (legacy).
    pub self_attention_model: String,
    pub pos_emb_max_len: usize,
    pub conv_kernel_size: usize,
    /// `"batch_norm"` (FastConformer-default) или `"layer_norm"`.
    pub conv_norm_type: String,
    #[serde(default = "default_subsampling_conv_channels")]
    pub subsampling_conv_channels: usize,
    /// `x = x * sqrt(d_model)` после pre_encode. NeMo default = true.
    #[serde(default = "default_xscaling")]
    pub xscaling: bool,
}

fn default_subsampling_conv_channels() -> usize {
    256
}
fn default_xscaling() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortformerHeadConfig {
    /// Размер входных эмбеддингов из энкодера (= encoder.d_model).
    pub feat_in: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    /// Размер FFN внутри transformer-слоя head'а (обычно 4*d_model).
    pub ff_expansion_factor: usize,
    pub max_speakers: usize,
    #[serde(default)]
    pub dropout: f32,
}

/// Параметры streaming-инференса из NeMo `sortformer_modules`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    #[serde(default = "default_spkcache_len")]
    pub spkcache_len: usize,
    #[serde(default)]
    pub fifo_len: usize,
    #[serde(default = "default_chunk_len")]
    pub chunk_len: usize,
    #[serde(default = "default_spkcache_update_period")]
    pub spkcache_update_period: usize,
    #[serde(default = "default_chunk_context")]
    pub chunk_left_context: usize,
    #[serde(default = "default_chunk_context")]
    pub chunk_right_context: usize,
    #[serde(default = "default_spkcache_sil")]
    pub spkcache_sil_frames_per_spk: usize,
    #[serde(default = "default_pred_score_threshold")]
    pub pred_score_threshold: f32,
    #[serde(default = "default_scores_boost_latest")]
    pub scores_boost_latest: f32,
    #[serde(default = "default_sil_threshold")]
    pub sil_threshold: f32,
    #[serde(default = "default_strong_boost_rate")]
    pub strong_boost_rate: f32,
    #[serde(default = "default_weak_boost_rate")]
    pub weak_boost_rate: f32,
    #[serde(default = "default_min_pos_scores_rate")]
    pub min_pos_scores_rate: f32,
    #[serde(default = "default_max_index")]
    pub max_index: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            spkcache_len: default_spkcache_len(),
            fifo_len: 0,
            chunk_len: default_chunk_len(),
            spkcache_update_period: default_spkcache_update_period(),
            chunk_left_context: default_chunk_context(),
            chunk_right_context: default_chunk_context(),
            spkcache_sil_frames_per_spk: default_spkcache_sil(),
            pred_score_threshold: default_pred_score_threshold(),
            scores_boost_latest: default_scores_boost_latest(),
            sil_threshold: default_sil_threshold(),
            strong_boost_rate: default_strong_boost_rate(),
            weak_boost_rate: default_weak_boost_rate(),
            min_pos_scores_rate: default_min_pos_scores_rate(),
            max_index: default_max_index(),
        }
    }
}

fn default_spkcache_len() -> usize { 188 }
fn default_chunk_len() -> usize { 188 }
fn default_spkcache_update_period() -> usize { 188 }
fn default_chunk_context() -> usize { 1 }
fn default_spkcache_sil() -> usize { 3 }
fn default_pred_score_threshold() -> f32 { 0.25 }
fn default_scores_boost_latest() -> f32 { 0.05 }
fn default_sil_threshold() -> f32 { 0.2 }
fn default_strong_boost_rate() -> f32 { 0.75 }
fn default_weak_boost_rate() -> f32 { 1.5 }
fn default_min_pos_scores_rate() -> f32 { 0.5 }
fn default_max_index() -> usize { 99999 }

impl SortformerConfig {
    /// Hardcoded defaults для diar_streaming_sortformer_4spk-v2.1 (NeMo model_config.yaml).
    pub fn streaming_4spk_v21_default() -> Self {
        Self {
            model_name: "sortformer-streaming-4spk-v2.1".to_string(),
            sample_rate: 16000,
            max_speakers: 4,
            // 8x downsampling: 100 Hz (STFT hop=160) / 8 = 12.5 Hz.
            frame_rate_hz: 12.5,
            preprocessor: NemoPreprocessorConfig {
                sample_rate: 16000,
                n_window_size: 400,
                n_window_stride: 160,
                n_fft: 512,
                n_mels: 128,
                log: true,
                dither: 1e-5,
                preemph: 0.97,
                normalize: "NA".to_string(),
                pad_to: 0,
            },
            encoder: FastConformerConfig {
                feat_in: 128,
                n_layers: 17,
                d_model: 512,
                n_heads: 8,
                subsampling: "dw_striding".to_string(),
                subsampling_factor: 8,
                subs_kernel_size: 9,
                ff_expansion_factor: 4,
                self_attention_model: "rel_pos".to_string(),
                pos_emb_max_len: 5000,
                conv_kernel_size: 9,
                conv_norm_type: "batch_norm".to_string(),
                subsampling_conv_channels: 256,
                xscaling: true,
            },
            head: SortformerHeadConfig {
                feat_in: 512,
                n_layers: 18,
                d_model: 192,
                n_heads: 8,
                ff_expansion_factor: 4,
                max_speakers: 4,
                dropout: 0.0,
            },
            streaming: StreamingConfig::default(),
        }
    }

    /// Парсинг `config.json` (если присутствует в бандле).
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, SortformerError> {
        serde_json::from_slice(bytes).map_err(|e| SortformerError::Config(e.to_string()))
    }

    /// Конфиг из `.syn`-бандла: `config.json`, иначе hardcoded v2.1-defaults.
    pub fn from_bundle(bundle: &Bundle) -> Result<Self, SortformerError> {
        match bundle.read_file("config.json") {
            Ok(bytes) => Self::from_json_bytes(&bytes),
            Err(_) => Ok(Self::streaming_4spk_v21_default()),
        }
    }

    /// Размерность головы attention энкодера (d_model / n_heads).
    pub fn d_k(&self) -> usize {
        self.encoder.d_model / self.encoder.n_heads
    }

    /// Размерность FFN энкодера (d_model * ff_expansion_factor).
    pub fn d_ff(&self) -> usize {
        self.encoder.d_model * self.encoder.ff_expansion_factor
    }
}
