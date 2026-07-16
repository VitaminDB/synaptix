//! Конфиги OmniVoice (парсятся из `.syn`: `config.json` + `audio_tokenizer/config.json`).
//! Значения по умолчанию — из `omnivoice.syn` (см. SPEC.md «Конфиг»).

use serde::Deserialize;

use synaptix_bundle::Bundle;

use crate::OmniVoiceError;

/// Qwen3-бэкбон (используется двунаправленно). Подмножество `llm_config` из config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct BackboneConfig {
    #[serde(default = "d_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "d_head_dim")]
    pub head_dim: usize,
    #[serde(default = "d_inter")]
    pub intermediate_size: usize,
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "d_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "d_tie")]
    pub tie_word_embeddings: bool,
}

fn d_hidden() -> usize { 1024 }
fn d_layers() -> usize { 28 }
fn d_heads() -> usize { 16 }
fn d_kv_heads() -> usize { 8 }
fn d_head_dim() -> usize { 128 }
fn d_inter() -> usize { 3072 }
fn d_vocab() -> usize { 151676 }
fn d_rms_eps() -> f64 { 1e-6 }
fn d_max_pos() -> usize { 40960 }
fn d_tie() -> bool { true }

/// rope_theta лежит в `llm_config.rope_parameters.rope_theta`.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "d_rope_theta")]
    pub rope_theta: f64,
}
fn d_rope_theta() -> f64 { 1_000_000.0 }

#[derive(Debug, Clone, Deserialize)]
struct LlmConfigRaw {
    #[serde(flatten)]
    backbone: BackboneConfig,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Clone, Deserialize)]
struct OmniVoiceConfigRaw {
    #[serde(default = "d_audio_vocab")]
    audio_vocab_size: usize,
    #[serde(default = "d_audio_mask")]
    audio_mask_id: usize,
    #[serde(default = "d_num_cb")]
    num_audio_codebook: usize,
    #[serde(default = "d_cb_weights")]
    audio_codebook_weights: Vec<f32>,
    #[serde(default = "d_eos")]
    eos_token_id: u32,
    #[serde(default = "d_pad")]
    pad_token_id: u32,
    llm_config: LlmConfigRaw,
}

fn d_audio_vocab() -> usize { 1025 }
fn d_audio_mask() -> usize { 1024 }
fn d_num_cb() -> usize { 8 }
fn d_cb_weights() -> Vec<f32> { vec![8.0, 8.0, 6.0, 6.0, 4.0, 4.0, 2.0, 2.0] }
fn d_eos() -> u32 { 151645 }
fn d_pad() -> u32 { 151643 }

#[derive(Debug, Clone)]
pub struct OmniVoiceConfig {
    pub audio_vocab_size: usize,
    pub audio_mask_id: usize,
    pub num_audio_codebook: usize,
    pub audio_codebook_weights: Vec<f32>,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub backbone: BackboneConfig,
    pub rope_theta: f64,
}

impl OmniVoiceConfig {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, OmniVoiceError> {
        let raw: OmniVoiceConfigRaw =
            serde_json::from_slice(bytes).map_err(|e| OmniVoiceError::Config(e.to_string()))?;
        let rope_theta = raw
            .llm_config
            .rope_parameters
            .map(|r| r.rope_theta)
            .unwrap_or_else(d_rope_theta);
        Ok(Self {
            audio_vocab_size: raw.audio_vocab_size,
            audio_mask_id: raw.audio_mask_id,
            num_audio_codebook: raw.num_audio_codebook,
            audio_codebook_weights: raw.audio_codebook_weights,
            eos_token_id: raw.eos_token_id,
            pad_token_id: raw.pad_token_id,
            backbone: raw.llm_config.backbone,
            rope_theta,
        })
    }

    pub fn from_bundle(bundle: &Bundle) -> Result<Self, OmniVoiceError> {
        let bytes = bundle
            .read_file("config.json")
            .map_err(|e| OmniVoiceError::Bundle(e.to_string()))?;
        Self::from_json_bytes(&bytes)
    }
}

/// HiggsAudioV2 codec (`audio_tokenizer/config.json`). DAC-акустика + HuBERT-semantic.
#[derive(Debug, Clone, Deserialize)]
pub struct AcousticConfig {
    #[serde(default = "d_ac_enc_hidden")]
    pub encoder_hidden_size: usize,
    #[serde(default = "d_ac_dec_hidden")]
    pub decoder_hidden_size: usize,
    #[serde(default = "d_ac_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_ac_ratios")]
    pub downsampling_ratios: Vec<usize>,
    #[serde(default = "d_ac_ratios")]
    pub upsampling_ratios: Vec<usize>,
    #[serde(default = "d_ac_hop")]
    pub hop_length: usize,
    #[serde(default = "d_ac_ncb")]
    pub n_codebooks: usize,
    #[serde(default = "d_ac_cbsize")]
    pub codebook_size: usize,
    #[serde(default = "d_ac_sr")]
    pub sampling_rate: usize,
}

fn d_ac_enc_hidden() -> usize { 64 }
fn d_ac_dec_hidden() -> usize { 1024 }
fn d_ac_hidden() -> usize { 256 }
fn d_ac_ratios() -> Vec<usize> { vec![8, 5, 4, 2, 3] }
fn d_ac_hop() -> usize { 960 }
fn d_ac_ncb() -> usize { 9 }
fn d_ac_cbsize() -> usize { 1024 }
fn d_ac_sr() -> usize { 16000 }

/// HuBERT semantic-model (`semantic_model_config`). Подмножество для encode-пути.
#[derive(Debug, Clone, Deserialize)]
pub struct SemanticConfig {
    #[serde(default = "d_sem_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_sem_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_sem_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_sem_inter")]
    pub intermediate_size: usize,
    #[serde(default = "d_sem_conv_dim")]
    pub conv_dim: Vec<usize>,
    #[serde(default = "d_sem_conv_kernel")]
    pub conv_kernel: Vec<usize>,
    #[serde(default = "d_sem_conv_stride")]
    pub conv_stride: Vec<usize>,
    #[serde(default = "d_sem_conv_bias")]
    pub conv_bias: bool,
    #[serde(default = "d_sem_pos_kernel")]
    pub num_conv_pos_embeddings: usize,
    #[serde(default = "d_sem_pos_groups")]
    pub num_conv_pos_embedding_groups: usize,
    #[serde(default = "d_sem_ln_eps")]
    pub layer_norm_eps: f64,
}

fn d_sem_hidden() -> usize { 768 }
fn d_sem_layers() -> usize { 12 }
fn d_sem_heads() -> usize { 12 }
fn d_sem_inter() -> usize { 3072 }
fn d_sem_conv_dim() -> Vec<usize> { vec![512, 512, 512, 512, 512, 512, 512] }
fn d_sem_conv_kernel() -> Vec<usize> { vec![10, 3, 3, 3, 3, 2, 2] }
fn d_sem_conv_stride() -> Vec<usize> { vec![5, 2, 2, 2, 2, 2, 2] }
fn d_sem_conv_bias() -> bool { false }
fn d_sem_pos_kernel() -> usize { 128 }
fn d_sem_pos_groups() -> usize { 16 }
fn d_sem_ln_eps() -> f64 { 1e-5 }

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            hidden_size: d_sem_hidden(),
            num_hidden_layers: d_sem_layers(),
            num_attention_heads: d_sem_heads(),
            intermediate_size: d_sem_inter(),
            conv_dim: d_sem_conv_dim(),
            conv_kernel: d_sem_conv_kernel(),
            conv_stride: d_sem_conv_stride(),
            conv_bias: d_sem_conv_bias(),
            num_conv_pos_embeddings: d_sem_pos_kernel(),
            num_conv_pos_embedding_groups: d_sem_pos_groups(),
            layer_norm_eps: d_sem_ln_eps(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct HiggsAudioConfig {
    #[serde(default = "d_hi_sr")]
    pub sample_rate: usize,
    #[serde(default = "d_hi_downsample")]
    pub downsample_factor: usize,
    #[serde(default = "d_hi_cbdim")]
    pub codebook_dim: usize,
    #[serde(default = "d_hi_cbsize")]
    pub codebook_size: usize,
    #[serde(default = "d_hi_semsr")]
    pub semantic_sample_rate: usize,
    #[serde(default = "d_hi_bandwidths")]
    pub target_bandwidths: Vec<f32>,
    pub acoustic_model_config: AcousticConfig,
    #[serde(default)]
    pub semantic_model_config: SemanticConfig,
}

fn d_hi_sr() -> usize { 24000 }
fn d_hi_downsample() -> usize { 320 }
fn d_hi_cbdim() -> usize { 64 }
fn d_hi_cbsize() -> usize { 1024 }
fn d_hi_semsr() -> usize { 16000 }
fn d_hi_bandwidths() -> Vec<f32> { vec![0.5, 1.0, 1.5, 2.0] }

impl HiggsAudioConfig {
    /// hop_length = ∏ downsampling_ratios.
    pub fn hop_length(&self) -> usize {
        self.acoustic_model_config.downsampling_ratios.iter().product()
    }

    /// frame_rate = ceil(sample_rate / hop_length).
    pub fn frame_rate(&self) -> usize {
        self.sample_rate.div_ceil(self.hop_length())
    }

    /// codebook_nbits = ceil(log2(codebook_size)).
    pub fn codebook_nbits(&self) -> usize {
        let cs = self.codebook_size.max(1);
        (usize::BITS - (cs - 1).leading_zeros()) as usize
    }

    /// semantic_downsample_factor = hop / (sr/sem_sr) / downsample_factor.
    pub fn semantic_downsample_factor(&self) -> usize {
        let r = self.sample_rate as f64 / self.semantic_sample_rate as f64;
        (self.hop_length() as f64 / r / self.downsample_factor as f64) as usize
    }

    /// n_q для encode (bandwidth = target_bandwidths[-1]): floor(bw / bw_per_q),
    /// bw_per_q = nbits·frame_rate/1000.
    pub fn num_quantizers_for_encode(&self) -> usize {
        let bw = *self.target_bandwidths.last().unwrap_or(&2.0) as f64;
        let bw_per_q = self.codebook_nbits() as f64 * self.frame_rate() as f64 / 1000.0;
        if bw > 0.0 {
            ((bw / bw_per_q).floor() as usize).max(1)
        } else {
            self.acoustic_model_config.n_codebooks
        }
    }

    pub fn from_bundle(bundle: &Bundle) -> Result<Self, OmniVoiceError> {
        let bytes = bundle
            .read_file("audio_tokenizer/config.json")
            .map_err(|e| OmniVoiceError::Bundle(e.to_string()))?;
        Self::from_json_bytes(&bytes)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, OmniVoiceError> {
        serde_json::from_slice(bytes).map_err(|e| OmniVoiceError::Config(e.to_string()))
    }
}

/// Параметры генерации (defaults = upstream `OmniVoiceGenerationConfig`).
#[derive(Debug, Clone)]
pub struct OmniVoiceGenerationConfig {
    pub num_step: usize,
    pub guidance_scale: f32,
    pub t_shift: f32,
    pub layer_penalty_factor: f32,
    pub position_temperature: f32,
    pub class_temperature: f32,
    pub denoise: bool,
    pub preprocess_prompt: bool,
    pub postprocess_output: bool,
    pub audio_chunk_duration: f32,
    pub audio_chunk_threshold: f32,
}

impl Default for OmniVoiceGenerationConfig {
    fn default() -> Self {
        Self {
            num_step: 32,
            guidance_scale: 2.0,
            t_shift: 0.1,
            layer_penalty_factor: 5.0,
            position_temperature: 5.0,
            class_temperature: 0.0,
            denoise: true,
            preprocess_prompt: true,
            postprocess_output: true,
            audio_chunk_duration: 15.0,
            audio_chunk_threshold: 30.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNPACK: &str = "tmp/ov_unpack";

    #[test]
    fn parse_real_omnivoice_config() {
        let p = format!("{UNPACK}/config.json");
        let Ok(bytes) = std::fs::read(&p) else {
            return;
        };
        let cfg = OmniVoiceConfig::from_json_bytes(&bytes).expect("parse config.json");
        assert_eq!(cfg.audio_vocab_size, 1025);
        assert_eq!(cfg.audio_mask_id, 1024);
        assert_eq!(cfg.num_audio_codebook, 8);
        assert_eq!(cfg.backbone.hidden_size, 1024);
        assert_eq!(cfg.backbone.num_hidden_layers, 28);
        assert_eq!(cfg.backbone.num_attention_heads, 16);
        assert_eq!(cfg.backbone.num_key_value_heads, 8);
        assert_eq!(cfg.backbone.head_dim, 128);
        assert_eq!(cfg.backbone.vocab_size, 151676);
        assert!(cfg.backbone.tie_word_embeddings);
        assert_eq!(cfg.rope_theta, 1_000_000.0);
    }

    #[test]
    fn parse_real_higgs_config() {
        let p = format!("{UNPACK}/audio_tokenizer/config.json");
        let Ok(bytes) = std::fs::read(&p) else {
            return;
        };
        let cfg: HiggsAudioConfig =
            serde_json::from_slice(&bytes).expect("parse audio_tokenizer/config.json");
        assert_eq!(cfg.sample_rate, 24000);
        assert_eq!(cfg.downsample_factor, 320);
        assert_eq!(cfg.acoustic_model_config.hop_length, 960);
        assert_eq!(cfg.acoustic_model_config.n_codebooks, 9);
        assert_eq!(cfg.acoustic_model_config.downsampling_ratios, vec![8, 5, 4, 2, 3]);
    }
}
