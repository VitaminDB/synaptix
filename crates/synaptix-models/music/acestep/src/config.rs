
use crate::AceError;

#[derive(Debug, Clone)]
pub struct DitConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub in_channels: usize,
    pub audio_acoustic_hidden_dim: usize,
    pub patch_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    pub encoder_hidden_size: usize,
    pub encoder_num_attention_heads: usize,
    pub encoder_num_key_value_heads: usize,
    pub encoder_intermediate_size: usize,
    pub num_lyric_encoder_hidden_layers: usize,
    pub num_timbre_encoder_hidden_layers: usize,
    pub num_attention_pooler_hidden_layers: usize,
    pub pool_window_size: usize,
    pub text_hidden_dim: usize,
    pub timbre_hidden_dim: usize,
    pub timbre_fix_frame: usize,
    pub fsq_dim: usize,
    pub fsq_input_levels: Vec<usize>,
    pub vocab_size: usize,
    pub is_turbo: bool,
}

impl DitConfig {
    pub fn xl_base() -> Self {
        Self {
            hidden_size: 2560,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 9728,
            in_channels: 192,
            audio_acoustic_hidden_dim: 64,
            patch_size: 2,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            sliding_window: 128,
            encoder_hidden_size: 2048,
            encoder_num_attention_heads: 16,
            encoder_num_key_value_heads: 8,
            encoder_intermediate_size: 6144,
            num_lyric_encoder_hidden_layers: 8,
            num_timbre_encoder_hidden_layers: 4,
            num_attention_pooler_hidden_layers: 2,
            pool_window_size: 5,
            text_hidden_dim: 1024,
            timbre_hidden_dim: 64,
            timbre_fix_frame: 750,
            fsq_dim: 2048,
            fsq_input_levels: vec![8, 8, 8, 5, 5, 5],
            vocab_size: 64003,
            is_turbo: false,
        }
    }

    pub fn xl_turbo() -> Self {
        Self {
            hidden_size: 2048,
            num_attention_heads: 16,
            intermediate_size: 6144,
            is_turbo: true,
            ..Self::xl_base()
        }
    }

    pub fn n_rep(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn layer_is_sliding(&self, idx: usize) -> bool {
        idx % 2 == 0
    }
}

#[derive(Debug, Clone)]
pub struct LmConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

impl LmConfig {
    pub fn lm_1_7b() -> Self {
        Self {
            hidden_size: 2048,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 6144,
            max_position_embeddings: 40960,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            vocab_size: 217204,
            bos_token_id: 151643,
            eos_token_id: 151645,
        }
    }

    /// Конфиг из HF `config.json`, лежащего в .syn-бандле рядом с весами.
    ///
    /// Одним загрузчиком читаются оба 5Hz-LM: `acestep_5hz_lm_1.7b.syn`
    /// (Qwen3Model: 28 слоёв, hidden 2048, 16 голов) и `acestep_5hz_lm_4b.syn`
    /// (Qwen3ForCausalLM: 36 слоёв, hidden 2560, 32 головы, intermediate 9728).
    /// Поля, которых в json нет, берутся из [`Self::lm_1_7b`]; `eos_token_id`
    /// может быть списком — берём первый.
    pub fn from_hf_json(bytes: &[u8]) -> Result<Self, AceError> {
        let v: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| AceError::Config(format!("config.json: {e}")))?;
        let base = Self::lm_1_7b();
        let num = |k: &str| -> Option<u64> {
            let x = v.get(k)?;
            x.as_u64().or_else(|| x.as_array()?.first()?.as_u64())
        };
        let u = |k: &str, d: usize| num(k).map(|x| x as usize).unwrap_or(d);
        let f = |k: &str, d: f32| v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32).unwrap_or(d);
        let hidden_size = u("hidden_size", base.hidden_size);
        let num_attention_heads = u("num_attention_heads", base.num_attention_heads).max(1);
        Ok(Self {
            hidden_size,
            num_hidden_layers: u("num_hidden_layers", base.num_hidden_layers),
            num_attention_heads,
            num_key_value_heads: u("num_key_value_heads", base.num_key_value_heads),
            head_dim: u("head_dim", hidden_size / num_attention_heads),
            intermediate_size: u("intermediate_size", base.intermediate_size),
            max_position_embeddings: u("max_position_embeddings", base.max_position_embeddings),
            rope_theta: f("rope_theta", base.rope_theta),
            rms_norm_eps: f("rms_norm_eps", base.rms_norm_eps),
            vocab_size: u("vocab_size", base.vocab_size),
            bos_token_id: u("bos_token_id", base.bos_token_id as usize) as u32,
            eos_token_id: u("eos_token_id", base.eos_token_id as usize) as u32,
        })
    }
}

#[derive(Debug, Clone)]
pub struct VaeConfig {
    pub audio_channels: usize,
    pub sampling_rate: usize,
    pub encoder_hidden_size: usize,
    pub decoder_channels: usize,
    pub decoder_input_channels: usize,
    pub channel_multiples: Vec<usize>,
    pub downsampling_ratios: Vec<usize>,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            audio_channels: 2,
            sampling_rate: 48000,
            encoder_hidden_size: 128,
            decoder_channels: 128,
            decoder_input_channels: 64,
            channel_multiples: vec![1, 2, 4, 8, 16],
            downsampling_ratios: vec![2, 4, 4, 6, 10],
        }
    }
}

impl VaeConfig {
    pub fn hop_length(&self) -> usize {
        self.downsampling_ratios.iter().product()
    }
}
