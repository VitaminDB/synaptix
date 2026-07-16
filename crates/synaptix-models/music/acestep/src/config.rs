
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
