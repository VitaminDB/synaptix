//! Конфигурации SheetSage2 и его энкодера MERT2 — из `sheetsage2/config.json`
//! в бандле (тот же `config.json` релиза: энкодер лежит в `backbone_config`).

use serde_json::Value;

use crate::SheetError;

fn usize_at(v: &Value, key: &str, default: usize) -> usize {
    v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(default)
}

fn f64_at(v: &Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
}

fn list_at(v: &Value, key: &str, default: &[usize]) -> Vec<usize> {
    v.get(key)
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect::<Vec<_>>())
        .filter(|a: &Vec<usize>| !a.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

/// Энкодер MERT-v2 (`model_type: "mert2"`, вариант `fs` — песня целиком).
#[derive(Debug, Clone, PartialEq)]
pub struct Mert2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_mel_bins: usize,
    pub sampling_rate: u32,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    /// Ширины трёх ступеней ConvNeXt-сабсэмплера (последняя = `hidden_size`).
    pub subsampling_channels: Vec<usize>,
    pub subsampling_depths: Vec<usize>,
    pub conv_depthwise_kernel_size: usize,
    pub rotary_embedding_base: f32,
    pub layer_norm_eps: f32,
    pub subsampling_layer_norm_eps: f32,
}

impl Default for Mert2Config {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_mel_bins: 128,
            sampling_rate: 24000,
            n_fft: 2048,
            win_length: 2048,
            hop_length: 240,
            subsampling_channels: vec![128, 512, 1024],
            subsampling_depths: vec![3, 4, 5],
            conv_depthwise_kernel_size: 31,
            rotary_embedding_base: 10000.0,
            layer_norm_eps: 1e-5,
            subsampling_layer_norm_eps: 1e-6,
        }
    }
}

impl Mert2Config {
    fn from_value(v: &Value) -> Result<Self, SheetError> {
        let d = Self::default();
        let cfg = Self {
            hidden_size: usize_at(v, "hidden_size", d.hidden_size),
            intermediate_size: usize_at(v, "intermediate_size", d.intermediate_size),
            num_hidden_layers: usize_at(v, "num_hidden_layers", d.num_hidden_layers),
            num_attention_heads: usize_at(v, "num_attention_heads", d.num_attention_heads),
            num_mel_bins: usize_at(v, "num_mel_bins", d.num_mel_bins),
            sampling_rate: usize_at(v, "sampling_rate", d.sampling_rate as usize) as u32,
            n_fft: usize_at(v, "n_fft", d.n_fft),
            win_length: usize_at(v, "win_length", d.win_length),
            hop_length: usize_at(v, "hop_length", d.hop_length),
            subsampling_channels: list_at(v, "subsampling_channels", &d.subsampling_channels),
            subsampling_depths: list_at(v, "subsampling_depths", &d.subsampling_depths),
            conv_depthwise_kernel_size: usize_at(v, "conv_depthwise_kernel_size", d.conv_depthwise_kernel_size),
            rotary_embedding_base: f64_at(v, "rotary_embedding_base", d.rotary_embedding_base as f64) as f32,
            layer_norm_eps: f64_at(v, "layer_norm_eps", d.layer_norm_eps as f64) as f32,
            subsampling_layer_norm_eps: f64_at(
                v,
                "subsampling_layer_norm_eps",
                d.subsampling_layer_norm_eps as f64,
            ) as f32,
        };
        if cfg.subsampling_channels.len() != 3 || cfg.subsampling_depths.len() != 3 {
            return Err(SheetError::Config("сабсэмплер MERT2: нужно три ступени".into()));
        }
        if *cfg.subsampling_channels.last().unwrap() != cfg.hidden_size
            || cfg.hidden_size % cfg.num_attention_heads != 0
            || cfg.n_fft < cfg.win_length
            || cfg.conv_depthwise_kernel_size % 2 != 1
        {
            return Err(SheetError::Config("MERT2: несогласованная геометрия".into()));
        }
        Ok(cfg)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Сэмплов на выходной кадр: хоп мел-спектра × 4 (два страйда сабсэмплера).
    pub fn inputs_to_logits_ratio(&self) -> usize {
        self.hop_length * 4
    }

    /// Минимальная длина входа — половина окна БПФ + 1.
    pub fn minimum_input_samples(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

/// SheetSage2 целиком (`model_type: "sheetsage2"`).
#[derive(Debug, Clone, PartialEq)]
pub struct SheetSage2Config {
    pub vocab_size: usize,
    /// Ширина декодера.
    pub hidden_size: usize,
    pub decoder_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    /// Окно модели в секундах: энкодер всегда видит ровно столько, дополняя тишиной.
    pub input_audio_length: f64,
    pub max_output_seq_len: usize,
    pub time_hz: usize,
    pub sampling_rate: u32,
    pub tokenizer_fingerprint: String,
    pub backbone: Mert2Config,
}

impl Default for SheetSage2Config {
    fn default() -> Self {
        Self {
            vocab_size: 31678,
            hidden_size: 512,
            decoder_layers: 6,
            num_attention_heads: 8,
            intermediate_size: 2048,
            input_audio_length: 300.0,
            max_output_seq_len: 5120,
            time_hz: 100,
            sampling_rate: 24000,
            tokenizer_fingerprint: "5ba3325af0344c7f".into(),
            backbone: Mert2Config::default(),
        }
    }
}

impl SheetSage2Config {
    pub fn from_json(bytes: &[u8]) -> Result<Self, SheetError> {
        let v: Value = serde_json::from_slice(bytes)
            .map_err(|e| SheetError::Config(format!("config.json: {e}")))?;
        let model_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
        if !model_type.is_empty() && model_type != "sheetsage2" {
            return Err(SheetError::Config(format!(
                "config.json: ожидался model_type=sheetsage2, а там `{model_type}`"
            )));
        }
        if let Some(schema) = v.get("tokenizer_schema_version").and_then(|x| x.as_str()) {
            if schema != "v1" {
                return Err(SheetError::Config(format!("схема словаря `{schema}` не поддержана")));
            }
        }
        let d = Self::default();
        let backbone = match v.get("backbone_config") {
            Some(b) => Mert2Config::from_value(b)?,
            None => Mert2Config::default(),
        };
        let cfg = Self {
            vocab_size: usize_at(&v, "vocab_size", d.vocab_size),
            hidden_size: usize_at(&v, "hidden_size", d.hidden_size),
            decoder_layers: usize_at(&v, "decoder_layers", d.decoder_layers),
            num_attention_heads: usize_at(&v, "num_attention_heads", d.num_attention_heads),
            intermediate_size: usize_at(&v, "intermediate_size", d.intermediate_size),
            input_audio_length: f64_at(&v, "input_audio_length", d.input_audio_length),
            max_output_seq_len: usize_at(&v, "max_output_seq_len", d.max_output_seq_len),
            time_hz: usize_at(&v, "time_hz", d.time_hz),
            sampling_rate: usize_at(&v, "sampling_rate", d.sampling_rate as usize) as u32,
            tokenizer_fingerprint: v
                .get("tokenizer_fingerprint")
                .and_then(|x| x.as_str())
                .unwrap_or(&d.tokenizer_fingerprint)
                .to_string(),
            backbone,
        };
        if cfg.hidden_size % cfg.num_attention_heads != 0 {
            return Err(SheetError::Config("hidden_size не делится на число голов".into()));
        }
        if cfg.sampling_rate != cfg.backbone.sampling_rate {
            return Err(SheetError::Config("частоты дискретизации декодера и энкодера разные".into()));
        }
        Ok(cfg)
    }

    /// Окно в сэмплах, дополненное до кратного шагу кадра энкодера.
    pub fn window_samples(&self) -> usize {
        let window = (self.input_audio_length * self.sampling_rate as f64).round() as usize;
        let stride = self.backbone.inputs_to_logits_ratio();
        window.div_ceil(stride) * stride
    }
}
