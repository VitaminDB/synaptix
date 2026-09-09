//! Разбор `config.json` Gemma-4 и перевод его в [`DecoderConfig`].
//!
//! Чем Gemma-4 отличается от Gemma-3 (и почему обычным `DecoderConfig` не
//! обойтись):
//!
//! * **разная геометрия у sliding- и global-слоёв.** Sliding: 16 голов по 256,
//!   8 KV-голов. Global: те же 16 голов, но по 512, и всего 2 KV-головы,
//!   причём значения берутся из проекции ключей (`attention_k_eq_v`) — своей
//!   матрицы V у таких слоёв в чекпойнте нет;
//! * **proportional RoPE** на global-слоях: вращается только четверть головы
//!   (`partial_rotary_factor = 0.25`), остальное едет без вращения. В терминах
//!   кэша это обычный RoPE на полную голову с нулевыми частотами в хвосте;
//! * **MoE-ветка рядом с плотным MLP**: плотный MLP играет роль всегда
//!   активного эксперта, у каждой ветки своя пара норм, выходы складываются;
//! * **RMS-нормы без `1 +`** (в Gemma-3 вес нормы хранился со сдвигом);
//! * **масштаб внимания 1.0** — его роль играет обучаемая Q-норма;
//! * **RMS-норма без веса поверх V**.

use synaptix_llm_common::{
    Activation, DecoderConfig, DecoderExt, GlobalAttn, LayerKind, MoeBranch, NormGain, RopeSpec,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerType {
    Sliding,
    Full,
}

/// Параметры RoPE одного типа слоёв.
#[derive(Debug, Clone)]
pub struct RopeParams {
    pub theta: f32,
    /// `default` — вращается вся голова; `proportional` — только первые
    /// `partial_rotary_factor · head_dim` измерений.
    pub proportional: bool,
    pub partial_rotary_factor: f32,
}

impl Default for RopeParams {
    fn default() -> Self {
        Self { theta: 10_000.0, proportional: false, partial_rotary_factor: 1.0 }
    }
}

#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Голова global-слоёв (обычно шире sliding).
    pub global_head_dim: usize,
    /// KV-головы global-слоёв.
    pub num_global_key_value_heads: usize,
    /// `attention_k_eq_v`: global-слои берут V из проекции K.
    pub attention_k_eq_v: bool,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub sliding_window: usize,
    pub layer_types: Vec<LayerType>,
    pub rope_sliding: RopeParams,
    pub rope_full: RopeParams,
    pub hidden_activation: String,
    pub final_logit_softcapping: Option<f32>,
    /// MoE включён (`enable_moe_block`).
    pub enable_moe_block: bool,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub moe_intermediate_size: usize,
    /// PLE (`hidden_size_per_layer_input`); у 26B A4B выключен (`0`).
    pub hidden_size_per_layer_input: usize,
    /// Слои, делящие KV с предыдущими (`num_kv_shared_layers`); у 26B A4B `0`.
    pub num_kv_shared_layers: usize,
    pub tie_word_embeddings: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    /// `image_token_id` — по нему в промпт вклеиваются мягкие токены картинки.
    pub image_token_id: Option<u32>,
    /// Есть ли в конфиге башня зрения.
    pub has_vision: bool,
}

fn f32_at(v: &serde_json::Value, key: &str) -> Option<f32> {
    v.get(key).and_then(|x| x.as_f64()).map(|x| x as f32)
}

fn usize_at(v: &serde_json::Value, key: &str) -> Option<usize> {
    v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
}

fn rope_params(text: &serde_json::Value, layer_type: &str, default_theta: f32) -> RopeParams {
    let Some(p) = text.get("rope_parameters").and_then(|v| v.get(layer_type)) else {
        return RopeParams { theta: default_theta, ..Default::default() };
    };
    let kind = p.get("rope_type").and_then(|x| x.as_str()).unwrap_or("default");
    RopeParams {
        theta: f32_at(p, "rope_theta").unwrap_or(default_theta),
        proportional: kind == "proportional",
        partial_rotary_factor: f32_at(p, "partial_rotary_factor").unwrap_or(1.0),
    }
}

impl Gemma4Config {
    /// Разбирает HF-конфиг. Арх-параметры лежат в `text_config`, токены и
    /// мультимодальные поля — на верхнем уровне.
    pub fn from_hf_bytes(bytes: &[u8]) -> Result<Self, ConfigError> {
        let root: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| ConfigError::Parse(format!("config.json: {e}")))?;
        let text = root.get("text_config").unwrap_or(&root).clone();

        let num_hidden_layers = usize_at(&text, "num_hidden_layers")
            .ok_or_else(|| ConfigError::Invalid("нет num_hidden_layers".into()))?;
        let layer_types: Vec<LayerType> = match text.get("layer_types").and_then(|v| v.as_array()) {
            Some(a) => a
                .iter()
                .map(|x| match x.as_str() {
                    Some("full_attention") => LayerType::Full,
                    _ => LayerType::Sliding,
                })
                .collect(),
            // Без списка считаем, что каждый шестой слой — global (раскладка Gemma).
            None => (0..num_hidden_layers)
                .map(|i| if (i + 1) % 6 == 0 { LayerType::Full } else { LayerType::Sliding })
                .collect(),
        };
        if layer_types.len() != num_hidden_layers {
            return Err(ConfigError::Invalid(format!(
                "layer_types: {} записей при {num_hidden_layers} слоях",
                layer_types.len()
            )));
        }

        let head_dim = usize_at(&text, "head_dim").unwrap_or(256);
        let eos_token_ids = match root.get("eos_token_id").or_else(|| text.get("eos_token_id")) {
            Some(serde_json::Value::Array(a)) => {
                a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()
            }
            Some(serde_json::Value::Number(n)) => {
                n.as_u64().map(|x| vec![x as u32]).unwrap_or_default()
            }
            _ => Vec::new(),
        };

        let cfg = Self {
            vocab_size: usize_at(&text, "vocab_size").unwrap_or(262_144),
            hidden_size: usize_at(&text, "hidden_size")
                .ok_or_else(|| ConfigError::Invalid("нет hidden_size".into()))?,
            intermediate_size: usize_at(&text, "intermediate_size")
                .ok_or_else(|| ConfigError::Invalid("нет intermediate_size".into()))?,
            num_hidden_layers,
            num_attention_heads: usize_at(&text, "num_attention_heads")
                .ok_or_else(|| ConfigError::Invalid("нет num_attention_heads".into()))?,
            num_key_value_heads: usize_at(&text, "num_key_value_heads").unwrap_or(1),
            head_dim,
            global_head_dim: usize_at(&text, "global_head_dim").unwrap_or(head_dim),
            num_global_key_value_heads: usize_at(&text, "num_global_key_value_heads")
                .unwrap_or_else(|| usize_at(&text, "num_key_value_heads").unwrap_or(1)),
            attention_k_eq_v: text
                .get("attention_k_eq_v")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            max_position_embeddings: usize_at(&text, "max_position_embeddings").unwrap_or(131_072),
            rms_norm_eps: f32_at(&text, "rms_norm_eps").unwrap_or(1e-6),
            sliding_window: usize_at(&text, "sliding_window").unwrap_or(1024),
            layer_types,
            rope_sliding: rope_params(&text, "sliding_attention", 10_000.0),
            rope_full: rope_params(&text, "full_attention", 1_000_000.0),
            hidden_activation: text
                .get("hidden_activation")
                .and_then(|x| x.as_str())
                .unwrap_or("gelu_pytorch_tanh")
                .to_string(),
            final_logit_softcapping: f32_at(&text, "final_logit_softcapping"),
            enable_moe_block: text
                .get("enable_moe_block")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            num_experts: usize_at(&text, "num_experts").unwrap_or(0),
            top_k_experts: usize_at(&text, "top_k_experts").unwrap_or(0),
            moe_intermediate_size: usize_at(&text, "moe_intermediate_size").unwrap_or(0),
            hidden_size_per_layer_input: usize_at(&text, "hidden_size_per_layer_input")
                .unwrap_or(0),
            num_kv_shared_layers: usize_at(&text, "num_kv_shared_layers").unwrap_or(0),
            tie_word_embeddings: text
                .get("tie_word_embeddings")
                .or_else(|| root.get("tie_word_embeddings"))
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
            bos_token_id: usize_at(&text, "bos_token_id").map(|x| x as u32),
            eos_token_ids,
            image_token_id: usize_at(&root, "image_token_id").map(|x| x as u32),
            has_vision: root.get("vision_config").is_some_and(|v| !v.is_null()),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Дополняет список стоп-токенов из `generation_config.json`: у Gemma-4
    /// он шире, чем в `config.json` (там нет `<turn|>`-варианта, которым
    /// модель на самом деле закрывает ход).
    pub fn merge_generation_config(&mut self, bytes: &[u8]) {
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(bytes) else { return };
        let extra: Vec<u32> = match v.get("eos_token_id") {
            Some(serde_json::Value::Array(a)) => {
                a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()
            }
            Some(serde_json::Value::Number(n)) => {
                n.as_u64().map(|x| vec![x as u32]).unwrap_or_default()
            }
            _ => Vec::new(),
        };
        for id in extra {
            if !self.eos_token_ids.contains(&id) {
                self.eos_token_ids.push(id);
            }
        }
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.hidden_size_per_layer_input != 0 {
            return Err(ConfigError::Invalid(
                "PLE (hidden_size_per_layer_input > 0) не поддержан: это раскладка E2B/E4B".into(),
            ));
        }
        if self.num_kv_shared_layers != 0 {
            return Err(ConfigError::Invalid(
                "num_kv_shared_layers > 0 не поддержан: слои, делящие KV, не реализованы".into(),
            ));
        }
        if self.enable_moe_block && (self.num_experts == 0 || self.top_k_experts == 0) {
            return Err(ConfigError::Invalid("enable_moe_block без экспертов".into()));
        }
        if self.hidden_activation != "gelu_pytorch_tanh" {
            return Err(ConfigError::Invalid(format!(
                "hidden_activation {:?}: ожидался gelu_pytorch_tanh",
                self.hidden_activation
            )));
        }
        // Общий декодер описывает раскладку слоёв периодом, а не списком.
        let p = self.sliding_pattern();
        for (i, t) in self.layer_types.iter().enumerate() {
            let want = if p <= 1 { LayerType::Full } else if (i + 1) % p == 0 { LayerType::Full } else { LayerType::Sliding };
            if *t != want {
                return Err(ConfigError::Invalid(format!(
                    "layer_types не периодические: слой {i} — {t:?}, а по периоду {p} ожидался {want:?}"
                )));
            }
        }
        Ok(())
    }

    /// Период чередования: каждый `p`-й слой (считая с единицы) — global.
    pub fn sliding_pattern(&self) -> usize {
        match self.layer_types.iter().position(|t| *t == LayerType::Full) {
            Some(first) => first + 1,
            None => 0,
        }
    }

    pub fn is_global(&self, idx: usize) -> bool {
        self.layer_types.get(idx) == Some(&LayerType::Full)
    }

    /// Обратные частоты global-слоёв: первые `partial · head_dim / 2` — обычный
    /// RoPE, остальные нули (эквивалент «повернуть только часть головы», но с
    /// разбиением пар пополам от ПОЛНОЙ головы, как в HF `rotate_half`).
    fn proportional_freqs(&self) -> Option<Vec<f32>> {
        let p = &self.rope_full;
        if !p.proportional {
            return None;
        }
        let hd = self.global_head_dim;
        let half = hd / 2;
        let rotated = ((p.partial_rotary_factor * hd as f32) as usize / 2).min(half);
        let mut freqs = vec![0.0_f32; half];
        for (j, f) in freqs.iter_mut().enumerate().take(rotated) {
            *f = p.theta.powf(-(2.0 * j as f32) / hd as f32);
        }
        Some(freqs)
    }

    pub fn to_decoder_config(&self) -> DecoderConfig {
        let global_attn = (self.global_head_dim != self.head_dim
            || self.num_global_key_value_heads != self.num_key_value_heads
            || self.attention_k_eq_v)
            .then(|| GlobalAttn {
                head_dim: self.global_head_dim,
                num_key_value_heads: self.num_global_key_value_heads,
                k_eq_v: self.attention_k_eq_v,
            });
        let moe = self.enable_moe_block.then(|| MoeBranch {
            num_experts: self.num_experts,
            num_experts_per_tok: self.top_k_experts,
            moe_intermediate_size: self.moe_intermediate_size,
        });
        DecoderConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            // Веса норм в Gemma-4 хранятся как есть (в Gemma-3 были со сдвигом).
            norm_gain: NormGain::Plain,
            activation: Activation::GeluTanh,
            sandwich_norms: true,
            post_norm_eps: None,
            qk_norm: true,
            attn_output_gate: false,
            // Масштаба 1/sqrt(d) нет: его роль играет обучаемая Q-норма.
            attn_scale: 1.0,
            embed_scale: Some((self.hidden_size as f32).sqrt()),
            embed_rms_norm: false,
            logit_scale: None,
            logit_softcap: self.final_logit_softcapping,
            rope_global: RopeSpec {
                theta: self.rope_full.theta,
                rotary_dim: self.global_head_dim,
                scaled_freqs: self.proportional_freqs(),
            },
            rope_local: Some(RopeSpec {
                theta: self.rope_sliding.theta,
                rotary_dim: self.head_dim,
                scaled_freqs: None,
            }),
            sliding_window: Some(self.sliding_window),
            sliding_window_pattern: self.sliding_pattern(),
            layer_kinds: vec![LayerKind::Full; self.num_hidden_layers],
            linear: None,
            tie_word_embeddings: self.tie_word_embeddings,
            bos_token_id: self.bos_token_id,
            eos_token_ids: self.eos_token_ids.clone(),
            ext: Some(DecoderExt {
                global_attn,
                v_rms_norm: true,
                moe,
                layer_scalar: true,
            }),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("config parse: {0}")]
    Parse(String),
    #[error("config invalid: {0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "image_token_id": 258880,
        "eos_token_id": [1, 106],
        "text_config": {
            "attention_k_eq_v": true,
            "enable_moe_block": true,
            "final_logit_softcapping": 30.0,
            "global_head_dim": 512,
            "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 2816,
            "hidden_size_per_layer_input": 0,
            "intermediate_size": 2112,
            "layer_types": [
                "sliding_attention","sliding_attention","sliding_attention",
                "sliding_attention","sliding_attention","full_attention",
                "sliding_attention","sliding_attention","sliding_attention",
                "sliding_attention","sliding_attention","full_attention"
            ],
            "max_position_embeddings": 262144,
            "moe_intermediate_size": 704,
            "num_attention_heads": 16,
            "num_experts": 128,
            "num_global_key_value_heads": 2,
            "num_hidden_layers": 12,
            "num_key_value_heads": 8,
            "num_kv_shared_layers": 0,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": { "rope_theta": 10000.0, "rope_type": "default" }
            },
            "sliding_window": 1024,
            "tie_word_embeddings": true,
            "top_k_experts": 8,
            "vocab_size": 262144
        },
        "vision_config": { "hidden_size": 1152 }
    }"#;

    #[test]
    fn parses_and_maps() {
        let cfg = Gemma4Config::from_hf_bytes(SAMPLE.as_bytes()).expect("parse");
        assert_eq!(cfg.sliding_pattern(), 6);
        assert!(cfg.is_global(5) && cfg.is_global(11) && !cfg.is_global(4));
        assert!(cfg.has_vision);
        assert_eq!(cfg.eos_token_ids, vec![1, 106]);

        let d = cfg.to_decoder_config();
        assert_eq!(d.head_dim_at(0), 256);
        assert_eq!(d.head_dim_at(5), 512);
        assert_eq!(d.kv_heads_at(0), 8);
        assert_eq!(d.kv_heads_at(5), 2);
        assert!(!d.k_eq_v_at(0) && d.k_eq_v_at(5));
        assert_eq!(d.window_for(0), Some(1024));
        assert_eq!(d.window_for(5), None);
        assert_eq!(d.logit_softcap, Some(30.0));

        // Proportional RoPE: 64 живых частоты, остальные 192 — нули.
        let freqs = d.rope_global.scaled_freqs.as_ref().expect("proportional freqs");
        assert_eq!(freqs.len(), 256);
        assert!((freqs[0] - 1.0).abs() < 1e-6);
        assert!(freqs[63] > 0.0);
        assert!(freqs[64..].iter().all(|f| *f == 0.0));

        let moe = d.moe_branch().expect("moe");
        assert_eq!((moe.num_experts, moe.num_experts_per_tok), (128, 8));
    }

    #[test]
    fn rejects_ple_layout() {
        let bad = SAMPLE.replace("\"hidden_size_per_layer_input\": 0", "\"hidden_size_per_layer_input\": 256");
        assert!(Gemma4Config::from_hf_bytes(bad.as_bytes()).is_err());
    }
}
