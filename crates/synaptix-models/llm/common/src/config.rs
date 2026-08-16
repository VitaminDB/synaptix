#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormGain {
    Plain,
    OnePlus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Silu,
    GeluTanh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Full,
    Linear,
}

#[derive(Debug, Clone)]
pub struct RopeSpec {
    pub theta: f32,
    pub rotary_dim: usize,
    pub scaled_freqs: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct LinearAttnConfig {
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel: usize,
}

impl LinearAttnConfig {
    pub fn key_dim(&self) -> usize {
        self.num_key_heads * self.key_head_dim
    }
    pub fn value_dim(&self) -> usize {
        self.num_value_heads * self.value_head_dim
    }
    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
    pub fn group(&self) -> usize {
        self.num_value_heads / self.num_key_heads
    }
}

#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,

    pub norm_gain: NormGain,
    pub activation: Activation,
    pub sandwich_norms: bool,
    pub post_norm_eps: Option<f32>,
    pub qk_norm: bool,
    pub attn_output_gate: bool,
    pub attn_scale: f32,
    pub embed_scale: Option<f32>,
    pub embed_rms_norm: bool,
    pub logit_scale: Option<f32>,
    pub logit_softcap: Option<f32>,

    pub rope_global: RopeSpec,
    pub rope_local: Option<RopeSpec>,
    pub sliding_window: Option<usize>,
    pub sliding_window_pattern: usize,

    pub layer_kinds: Vec<LayerKind>,
    pub linear: Option<LinearAttnConfig>,

    pub tie_word_embeddings: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
}

impl DecoderConfig {
    pub fn group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
    pub fn q_total_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    pub fn kv_total_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    pub fn layer_kind(&self, idx: usize) -> LayerKind {
        self.layer_kinds.get(idx).copied().unwrap_or(LayerKind::Full)
    }
    pub fn is_global_layer(&self, idx: usize) -> bool {
        if self.sliding_window.is_none() {
            return true;
        }
        let p = self.sliding_window_pattern;
        p <= 1 || (idx + 1) % p == 0
    }
    pub fn rope_for(&self, idx: usize) -> &RopeSpec {
        if self.is_global_layer(idx) {
            &self.rope_global
        } else {
            self.rope_local.as_ref().unwrap_or(&self.rope_global)
        }
    }
    pub fn window_for(&self, idx: usize) -> Option<usize> {
        if self.is_global_layer(idx) {
            None
        } else {
            self.sliding_window
        }
    }
    pub fn simple_profile(&self) -> bool {
        !self.attn_output_gate
            && !self.sandwich_norms
            && self.sliding_window.is_none()
            && self.linear.is_none()
            && self.rope_global.rotary_dim == self.head_dim
            && self.layer_kinds.iter().all(|k| *k == LayerKind::Full)
    }

    /// Профиль, поддержанный device-резидентным `forward_decode_dev` (CUDA-graph).
    /// Шире [`Self::simple_profile`]: допускает linear-слои (GatedDeltaNet),
    /// attn-output-gate, partial-RoPE и Q/K-norm. НЕ поддержаны sandwich-нормы,
    /// sliding-window и отдельный local-RoPE (нужен per-layer rope-кэш в графе).
    pub fn graph_decode_ok(&self) -> bool {
        !self.sandwich_norms
            && self.sliding_window.is_none()
            && self.rope_local.is_none()
            && !self.embed_rms_norm
            && self.logit_softcap.is_none()
            && self.logit_scale.is_none()
    }

    /// Профиль, поддержанный device-резидентным `forward_prefill_dev` (CUDA-graph
    /// prefill chunk'а). Strictly subset of `graph_decode_ok`: дополнительно
    /// требует **отсутствия linear-слоёв** — chunked GatedDeltaNet prefill ещё не
    /// портирован на device-резидентный путь (host-loop через
    /// `gated_delta_rule_prefill` остаётся валиден для hybrid). Hybrid-prefill
    /// граф — отдельная сессия.
    pub fn graph_prefill_ok(&self) -> bool {
        self.graph_decode_ok()
    }
}
