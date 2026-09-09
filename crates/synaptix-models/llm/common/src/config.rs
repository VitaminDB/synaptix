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

/// Геометрия внимания global-слоёв, когда она отличается от sliding-слоёв
/// (Gemma-4: голова 512 против 256, две KV-головы против восьми, V берётся
/// из той же проекции, что и K).
#[derive(Debug, Clone)]
pub struct GlobalAttn {
    pub head_dim: usize,
    pub num_key_value_heads: usize,
    /// `attention_k_eq_v`: своей матрицы V у слоя нет, значения — выход
    /// `k_proj` ДО Q/K-нормы и RoPE.
    pub k_eq_v: bool,
}

/// MoE-ветка, считающаяся ПАРАЛЛЕЛЬНО плотному MLP (Gemma-4: плотный MLP
/// играет роль всегда активного эксперта, у каждой ветки свои нормы).
#[derive(Debug, Clone)]
pub struct MoeBranch {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
}

/// Профильные расширения, которых нет у обычного dense-декодера. `None` в
/// [`DecoderConfig::ext`] — модель без них (весь прежний парк).
#[derive(Debug, Clone, Default)]
pub struct DecoderExt {
    pub global_attn: Option<GlobalAttn>,
    /// RMS-норма без веса поверх V (Gemma-4 `v_norm`).
    pub v_rms_norm: bool,
    pub moe: Option<MoeBranch>,
    /// Выход блока умножается на скаляр `layers.N.layer_scalar` из весов.
    pub layer_scalar: bool,
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

    pub ext: Option<DecoderExt>,
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
    /// Голова слоя: у global-слоя она может быть шире (Gemma-4).
    pub fn head_dim_at(&self, idx: usize) -> usize {
        match self.global_attn_at(idx) {
            Some(g) => g.head_dim,
            None => self.head_dim,
        }
    }
    /// Сколько KV-голов у слоя.
    pub fn kv_heads_at(&self, idx: usize) -> usize {
        match self.global_attn_at(idx) {
            Some(g) => g.num_key_value_heads,
            None => self.num_key_value_heads,
        }
    }
    /// Берёт ли слой V из проекции K (своей матрицы V у него нет).
    pub fn k_eq_v_at(&self, idx: usize) -> bool {
        self.global_attn_at(idx).is_some_and(|g| g.k_eq_v)
    }
    fn global_attn_at(&self, idx: usize) -> Option<&GlobalAttn> {
        let g = self.ext.as_ref()?.global_attn.as_ref()?;
        self.is_global_layer(idx).then_some(g)
    }
    /// Самая широкая голова модели — по ней считается ёмкость RoPE-кэша и
    /// верхняя оценка KV.
    pub fn max_head_dim(&self) -> usize {
        let g = self.ext.as_ref().and_then(|e| e.global_attn.as_ref()).map_or(0, |g| g.head_dim);
        self.head_dim.max(g)
    }
    pub fn moe_branch(&self) -> Option<&MoeBranch> {
        self.ext.as_ref()?.moe.as_ref()
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
        self.ext.is_none()
            && !self.attn_output_gate
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
    /// Профиль, поддержанный device-резидентным `forward_decode_dev`
    /// (CUDA-graph). Два реальных RoPE допустимы: `DecodeState` держит по паре
    /// таблиц на тип слоя. Готовность MoE-ветки проверяет уже модель
    /// (`DecoderModel::graph_decode_ready`) — конфиг про веса не знает.
    pub fn graph_decode_ok(&self) -> bool {
        true
    }

    /// Профиль, поддержанный device-резидентным `forward_prefill_dev` (CUDA-graph
    /// prefill chunk'а). Строже decode-профиля: без sandwich-норм, sliding-окон,
    /// local-RoPE, embed-нормы и softcap'а — chunked-prefill dev-путь их не
    /// реализует (муза префиллится host-путём). Linear-слои допустимы только в
    /// MTP-verify гибрида.
    pub fn graph_prefill_ok(&self) -> bool {
        self.ext.is_none()
            && !self.sandwich_norms
            && self.sliding_window.is_none()
            && self.rope_local.is_none()
            && !self.embed_rms_norm
            && self.logit_softcap.is_none()
            && self.logit_scale.is_none()
    }
}
