//! Доступ обучения к весам модели: ссылки на проекции, нормы и параметры
//! слоя в том виде, в каком их собрал загрузчик (кванты, свёрнутые нормы).
//! Обучающий проход (`synaptix-llm-train`) считает свой прямой и обратный
//! ход поверх этих весов и не трогает пути инференса.

use synaptix_core::tensor::Tensor;
use synaptix_ops::pos::rope_cache::RopeCache;

use super::{Block, DecoderModel, Ffn, FullAttn, LinearAttn, Mixer, ModelError};
use crate::config::Activation;
use crate::weights::QLinear;

/// Полное внимание слоя.
pub struct AttnRefs<'a> {
    pub q_proj: &'a QLinear,
    pub k_proj: &'a QLinear,
    /// `None` — V берётся из проекции K (Gemma-4).
    pub v_proj: Option<&'a QLinear>,
    pub o_proj: &'a QLinear,
    pub q_bias: Option<&'a Tensor>,
    pub k_bias: Option<&'a Tensor>,
    pub v_bias: Option<&'a Tensor>,
    pub q_norm: Option<&'a Tensor>,
    pub k_norm: Option<&'a Tensor>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    /// Выход q-проекции — `[q | gate]` на голову, выход внимания
    /// умножается на `sigmoid(gate)` (Qwen3.5/3.6/3.8).
    pub attn_output_gate: bool,
    pub attn_scale: f32,
    pub rms_eps: f32,
    pub rope: &'a RopeCache,
}

/// Линейное внимание Gated DeltaNet (Qwen3-Next / Qwen3.5+).
pub struct LinearRefs<'a> {
    pub in_proj_qkv: &'a QLinear,
    pub in_proj_a: &'a QLinear,
    pub in_proj_b: &'a QLinear,
    pub in_proj_z: &'a QLinear,
    pub out_proj: &'a QLinear,
    /// Веса свёртки `[conv_dim, conv_k]`.
    pub conv_w: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    /// Вес RMSNorm выхода (на голову `dv`), после неё — гейт `silu(z)`.
    pub norm_weight: &'a Tensor,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub dk: usize,
    pub dv: usize,
    pub conv_k: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_dim: usize,
    pub q_scale: f32,
    pub rms_eps: f32,
}

/// Смешивание токенов слоя.
pub enum MixerRefs<'a> {
    Full(AttnRefs<'a>),
    Linear(LinearRefs<'a>),
}

/// Плотный MLP `down(act(gate) · up)`.
pub struct MlpRefs<'a> {
    pub gate_proj: &'a QLinear,
    pub up_proj: &'a QLinear,
    pub down_proj: &'a QLinear,
    pub activation: Activation,
}

/// FFN слоя: плотный MLP или MoE (роутер, эксперты, shared-эксперт).
pub enum FfnRefs<'a> {
    Dense(MlpRefs<'a>),
    Moe(&'a crate::moe::MoeFfn),
}

/// Слой декодера: полное или линейное внимание и FFN.
pub struct BlockRefs<'a> {
    pub pre_attn_norm: &'a Tensor,
    pub post_attn_norm: Option<&'a Tensor>,
    pub pre_mlp_norm: &'a Tensor,
    pub post_mlp_norm: Option<&'a Tensor>,
    pub mixer: MixerRefs<'a>,
    pub mlp: FfnRefs<'a>,
    pub rms_eps: f32,
    pub post_eps: f32,
}

impl FullAttn {
    fn refs<'a>(&'a self, rope: &'a RopeCache) -> Result<AttnRefs<'a>, ModelError> {
        if self.sliding_window.is_some() {
            return Err(ModelError::Build("обучение: скользящее окно внимания пока не поддержано".into()));
        }
        if self.kv_shared {
            return Err(ModelError::Build("обучение: общий KV между слоями пока не поддержан".into()));
        }
        if self.v_norm.is_some() {
            return Err(ModelError::Build("обучение: норма V пока не поддержана".into()));
        }
        Ok(AttnRefs {
            q_proj: &self.q_proj,
            k_proj: &self.k_proj,
            v_proj: self.v_proj.as_ref(),
            o_proj: &self.o_proj,
            q_bias: self.q_bias.as_ref(),
            k_bias: self.k_bias.as_ref(),
            v_bias: self.v_bias.as_ref(),
            q_norm: self.q_norm.as_ref(),
            k_norm: self.k_norm.as_ref(),
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            rotary_dim: self.rotary_dim,
            attn_output_gate: self.attn_output_gate,
            attn_scale: self.attn_scale,
            rms_eps: self.rms_eps,
            rope,
        })
    }
}

impl LinearAttn {
    fn refs(&self) -> LinearRefs<'_> {
        LinearRefs {
            in_proj_qkv: &self.in_proj_qkv,
            in_proj_a: &self.in_proj_a,
            in_proj_b: &self.in_proj_b,
            in_proj_z: &self.in_proj_z,
            out_proj: &self.out_proj,
            conv_w: &self.conv_w,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            norm_weight: &self.norm_weight,
            num_k_heads: self.num_k_heads,
            num_v_heads: self.num_v_heads,
            dk: self.dk,
            dv: self.dv,
            conv_k: self.conv_k,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            conv_dim: self.conv_dim,
            q_scale: self.q_scale,
            rms_eps: self.rms_eps,
        }
    }
}

impl Block {
    fn refs<'a>(&'a self, rope: &'a RopeCache) -> Result<BlockRefs<'a>, ModelError> {
        if self.moe.is_some() || self.ple.is_some() {
            return Err(ModelError::Build("обучение: боковая MoE-ветка (Gemma-4) и PLE пока не поддержаны".into()));
        }
        if self.layer_scalar.is_some_and(|a| a != 1.0) {
            return Err(ModelError::Build("обучение: множитель слоя пока не поддержан".into()));
        }
        let mixer = match &self.mixer {
            Mixer::Full(fa) => MixerRefs::Full(fa.refs(rope)?),
            Mixer::Linear(la) => MixerRefs::Linear(la.refs()),
        };
        let mlp = match &self.mlp {
            Ffn::Dense(m) => FfnRefs::Dense(MlpRefs {
                gate_proj: &m.gate_proj,
                up_proj: &m.up_proj,
                down_proj: &m.down_proj,
                activation: m.activation,
            }),
            Ffn::Moe(m) => {
                m.train_check()?;
                FfnRefs::Moe(m)
            }
        };
        Ok(BlockRefs {
            pre_attn_norm: &self.pre_attn_norm,
            post_attn_norm: self.post_attn_norm.as_ref(),
            pre_mlp_norm: &self.pre_mlp_norm,
            post_mlp_norm: self.post_mlp_norm.as_ref(),
            mixer,
            mlp,
            rms_eps: self.rms_eps,
            post_eps: self.post_eps,
        })
    }
}

impl DecoderModel {
    /// Веса слоя `idx` для обучения. Ошибка — у слоя есть то, чего обучающий
    /// проход ещё не умеет (линейное внимание, MoE, окно, ...).
    pub fn train_block(&self, idx: usize) -> Result<BlockRefs<'_>, ModelError> {
        if !self.blocks_all_resident() {
            return Err(ModelError::Build("обучение: модель должна целиком лежать на карте".into()));
        }
        let blk = self
            .blocks
            .get(idx)
            .ok_or_else(|| ModelError::Build(format!("обучение: нет слоя {idx}")))?;
        blk.refs(self.rope_at(idx))
    }

    pub fn train_final_norm(&self) -> &Tensor {
        &self.final_norm
    }

    pub fn train_lm_head(&self) -> &QLinear {
        &self.lm_head
    }

    /// Проверка модели целиком до начала обучения: всё, чего проход не
    /// умеет, — ошибкой сразу, а не на середине первого шага.
    pub fn train_check(&self) -> Result<(), ModelError> {
        if self.ple.is_some() {
            return Err(ModelError::Build("обучение: PLE пока не поддержан".into()));
        }
        for i in 0..self.blocks.len() {
            self.train_block(i)?;
        }
        Ok(())
    }
}
