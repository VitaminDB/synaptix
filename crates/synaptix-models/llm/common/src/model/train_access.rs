//! Доступ обучения к весам модели: ссылки на проекции, нормы и параметры
//! слоя в том виде, в каком их собрал загрузчик (кванты, свёрнутые нормы).
//! Обучающий проход (`synaptix-llm-train`) считает свой прямой и обратный
//! ход поверх этих весов и не трогает пути инференса.

use synaptix_core::tensor::Tensor;
use synaptix_ops::pos::rope_cache::RopeCache;

use super::{Block, DecoderModel, Ffn, FullAttn, Mixer, ModelError};
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

/// Плотный MLP `down(act(gate) · up)`.
pub struct MlpRefs<'a> {
    pub gate_proj: &'a QLinear,
    pub up_proj: &'a QLinear,
    pub down_proj: &'a QLinear,
    pub activation: Activation,
}

/// Слой декодера (пока — полное внимание и плотный MLP).
pub struct BlockRefs<'a> {
    pub pre_attn_norm: &'a Tensor,
    pub post_attn_norm: Option<&'a Tensor>,
    pub pre_mlp_norm: &'a Tensor,
    pub post_mlp_norm: Option<&'a Tensor>,
    pub attn: AttnRefs<'a>,
    pub mlp: MlpRefs<'a>,
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

impl Block {
    fn refs<'a>(&'a self, rope: &'a RopeCache) -> Result<BlockRefs<'a>, ModelError> {
        if self.moe.is_some() || self.ple.is_some() {
            return Err(ModelError::Build("обучение: MoE-ветка и PLE пока не поддержаны".into()));
        }
        if self.layer_scalar.is_some_and(|a| a != 1.0) {
            return Err(ModelError::Build("обучение: множитель слоя пока не поддержан".into()));
        }
        let attn = match &self.mixer {
            Mixer::Full(fa) => fa.refs(rope)?,
            Mixer::Linear(_) => {
                return Err(ModelError::Build("обучение: линейное внимание (GDN) пока не поддержано".into()))
            }
        };
        let mlp = match &self.mlp {
            Ffn::Dense(m) => MlpRefs {
                gate_proj: &m.gate_proj,
                up_proj: &m.up_proj,
                down_proj: &m.down_proj,
                activation: m.activation,
            },
            Ffn::Moe(_) => return Err(ModelError::Build("обучение: MoE пока не поддержан".into())),
        };
        Ok(BlockRefs {
            pre_attn_norm: &self.pre_attn_norm,
            post_attn_norm: self.post_attn_norm.as_ref(),
            pre_mlp_norm: &self.pre_mlp_norm,
            post_mlp_norm: self.post_mlp_norm.as_ref(),
            attn,
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
