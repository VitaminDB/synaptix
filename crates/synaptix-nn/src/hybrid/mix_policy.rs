use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

/// Тип слоя в гибридной модели.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Attention,
    Ssm,
    MoE,
}

/// Schedule-таблица, описывающая что выполнять на каждом слое гибрида.
///
/// `apply` принимает три замыкания (по одному на тип слоя) и dispatch'ит
/// один из них по индексу слоя. Это **functional** dispatch — runtime не
/// знает о конкретных модулях; caller передаёт замыкания, замыкающие
/// захваченные модули (Mamba/Attn/MoE).
pub struct MixPolicy {
    pub schedule: Vec<LayerKind>,
}

impl MixPolicy {
    pub fn new(schedule: Vec<LayerKind>) -> Self {
        Self { schedule }
    }

    pub fn kind_at(&self, layer_idx: usize) -> Option<&LayerKind> {
        self.schedule.get(layer_idx)
    }

    pub fn len(&self) -> usize {
        self.schedule.len()
    }

    pub fn is_empty(&self) -> bool {
        self.schedule.is_empty()
    }

    /// Functional dispatch: пользователь передаёт три замыкания, каждое для
    /// своего `LayerKind`. Полезно для тестов и для caller'а, у которого
    /// модули SSM/Attn/MoE захвачены в env.
    pub fn apply<FA, FS, FM>(
        &self,
        x: &Tensor,
        layer_idx: usize,
        attn_fn: FA,
        ssm_fn: FS,
        moe_fn: FM,
    ) -> Result<Tensor>
    where
        FA: FnOnce(&Tensor) -> Result<Tensor>,
        FS: FnOnce(&Tensor) -> Result<Tensor>,
        FM: FnOnce(&Tensor) -> Result<Tensor>,
    {
        match self.kind_at(layer_idx) {
            Some(LayerKind::Attention) => attn_fn(x),
            Some(LayerKind::Ssm) => ssm_fn(x),
            Some(LayerKind::MoE) => moe_fn(x),
            None => Err(SynaptixError::Unsupported("MixPolicy::apply: layer_idx out of range")),
        }
    }

    /// Подсчёт количества слоёв каждого типа.
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut a = 0;
        let mut s = 0;
        let mut m = 0;
        for kind in &self.schedule {
            match kind {
                LayerKind::Attention => a += 1,
                LayerKind::Ssm => s += 1,
                LayerKind::MoE => m += 1,
            }
        }
        (a, s, m)
    }
}
