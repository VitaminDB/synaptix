//! Регуляторы длины (`modules/v2/length_regulator.py: InterpolateRegulator`).
//!
//! CFM: эмбеддинг широкого токена (2048 × 512) → растяжение «ближайшим
//! соседом» до числа мел-кадров → 4 × (Conv1d k3 → GroupNorm(1) → Mish).
//! AR: только эмбеддинг узкого токена (32 × 768), без растяжения и свёрток.

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_ops::activation::mish::mish;
use synaptix_ops::conv::conv1d::conv1d;
use synaptix_ops::norm::group_norm::group_norm;

use crate::loader::Comp;
use crate::{SeedVcError, R};

/// Таблица эмбеддингов на хосте.
pub struct Embedding {
    table: Vec<f32>,
    pub dim: usize,
    pub vocab: usize,
}

impl Embedding {
    pub fn load(c: &Comp, name: &str) -> R<Self> {
        let (table, d) = c.host(name)?;
        Ok(Self { table, vocab: d[0], dim: d[1] })
    }

    pub fn row(&self, id: u32) -> R<&[f32]> {
        let i = id as usize;
        if i >= self.vocab {
            return Err(SeedVcError::Other(format!("токен {i} вне словаря {}", self.vocab)));
        }
        Ok(&self.table[i * self.dim..(i + 1) * self.dim])
    }

    /// Строки по токенам → `[1, n, dim]`.
    pub fn lookup(&self, ids: &[u32], device: Device) -> R<Tensor> {
        let mut v = Vec::with_capacity(ids.len() * self.dim);
        for &id in ids {
            v.extend_from_slice(self.row(id)?);
        }
        Ok(Tensor::from_vec(v, vec![1, ids.len(), self.dim], device)?)
    }
}

struct Stage {
    w: Tensor,
    b: Tensor,
    gn_w: Tensor,
    gn_b: Tensor,
}

pub struct CfmRegulator {
    emb: Embedding,
    stages: Vec<Stage>,
    device: Device,
}

impl CfmRegulator {
    /// Ключи — с приставкой `length_regulator.` в компоненте `cfm`.
    pub fn load(c: &Comp) -> R<Self> {
        let p = "length_regulator";
        let mut stages = Vec::with_capacity(4);
        for i in 0..4 {
            let (cv, gn) = (3 * i, 3 * i + 1);
            stages.push(Stage {
                w: c.get(&format!("{p}.model.{cv}.weight"), DType::F32)?,
                b: c.get(&format!("{p}.model.{cv}.bias"), DType::F32)?,
                gn_w: c.get(&format!("{p}.model.{gn}.weight"), DType::F32)?,
                gn_b: c.get(&format!("{p}.model.{gn}.bias"), DType::F32)?,
            });
        }
        Ok(Self { emb: Embedding::load(c, &format!("{p}.embedding.weight"))?, stages, device: c.device() })
    }

    /// Токены → условие `[1, frames, 512]`.
    pub fn forward(&self, tokens: &[u32], frames: usize) -> R<Tensor> {
        let (n, d) = (tokens.len(), self.emb.dim);
        if n == 0 || frames == 0 {
            return Err(SeedVcError::Input("пустая последовательность токенов".into()));
        }
        // `F.interpolate(mode="nearest")`: src = ⌊dst · (n / frames)⌋, арифметика в f32 как в PyTorch.
        let scale = n as f32 / frames as f32;
        let mut x = vec![0f32; d * frames];
        for t in 0..frames {
            let src = ((t as f32 * scale).floor() as usize).min(n - 1);
            let row = self.emb.row(tokens[src])?;
            for (ch, &v) in row.iter().enumerate() {
                x[ch * frames + t] = v;
            }
        }
        let mut x = Tensor::from_vec(x, vec![1, d, frames], self.device)?;
        for s in &self.stages {
            x = conv1d(&x, &s.w, Some(&s.b), 1, 1)?;
            x = mish(&group_norm(&x, Some(&s.gn_w), Some(&s.gn_b), 1, 1e-5)?)?;
        }
        Ok(x.transpose(1, 2)?.contiguous()?)
    }
}
