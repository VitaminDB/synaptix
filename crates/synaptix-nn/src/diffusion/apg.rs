use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

/// Adaptive Projected Guidance (Sadat et al., 2024) — stateless orthogonal-only
/// вариант (без momentum-буфера).
///
/// 1. `diff = cond − uncond`.
/// 2. Rescale: если `‖diff‖₂ > norm_threshold`, то `diff ← diff · (norm_threshold / ‖diff‖₂)`.
/// 3. Orthogonal projection: убираем компоненту вдоль `cond` —
///    `parallel = ((diff · cond) / max(‖cond‖₂², ε)) · cond`,
///    `ortho = diff − parallel`.
/// 4. `eps = cond + scale · ortho`.
///
/// Поле `momentum` сохранено в struct для совместимости с loader-API; stateless
/// `apply()` его игнорирует. Для stateful-режима использовать
/// `apply_with_momentum(prev_diff)` явно.
pub struct Apg {
    pub scale: f32,
    pub momentum: f32,
    pub norm_threshold: f32,
    pub eps: f32,
}

impl Apg {
    pub fn new(scale: f32, momentum: f32) -> Self {
        Self {
            scale,
            momentum,
            norm_threshold: 2.5,
            eps: 1e-8,
        }
    }

    pub fn with_norm_threshold(mut self, threshold: f32) -> Self {
        self.norm_threshold = threshold;
        self
    }

    pub fn apply(&self, cond: &Tensor, uncond: &Tensor) -> Result<Tensor> {
        let diff = cond.sub(uncond)?;
        let rescaled = self.rescale(&diff)?;
        let ortho = self.orthogonal(&rescaled, cond)?;
        let scaled = ortho.affine(self.scale, 0.0)?;
        cond.add(&scaled)
    }

    /// Stateful-вариант с явным momentum-буфером:
    /// `diff ← diff + momentum · prev_diff`, далее как в `apply`. Возвращает
    /// `(output, new_diff)` — `new_diff` следует сохранить и передать на
    /// следующий шаг.
    pub fn apply_with_momentum(
        &self,
        cond: &Tensor,
        uncond: &Tensor,
        prev_diff: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let mut diff = cond.sub(uncond)?;
        if let Some(prev) = prev_diff {
            let prev_scaled = prev.affine(self.momentum, 0.0)?;
            diff = diff.add(&prev_scaled)?;
        }
        let rescaled = self.rescale(&diff)?;
        let ortho = self.orthogonal(&rescaled, cond)?;
        let scaled = ortho.affine(self.scale, 0.0)?;
        let out = cond.add(&scaled)?;
        Ok((out, diff))
    }

    fn rescale(&self, diff: &Tensor) -> Result<Tensor> {
        let norm_sq = diff.mul(diff)?.sum_all()?;
        let norm_t = norm_sq.sqrt()?;
        let norm = scalar_f32(&norm_t)?;
        if norm > self.norm_threshold && norm > 0.0 {
            diff.affine(self.norm_threshold / norm, 0.0)
        } else {
            Ok(diff.clone())
        }
    }

    fn orthogonal(&self, diff: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let dot_t = diff.mul(cond)?.sum_all()?;
        let cond_sq_t = cond.mul(cond)?.sum_all()?;
        let dot = scalar_f32(&dot_t)?;
        let cond_sq = scalar_f32(&cond_sq_t)?;
        let denom = if cond_sq > self.eps { cond_sq } else { self.eps };
        let parallel = cond.affine(dot / denom, 0.0)?;
        diff.sub(&parallel)
    }
}

fn scalar_f32(t: &Tensor) -> Result<f32> {
    let flat = t.reshape(vec![1])?.to_vec1::<f32>()?;
    Ok(flat[0])
}
