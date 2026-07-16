use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

/// Perturbed-Attention Guidance (Ahn et al., 2024).
///
/// `eps = eps_cond + scale · (eps_cond − eps_perturbed)`,
/// где `eps_perturbed` — выход модели с заменой self-attention на identity-карту
/// внутри выбранных блоков. Часто комбинируется с CFG аддитивно.
pub struct Pag {
    pub scale: f32,
}

impl Pag {
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }

    pub fn apply(&self, cond: &Tensor, perturbed: &Tensor) -> Result<Tensor> {
        let diff = cond.sub(perturbed)?;
        let scaled = diff.affine(self.scale, 0.0)?;
        cond.add(&scaled)
    }
}
