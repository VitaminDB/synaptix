use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

/// Classifier-Free Guidance (Ho & Salimans, 2022).
///
/// `eps = eps_uncond + scale · (eps_cond − eps_uncond)`.
///
/// При `scale = 1.0` выход равен `cond`; `scale > 1` усиливает условную ветку.
pub struct Cfg {
    pub scale: f32,
}

impl Cfg {
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }

    pub fn apply(&self, cond: &Tensor, uncond: &Tensor) -> Result<Tensor> {
        let diff = cond.sub(uncond)?;
        let scaled = diff.affine(self.scale, 0.0)?;
        uncond.add(&scaled)
    }
}
