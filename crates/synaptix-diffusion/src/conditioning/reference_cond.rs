use synaptix_core::tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceMode {
    Write,
    Read,
}

#[derive(Debug, Clone)]
pub struct ReferenceConditioning {
    pub ref_image_latents: Tensor,
    pub attention_scale: f32,
    pub adain_scale: f32,
    pub mode: ReferenceMode,
}

impl ReferenceConditioning {
    pub fn new(ref_image_latents: Tensor) -> Self {
        Self {
            ref_image_latents,
            attention_scale: 1.0,
            adain_scale: 1.0,
            mode: ReferenceMode::Write,
        }
    }

    pub fn with_attention_scale(mut self, s: f32) -> Self {
        self.attention_scale = s;
        self
    }

    pub fn with_adain_scale(mut self, s: f32) -> Self {
        self.adain_scale = s;
        self
    }
}
