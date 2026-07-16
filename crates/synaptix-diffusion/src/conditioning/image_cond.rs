use synaptix_core::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct ImageConditioning {
    pub image_embeds: Tensor,
    pub image_latents: Option<Tensor>,
    pub mask: Option<Tensor>,
    pub scale: f32,
}

impl ImageConditioning {
    pub fn new(image_embeds: Tensor) -> Self {
        Self { image_embeds, image_latents: None, mask: None, scale: 1.0 }
    }

    pub fn with_latents(mut self, latents: Tensor) -> Self {
        self.image_latents = Some(latents);
        self
    }

    pub fn with_mask(mut self, mask: Tensor) -> Self {
        self.mask = Some(mask);
        self
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
}

#[derive(Debug, Clone)]
pub struct InpaintConditioning {
    pub masked_latents: Tensor,
    pub mask: Tensor,
}

impl InpaintConditioning {
    pub fn new(masked_latents: Tensor, mask: Tensor) -> Self {
        Self { masked_latents, mask }
    }
}
