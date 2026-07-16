use synaptix_core::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct ControlNetResiduals {
    pub down_block_res_samples: Vec<Tensor>,
    pub mid_block_res_sample: Tensor,
    pub conditioning_scale: f32,
}

impl ControlNetResiduals {
    pub fn new(down_block_res_samples: Vec<Tensor>, mid_block_res_sample: Tensor) -> Self {
        Self { down_block_res_samples, mid_block_res_sample, conditioning_scale: 1.0 }
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.conditioning_scale = scale;
        self
    }
}

#[derive(Debug, Clone)]
pub struct ControlNetInput {
    pub image: Tensor,
    pub conditioning_scale: f32,
}

impl ControlNetInput {
    pub fn new(image: Tensor) -> Self {
        Self { image, conditioning_scale: 1.0 }
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.conditioning_scale = scale;
        self
    }
}
