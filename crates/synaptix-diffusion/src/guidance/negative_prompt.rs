use synaptix_core::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct NegativePromptCond {
    pub embeds: Tensor,
    pub pooled: Option<Tensor>,
    pub attention_mask: Option<Tensor>,
}

impl NegativePromptCond {
    pub fn new(embeds: Tensor) -> Self {
        Self { embeds, pooled: None, attention_mask: None }
    }

    pub fn with_pooled(mut self, pooled: Tensor) -> Self {
        self.pooled = Some(pooled);
        self
    }

    pub fn with_mask(mut self, mask: Tensor) -> Self {
        self.attention_mask = Some(mask);
        self
    }
}
