use synaptix_core::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct TextConditioning {
    pub prompt_embeds: Tensor,
    pub pooled_prompt_embeds: Option<Tensor>,
    pub attention_mask: Option<Tensor>,
}

impl TextConditioning {
    pub fn new(prompt_embeds: Tensor) -> Self {
        Self { prompt_embeds, pooled_prompt_embeds: None, attention_mask: None }
    }

    pub fn with_pooled(mut self, pooled: Tensor) -> Self {
        self.pooled_prompt_embeds = Some(pooled);
        self
    }

    pub fn with_mask(mut self, mask: Tensor) -> Self {
        self.attention_mask = Some(mask);
        self
    }

    pub fn seq_len(&self) -> usize {
        self.prompt_embeds.dims().get(1).copied().unwrap_or(0)
    }

    pub fn embed_dim(&self) -> usize {
        self.prompt_embeds.dims().get(2).copied().unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct DualTextConditioning {
    pub cond: TextConditioning,
    pub cond2: TextConditioning,
}

impl DualTextConditioning {
    pub fn new(cond: TextConditioning, cond2: TextConditioning) -> Self {
        Self { cond, cond2 }
    }
}

#[derive(Debug, Clone)]
pub struct CfgTextConditioning {
    pub uncond: TextConditioning,
    pub cond: TextConditioning,
}

impl CfgTextConditioning {
    pub fn new(uncond: TextConditioning, cond: TextConditioning) -> Self {
        Self { uncond, cond }
    }
}
