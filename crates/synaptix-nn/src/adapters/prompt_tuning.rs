use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::parameter::Parameter;

/// Soft Prompt Tuning — обучаемые `num_tokens` эмбеддингов, которые
/// конкатенируются к входу по последовательности слева.
pub struct PromptTuning {
    pub soft_prompts: Parameter,
    pub num_tokens: usize,
    pub hidden_size: usize,
}

impl PromptTuning {
    pub fn new(num_tokens: usize, hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        let sp = crate::init::init_tensor(
            &[num_tokens, hidden_size],
            InitMethod::Normal { mean: 0.0, std: 0.02 },
            dtype, 0, device,
        )?;
        Ok(Self {
            soft_prompts: Parameter::new(sp),
            num_tokens,
            hidden_size,
        })
    }

    pub fn from_weights(soft_prompts: Tensor) -> Result<Self> {
        if soft_prompts.rank() != 2 {
            return Err(SynaptixError::Unsupported("PromptTuning: soft_prompts must be [num_tokens, hidden]"));
        }
        let num_tokens = soft_prompts.dims()[0];
        let hidden_size = soft_prompts.dims()[1];
        Ok(Self {
            soft_prompts: Parameter::new(soft_prompts),
            num_tokens,
            hidden_size,
        })
    }

    /// `x: [B, T, H]` → `[B, num_tokens + T, H]`.
    pub fn prepend(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 {
            return Err(SynaptixError::Unsupported("PromptTuning::prepend: input must be [B, T, H]"));
        }
        if x.dims()[2] != self.hidden_size {
            return Err(SynaptixError::shape_mismatch(&[x.dims()[0], x.dims()[1], self.hidden_size], x.dims()));
        }
        let batch = x.dims()[0];
        let sp = self.soft_prompts.tensor();
        let sp_b = sp.unsqueeze(0)?.expand(&[batch, self.num_tokens, self.hidden_size])?.contiguous()?;
        Tensor::cat(&[&sp_b, x], 1)
    }
}
