use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::parameter::Parameter;

pub enum PoolingStrategy {
    Cls,
    Mean,
    Max,
    Last,
}

pub struct AttentionPooling {
    pub query: Parameter,
    pub proj: Linear,
}

impl AttentionPooling {
    pub fn new(hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        let q = crate::init::init_tensor(&[1, hidden_size], InitMethod::Normal { mean: 0.0, std: 0.02 }, dtype, 0, device)?;
        Ok(Self {
            query: Parameter::new(q),
            proj: Linear::from_init(hidden_size, hidden_size, false, InitMethod::XavierUniform { fan_in: hidden_size, fan_out: hidden_size }, InitMethod::Zeros, device, dtype, 0)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Cross-attention with learned query: q: [1, H] expanded to [B, 1, H]
        // Simplified: dot-product with query → softmax → weighted sum
        use synaptix_ops::attention::softmax_dim;
        let q = self.query.tensor(); // [1, H]
        let batch = x.dims()[0];
        let h = x.dims()[x.rank() - 1];
        let q_b = q.unsqueeze(0)?.reshape(vec![1, 1, h])?.broadcast_mul(
            &Tensor::ones(vec![batch, 1, h], q.dtype(), q.device())?
        )?; // [B, 1, H]
        let x_t = x.permute(vec![0, 2, 1])?.contiguous()?; // [B, H, S]
        let scores = q_b.matmul(&x_t)?; // [B, 1, S]
        let scale = (h as f32).sqrt().recip();
        let scores = scores.mul_scalar(scale)?;
        let attn = softmax_dim(&scores, 2)?; // [B, 1, S]
        let pooled = attn.matmul(&x)?.squeeze(1)?; // [B, H]
        crate::module::Module::forward(&self.proj, &pooled)
    }
}

pub fn pool(x: &Tensor, strategy: &PoolingStrategy) -> Result<Tensor> {
    match strategy {
        PoolingStrategy::Cls => crate::pooling::cls_pool(x),
        PoolingStrategy::Last => crate::pooling::last_pool(x),
        PoolingStrategy::Mean => crate::pooling::mean_pool(x, x.rank() - 2),
        PoolingStrategy::Max => crate::pooling::max_pool(x, x.rank() - 2),
    }
}
