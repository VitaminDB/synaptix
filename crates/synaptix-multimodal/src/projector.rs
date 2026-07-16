use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct MlpProjectorWeights {
    pub fc1_weight: Tensor,
    pub fc1_bias: Option<Tensor>,
    pub fc2_weight: Tensor,
    pub fc2_bias: Option<Tensor>,
}

pub fn mlp_projector(input: &Tensor, weights: &MlpProjectorWeights) -> Result<Tensor> {
    let mut h = input.matmul(&weights.fc1_weight)?;
    if let Some(b) = weights.fc1_bias.as_ref() {
        h = h.broadcast_add(b)?;
    }
    h = h.gelu_exact()?;
    let mut out = h.matmul(&weights.fc2_weight)?;
    if let Some(b) = weights.fc2_bias.as_ref() {
        out = out.broadcast_add(b)?;
    }
    Ok(out)
}
