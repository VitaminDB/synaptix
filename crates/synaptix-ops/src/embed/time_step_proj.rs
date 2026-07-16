use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::activation::silu::silu;
use crate::embed::timestep_embed::timestep_embedding;

pub fn timestep_projection(
    timesteps: &Tensor,
    embed_dim: usize,
    max_period: f32,
    w1: &Tensor,
    b1: Option<&Tensor>,
    w2: &Tensor,
    b2: Option<&Tensor>,
) -> Result<Tensor> {
    let emb = timestep_embedding(timesteps, embed_dim, max_period)?;
    let w1_t = w1.transpose(0, 1)?.contiguous()?;
    let mut h = emb.matmul(&w1_t)?;
    if let Some(b) = b1 {
        h = h.broadcast_add(b)?;
    }
    let h_act = silu(&h)?;
    let w2_t = w2.transpose(0, 1)?.contiguous()?;
    let mut out = h_act.matmul(&w2_t)?;
    if let Some(b) = b2 {
        out = out.broadcast_add(b)?;
    }
    Ok(out)
}
