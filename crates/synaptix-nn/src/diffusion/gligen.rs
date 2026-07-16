use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// GLIGEN (Li et al., 2023) — grounded generation через gated additive injection.
///
/// `tokenize(boxes, entity_emb) = box_proj(boxes) + entity_proj(entity_emb)`,
/// форма `[B, N, hidden]`. Pooled-репрезентация (среднее по `N`) бродкастится
/// по seq-измерению и прибавляется к `x` с gated-масштабом `tanh(gate) · scale`:
///
/// `y = x + tanh(gate) · scale · broadcast(mean(grounded, dim=1))`.
///
/// Это semantic-плоский inference-stub полного gated self-attention из оригинала;
/// gate-параметр и форма входов/выходов выдержаны точно.
pub struct Gligen {
    pub entity_proj: Linear,
    pub box_proj: Linear,
    pub gate: Parameter,
    pub scale: f32,
}

impl Gligen {
    pub fn new(entity_dim: usize, hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        let gate = crate::init::init_tensor(&[1], InitMethod::Zeros, dtype, 0, device)?;
        Ok(Self {
            entity_proj: Linear::from_init(
                entity_dim, hidden_size, true,
                InitMethod::XavierUniform { fan_in: entity_dim, fan_out: hidden_size },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            box_proj: Linear::from_init(
                4, hidden_size, true,
                InitMethod::XavierUniform { fan_in: 4, fan_out: hidden_size },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            gate: Parameter::new(gate),
            scale: 1.0,
        })
    }

    pub fn from_weights(
        entity_proj_w: Tensor,
        entity_proj_b: Option<Tensor>,
        box_proj_w: Tensor,
        box_proj_b: Option<Tensor>,
        gate: Tensor,
        scale: f32,
    ) -> Result<Self> {
        Ok(Self {
            entity_proj: Linear::new(entity_proj_w, entity_proj_b)?,
            box_proj: Linear::new(box_proj_w, box_proj_b)?,
            gate: Parameter::new(gate),
            scale,
        })
    }

    /// Объединение `boxes` и `entity_emb` в grounded-токены `[B, N, hidden]`.
    pub fn tokenize(&self, boxes: &Tensor, entity_emb: &Tensor) -> Result<Tensor> {
        let e = self.entity_proj.forward(entity_emb)?;
        let p = self.box_proj.forward(boxes)?;
        e.add(&p)
    }

    /// `x: [B, T, H]`, `boxes: [B, N, 4]`, `entity_emb: [B, N, entity_dim]`.
    pub fn forward(&self, x: &Tensor, boxes: &Tensor, entity_emb: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 {
            return Err(SynaptixError::Unsupported("Gligen::forward: x must be [B, T, H]"));
        }
        let grounded = self.tokenize(boxes, entity_emb)?;
        let pooled = grounded.mean_keepdim(1)?;
        let pooled_b = pooled
            .expand(&[x.dims()[0], x.dims()[1], x.dims()[2]])?
            .contiguous()?;
        let gate_t = self.gate.tensor().tanh()?;
        let gate_scalar = scalar_f32(&gate_t)?;
        let scaled = pooled_b.affine(gate_scalar * self.scale, 0.0)?;
        x.add(&scaled)
    }
}

fn scalar_f32(t: &Tensor) -> Result<f32> {
    let flat = t.reshape(vec![1])?.to_vec1::<f32>()?;
    Ok(flat[0])
}
