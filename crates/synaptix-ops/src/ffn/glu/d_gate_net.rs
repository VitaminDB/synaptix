use synaptix_core::{error::Result, tensor::Tensor};

/// Dynamic gate network (data-dependent gating): `y = x ⊙ sigmoid(x · W)`.
/// `x:[..,D]`, `gate_weight:[D,D]` (используется как есть, без транспонирования).
pub fn d_gate_net(x: &Tensor, gate_weight: &Tensor) -> Result<Tensor> {
    let gate = x.matmul(gate_weight)?.sigmoid()?;
    x.mul(&gate)
}
