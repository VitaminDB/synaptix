use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::norm::layer_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// Jamba minimal block (pre-LN top-K MoE-стиль + residual).
///
/// Полная Jamba (AI21) чередует Mamba/Attention блоки с MoE-FFN. Здесь stub:
/// 2 экспертных Linear-ветви + softmax-router из gate; каждый токен взвешивается
/// `softmax(gate(LN(x)))` по двум экспертам.
///
/// `forward(x: [B, T, hidden])` →
/// `x + sum_e( softmax(gate(LN(x)))[e] · expert_e(LN(x)) )`.
pub struct Jamba {
    pub norm_w: Parameter,
    pub norm_b: Parameter,
    pub gate: Linear,
    pub expert0: Linear,
    pub expert1: Linear,
    pub hidden_size: usize,
    pub eps: f32,
}

impl Jamba {
    pub fn new(hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            norm_w: Parameter::new(Tensor::ones(vec![hidden_size], dtype, device)?),
            norm_b: Parameter::new(Tensor::zeros(vec![hidden_size], dtype, device)?),
            gate: Linear::from_init(
                hidden_size, 2, false,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 0,
            )?,
            expert0: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            expert1: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros, device, dtype, 2,
            )?,
            hidden_size, eps: 1e-5,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        norm_w: Tensor, norm_b: Tensor,
        gate_w: Tensor,
        expert0_w: Tensor, expert0_b: Option<Tensor>,
        expert1_w: Tensor, expert1_b: Option<Tensor>,
        eps: f32,
    ) -> Result<Self> {
        let gate = Linear::new(gate_w, None)?;
        let expert0 = Linear::new(expert0_w, expert0_b)?;
        let expert1 = Linear::new(expert1_w, expert1_b)?;
        let hidden_size = gate.in_features();
        if gate.out_features() != 2 {
            return Err(SynaptixError::Unsupported("Jamba: gate must have 2 outputs (= num experts)"));
        }
        Ok(Self {
            norm_w: Parameter::new(norm_w),
            norm_b: Parameter::new(norm_b),
            gate, expert0, expert1, hidden_size, eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 || x.dims()[2] != self.hidden_size {
            return Err(SynaptixError::Unsupported("Jamba: expects x [B, T, hidden]"));
        }
        let h = layer_norm(x, Some(&self.norm_w.tensor()), Some(&self.norm_b.tensor()), self.eps)?;
        let g = self.gate.forward(&h)?;                          // [B, T, 2]
        let last = g.rank() - 1;
        let probs = synaptix_ops::attention::softmax_dim(&g, last)?;
        let w0 = probs.narrow(last, 0, 1)?;                       // [B, T, 1]
        let w1 = probs.narrow(last, 1, 1)?;                       // [B, T, 1]
        let e0 = self.expert0.forward(&h)?;
        let e1 = self.expert1.forward(&h)?;
        let blended = e0.broadcast_mul(&w0)?.add(&e1.broadcast_mul(&w1)?)?;
        x.add(&blended)
    }
}
