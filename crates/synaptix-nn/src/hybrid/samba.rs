use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::norm::layer_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// Samba minimal block (pre-LN GLU + residual + masking-gate).
///
/// Полный Samba (Microsoft) — Mamba + sliding-window-attention; здесь stub
/// делает pre-LN, GLU-разбиение через `fc_in [hidden → 2·hidden]`, и
/// per-token gated scaling по learnable `window_gate` (scalar в struct, но
/// загружается через 1-элементный Parameter).
///
/// `forward(x: [B, T, hidden])` → `x + sigmoid(window_gate) · fc_out(silu(a) ⊙ b)`.
pub struct Samba {
    pub norm_w: Parameter,
    pub norm_b: Parameter,
    pub fc_in: Linear,
    pub fc_out: Linear,
    pub window_gate: Parameter,
    pub hidden_size: usize,
    pub eps: f32,
}

impl Samba {
    pub fn new(hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            norm_w: Parameter::new(Tensor::ones(vec![hidden_size], dtype, device)?),
            norm_b: Parameter::new(Tensor::zeros(vec![hidden_size], dtype, device)?),
            fc_in: Linear::from_init(
                hidden_size, hidden_size * 2, true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            fc_out: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 1,
            )?,
            window_gate: Parameter::new(
                crate::init::init_tensor(&[1], InitMethod::Zeros, dtype, 0, device)?,
            ),
            hidden_size, eps: 1e-5,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        norm_w: Tensor, norm_b: Tensor,
        fc_in_w: Tensor, fc_in_b: Option<Tensor>,
        fc_out_w: Tensor, fc_out_b: Option<Tensor>,
        window_gate: Tensor,
        eps: f32,
    ) -> Result<Self> {
        let fc_in = Linear::new(fc_in_w, fc_in_b)?;
        let fc_out = Linear::new(fc_out_w, fc_out_b)?;
        let hidden_size = fc_in.in_features();
        Ok(Self {
            norm_w: Parameter::new(norm_w),
            norm_b: Parameter::new(norm_b),
            fc_in, fc_out,
            window_gate: Parameter::new(window_gate),
            hidden_size, eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 || x.dims()[2] != self.hidden_size {
            return Err(SynaptixError::Unsupported("Samba: expects x [B, T, hidden]"));
        }
        let h = layer_norm(x, Some(&self.norm_w.tensor()), Some(&self.norm_b.tensor()), self.eps)?;
        let ab = self.fc_in.forward(&h)?;
        let a = ab.narrow(2, 0, self.hidden_size)?.contiguous()?;
        let b = ab.narrow(2, self.hidden_size, self.hidden_size)?.contiguous()?;
        let gated = a.silu()?.mul(&b)?;
        let out = self.fc_out.forward(&gated)?;
        let gate_t = self.window_gate.tensor().sigmoid()?;
        let scale = gate_t.reshape(vec![1])?.to_vec1::<f32>()?[0];
        let scaled = out.affine(scale, 0.0)?;
        x.add(&scaled)
    }
}
