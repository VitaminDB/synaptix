use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// VeRA — Vector-based Random Matrix Adaptation.
///
/// `A_shared` ([r, in]) и `B_shared` ([out, r]) — фиксированные random-проекции,
/// общие для всех модулей с одинаковой формой; на каждом модуле обучаются только
/// диагональные масштабы `lambda_d` ([r]) и `lambda_b` ([out]).
/// Forward: `y = base(x) + (lambda_b ⊙ (B_shared · (lambda_d ⊙ (A_shared · x))))`.
pub struct VeraLinear {
    pub base: Linear,
    pub a_shared: Parameter,
    pub b_shared: Parameter,
    pub lambda_d: Parameter,
    pub lambda_b: Parameter,
}

impl VeraLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        r: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let a = crate::init::init_tensor(
            &[r, in_features],
            InitMethod::KaimingUniform { fan_in: in_features, a: 0.0 },
            dtype, 100, device,
        )?;
        let b = crate::init::init_tensor(
            &[out_features, r],
            InitMethod::KaimingUniform { fan_in: r, a: 0.0 },
            dtype, 101, device,
        )?;
        let ld = crate::init::init_tensor(&[r], InitMethod::Ones, dtype, 0, device)?;
        let lb = crate::init::init_tensor(&[out_features], InitMethod::Zeros, dtype, 1, device)?;
        Ok(Self {
            base: Linear::from_init(
                in_features, out_features, false,
                InitMethod::KaimingUniform { fan_in: in_features, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            a_shared: Parameter::new(a),
            b_shared: Parameter::new(b),
            lambda_d: Parameter::new(ld),
            lambda_b: Parameter::new(lb),
        })
    }

    pub fn from_weights(
        base_w: Tensor,
        a_shared: Tensor,
        b_shared: Tensor,
        lambda_d: Tensor,
        lambda_b: Tensor,
    ) -> Result<Self> {
        Ok(Self {
            base: Linear::new(base_w, None)?,
            a_shared: Parameter::new(a_shared),
            b_shared: Parameter::new(b_shared),
            lambda_d: Parameter::new(lambda_d),
            lambda_b: Parameter::new(lambda_b),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(x)?;
        let a_w = self.a_shared.tensor();
        let a_t = a_w.transpose(0, 1)?.contiguous()?;
        let ax = x.matmul(&a_t)?;
        let ld = self.lambda_d.tensor();
        let ax_scaled = ax.broadcast_mul(&ld)?;
        let b_w = self.b_shared.tensor();
        let b_t = b_w.transpose(0, 1)?.contiguous()?;
        let bx = ax_scaled.matmul(&b_t)?;
        let lb = self.lambda_b.tensor();
        let bx_scaled = bx.broadcast_mul(&lb)?;
        base_out.add(&bx_scaled)
    }
}
