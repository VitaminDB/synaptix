use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::parameter::Parameter;

pub struct DoraLinear {
    pub base: Linear,
    pub lora_a: Linear,
    pub lora_b: Linear,
    pub magnitude: Parameter,
    pub scaling: f32,
    pub eps: f32,
}

impl DoraLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        r: usize,
        alpha: f32,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let mag = crate::init::init_tensor(
            &[out_features],
            InitMethod::Ones,
            dtype, 0, device,
        )?;
        Ok(Self {
            base: Linear::from_init(
                in_features, out_features, false,
                InitMethod::KaimingUniform { fan_in: in_features, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            lora_a: Linear::from_init(
                in_features, r, false,
                InitMethod::KaimingUniform { fan_in: in_features, a: 0.0 },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            lora_b: Linear::from_init(
                r, out_features, false,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 2,
            )?,
            magnitude: Parameter::new(mag),
            scaling: alpha / r as f32,
            eps: 1e-8,
        })
    }

    pub fn from_weights(
        base_w: Tensor,
        lora_a_w: Tensor,
        lora_b_w: Tensor,
        magnitude: Tensor,
        scaling: f32,
    ) -> Result<Self> {
        Ok(Self {
            base: Linear::new(base_w, None)?,
            lora_a: Linear::new(lora_a_w, None)?,
            lora_b: Linear::new(lora_b_w, None)?,
            magnitude: Parameter::new(magnitude),
            scaling,
            eps: 1e-8,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.base.weight();
        let a = self.lora_a.weight();
        let b = self.lora_b.weight();
        let delta = b.matmul(&a)?.affine(self.scaling, 0.0)?;
        let v = w.add(&delta)?;

        let v_sq = v.mul(&v)?;
        let v_sq_sum = v_sq.sum_keepdim(1)?;
        let v_norm = v_sq_sum.sqrt()?.affine(1.0, self.eps)?;
        let v_normalized = v.broadcast_div(&v_norm)?;

        let mag = self.magnitude.tensor();
        let mag_col = mag.unsqueeze(1)?;
        let scaled = v_normalized.broadcast_mul(&mag_col)?;

        let scaled_t = scaled.transpose(0, 1)?.contiguous()?;
        x.matmul(&scaled_t)
    }
}
