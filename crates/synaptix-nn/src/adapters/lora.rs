use crate::module::Module;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;

pub struct LoraConfig {
    pub r: usize,
    pub alpha: f32,
    pub dropout: f64,
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
        }
    }
}

pub struct LoraLinear {
    pub base: Linear,
    pub lora_a: Linear,
    pub lora_b: Linear,
    pub scaling: f32,
}

impl LoraLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        r: usize,
        alpha: f32,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            base: Linear::from_init(in_features, out_features, false, InitMethod::KaimingUniform { fan_in: in_features, a: 0.0 }, InitMethod::Zeros, device, dtype, 0)?,
            lora_a: Linear::from_init(in_features, r, false, InitMethod::KaimingUniform { fan_in: in_features, a: 0.0 }, InitMethod::Zeros, device, dtype, 1)?,
            lora_b: Linear::from_init(r, out_features, false, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 2)?,
            scaling: alpha / r as f32,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(x)?;
        let lora_out = self.lora_b.forward(&self.lora_a.forward(x)?)?;
        let lora_scaled = lora_out.affine(self.scaling, 0.0)?;
        base_out.add(&lora_scaled)
    }

    pub fn merge_weights(&self) -> Result<Tensor> {
        // merged = base_weight + (lora_b_weight @ lora_a_weight) * scaling
        // lora_a: [r, in],  lora_b: [out, r]
        // lora_b @ lora_a = [out, in]
        let delta = self.lora_b.weight().matmul(&self.lora_a.weight())?;
        self.base.weight().add(&delta.mul_scalar(self.scaling)?)
    }
}
