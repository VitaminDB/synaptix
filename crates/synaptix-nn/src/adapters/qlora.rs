use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::adapters::lora::LoraLinear;

/// QLoRA = LoRA over a NF4/INT8-quantized frozen base.
///
/// Dequantization of the packed base weight lives in the loader path
/// (`ai_quant` / `synaptix-bundle`); by the time the module reaches the runtime
/// the base is already a dense floating-point tensor, so for inference QLoRA
/// is bit-exact equivalent to a plain LoRA on top of that dequantized base.
pub struct QLoraLinear {
    pub lora: LoraLinear,
}

impl QLoraLinear {
    pub fn new(in_features: usize, out_features: usize, r: usize, alpha: f32, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            lora: LoraLinear::new(in_features, out_features, r, alpha, device, dtype)?,
        })
    }

    pub fn from_weights(
        dequantized_base_w: Tensor,
        lora_a_w: Tensor,
        lora_b_w: Tensor,
        scaling: f32,
    ) -> Result<Self> {
        Ok(Self {
            lora: LoraLinear {
                base: crate::linear::Linear::new(dequantized_base_w, None)?,
                lora_a: crate::linear::Linear::new(lora_a_w, None)?,
                lora_b: crate::linear::Linear::new(lora_b_w, None)?,
                scaling,
            },
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.lora.forward(x)
    }
}
