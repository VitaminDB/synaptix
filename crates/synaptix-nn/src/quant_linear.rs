use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::Tensor;

use crate::linear::Linear;
use crate::module::Module;

pub enum QuantLinear {
    Dense(Linear),
    Quant { w: QuantWeight, bias: Option<Tensor> },
}

const F16_TARGET: f32 = 64.0;

fn prescale_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("SYNAPTIX_NO_ACT_PRESCALE").as_deref(), Ok("1")))
}

fn nvfp4_weight_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("SYNAPTIX_NVFP4_WO").as_deref(), Ok("1")))
}

fn quant_matmul(x: &Tensor, w: &QuantWeight) -> Result<Tensor> {
    let in_dt = x.dtype();
    if in_dt == DType::F16 {
        return x.linear_quant(w);
    }
    let native = in_dt == DType::BF16 && w.dtype() == DType::NVFP4 && !nvfp4_weight_only();
    if !prescale_enabled() {
        if native {
            return x.linear_quant(w);
        }
        return x.to_dtype(DType::F16)?.linear_quant(w)?.to_dtype(in_dt);
    }
    let scale = match x
        .abs()
        .and_then(|t| t.max_all())
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.mul_scalar(1.0 / F16_TARGET))
        .and_then(|t| t.add_scalar(f32::MIN_POSITIVE))
        .and_then(|t| t.reshape(vec![1]))
    {
        Ok(s) => s,
        Err(_) => return x.to_dtype(DType::F16)?.linear_quant(w)?.to_dtype(in_dt),
    };
    let inv = scale.recip()?.to_dtype(in_dt)?;
    let scale = scale.to_dtype(in_dt)?;
    let xs = x.broadcast_mul(&inv)?;
    let y = if native {
        xs.linear_quant(w)?
    } else {
        xs.to_dtype(DType::F16)?.linear_quant(w)?.to_dtype(in_dt)?
    };
    y.broadcast_mul(&scale)
}

impl QuantLinear {
    pub fn build(
        weight: Tensor,
        bias: Option<Tensor>,
        quant_dtype: DType,
        compute: DType,
    ) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(SynaptixError::Unsupported("QuantLinear: weight must be 2D"));
        }
        let (n, k) = (dims[0], dims[1]);
        let bias_c = match bias {
            Some(b) => Some(b.to_dtype(compute)?),
            None => None,
        };
        let to_f16 = |w: Tensor| -> Result<Tensor> {
            if w.dtype() == DType::F16 { Ok(w) } else { w.to_dtype(DType::F16) }
        };
        let quant = match quant_dtype {
            DType::NVFP4 if n % 64 == 0 && k % 64 == 0 => Some(to_f16(weight.clone())?.quantize_to_nvfp4()?),
            DType::MXFP8 if k % 32 == 0 => Some(to_f16(weight.clone())?.quantize_to_mxfp8()?),
            _ => None,
        };
        match quant {
            Some(w) => Ok(QuantLinear::Quant { w, bias: bias_c }),
            None => {
                let w = if weight.dtype() == compute {
                    weight
                } else {
                    weight.to_dtype(compute)?
                };
                Ok(QuantLinear::Dense(Linear::new(w, bias_c)?))
            }
        }
    }

    pub fn dense(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(QuantLinear::Dense(Linear::new(weight, bias)?))
    }

    pub fn is_quant(&self) -> bool {
        matches!(self, QuantLinear::Quant { .. })
    }

    pub fn forward_add(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        match self {
            QuantLinear::Dense(l) => l.forward_add(x, residual),
            QuantLinear::Quant { w, bias } => {
                let y = quant_matmul(x, w)?;
                let y = match bias {
                    Some(b) => y.broadcast_add(b)?,
                    None => y,
                };
                y.add(residual)
            }
        }
    }

    pub fn to_device(&self, dev: Device) -> Result<Self> {
        match self {
            QuantLinear::Dense(l) => {
                let w = l.weight().to_device(dev)?;
                let b = l.bias().map(|t| t.to_device(dev)).transpose()?;
                Ok(QuantLinear::Dense(Linear::new(w, b)?))
            }
            QuantLinear::Quant { w, bias } => Ok(QuantLinear::Quant {
                w: w.to_device(dev)?,
                bias: bias.as_ref().map(|t| t.to_device(dev)).transpose()?,
            }),
        }
    }
}

impl QuantLinear {
    pub fn is_nvfp4(&self) -> bool {
        matches!(self, QuantLinear::Quant { w, .. } if w.dtype() == DType::NVFP4)
    }
    pub fn is_mxfp8(&self) -> bool {
        matches!(self, QuantLinear::Quant { w, .. } if w.dtype() == DType::MXFP8)
    }
    pub fn quant_dtype(&self) -> Option<DType> {
        match self {
            QuantLinear::Quant { w, .. } => Some(w.dtype()),
            QuantLinear::Dense(_) => None,
        }
    }
    pub fn forward_prequant(
        &self,
        packed: &Tensor,
        scales: &Tensor,
        m: usize,
        out_dt: DType,
    ) -> Result<Tensor> {
        match self {
            QuantLinear::Quant { w, bias } => {
                let gemm_dt = if matches!(out_dt, DType::F16 | DType::BF16) {
                    out_dt
                } else {
                    DType::F16
                };
                let y = packed.linear_quant_prequant(scales, w, m, gemm_dt)?;
                let y = if y.dtype() == out_dt { y } else { y.to_dtype(out_dt)? };
                match bias {
                    Some(b) => y.broadcast_add(b),
                    None => Ok(y),
                }
            }
            QuantLinear::Dense(_) => Err(SynaptixError::Unsupported("forward_prequant: Dense")),
        }
    }
}

impl Module for QuantLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            QuantLinear::Dense(l) => l.forward(x),
            QuantLinear::Quant { w, bias } => {
                let y = quant_matmul(x, w)?;
                match bias {
                    Some(b) => y.broadcast_add(b),
                    None => Ok(y),
                }
            }
        }
    }
}
