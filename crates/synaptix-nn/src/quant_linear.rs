//! Linear с опциональным квантованием веса (NVFP4/MXFP8) + bias — общий для всех
//! моделей (LLM/image: FLUX, SDXL, …). `Dense` = плотный [`Linear`] (вес в
//! compute-dtype, поддерживает streaming/offload через [`to_device`]). `Quant` =
//! вес NVFP4 (N%64==0,K%64==0) либо MXFP8 (K%32==0); активация считается в F16,
//! bias добавляется после (broadcast). Квантованные веса РЕЗИДЕНТНЫ (малы → нет
//! смысла стримить), поэтому `to_device` на `Quant` не поддержан.

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

/// Целевой максимум активации перед F16-кастом. И вход, и выход `linear_quant`
/// живут в F16 (потолок 65504), поэтому запас нужен не только под саму
/// активацию, но и под усиление слоя: |y| <= amax * ||w_row||, а у трансформера
/// норма строки порядка 5–10. При 64 на входе выход остаётся в районе сотен.
const F16_TARGET: f32 = 64.0;

/// Квант-ядра принимают и отдают активацию в F16, но у диффузионных
/// трансформеров massive activations уходят далеко за 65504 — в середине стека
/// это превращалось в Inf. GEMM линеен по активации, поэтому вход приводится к
/// целевому масштабу до каста и результат возвращается обратно после.
///
/// Масштаб остаётся тензором на устройстве: читать его на хост означало бы
/// синхронизацию на каждый Linear, а их сотни за шаг.
fn quant_matmul(x: &Tensor, w: &QuantWeight) -> Result<Tensor> {
    let in_dt = x.dtype();
    if in_dt == DType::F16 {
        return x.linear_quant(w);
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
    x.broadcast_mul(&inv)?
        .to_dtype(DType::F16)?
        .linear_quant(w)?
        .to_dtype(in_dt)?
        .broadcast_mul(&scale)
}

impl QuantLinear {
    /// Из плотного `[out,in]` веса (+ опц. bias `[out]`). `quant_dtype` выбирает
    /// схему; несовместимая форма → тихий fallback в Dense (вес кастуется в
    /// `compute`). bias всегда в `compute`-dtype.
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
        // Квант-ядра требуют F16-вес. bf16→f16 без потерь (f16: 10 бит мантиссы > bf16: 7),
        // веса трансформера в диапазоне f16 → точный каст.
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

    /// Плотный Linear без квантования (вес как есть). Для слоёв, которые не квантуем.
    pub fn dense(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(QuantLinear::Dense(Linear::new(weight, bias)?))
    }

    pub fn is_quant(&self) -> bool {
        matches!(self, QuantLinear::Quant { .. })
    }

    /// `forward(x) + residual` за один проход. Dense → fused-эпилог [`Linear::forward_add`]
    /// (bit-identical). Quant → linear_quant + bias + residual (broadcast).
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

    /// Перенос на `dev` (layer-streaming). `Quant` переносит packed+scales
    /// побайтово (квантуем 1× → host-RAM → стрим обратно, bit-identical).
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
    /// Вес NVFP4 → активацию можно подать prequant-парой (packed, scales).
    pub fn is_nvfp4(&self) -> bool {
        matches!(self, QuantLinear::Quant { w, .. } if w.dtype() == DType::NVFP4)
    }
    /// Вес MXFP8 → активацию можно подать prequant-парой (packed, natural scales).
    pub fn is_mxfp8(&self) -> bool {
        matches!(self, QuantLinear::Quant { w, .. } if w.dtype() == DType::MXFP8)
    }
    /// Формат квант-веса (NVFP4|MXFP8), `None` для Dense — выбор формата
    /// prequant-пары на call-site (fused norm-quant).
    pub fn quant_dtype(&self) -> Option<DType> {
        match self {
            QuantLinear::Quant { w, .. } => Some(w.dtype()),
            QuantLinear::Dense(_) => None,
        }
    }
    /// Проекция из УЖЕ квантованной активации (packed, scales) — пропускает
    /// f16-каст и квант. Возвращает `[m, n]` в `out_dt`. Формат пары должен
    /// совпадать с форматом веса (NVFP4|MXFP8).
    pub fn forward_prequant(
        &self,
        packed: &Tensor,
        scales: &Tensor,
        m: usize,
        out_dt: DType,
    ) -> Result<Tensor> {
        match self {
            QuantLinear::Quant { w, bias } => {
                let y = packed.linear_quant_prequant(scales, w, m)?;
                let y = if out_dt == DType::F16 { y } else { y.to_dtype(out_dt)? };
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
