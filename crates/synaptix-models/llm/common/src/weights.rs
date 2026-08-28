use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::Tensor;
use synaptix_nn::linear::Linear;

use crate::model::ModelError;

pub trait WeightSource {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, ModelError>;
    fn contains(&self, key: &str) -> bool;

    /// Готовый квант-вес, если он лежит в источнике уже упакованным
    /// (бандл собран с `syn-quant-v1`).
    ///
    /// `None` — источник хранит плотные веса, работает обычный путь:
    /// прочитать F16 и посчитать квант на GPU. `Some(Err(_))` — вес
    /// квантован, но прочитать его не удалось; свалиться на плотный путь
    /// нельзя, плотной копии в таком бандле нет.
    ///
    /// Реализация по умолчанию отвечает `None`, поэтому источники, которые
    /// про квант ничего не знают, продолжают работать как раньше.
    fn quant(&self, _key: &str, _device: Device) -> Option<Result<QuantWeight, ModelError>> {
        None
    }
}

pub enum QLinear {
    Dense(Linear),
    Quant(QuantWeight),
}

impl QLinear {
    /// `quant_dtype` задаёт схему кванта веса: NVFP4 (требует N%64==0,K%64==0),
    /// MXFP8 (требует K%32==0), либо любой неквантованный dtype → плотный Linear
    /// (вес кастуется в `compute`). Если форма не подходит под выбранную схему —
    /// тихий fallback в Dense (тот же путь, что был у NVFP4 с неподходящими N/K).
    pub fn build(weight: Tensor, quant_dtype: DType, compute: DType) -> Result<Self, ModelError> {
        let n = weight.dims()[0];
        let k = weight.dims()[1];
        let quant = match quant_dtype {
            DType::NVFP4 if n % 64 == 0 && k % 64 == 0 => Some(
                weight
                    .quantize_to_nvfp4()
                    .map_err(|e| ModelError::Build(format!("quantize_to_nvfp4: {e}")))?,
            ),
            DType::MXFP8 if k % 32 == 0 => Some(
                weight
                    .quantize_to_mxfp8()
                    .map_err(|e| ModelError::Build(format!("quantize_to_mxfp8: {e}")))?,
            ),
            _ => None,
        };
        if let Some(qw) = quant {
            Ok(QLinear::Quant(qw))
        } else {
            let w = if weight.dtype() == compute {
                weight
            } else {
                weight.to_dtype(compute).map_err(|e| ModelError::Build(e.to_string()))?
            };
            Ok(QLinear::Dense(
                Linear::new(w, None).map_err(|e| ModelError::Build(e.to_string()))?,
            ))
        }
    }

    /// NVFP4-Quant ли вес — можно ли использовать prequant-путь (общая квант-активация).
    pub fn is_nvfp4(&self) -> bool {
        matches!(self, QLinear::Quant(w) if w.dtype() == DType::NVFP4)
    }

    /// Формат квант-веса (NVFP4|MXFP8), `None` для Dense — выбор формата
    /// prequant-пары на call-site.
    pub fn quant_dtype(&self) -> Option<DType> {
        match self {
            QLinear::Quant(w) => Some(w.dtype()),
            QLinear::Dense(_) => None,
        }
    }

    /// GEMM/GEMV из УЖЕ квантованной активации (`packed`/`scales` от
    /// `Tensor::{nvfp4,mxfp8}_quantize_act` / fused norm-quant, посчитанной 1×
    /// для общего `h`; формат пары = формат веса). Для Dense — `Unsupported`
    /// (caller падает в [`Self::forward`]). `m` строк.
    pub fn forward_prequant(
        &self,
        packed: &Tensor,
        scales: &Tensor,
        m: usize,
    ) -> Result<Tensor, ModelError> {
        match self {
            QLinear::Quant(w) if matches!(w.dtype(), DType::NVFP4 | DType::MXFP8) => packed
                .linear_quant_prequant(scales, w, m, DType::F16)
                .map_err(|e| ModelError::Forward(e.to_string())),
            _ => Err(ModelError::Forward("forward_prequant: вес не NVFP4/MXFP8".into())),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, ModelError> {
        match self {
            QLinear::Dense(l) => {
                use synaptix_nn::module::Module;
                l.forward(x).map_err(|e| ModelError::Forward(e.to_string()))
            }
            QLinear::Quant(w) => {
                let in_dt = x.dtype();
                if in_dt == DType::F16 {
                    x.linear_quant(w).map_err(|e| ModelError::Forward(e.to_string()))
                } else {
                    let xf = x.to_dtype(DType::F16).map_err(|e| ModelError::Forward(e.to_string()))?;
                    let yf = xf.linear_quant(w).map_err(|e| ModelError::Forward(e.to_string()))?;
                    yf.to_dtype(in_dt).map_err(|e| ModelError::Forward(e.to_string()))
                }
            }
        }
    }

    /// Перенос весов на устройство (host-stream блоков: CPU-резидент → GPU по
    /// требованию в forward, как DiT-блоки LTX).
    pub fn to_device(&self, dev: Device) -> Result<Self, ModelError> {
        Ok(match self {
            QLinear::Dense(l) => {
                let w = l.weight().to_device(dev).map_err(|e| ModelError::Load(e.to_string()))?;
                let b = match l.bias() {
                    Some(b) => Some(b.to_device(dev).map_err(|e| ModelError::Load(e.to_string()))?),
                    None => None,
                };
                QLinear::Dense(Linear::new(w, b).map_err(|e| ModelError::Load(e.to_string()))?)
            }
            QLinear::Quant(w) => QLinear::Quant(
                w.to_device(dev).map_err(|e| ModelError::Load(e.to_string()))?,
            ),
        })
    }
}
