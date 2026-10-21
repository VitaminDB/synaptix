//! Тензорные обёртки операций обучения ([`TrainOp`]) и деквант весов для
//! обратного прохода через замороженную базу.

use std::sync::Arc;

use crate::backend::registry;
use crate::backend::train::TrainOp;
use crate::dtype::DType;
use crate::error::{Result, SynaptixError};
use crate::stream::Stream;
use crate::tensor::layout::Layout;
use crate::tensor::quant::QuantWeight;
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

impl Tensor {
    /// Выполнить [`TrainOp`]: входы делаются contiguous, выходы — новые
    /// тензоры заданных форм и типов (ядро пишет их целиком).
    pub fn train_op(op: TrainOp, inputs: &[&Tensor], outputs: &[(Vec<usize>, DType)]) -> Result<Vec<Tensor>> {
        let first = inputs.first().ok_or(SynaptixError::Unsupported("train_op: нет входов"))?;
        let device = first.device();
        let ins: Vec<Tensor> = inputs
            .iter()
            .map(|t| {
                if t.device() != device {
                    return Err(SynaptixError::device_mismatch(device, t.device()));
                }
                if t.is_contiguous() && t.layout.byte_offset() == 0 {
                    Ok((*t).clone())
                } else {
                    t.contiguous()
                }
            })
            .collect::<Result<_>>()?;
        let backend = registry::backend_for(device)?;
        let stream = Stream::default_for(device)?;
        let layouts: Vec<Layout> = outputs
            .iter()
            .map(|(dims, dt)| Layout::contiguous(Shape::new(dims.clone()), *dt))
            .collect();
        let mut storages = layouts
            .iter()
            .map(|l| backend.alloc_uninit(l.dtype().bytes_for_numel(l.numel()).max(1), device))
            .collect::<Result<Vec<_>>>()?;
        {
            let in_refs: Vec<_> = ins.iter().map(|t| (&*t.storage, &t.layout)).collect();
            let mut out_refs: Vec<_> = storages.iter_mut().zip(layouts.iter()).map(|(s, l)| (s, l)).collect();
            backend.train_op(op, &in_refs, &mut out_refs, &stream)?;
        }
        Ok(storages
            .into_iter()
            .zip(layouts)
            .map(|(s, l)| Tensor::from_parts(Arc::new(s), l))
            .collect())
    }
}

impl QuantWeight {
    /// Плотная копия веса `[n, k]` в BF16 на его устройстве — для обратного
    /// прохода `dx = dy · W`. Работает и после того, как первое умножение
    /// заменило линейный packed NVFP4 перемешанной копией.
    pub fn dequantize_bf16(&self) -> Result<Tensor> {
        if self.dtype() == DType::NVFP4 && self.packed_arc().is_none() {
            let shuf = self
                .shuffled_arc()
                .ok_or(SynaptixError::Unsupported("dequantize_bf16: у NVFP4 нет ни packed, ни перемешанной копии"))?;
            let scales = self.scales_arc().ok_or(SynaptixError::Unsupported("dequantize_bf16: NVFP4 без шкал"))?;
            let (n, k) = (self.n(), self.k());
            let sh = Tensor::from_parts(shuf.clone(), Layout::contiguous(Shape::new(vec![shuf.byte_len()]), DType::U8));
            let sc = Tensor::from_parts(scales.clone(), Layout::contiguous(Shape::new(vec![scales.byte_len()]), DType::U8));
            let out = Tensor::train_op(TrainOp::Nvfp4DequantShuffled { n, k }, &[&sh, &sc], &[(vec![n, k], DType::BF16)])?;
            return Ok(out.into_iter().next().unwrap());
        }
        match self.dequantize(DType::BF16) {
            Ok(t) => Ok(t),
            // MXFP8 деквантуется только в F16.
            Err(SynaptixError::Unsupported(_)) => self.dequantize(DType::F16)?.to_dtype(DType::BF16),
            Err(e) => Err(e),
        }
    }
}
