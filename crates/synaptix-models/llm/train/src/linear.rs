//! Замороженная проекция [`QLinear`] на обучении.
//!
//! Квантованная база по умолчанию умножается как в QLoRA: вес слоя
//! деквантуется в BF16 и умножается на BF16-активацию (квантованы только
//! веса, W4A16). Режим [`BaseMatmul::Native`] берёт ядра инференса — у NVFP4
//! они квантуют и активацию (W4A4): быстрее, но заметно грубее. Обратный
//! проход по входу `dx = dy · W` — всегда через плотную копию веса.
//!
//! Плотные копии весов слоя живут в [`Weights`] — одна на слой на время его
//! прямого и обратного хода, так что деквант делается раз на матрицу.

use std::cell::RefCell;

use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::QLinear;

use crate::Result;

/// Как умножать на квантованную базу в прямом проходе.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BaseMatmul {
    /// Деквант веса в BF16 + BF16-GEMM (QLoRA, W4A16).
    #[default]
    Dequant,
    /// Ядра инференса (у NVFP4 — с квантом активации).
    Native,
}

/// Кэш плотных копий весов одного слоя.
pub struct Weights {
    mode: BaseMatmul,
    cache: RefCell<Vec<(usize, Tensor)>>,
}

impl Weights {
    pub fn new(mode: BaseMatmul) -> Self {
        Self { mode, cache: RefCell::new(Vec::new()) }
    }

    /// Плотный вес `[N, K]` (BF16; у F32-базы — F32).
    pub fn dense(&self, q: &QLinear) -> Result<Tensor> {
        let key = q as *const QLinear as usize;
        if let Some((_, w)) = self.cache.borrow().iter().find(|(k, _)| *k == key) {
            return Ok(w.clone());
        }
        let w = crate::prof::prof(q_device(q), "dequant", || dense_weight(q))?;
        self.cache.borrow_mut().push((key, w.clone()));
        Ok(w)
    }

    /// `y = x · Wᵀ` в compute-типе `x`, `x` — `[T, K]`.
    pub fn forward(&self, q: &QLinear, x: &Tensor) -> Result<Tensor> {
        let y = match (q, self.mode) {
            (QLinear::Quant(_), BaseMatmul::Dequant) => {
                let w = self.dense(q)?;
                crate::prof::prof(x.device(), "w_fwd", || -> Result<Tensor> { Ok(x.to_dtype(w.dtype())?.matmul(&w.t()?)?) })?
            }
            _ => crate::prof::prof(x.device(), "w_fwd", || q.forward(x))?,
        };
        Ok(if y.dtype() != x.dtype() { y.to_dtype(x.dtype())? } else { y })
    }

    /// `dx = dy · W`: `dy` — `[T, N]` F32, результат `[T, K]` F32.
    pub fn backward_input(&self, q: &QLinear, dy: &Tensor) -> Result<Tensor> {
        let w = self.dense(q)?;
        crate::prof::prof(dy.device(), "w_bwd", || backward_input_with(&w, dy))
    }
}

/// Плотный вес `[N, K]` в BF16 (F32-база остаётся в F32 — для сверки с
/// эталоном).
pub fn dense_weight(q: &QLinear) -> Result<Tensor> {
    Ok(match q {
        QLinear::Dense(l) => {
            let w = l.weight();
            if matches!(w.dtype(), DType::BF16 | DType::F32) { w } else { w.to_dtype(DType::BF16)? }
        }
        QLinear::Quant(w) => w.dequantize_bf16()?,
    })
}

/// `dx = dy · W` с готовой плотной копией веса.
pub fn backward_input_with(w: &Tensor, dy: &Tensor) -> Result<Tensor> {
    if w.dtype() == DType::F32 {
        return Ok(dy.to_dtype(DType::F32)?.matmul(w)?);
    }
    Ok(dy.to_dtype(w.dtype())?.matmul(w)?.to_dtype(DType::F32)?)
}

/// `(N, K)` проекции.
pub fn dims(q: &QLinear) -> (usize, usize) {
    match q {
        QLinear::Dense(l) => {
            let w = l.weight();
            (w.dims()[0], w.dims()[1])
        }
        QLinear::Quant(w) => (w.n(), w.k()),
    }
}

fn q_device(q: &QLinear) -> synaptix_core::device::Device {
    match q {
        QLinear::Dense(l) => l.weight().device(),
        QLinear::Quant(w) => w.device(),
    }
}
