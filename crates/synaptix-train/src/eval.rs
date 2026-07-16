use crate::error::Result;

pub struct EvalMetrics { pub loss: f64, pub accuracy: f64 }

pub fn evaluate() -> Result<EvalMetrics> {
    Ok(EvalMetrics { loss: 0.0, accuracy: 0.0 })
}
