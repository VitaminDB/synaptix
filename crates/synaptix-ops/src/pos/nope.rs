use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

/// NoPE (No Positional Encoding): позиционное кодирование не применяется —
/// операция тождественна (для слоёв без RoPE/ALiBi).
pub fn nope(x: &Tensor) -> Result<Tensor> { Ok(x.clone()) }
