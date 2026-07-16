use synaptix_core::tensor::Tensor;

pub struct AsyncTpHandle;

pub fn async_all_gather(_x: Tensor) -> AsyncTpHandle { AsyncTpHandle }
