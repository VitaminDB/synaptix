use synaptix_core::tensor::Tensor;
use synaptix_core::error::Result;
use std::collections::VecDeque;

pub struct GradTape {
    ops: VecDeque<Box<dyn Fn() -> Result<Vec<Tensor>> + Send>>,
}

impl GradTape {
    pub fn new() -> Self { Self { ops: VecDeque::new() } }
    pub fn record(&mut self, op: impl Fn() -> Result<Vec<Tensor>> + Send + 'static) {
        self.ops.push_back(Box::new(op));
    }
    pub fn backward(&mut self) -> Result<()> {
        while let Some(op) = self.ops.pop_back() { op()?; }
        Ok(())
    }
    pub fn len(&self) -> usize { self.ops.len() }
    pub fn is_empty(&self) -> bool { self.ops.is_empty() }
}

impl Default for GradTape { fn default() -> Self { Self::new() } }
