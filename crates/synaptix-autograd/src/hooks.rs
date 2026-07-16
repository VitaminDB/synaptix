pub use synaptix_core::grad::BackwardHook;

use synaptix_core::tensor::Tensor;

pub type ForwardHook = Box<dyn Fn(&Tensor, &Tensor) + Send + Sync>;

pub struct HookRegistry {
    forward: Vec<ForwardHook>,
    backward: Vec<BackwardHook>,
}

impl HookRegistry {
    pub fn new() -> Self { Self { forward: Vec::new(), backward: Vec::new() } }
    pub fn register_forward(&mut self, h: ForwardHook) { self.forward.push(h); }
    pub fn register_backward(&mut self, h: BackwardHook) { self.backward.push(h); }
    pub fn run_forward(&self, input: &Tensor, output: &Tensor) { for h in &self.forward { h(input, output); } }
    pub fn run_backward(&self, grad: &Tensor) { for h in &self.backward { h(grad); } }
}

impl Default for HookRegistry { fn default() -> Self { Self::new() } }
