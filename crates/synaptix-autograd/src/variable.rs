use synaptix_core::tensor::Tensor;

pub struct Variable {
    pub data: Tensor,
    pub grad: Option<Tensor>,
    pub requires_grad: bool,
}

impl Variable {
    pub fn new(data: Tensor, requires_grad: bool) -> Self {
        Self { data, grad: None, requires_grad }
    }
    pub fn detach(mut self) -> Self { self.requires_grad = false; self }
    pub fn zero_grad(&mut self) { self.grad = None; }
    pub fn data(&self) -> &Tensor { &self.data }
    pub fn grad(&self) -> Option<&Tensor> { self.grad.as_ref() }
}
