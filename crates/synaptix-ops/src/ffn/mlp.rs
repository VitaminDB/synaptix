use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::activation::{
    gelu_exact, gelu_tanh, mish, quick_gelu, relu, silu, softplus, tanh,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Gelu,
    GeluExact,
    QuickGelu,
    Silu,
    Relu,
    Mish,
    Tanh,
    Softplus,
}

pub fn apply_activation(x: &Tensor, act: Activation) -> Result<Tensor> {
    match act {
        Activation::Gelu => gelu_tanh(x),
        Activation::GeluExact => gelu_exact(x),
        Activation::QuickGelu => quick_gelu(x),
        Activation::Silu => silu(x),
        Activation::Relu => relu(x),
        Activation::Mish => mish(x),
        Activation::Tanh => tanh(x),
        Activation::Softplus => softplus(x, 1.0, 20.0),
    }
}

pub fn mlp(
    x: &Tensor,
    w1: &Tensor,
    b1: Option<&Tensor>,
    w2: &Tensor,
    b2: Option<&Tensor>,
    act: Activation,
) -> Result<Tensor> {
    let w1_t = w1.transpose(0, 1)?.contiguous()?;
    let mut h = x.matmul(&w1_t)?;
    if let Some(b) = b1 {
        h = h.broadcast_add(b)?;
    }
    let h_act = apply_activation(&h, act)?;
    let w2_t = w2.transpose(0, 1)?.contiguous()?;
    let mut out = h_act.matmul(&w2_t)?;
    if let Some(b) = b2 {
        out = out.broadcast_add(b)?;
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct Mlp {
    w1: Tensor,
    b1: Option<Tensor>,
    w2: Tensor,
    b2: Option<Tensor>,
    act: Activation,
}

impl Mlp {
    pub fn new(w1: Tensor, b1: Option<Tensor>, w2: Tensor, b2: Option<Tensor>, act: Activation) -> Self {
        Self { w1, b1, w2, b2, act }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        mlp(x, &self.w1, self.b1.as_ref(), &self.w2, self.b2.as_ref(), self.act)
    }
}
