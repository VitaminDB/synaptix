use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn relu(x: &Tensor) -> Result<Tensor> {
    x.relu()
}

pub fn relu2(x: &Tensor) -> Result<Tensor> {
    x.relu2()
}

pub fn leaky_relu(x: &Tensor, alpha: f32) -> Result<Tensor> {
    x.leaky_relu(alpha)
}

pub fn silu(x: &Tensor) -> Result<Tensor> {
    x.silu()
}

pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    x.gelu_tanh()
}

pub fn gelu_exact(x: &Tensor) -> Result<Tensor> {
    x.gelu_exact()
}

pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    x.sigmoid()
}

pub fn tanh(x: &Tensor) -> Result<Tensor> {
    x.tanh()
}

pub fn softplus(x: &Tensor) -> Result<Tensor> {
    x.exp()?.add_scalar(1.0)?.log()
}

pub fn softmax(x: &Tensor, dim: usize) -> Result<Tensor> {
    let exp = x.exp()?;
    let sum_exp = exp.sum_keepdim(dim)?;
    exp.broadcast_div(&sum_exp)
}

pub fn log_softmax(x: &Tensor, dim: usize) -> Result<Tensor> {
    let exp = x.exp()?;
    let sum_exp = exp.sum_keepdim(dim)?;
    let log_sum = sum_exp.log()?;
    x.broadcast_sub(&log_sum)
}

pub fn swiglu(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
    gate.silu()?.mul(x)
}

pub fn geglu(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
    gate.gelu_exact()?.mul(x)
}

pub fn reglu(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
    gate.relu()?.mul(x)
}
