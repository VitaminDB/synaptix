use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn euclidean(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.sub(b)?.sqr()?.sum(vec![1])?.sqrt()
}

pub fn manhattan(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.sub(b)?.abs()?.sum(vec![1])
}

pub fn cosine(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dot = a.mul(b)?.sum(vec![1])?;
    let na = a.sqr()?.sum(vec![1])?.sqrt()?;
    let nb = b.sqr()?.sum(vec![1])?.sqrt()?;
    let denom = na.mul(&nb)?.add_scalar(1e-12)?;
    dot.div(&denom)
}
