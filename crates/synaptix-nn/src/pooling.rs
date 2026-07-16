use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn mean_pool(x: &Tensor, dim: usize) -> Result<Tensor> {
    x.mean_keepdim(dim)?.squeeze(dim)
}

pub fn max_pool(x: &Tensor, dim: usize) -> Result<Tensor> {
    x.max_keepdim(dim)?.squeeze(dim)
}

pub fn cls_pool(x: &Tensor) -> Result<Tensor> {
    // Extract first token: [B, S, H] → [B, H]
    let seq_dim = x.rank() - 2;
    x.narrow(seq_dim, 0, 1)?.squeeze(seq_dim)
}

pub fn last_pool(x: &Tensor) -> Result<Tensor> {
    // Extract last token: [B, S, H] → [B, H]
    let seq_dim = x.rank() - 2;
    let s = x.dims()[seq_dim];
    x.narrow(seq_dim, s - 1, 1)?.squeeze(seq_dim)
}
