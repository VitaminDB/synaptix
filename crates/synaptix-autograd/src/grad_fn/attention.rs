use synaptix_core::{dtype::DType, error::Result, tensor::Tensor};

pub fn softmax_backward(grad_output: &Tensor, softmax_out: &Tensor, dim: usize) -> Result<Tensor> {
    // d_softmax = softmax * (grad - (grad * softmax).sum(dim, keepdim=True))
    let s = softmax_out.to_dtype(DType::F32)?;
    let g = grad_output.to_dtype(DType::F32)?;
    let gs = g.mul(&s)?;
    let gs_sum = gs.sum_keepdim(dim)?;
    let out = s.mul(&g.broadcast_sub(&gs_sum)?)?;
    out.to_dtype(grad_output.dtype())
}

pub fn attention_backward(
    grad_output: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_weights: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    // Standard scaled-dot-product attention backward (no FlashAttention memory trick).
    // attn_weights: [B, H, Sq, Sk] (post-softmax)
    // grad_output:  [B, H, Sq, Dv]
    // dV = attn_weights^T @ grad_output  → [B, H, Sk, Dv]
    let ak = attn_weights.rank();
    let attn_t = attn_weights.transpose(ak - 2, ak - 1)?.contiguous()?;
    let dv = attn_t.matmul(grad_output)?;

    // d_attn = grad_output @ V^T  → [B, H, Sq, Sk]
    let vr = v.rank();
    let v_t = v.transpose(vr - 2, vr - 1)?.contiguous()?;
    let d_attn = grad_output.matmul(&v_t)?;

    // d_attn through softmax
    let d_softmax = softmax_backward(&d_attn, attn_weights, ak - 1)?;

    // dQ = d_softmax @ K / scale  (scale = 1/sqrt(Dk) embedded in Q*K^T, pass as-is)
    let kr = k.rank();
    let k_t = k.transpose(kr - 2, kr - 1)?.contiguous()?;
    let _ = k_t; // not used directly — dQ = d_softmax @ K
    let dq = d_softmax.matmul(k)?;

    // dK = d_softmax^T @ Q
    let ds_t = d_softmax.transpose(ak - 2, ak - 1)?.contiguous()?;
    let dk = ds_t.matmul(q)?;

    Ok((dq.to_dtype(q.dtype())?, dk.to_dtype(k.dtype())?, dv.to_dtype(v.dtype())?))
}
