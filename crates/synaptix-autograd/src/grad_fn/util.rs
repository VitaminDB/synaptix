use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn unbroadcast_to(mut grad: Tensor, target_shape: &[usize]) -> Result<Tensor> {
    let g_dims = grad.dims().to_vec();
    let g_rank = g_dims.len();
    let t_rank = target_shape.len();
    if g_rank > t_rank {
        let extra = g_rank - t_rank;
        for _ in 0..extra {
            grad = grad.sum_keepdim(0)?.squeeze(0)?;
        }
    }
    let g_dims_after: Vec<usize> = grad.dims().to_vec();
    for (i, (&g, &t)) in g_dims_after.iter().zip(target_shape).enumerate() {
        if t == 1 && g != 1 {
            grad = grad.sum_keepdim(i)?;
        }
    }
    Ok(grad)
}
