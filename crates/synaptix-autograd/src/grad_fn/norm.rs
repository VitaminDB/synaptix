use synaptix_core::{
    dtype::DType,
    error::Result,
    tensor::Tensor,
};

pub fn layer_norm_backward(
    grad_output: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    // Standard layer norm backward (last dim normalised).
    // d  = x - mean(x),  var = mean(d²) + eps,  inv_std = 1/sqrt(var)
    // x_hat = d * inv_std,  y = w * x_hat + b
    // grad_bias   = sum_batch(grad_output, all dims except last)
    // grad_weight = sum_batch(grad_output * x_hat)
    // grad_input  = (1/N) * w * inv_std * (N*grad_output - sum(grad_output) - x_hat*sum(grad_output*x_hat))
    let _ = bias;
    let rank = input.rank();
    let last = rank - 1;
    let n = input.dims()[last] as f32;

    let x_f32 = input.to_dtype(DType::F32)?;
    let g_f32 = grad_output.to_dtype(DType::F32)?;
    let w_f32 = weight.to_dtype(DType::F32)?;
    let eps_f = eps as f32;

    let mean = x_f32.mean_keepdim(last)?;
    let centered = x_f32.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(last)?;
    let inv_std = var.add_scalar(eps_f)?.sqrt()?.recip()?;
    let x_hat = centered.broadcast_mul(&inv_std)?;

    let grad_bias = sum_over_batch(&g_f32, last)?;
    let grad_weight = sum_over_batch(&g_f32.mul(&x_hat)?, last)?;

    // grad_input
    let wg = g_f32.broadcast_mul(&w_f32)?;
    let sum_wg = wg.sum_keepdim(last)?;
    let sum_wg_xhat = wg.mul(&x_hat)?.sum_keepdim(last)?;
    let grad_x = inv_std.broadcast_mul(
        &wg.broadcast_sub(&sum_wg.mul_scalar(1.0 / n)?)?
           .broadcast_sub(&x_hat.broadcast_mul(&sum_wg_xhat)?.mul_scalar(1.0 / n)?)?,
    )?.to_dtype(input.dtype())?;

    Ok((grad_x, grad_weight.to_dtype(weight.dtype())?, grad_bias.to_dtype(weight.dtype())?))
}

pub fn rms_norm_backward(
    grad_output: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    // y = w * x / rms(x),  rms = sqrt(mean(x²) + eps)
    // grad_weight = sum_batch(grad_output * x_hat)
    // grad_input  = (1/N) * w * inv_rms * (N*grad_output - x_hat * sum(grad_output * x_hat))
    let rank = input.rank();
    let last = rank - 1;
    let n = input.dims()[last] as f32;

    let x_f32 = input.to_dtype(DType::F32)?;
    let g_f32 = grad_output.to_dtype(DType::F32)?;
    let w_f32 = weight.to_dtype(DType::F32)?;

    let rms = x_f32.sqr()?.mean_keepdim(last)?.add_scalar(eps as f32)?.sqrt()?;
    let inv_rms = rms.recip()?;
    let x_hat = x_f32.broadcast_mul(&inv_rms)?;

    let grad_weight = sum_over_batch(&g_f32.mul(&x_hat)?, last)?;

    let wg = g_f32.broadcast_mul(&w_f32)?;
    let sum_wg_xhat = wg.mul(&x_hat)?.sum_keepdim(last)?;
    let grad_x = inv_rms.broadcast_mul(
        &wg.broadcast_sub(&x_hat.broadcast_mul(&sum_wg_xhat)?.mul_scalar(1.0 / n)?)?,
    )?.to_dtype(input.dtype())?;

    Ok((grad_x, grad_weight.to_dtype(weight.dtype())?))
}

fn sum_over_batch(x: &Tensor, last_dim: usize) -> Result<Tensor> {
    let mut g = x.clone();
    for _ in 0..last_dim {
        g = g.sum_keepdim(0)?.squeeze(0)?;
    }
    Ok(g)
}
