use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn build_rope_2d_cos_sin(
    grid_h: usize,
    grid_w: usize,
    head_dim: usize,
    theta_base: f32,
    device: Device,
) -> Result<(Tensor, Tensor)> {
    if head_dim % 4 != 0 {
        return Err(SynaptixError::Unsupported("rope_2d: head_dim must be divisible by 4"));
    }
    let quarter = head_dim / 4;
    let make_axis = |size: usize| -> Result<(Vec<f32>, Vec<f32>)> {
        let mut cos = vec![0.0_f32; size * quarter];
        let mut sin = vec![0.0_f32; size * quarter];
        for t in 0..size {
            for i in 0..quarter {
                let exponent = -(2.0 * i as f32) / (head_dim as f32);
                let freq = theta_base.powf(exponent);
                let angle = (t as f32) * freq;
                cos[t * quarter + i] = angle.cos();
                sin[t * quarter + i] = angle.sin();
            }
        }
        Ok((cos, sin))
    };
    let (cos_h, sin_h) = make_axis(grid_h)?;
    let (cos_w, sin_w) = make_axis(grid_w)?;
    let n = grid_h * grid_w;
    let mut cos = vec![0.0_f32; n * (head_dim / 2)];
    let mut sin = vec![0.0_f32; n * (head_dim / 2)];
    for h in 0..grid_h {
        for w in 0..grid_w {
            let pos = h * grid_w + w;
            for i in 0..quarter {
                cos[pos * (head_dim / 2) + i] = cos_h[h * quarter + i];
                sin[pos * (head_dim / 2) + i] = sin_h[h * quarter + i];
                cos[pos * (head_dim / 2) + quarter + i] = cos_w[w * quarter + i];
                sin[pos * (head_dim / 2) + quarter + i] = sin_w[w * quarter + i];
            }
        }
    }
    let cos_t = Tensor::from_vec(cos, (n, head_dim / 2), device)?;
    let sin_t = Tensor::from_vec(sin, (n, head_dim / 2), device)?;
    Ok((cos_t, sin_t))
}

pub fn apply_rope_2d(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    if x.rank() != 4 {
        return Err(SynaptixError::Unsupported("apply_rope_2d: requires rank-4 [B,H,S,D]"));
    }
    let head_dim = x.dims()[3];
    let s = x.dims()[2];
    let half = head_dim / 2;
    let cos_b = cos.reshape((1usize, 1, s, half))?;
    let sin_b = sin.reshape((1usize, 1, s, half))?;
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let a = x_f32.narrow(3, 0, half)?.contiguous()?;
    let b = x_f32.narrow(3, half, half)?.contiguous()?;
    let rot_a = a.broadcast_mul(&cos_b)?.sub(&b.broadcast_mul(&sin_b)?)?;
    let rot_b = a.broadcast_mul(&sin_b)?.add(&b.broadcast_mul(&cos_b)?)?;
    let out = Tensor::cat(&[&rot_a, &rot_b], 3)?;
    out.to_dtype(dtype_in)
}
