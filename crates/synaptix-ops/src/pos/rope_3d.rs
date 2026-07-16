use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn build_rope_3d_cos_sin(
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    head_dim: usize,
    theta_base: f32,
    device: Device,
) -> Result<(Tensor, Tensor)> {
    if head_dim % 6 != 0 {
        return Err(SynaptixError::Unsupported("rope_3d: head_dim must be divisible by 6"));
    }
    let per_axis = head_dim / 6;
    let make_axis = |size: usize| -> Vec<(f32, f32)> {
        let mut out = Vec::with_capacity(size * per_axis);
        for t in 0..size {
            for i in 0..per_axis {
                let exponent = -(2.0 * i as f32) / (head_dim as f32);
                let freq = theta_base.powf(exponent);
                let angle = (t as f32) * freq;
                out.push((angle.cos(), angle.sin()));
            }
        }
        out
    };
    let axis_t = make_axis(grid_t);
    let axis_h = make_axis(grid_h);
    let axis_w = make_axis(grid_w);
    let n = grid_t * grid_h * grid_w;
    let stride = head_dim / 2;
    let mut cos = vec![0.0_f32; n * stride];
    let mut sin = vec![0.0_f32; n * stride];
    for t in 0..grid_t {
        for h in 0..grid_h {
            for w in 0..grid_w {
                let pos = (t * grid_h + h) * grid_w + w;
                for i in 0..per_axis {
                    cos[pos * stride + i] = axis_t[t * per_axis + i].0;
                    sin[pos * stride + i] = axis_t[t * per_axis + i].1;
                    cos[pos * stride + per_axis + i] = axis_h[h * per_axis + i].0;
                    sin[pos * stride + per_axis + i] = axis_h[h * per_axis + i].1;
                    cos[pos * stride + 2 * per_axis + i] = axis_w[w * per_axis + i].0;
                    sin[pos * stride + 2 * per_axis + i] = axis_w[w * per_axis + i].1;
                }
            }
        }
    }
    let cos_t = Tensor::from_vec(cos, (n, stride), device)?;
    let sin_t = Tensor::from_vec(sin, (n, stride), device)?;
    Ok((cos_t, sin_t))
}

pub fn apply_rope_3d(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    if x.rank() != 4 {
        return Err(SynaptixError::Unsupported("apply_rope_3d: requires rank-4 [B,H,S,D]"));
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
