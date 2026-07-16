use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

/// GroupNorm с фьюзед SiLU в эпилоге (CUDA). Эквивалент `silu(group_norm(x))`,
/// но один kernel-launch (сохраняет unary-pass). CPU/неподдержка → fallback.
pub fn group_norm_silu(
    x: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    num_groups: usize,
    eps: f32,
) -> Result<Tensor> {
    if x.rank() < 2 {
        return Err(SynaptixError::Unsupported("group_norm_silu: rank must be >= 2"));
    }
    match x.group_norm_fused(weight, bias, num_groups, eps, true) {
        Ok(out) => return Ok(out),
        Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
        Err(e) => return Err(e),
    }
    group_norm(x, weight, bias, num_groups, eps)?.silu()
}

/// GroupNorm на NHWC-входе `[B,H,W,C]` (каналы — последняя ось). Fused CUDA-путь
/// (channels-last reduce); fallback — round-trip NHWC→NCHW→NHWC через decomposed.
pub fn group_norm_nhwc(
    x: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    num_groups: usize,
    eps: f32,
    silu: bool,
) -> Result<Tensor> {
    if x.rank() < 3 {
        return Err(SynaptixError::Unsupported("group_norm_nhwc: rank must be >= 3"));
    }
    match x.group_norm_fused_layout(weight, bias, num_groups, eps, silu, true) {
        Ok(out) => return Ok(out),
        Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
        Err(e) => return Err(e),
    }
    let r = x.rank();
    let mut perm_in: Vec<usize> = Vec::with_capacity(r);
    perm_in.push(0);
    perm_in.push(r - 1);
    perm_in.extend(1..r - 1);
    let x_nchw = x.permute(perm_in)?.contiguous()?;
    let gn = group_norm(&x_nchw, weight, bias, num_groups, eps)?;
    let gn = if silu { gn.silu()? } else { gn };
    let mut perm_out: Vec<usize> = Vec::with_capacity(r);
    perm_out.push(0);
    perm_out.extend(2..r);
    perm_out.push(1);
    gn.permute(perm_out)?.contiguous()
}

pub fn group_norm(
    x: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    num_groups: usize,
    eps: f32,
) -> Result<Tensor> {
    if x.rank() < 2 {
        return Err(SynaptixError::Unsupported("group_norm: rank must be >= 2"));
    }
    // Fused backend-путь (CUDA — один launch вместо ~12 ops + multi-dim reduce).
    // На CPU / неподдержке backend падаем в decomposed реализацию ниже.
    match x.group_norm_fused(weight, bias, num_groups, eps, false) {
        Ok(out) => return Ok(out),
        Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
        Err(e) => return Err(e),
    }
    let dims = x.dims().to_vec();
    let channels = dims[1];
    if channels % num_groups != 0 {
        return Err(SynaptixError::Other(format!(
            "group_norm: channels {channels} must be divisible by num_groups {num_groups}"
        )));
    }
    let per_group = channels / num_groups;
    let mut grouped_dims = Vec::with_capacity(dims.len() + 1);
    grouped_dims.push(dims[0]);
    grouped_dims.push(num_groups);
    grouped_dims.push(per_group);
    grouped_dims.extend_from_slice(&dims[2..]);
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?.reshape(grouped_dims.clone())?;
    let reduce_dims: Vec<usize> = (2..grouped_dims.len()).collect();
    let g_numel: usize = reduce_dims.iter().map(|&d| grouped_dims[d]).product();
    let sum = x_f32.sum(&reduce_dims[..])?;
    let mean = sum.mul_scalar(1.0 / (g_numel as f32))?;
    let mut keepdim_shape = vec![1usize; grouped_dims.len()];
    keepdim_shape[0] = grouped_dims[0];
    keepdim_shape[1] = grouped_dims[1];
    let mean_kd = mean.reshape(keepdim_shape.clone())?;
    let centered = x_f32.broadcast_sub(&mean_kd)?;
    let var_sum = centered.sqr()?.sum(&reduce_dims[..])?;
    let var = var_sum.mul_scalar(1.0 / (g_numel as f32))?;
    let var_kd = var.reshape(keepdim_shape)?;
    let inv = var_kd.add_scalar(eps)?.sqrt()?.recip()?;
    let normed = centered.broadcast_mul(&inv)?.reshape(dims.clone())?;
    let scaled = match weight {
        Some(w) => {
            let w_f32 = w.to_dtype(DType::F32)?;
            let mut w_shape = vec![1usize; dims.len()];
            w_shape[1] = channels;
            let w_view = w_f32.reshape(w_shape)?;
            normed.broadcast_mul(&w_view)?
        }
        None => normed,
    };
    let out = match bias {
        Some(b) => {
            let b_f32 = b.to_dtype(DType::F32)?;
            let mut b_shape = vec![1usize; dims.len()];
            b_shape[1] = channels;
            let b_view = b_f32.reshape(b_shape)?;
            scaled.broadcast_add(&b_view)?
        }
        None => scaled,
    };
    out.to_dtype(dtype_in)
}

#[derive(Debug, Clone)]
pub struct GroupNorm {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    num_groups: usize,
    eps: f32,
}

impl GroupNorm {
    pub fn new(num_groups: usize, weight: Option<Tensor>, bias: Option<Tensor>, eps: f32) -> Self {
        Self { weight, bias, num_groups, eps }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        group_norm(x, self.weight.as_ref(), self.bias.as_ref(), self.num_groups, self.eps)
    }
}
