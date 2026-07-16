use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::attention::softmax_dim;

pub struct DifferentialAttnConfig {
    pub lambda_init: f32,
}

impl Default for DifferentialAttnConfig {
    fn default() -> Self {
        Self { lambda_init: 0.8 }
    }
}

pub fn differential_attention(
    q1: &Tensor,
    q2: &Tensor,
    k1: &Tensor,
    k2: &Tensor,
    v: &Tensor,
    scale: f32,
    lambda_val: f32,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    if q1.rank() != 4 || q2.rank() != 4 || k1.rank() != 4 || k2.rank() != 4 || v.rank() != 4 {
        return Err(SynaptixError::Unsupported(
            "differential: requires rank-4 [B,H,S,D]",
        ));
    }
    if q1.dims() != q2.dims() || k1.dims() != k2.dims() {
        return Err(SynaptixError::shape_mismatch(q1.dims(), q2.dims()));
    }

    let dtype_in = q1.dtype();
    let q1f = q1.to_dtype(DType::F32)?;
    let q2f = q2.to_dtype(DType::F32)?;
    let k1f = k1.to_dtype(DType::F32)?;
    let k2f = k2.to_dtype(DType::F32)?;
    let v_f = v.to_dtype(DType::F32)?;

    let k1_t = k1f.transpose(2, 3)?.contiguous()?;
    let k2_t = k2f.transpose(2, 3)?.contiguous()?;

    let mut s1 = q1f.matmul(&k1_t)?.mul_scalar(scale)?;
    let mut s2 = q2f.matmul(&k2_t)?.mul_scalar(scale)?;
    if let Some(m) = mask {
        let mf = m.to_dtype(DType::F32)?;
        s1 = s1.broadcast_add(&mf)?;
        s2 = s2.broadcast_add(&mf)?;
    }
    let p1 = softmax_dim(&s1, 3)?;
    let p2 = softmax_dim(&s2, 3)?;
    let diff = p1.sub(&p2.affine(lambda_val, 0.0)?)?;
    let out = diff.matmul(&v_f)?;
    out.to_dtype(dtype_in)
}
