use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::attention::softmax_dim;

pub struct MlaConfig {
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub num_heads: usize,
}

pub fn mla_attention(
    q_nope: &Tensor,
    q_rope: &Tensor,
    k_nope: &Tensor,
    k_rope: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    if q_nope.rank() != 4 || q_rope.rank() != 4 || k_nope.rank() != 4 || k_rope.rank() != 4 || v.rank() != 4 {
        return Err(SynaptixError::Unsupported("mla: requires rank-4 [B,H,S,D]"));
    }
    let b = q_nope.dims()[0];
    let h = q_nope.dims()[1];
    let s_q = q_nope.dims()[2];
    let s_kv = k_nope.dims()[2];
    let d_nope = q_nope.dims()[3];
    let d_rope = q_rope.dims()[3];
    if q_rope.dims() != [b, h, s_q, d_rope] {
        return Err(SynaptixError::shape_mismatch(q_nope.dims(), q_rope.dims()));
    }
    if k_nope.dims() != [b, h, s_kv, d_nope] {
        return Err(SynaptixError::shape_mismatch(q_nope.dims(), k_nope.dims()));
    }
    if k_rope.dims() != [b, 1, s_kv, d_rope] && k_rope.dims() != [b, h, s_kv, d_rope] {
        return Err(SynaptixError::Unsupported(
            "mla: k_rope must be [B,1,S_kv,D_rope] (shared) or [B,H,S_kv,D_rope]",
        ));
    }
    if v.dims()[0] != b || v.dims()[1] != h || v.dims()[2] != s_kv {
        return Err(SynaptixError::shape_mismatch(q_nope.dims(), v.dims()));
    }

    let dtype_in = q_nope.dtype();
    let q_nope_f = q_nope.to_dtype(DType::F32)?;
    let q_rope_f = q_rope.to_dtype(DType::F32)?;
    let k_nope_f = k_nope.to_dtype(DType::F32)?;
    let k_rope_f = if k_rope.dims()[1] == 1 {
        k_rope.expand(vec![b, h, s_kv, d_rope])?.contiguous()?.to_dtype(DType::F32)?
    } else {
        k_rope.to_dtype(DType::F32)?
    };
    let v_f = v.to_dtype(DType::F32)?;

    let k_nope_t = k_nope_f.transpose(2, 3)?.contiguous()?;
    let k_rope_t = k_rope_f.transpose(2, 3)?.contiguous()?;
    let scores_nope = q_nope_f.matmul(&k_nope_t)?;
    let scores_rope = q_rope_f.matmul(&k_rope_t)?;
    let scores = scores_nope.add(&scores_rope)?.mul_scalar(scale)?;
    let masked = match mask {
        Some(m) => scores.broadcast_add(&m.to_dtype(DType::F32)?)?,
        None => scores,
    };
    let probs = softmax_dim(&masked, 3)?;
    let out = probs.matmul(&v_f)?;
    out.to_dtype(dtype_in)
}
