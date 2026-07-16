use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::attention::softmax_dim;

pub fn reformer_lsh_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    buckets: &Tensor,
    scale: f32,
    causal: bool,
) -> Result<Tensor> {
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Err(SynaptixError::Unsupported("reformer: requires rank-4 [B,H,S,D]"));
    }
    if buckets.rank() != 3 {
        return Err(SynaptixError::Unsupported("reformer: buckets must be rank-3 [B,H,S]"));
    }
    let (b, h, s_q, _) = (q.dims()[0], q.dims()[1], q.dims()[2], q.dims()[3]);
    let s_kv = k.dims()[2];
    if buckets.dims() != [b, h, s_q] {
        return Err(SynaptixError::shape_mismatch(&[b, h, s_q], buckets.dims()));
    }
    if s_q != s_kv {
        return Err(SynaptixError::Unsupported("reformer: requires s_q == s_kv"));
    }

    let dtype_in = q.dtype();
    let q_f = q.to_dtype(DType::F32)?;
    let k_f = k.to_dtype(DType::F32)?;
    let v_f = v.to_dtype(DType::F32)?;
    let buckets_i = buckets.to_dtype(DType::I64)?.contiguous()?.flatten_all()?.to_vec1::<i64>()?;

    let k_t = k_f.transpose(2, 3)?.contiguous()?;
    let scores = q_f.matmul(&k_t)?.mul_scalar(scale)?;

    let mut mask_data = vec![f32::NEG_INFINITY; b * h * s_q * s_kv];
    for bi in 0..b {
        for hi in 0..h {
            let off = (bi * h + hi) * s_q;
            for i in 0..s_q {
                let bi_bucket = buckets_i[off + i];
                for j in 0..s_kv {
                    if causal && j > i {
                        continue;
                    }
                    let bj_bucket = buckets_i[off + j];
                    if bi_bucket == bj_bucket {
                        mask_data[((bi * h + hi) * s_q + i) * s_kv + j] = 0.0;
                    }
                }
            }
        }
    }
    let mask = Tensor::from_vec::<_, f32>(mask_data, vec![b, h, s_q, s_kv], q.device())?;
    let masked = scores.add(&mask)?;
    let probs = softmax_dim(&masked, 3)?;
    let out = probs.matmul(&v_f)?;
    out.to_dtype(dtype_in)
}
