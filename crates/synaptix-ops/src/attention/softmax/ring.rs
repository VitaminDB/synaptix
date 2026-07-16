use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn ring_attention_local(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    block_size: usize,
    causal: bool,
) -> Result<Tensor> {
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Err(SynaptixError::Unsupported("ring: requires rank-4 [B,H,S,D]"));
    }
    if block_size == 0 {
        return Err(SynaptixError::Unsupported("ring: block_size must be > 0"));
    }
    let (b, h, s_q, d_k) = (q.dims()[0], q.dims()[1], q.dims()[2], q.dims()[3]);
    let s_kv = k.dims()[2];
    let d_v = v.dims()[3];
    if k.dims() != [b, h, s_kv, d_k] {
        return Err(SynaptixError::shape_mismatch(q.dims(), k.dims()));
    }
    if v.dims()[0] != b || v.dims()[1] != h || v.dims()[2] != s_kv {
        return Err(SynaptixError::shape_mismatch(q.dims(), v.dims()));
    }

    let dtype_in = q.dtype();
    let q_f = q.to_dtype(DType::F32)?;
    let k_f = k.to_dtype(DType::F32)?;
    let v_f = v.to_dtype(DType::F32)?;

    let mut acc_o = Tensor::zeros(vec![b, h, s_q, d_v], DType::F32, q.device())?;
    let neg_inf_vec = vec![f32::NEG_INFINITY; b * h * s_q];
    let mut acc_m = Tensor::from_vec::<_, f32>(neg_inf_vec, vec![b, h, s_q, 1], q.device())?;
    let mut acc_l = Tensor::zeros(vec![b, h, s_q, 1], DType::F32, q.device())?;

    let n_blocks = (s_kv + block_size - 1) / block_size;
    for bi in 0..n_blocks {
        let off = bi * block_size;
        let bsz = block_size.min(s_kv - off);
        let kj = k_f.narrow(2, off, bsz)?.contiguous()?;
        let vj = v_f.narrow(2, off, bsz)?.contiguous()?;
        let kj_t = kj.transpose(2, 3)?.contiguous()?;
        let mut s_ij = q_f.matmul(&kj_t)?.mul_scalar(scale)?;
        if causal {
            let mut mask_data = vec![0.0_f32; s_q * bsz];
            for qi in 0..s_q {
                for kj_local in 0..bsz {
                    let kj_global = off + kj_local;
                    if kj_global > qi {
                        mask_data[qi * bsz + kj_local] = f32::NEG_INFINITY;
                    }
                }
            }
            let mask = Tensor::from_vec::<_, f32>(mask_data, vec![s_q, bsz], q.device())?;
            s_ij = s_ij.broadcast_add(&mask)?;
        }
        let m_ij = s_ij.max_keepdim(3)?;
        let new_m = acc_m.maximum(&m_ij)?;
        let alpha = acc_m.sub(&new_m)?.exp()?;
        let p_ij = s_ij.broadcast_sub(&new_m)?.exp()?;
        let l_ij = p_ij.sum_keepdim(3)?;
        acc_l = acc_l.mul(&alpha)?.add(&l_ij)?;
        let alpha_o = alpha.broadcast_as(acc_o.shape().clone())?;
        let scaled_o = acc_o.mul(&alpha_o)?;
        let new_o = p_ij.matmul(&vj)?;
        acc_o = scaled_o.add(&new_o)?;
        acc_m = new_m;
    }

    let l_b = acc_l.broadcast_as(acc_o.shape().clone())?;
    let out = acc_o.div(&l_b)?;
    out.to_dtype(dtype_in)
}
