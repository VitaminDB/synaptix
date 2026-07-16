use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn lightning_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    slope: Option<&Tensor>,
    causal: bool,
) -> Result<Tensor> {
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Err(SynaptixError::Unsupported(
            "lightning: requires rank-4 [B,H,S,D]",
        ));
    }
    let b = q.dims()[0];
    let h = q.dims()[1];
    let s = q.dims()[2];
    let dk = q.dims()[3];
    let dv = v.dims()[3];
    if k.dims() != [b, h, s, dk] {
        return Err(SynaptixError::shape_mismatch(q.dims(), k.dims()));
    }
    if v.dims()[0] != b || v.dims()[1] != h || v.dims()[2] != s {
        return Err(SynaptixError::shape_mismatch(q.dims(), v.dims()));
    }
    if let Some(sl) = slope {
        if sl.dims() != [h] {
            return Err(SynaptixError::Unsupported(
                "lightning: slope must be [H]",
            ));
        }
    }

    let dtype_in = q.dtype();
    let q_f = q.to_dtype(DType::F32)?.contiguous()?;
    let k_f = k.to_dtype(DType::F32)?.contiguous()?;
    let v_f = v.to_dtype(DType::F32)?.contiguous()?;

    let q_flat = q_f.flatten_all()?.to_vec1::<f32>()?;
    let k_flat = k_f.flatten_all()?.to_vec1::<f32>()?;
    let v_flat = v_f.flatten_all()?.to_vec1::<f32>()?;
    let slopes: Vec<f32> = match slope {
        Some(sl) => sl.to_dtype(DType::F32)?.contiguous()?.to_vec1::<f32>()?,
        None => vec![0.0; h],
    };

    let mut out = vec![0.0_f32; b * h * s * dv];
    for bi in 0..b {
        for hi in 0..h {
            let lam = slopes[hi];
            let decay = (-lam).exp();
            let mut state = vec![0.0_f32; dk * dv];
            for t in 0..s {
                if causal {
                    for r in 0..dk {
                        for c in 0..dv {
                            state[r * dv + c] *= decay;
                        }
                    }
                    let k_off = ((bi * h + hi) * s + t) * dk;
                    let v_off = ((bi * h + hi) * s + t) * dv;
                    for r in 0..dk {
                        let kv = k_flat[k_off + r];
                        for c in 0..dv {
                            state[r * dv + c] += kv * v_flat[v_off + c];
                        }
                    }
                    let q_off = ((bi * h + hi) * s + t) * dk;
                    let out_off = ((bi * h + hi) * s + t) * dv;
                    for c in 0..dv {
                        let mut acc = 0.0_f32;
                        for r in 0..dk {
                            acc += q_flat[q_off + r] * state[r * dv + c];
                        }
                        out[out_off + c] = acc;
                    }
                }
            }
            if !causal {
                let mut s_full = vec![0.0_f32; dk * dv];
                for t in 0..s {
                    let k_off = ((bi * h + hi) * s + t) * dk;
                    let v_off = ((bi * h + hi) * s + t) * dv;
                    for r in 0..dk {
                        let kv = k_flat[k_off + r];
                        for c in 0..dv {
                            s_full[r * dv + c] += kv * v_flat[v_off + c];
                        }
                    }
                }
                for t in 0..s {
                    let q_off = ((bi * h + hi) * s + t) * dk;
                    let out_off = ((bi * h + hi) * s + t) * dv;
                    for c in 0..dv {
                        let mut acc = 0.0_f32;
                        for r in 0..dk {
                            acc += q_flat[q_off + r] * s_full[r * dv + c];
                        }
                        out[out_off + c] = acc;
                    }
                }
            }
        }
    }

    let out_t = Tensor::from_vec::<_, f32>(out, vec![b, h, s, dv], q.device())?;
    out_t.to_dtype(dtype_in)
}
