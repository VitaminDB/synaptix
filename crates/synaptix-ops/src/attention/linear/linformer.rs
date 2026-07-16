use synaptix_core::{
    error::{Result, SynaptixError},
    tensor::Tensor,
};

use super::linear::{linear_dims, softmax_inplace, to_f32_vec};

/// Linformer: проекция K,V вдоль оси последовательности с S до ранга r,
/// затем обычный softmax-attention. `K' = E·K` (`[r,dk]`), `V' = F·V` (`[r,dv]`),
/// `O = softmax(Q K'ᵀ / √dk) V'`. Non-causal (проекция нарушает причинность).
/// `e_proj`, `f_proj` — формы `[r, s]` (общие для всех B,H).
pub fn linformer_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    e_proj: &Tensor,
    f_proj: &Tensor,
) -> Result<Tensor> {
    let (b, h, s, dk, dv) = linear_dims(q, k, v)?;
    if e_proj.rank() != 2 || e_proj.dims()[1] != s {
        return Err(SynaptixError::Unsupported("linformer: e_proj must be [r, s]"));
    }
    if f_proj.rank() != 2 || f_proj.dims()[1] != s {
        return Err(SynaptixError::Unsupported("linformer: f_proj must be [r, s]"));
    }
    let r = e_proj.dims()[0];
    if f_proj.dims()[0] != r {
        return Err(SynaptixError::shape_mismatch(e_proj.dims(), f_proj.dims()));
    }
    let dtype_in = q.dtype();
    let qf = to_f32_vec(q)?;
    let kf = to_f32_vec(k)?;
    let vf = to_f32_vec(v)?;
    let ef = to_f32_vec(e_proj)?;
    let ff = to_f32_vec(f_proj)?;

    let scale = (dk as f32).powf(-0.5);
    let mut out = vec![0.0f32; b * h * s * dv];
    let mut scores = vec![0.0f32; r];
    for bi in 0..b {
        for hi in 0..h {
            let base = (bi * h + hi) * s;
            // K'[a,r2] = Σ_j E[a,j] k[j,r2];  V'[a,c] = Σ_j F[a,j] v[j,c]
            let mut kp = vec![0.0f32; r * dk];
            let mut vp = vec![0.0f32; r * dv];
            for a in 0..r {
                let e_row = a * s;
                let f_row = a * s;
                for j in 0..s {
                    let e = ef[e_row + j];
                    let fcoef = ff[f_row + j];
                    let k_off = (base + j) * dk;
                    let v_off = (base + j) * dv;
                    let kp_row = a * dk;
                    let vp_row = a * dv;
                    for r2 in 0..dk {
                        kp[kp_row + r2] += e * kf[k_off + r2];
                    }
                    for c in 0..dv {
                        vp[vp_row + c] += fcoef * vf[v_off + c];
                    }
                }
            }
            for t in 0..s {
                let q_off = (base + t) * dk;
                let o_off = (base + t) * dv;
                for a in 0..r {
                    let kp_row = a * dk;
                    let mut sc = 0.0f32;
                    for r2 in 0..dk {
                        sc += qf[q_off + r2] * kp[kp_row + r2];
                    }
                    scores[a] = sc * scale;
                }
                softmax_inplace(&mut scores);
                for c in 0..dv {
                    let mut acc = 0.0f32;
                    for a in 0..r {
                        acc += scores[a] * vp[a * dv + c];
                    }
                    out[o_off + c] = acc;
                }
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, h, s, dv], q.device())?.to_dtype(dtype_in)
}
