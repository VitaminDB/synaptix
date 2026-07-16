use synaptix_core::{
    error::{Result, SynaptixError},
    tensor::Tensor,
};

use super::linear::{linear_dims, to_f32_vec};

/// Hyena — это gated long-convolution, а не attention. Минимальная честная форма
/// порядка 2: `u = k ⊙ v` (gate-1) → causal long-conv с фильтром `filt[H,S]` по
/// каналам → `O = q ⊙ c` (gate-2):
///   `u_t = k_t ⊙ v_t`;  `c_t[ch] = Σ_{j≤t} filt[h, t−j]·u_j[ch]`;  `o_t = q_t ⊙ c_t`.
/// Требует `dk == dv` (поэлементное гейтирование).
pub fn hyena_attention(q: &Tensor, k: &Tensor, v: &Tensor, filt: &Tensor) -> Result<Tensor> {
    let (b, h, s, dk, dv) = linear_dims(q, k, v)?;
    if dk != dv {
        return Err(SynaptixError::Unsupported("hyena: requires dk == dv"));
    }
    if filt.dims() != [h, s] {
        return Err(SynaptixError::Unsupported("hyena: filt must be [H, S]"));
    }
    let d = dv;
    let dtype_in = q.dtype();
    let qf = to_f32_vec(q)?;
    let kf = to_f32_vec(k)?;
    let vf = to_f32_vec(v)?;
    let ff = to_f32_vec(filt)?;

    // u = k ⊙ v
    let mut u = vec![0.0f32; b * h * s * d];
    for idx in 0..u.len() {
        u[idx] = kf[idx] * vf[idx];
    }

    let mut out = vec![0.0f32; b * h * s * d];
    for bi in 0..b {
        for hi in 0..h {
            let kern = &ff[hi * s..hi * s + s];
            for i in 0..s {
                let o_off = ((bi * h + hi) * s + i) * d;
                // c_i[ch] = Σ_{j≤i} filt[h,i−j] u_j[ch]
                for j in 0..=i {
                    let coef = kern[i - j];
                    let u_off = ((bi * h + hi) * s + j) * d;
                    for ch in 0..d {
                        out[o_off + ch] += coef * u[u_off + ch];
                    }
                }
                // o_i = q_i ⊙ c_i
                let q_off = o_off;
                for ch in 0..d {
                    out[o_off + ch] *= qf[q_off + ch];
                }
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, h, s, d], q.device())?.to_dtype(dtype_in)
}
