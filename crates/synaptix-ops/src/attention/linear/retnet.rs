use synaptix_core::{error::Result, tensor::Tensor};

use super::linear::{linear_dims, to_f32_vec};

/// RetNet retention = causal linear-attention со скалярным экспоненциальным decay.
/// Рекуррентно: `S_t = γ·S_{t-1} + k_t v_tᵀ`, `o_t = q_tᵀ S_t` (без softmax/нормализации).
pub fn retnet_attention(q: &Tensor, k: &Tensor, v: &Tensor, gamma: f32) -> Result<Tensor> {
    let (b, h, s, dk, dv) = linear_dims(q, k, v)?;
    let dtype_in = q.dtype();
    let qf = to_f32_vec(q)?;
    let kf = to_f32_vec(k)?;
    let vf = to_f32_vec(v)?;

    let mut out = vec![0.0f32; b * h * s * dv];
    for bi in 0..b {
        for hi in 0..h {
            let mut state = vec![0.0f32; dk * dv];
            for t in 0..s {
                for x in state.iter_mut() {
                    *x *= gamma;
                }
                let qk_off = ((bi * h + hi) * s + t) * dk;
                let v_off = ((bi * h + hi) * s + t) * dv;
                for r in 0..dk {
                    let kk = kf[qk_off + r];
                    for c in 0..dv {
                        state[r * dv + c] += kk * vf[v_off + c];
                    }
                }
                for c in 0..dv {
                    let mut acc = 0.0f32;
                    for r in 0..dk {
                        acc += qf[qk_off + r] * state[r * dv + c];
                    }
                    out[v_off + c] = acc;
                }
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, h, s, dv], q.device())?.to_dtype(dtype_in)
}
