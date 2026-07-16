use synaptix_core::{
    error::{Result, SynaptixError},
    tensor::Tensor,
};

use super::linear::{linear_dims, to_f32_vec};

/// Gated Linear Attention: causal linear-attention с data-dependent per-channel
/// (по строкам состояния = по каналам ключа) забыванием.
///
/// `gate` — log-decay формы `[B,H,S,dk]` (значения ≤ 0). Рекуррентно:
/// `S_t[r,c] = exp(gate_t[r])·S_{t-1}[r,c] + k_t[r] v_t[c]`, `o_t[c] = Σ_r q_t[r] S_t[r,c]`.
pub fn gla_attention(q: &Tensor, k: &Tensor, v: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let (b, h, s, dk, dv) = linear_dims(q, k, v)?;
    if gate.dims() != [b, h, s, dk] {
        return Err(SynaptixError::shape_mismatch(q.dims(), gate.dims()));
    }
    let dtype_in = q.dtype();
    let qf = to_f32_vec(q)?;
    let kf = to_f32_vec(k)?;
    let vf = to_f32_vec(v)?;
    let gf = to_f32_vec(gate)?;

    let mut out = vec![0.0f32; b * h * s * dv];
    for bi in 0..b {
        for hi in 0..h {
            let mut state = vec![0.0f32; dk * dv];
            for t in 0..s {
                let qk_off = ((bi * h + hi) * s + t) * dk;
                let v_off = ((bi * h + hi) * s + t) * dv;
                for r in 0..dk {
                    let dec = gf[qk_off + r].exp();
                    let kk = kf[qk_off + r];
                    let row = r * dv;
                    for c in 0..dv {
                        state[row + c] = state[row + c] * dec + kk * vf[v_off + c];
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
