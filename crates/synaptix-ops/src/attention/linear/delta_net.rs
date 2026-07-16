use synaptix_core::{error::Result, tensor::Tensor};

use super::linear::{linear_dims, to_f32_vec};

/// DeltaNet (delta-rule) в раскладке состояния репозитория `S[r,c]`
/// (r = канал ключа, c = канал значения), causal:
///   `kv_old = S_{t-1}ᵀ k_t`;  `δ = β·(v_t − kv_old)`;
///   `S_t = S_{t-1} + k_t δᵀ`;  `o_t = S_tᵀ q_t`.
/// Без L2-нормализации q,k (отличие от gated-варианта).
pub fn delta_net_attention(q: &Tensor, k: &Tensor, v: &Tensor, beta: f32) -> Result<Tensor> {
    let (b, h, s, dk, dv) = linear_dims(q, k, v)?;
    let dtype_in = q.dtype();
    let qf = to_f32_vec(q)?;
    let kf = to_f32_vec(k)?;
    let vf = to_f32_vec(v)?;

    let mut out = vec![0.0f32; b * h * s * dv];
    let mut kv_old = vec![0.0f32; dv];
    let mut delta = vec![0.0f32; dv];
    for bi in 0..b {
        for hi in 0..h {
            let mut state = vec![0.0f32; dk * dv];
            for t in 0..s {
                let qk_off = ((bi * h + hi) * s + t) * dk;
                let v_off = ((bi * h + hi) * s + t) * dv;

                // kv_old[c] = Σ_r S[r,c] k_t[r]
                for c in 0..dv {
                    kv_old[c] = 0.0;
                }
                for r in 0..dk {
                    let kk = kf[qk_off + r];
                    let row = r * dv;
                    for c in 0..dv {
                        kv_old[c] += state[row + c] * kk;
                    }
                }
                // delta[c] = beta (v_t[c] - kv_old[c]);  S[r,c] += k_t[r] delta[c]
                for c in 0..dv {
                    delta[c] = beta * (vf[v_off + c] - kv_old[c]);
                }
                for r in 0..dk {
                    let kk = kf[qk_off + r];
                    let row = r * dv;
                    for c in 0..dv {
                        state[row + c] += kk * delta[c];
                    }
                }
                // o_t[c] = Σ_r S[r,c] q_t[r]
                for c in 0..dv {
                    let mut acc = 0.0f32;
                    for r in 0..dk {
                        acc += state[r * dv + c] * qf[qk_off + r];
                    }
                    out[v_off + c] = acc;
                }
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, h, s, dv], q.device())?.to_dtype(dtype_in)
}
