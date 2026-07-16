use synaptix_core::{
    error::{Result, SynaptixError},
    tensor::Tensor,
};

use super::linear::{linear_dims, to_f32_vec};

/// Положительные softmax-kernel features FAVOR+ для вектора `x` (длины dk).
/// `x_s = x·dk^{-1/4}`; `φ[a] = exp(Ω_a·x_s − ‖x_s‖²/2) / √m`, где `Ω = proj[m,dk]`.
fn favor_phi(x: &[f32], proj: &[f32], m: usize, dk: usize, scale: f32, inv_sqrt_m: f32, out: &mut [f32]) {
    let mut norm_sq = 0.0f32;
    for r in 0..dk {
        let xs = x[r] * scale;
        norm_sq += xs * xs;
    }
    let half_norm = 0.5 * norm_sq;
    for a in 0..m {
        let prow = a * dk;
        let mut dot = 0.0f32;
        for r in 0..dk {
            dot += proj[prow + r] * (x[r] * scale);
        }
        out[a] = (dot - half_norm).exp() * inv_sqrt_m;
    }
}

/// Performer (FAVOR+): non-causal нормализованная linear-attention со случайными
/// признаками. `proj` формы `[m, dk]` — фиксированная random-feature матрица Ω
/// (передаётся снаружи ради детерминизма; RNG внутри не используется).
/// `o_t = (φ(q_t)ᵀ Σ_j φ(k_j)v_jᵀ) / (φ(q_t)ᵀ Σ_j φ(k_j) + ε)`.
pub fn performer_attention(q: &Tensor, k: &Tensor, v: &Tensor, proj: &Tensor) -> Result<Tensor> {
    let (b, h, s, dk, dv) = linear_dims(q, k, v)?;
    if proj.rank() != 2 || proj.dims()[1] != dk {
        return Err(SynaptixError::Unsupported(
            "performer: proj must be rank-2 [m, dk]",
        ));
    }
    let m = proj.dims()[0];
    let dtype_in = q.dtype();
    let qf = to_f32_vec(q)?;
    let kf = to_f32_vec(k)?;
    let vf = to_f32_vec(v)?;
    let pf = to_f32_vec(proj)?;

    let scale = (dk as f32).powf(-0.25);
    let inv_sqrt_m = 1.0 / (m as f32).sqrt();

    let mut out = vec![0.0f32; b * h * s * dv];
    let mut pk = vec![0.0f32; m];
    let mut pq = vec![0.0f32; m];
    for bi in 0..b {
        for hi in 0..h {
            // Skv[m,dv] = Σ_j φ(k_j) v_jᵀ;  Sk[m] = Σ_j φ(k_j)
            let mut skv = vec![0.0f32; m * dv];
            let mut sk = vec![0.0f32; m];
            for j in 0..s {
                let qk_off = ((bi * h + hi) * s + j) * dk;
                let v_off = ((bi * h + hi) * s + j) * dv;
                favor_phi(&kf[qk_off..qk_off + dk], &pf, m, dk, scale, inv_sqrt_m, &mut pk);
                for a in 0..m {
                    let pka = pk[a];
                    let row = a * dv;
                    for c in 0..dv {
                        skv[row + c] += pka * vf[v_off + c];
                    }
                    sk[a] += pka;
                }
            }
            for t in 0..s {
                let qk_off = ((bi * h + hi) * s + t) * dk;
                let o_off = ((bi * h + hi) * s + t) * dv;
                favor_phi(&qf[qk_off..qk_off + dk], &pf, m, dk, scale, inv_sqrt_m, &mut pq);
                let mut den = 0.0f32;
                for a in 0..m {
                    den += pq[a] * sk[a];
                }
                let inv = 1.0 / (den + 1e-6);
                for c in 0..dv {
                    let mut num = 0.0f32;
                    for a in 0..m {
                        num += pq[a] * skv[a * dv + c];
                    }
                    out[o_off + c] = num * inv;
                }
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, h, s, dv], q.device())?.to_dtype(dtype_in)
}
