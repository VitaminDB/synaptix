use synaptix_core::{error::Result, tensor::Tensor};

use super::linear::{linear_dims, to_f32_vec};

/// Feature map Based: 2-й порядок Тейлора `φ(x) = [1, x, (x⊙x)/√2]`
/// (размерность `1 + 2·dk`). Пишет результат в `out[..m]`.
fn based_phi(x: &[f32], dk: usize, out: &mut [f32]) {
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    out[0] = 1.0;
    for r in 0..dk {
        let xr = x[r];
        out[1 + r] = xr;
        out[1 + dk + r] = xr * xr * INV_SQRT2;
    }
}

/// Based: causal нормализованная linear-attention с Тейлор-feature-map.
/// `num_t = Σ_{j≤t}(φ(q_t)·φ(k_j)) v_j`, `den_t = Σ_{j≤t} φ(q_t)·φ(k_j)`,
/// `o_t = num_t / (den_t + ε)`. Реализовано рекуррентно через state `[m,dv]` + `z[m]`.
pub fn based_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (b, h, s, dk, dv) = linear_dims(q, k, v)?;
    let dtype_in = q.dtype();
    let qf = to_f32_vec(q)?;
    let kf = to_f32_vec(k)?;
    let vf = to_f32_vec(v)?;

    let m = 1 + 2 * dk;
    let mut out = vec![0.0f32; b * h * s * dv];
    let mut pq = vec![0.0f32; m];
    let mut pk = vec![0.0f32; m];
    for bi in 0..b {
        for hi in 0..h {
            let mut state = vec![0.0f32; m * dv];
            let mut z = vec![0.0f32; m];
            for t in 0..s {
                let qk_off = ((bi * h + hi) * s + t) * dk;
                let v_off = ((bi * h + hi) * s + t) * dv;
                based_phi(&kf[qk_off..qk_off + dk], dk, &mut pk);
                based_phi(&qf[qk_off..qk_off + dk], dk, &mut pq);

                for a in 0..m {
                    let pka = pk[a];
                    let row = a * dv;
                    for c in 0..dv {
                        state[row + c] += pka * vf[v_off + c];
                    }
                    z[a] += pka;
                }
                let mut den = 0.0f32;
                for a in 0..m {
                    den += pq[a] * z[a];
                }
                let inv = 1.0 / (den + 1e-6);
                for c in 0..dv {
                    let mut num = 0.0f32;
                    for a in 0..m {
                        num += pq[a] * state[a * dv + c];
                    }
                    out[v_off + c] = num * inv;
                }
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, h, s, dv], q.device())?.to_dtype(dtype_in)
}
