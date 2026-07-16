use synaptix_core::{error::Result, tensor::Tensor};

use super::linear::{linear_dims, to_f32_vec};

/// cosFormer: ReLU feature map + cos/sin позиционное переваживание, causal, нормализованное.
/// `θ_t = (π/2)·(t/S)`; `cos(θ_t−θ_j)=cosθ_t cosθ_j + sinθ_t sinθ_j` раскладывает
/// ядро на две линейные ветви. Рекуррентно держим cos- и sin-состояния:
/// `num_t = cosθ_t (q̃_t·Sc) + sinθ_t (q̃_t·Ss)`,
/// `den_t = cosθ_t (q̃_t·zc) + sinθ_t (q̃_t·zs)`, `o_t = num/(den+ε)`,
/// где `q̃=relu(q)`, `k̃=relu(k)`.
pub fn cosformer_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (b, h, s, dk, dv) = linear_dims(q, k, v)?;
    let dtype_in = q.dtype();
    let qf = to_f32_vec(q)?;
    let kf = to_f32_vec(k)?;
    let vf = to_f32_vec(v)?;

    let half_pi = std::f32::consts::FRAC_PI_2;
    let m_len = s as f32;
    let mut out = vec![0.0f32; b * h * s * dv];
    let mut qr = vec![0.0f32; dk];
    for bi in 0..b {
        for hi in 0..h {
            let mut sc = vec![0.0f32; dk * dv];
            let mut ss = vec![0.0f32; dk * dv];
            let mut zc = vec![0.0f32; dk];
            let mut zs = vec![0.0f32; dk];
            for t in 0..s {
                let qk_off = ((bi * h + hi) * s + t) * dk;
                let v_off = ((bi * h + hi) * s + t) * dv;
                let theta = half_pi * (t as f32) / m_len;
                let ct = theta.cos();
                let st = theta.sin();

                for r in 0..dk {
                    let kr = kf[qk_off + r].max(0.0);
                    let row = r * dv;
                    let ckr = ct * kr;
                    let skr = st * kr;
                    for c in 0..dv {
                        let vv = vf[v_off + c];
                        sc[row + c] += ckr * vv;
                        ss[row + c] += skr * vv;
                    }
                    zc[r] += ckr;
                    zs[r] += skr;
                    qr[r] = qf[qk_off + r].max(0.0);
                }

                let mut den = 0.0f32;
                for r in 0..dk {
                    den += qr[r] * (ct * zc[r] + st * zs[r]);
                }
                let inv = 1.0 / (den + 1e-6);
                for c in 0..dv {
                    let mut num = 0.0f32;
                    for r in 0..dk {
                        num += qr[r] * (ct * sc[r * dv + c] + st * ss[r * dv + c]);
                    }
                    out[v_off + c] = num * inv;
                }
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, h, s, dv], q.device())?.to_dtype(dtype_in)
}
