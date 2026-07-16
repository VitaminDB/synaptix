use synaptix_core::{
    error::{Result, SynaptixError},
    tensor::Tensor,
};

use super::linear::{linear_dims, to_f32_vec};

/// Chunkwise-parallel форма обычного causal linear-attention (без decay).
/// Для чанка `[c0,c1)`: `o_t = q_t·S_inter + Σ_{j=c0..=t} (q_t·k_j) v_j`,
/// где `S_inter = Σ_{j<c0} k_j v_jᵀ` переносится между чанками.
/// Результат математически эквивалентен рекуррентному causal-скану
/// `o_t = Σ_{j≤t}(q_t·k_j) v_j` — на этом и строится проверочный тест.
pub fn chunk_scan(q: &Tensor, k: &Tensor, v: &Tensor, chunk_size: usize) -> Result<Tensor> {
    let (b, h, s, dk, dv) = linear_dims(q, k, v)?;
    if chunk_size == 0 {
        return Err(SynaptixError::Unsupported("chunk_scan: chunk_size must be > 0"));
    }
    let dtype_in = q.dtype();
    let qf = to_f32_vec(q)?;
    let kf = to_f32_vec(k)?;
    let vf = to_f32_vec(v)?;

    let mut out = vec![0.0f32; b * h * s * dv];
    let mut o = vec![0.0f32; dv];
    for bi in 0..b {
        for hi in 0..h {
            // S_inter — накопленное состояние от предыдущих чанков.
            let mut state = vec![0.0f32; dk * dv];
            let mut c0 = 0;
            while c0 < s {
                let c1 = (c0 + chunk_size).min(s);
                // выход для каждой позиции чанка
                for t in c0..c1 {
                    let qk_off_t = ((bi * h + hi) * s + t) * dk;
                    let o_off = ((bi * h + hi) * s + t) * dv;
                    // inter: q_t · S_inter
                    for c in 0..dv {
                        let mut acc = 0.0f32;
                        for r in 0..dk {
                            acc += qf[qk_off_t + r] * state[r * dv + c];
                        }
                        o[c] = acc;
                    }
                    // intra: Σ_{j=c0..=t} (q_t·k_j) v_j
                    for j in c0..=t {
                        let qk_off_j = ((bi * h + hi) * s + j) * dk;
                        let v_off_j = ((bi * h + hi) * s + j) * dv;
                        let mut score = 0.0f32;
                        for r in 0..dk {
                            score += qf[qk_off_t + r] * kf[qk_off_j + r];
                        }
                        for c in 0..dv {
                            o[c] += score * vf[v_off_j + c];
                        }
                    }
                    out[o_off..o_off + dv].copy_from_slice(&o[..dv]);
                }
                // обновляем S_inter вкладом текущего чанка
                for j in c0..c1 {
                    let qk_off_j = ((bi * h + hi) * s + j) * dk;
                    let v_off_j = ((bi * h + hi) * s + j) * dv;
                    for r in 0..dk {
                        let kk = kf[qk_off_j + r];
                        let row = r * dv;
                        for c in 0..dv {
                            state[row + c] += kk * vf[v_off_j + c];
                        }
                    }
                }
                c0 = c1;
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, h, s, dv], q.device())?.to_dtype(dtype_in)
}
