use synaptix_core::{
    error::{Result, SynaptixError},
    tensor::Tensor,
};

use super::linear::{linear_dims, softmax_inplace, to_f32_vec};

/// ABC (Attention with Bounded Context): двухстадийный attention через `m`
/// memory-слотов. Non-causal. `slot_proj` формы `[dk, m]`:
///   `sl[j,slot] = Σ_r k[j,r]·Φ[r,slot]`;  `φ = softmax_seq(sl)` (по оси j);
///   `Mem[slot,:] = Σ_j φ[j,slot] v_j`,  `Memk[slot,:] = Σ_j φ[j,slot] k_j`;
///   `α[i,:] = softmax(q_i·Memkᵀ / √dk)`;  `o_i = Σ_slot α[i,slot] Mem[slot,:]`.
pub fn abc_attention(q: &Tensor, k: &Tensor, v: &Tensor, slot_proj: &Tensor) -> Result<Tensor> {
    let (b, h, s, dk, dv) = linear_dims(q, k, v)?;
    if slot_proj.rank() != 2 || slot_proj.dims()[0] != dk {
        return Err(SynaptixError::Unsupported("abc: slot_proj must be [dk, m]"));
    }
    let m = slot_proj.dims()[1];
    let dtype_in = q.dtype();
    let qf = to_f32_vec(q)?;
    let kf = to_f32_vec(k)?;
    let vf = to_f32_vec(v)?;
    let phi_w = to_f32_vec(slot_proj)?;

    let scale = (dk as f32).powf(-0.5);
    let mut out = vec![0.0f32; b * h * s * dv];
    for bi in 0..b {
        for hi in 0..h {
            let base = (bi * h + hi) * s;
            // slot-логиты sl[j,slot] и softmax по оси j (отдельно для каждого слота).
            let mut sl = vec![0.0f32; s * m];
            for j in 0..s {
                let k_off = (base + j) * dk;
                for slot in 0..m {
                    let mut acc = 0.0f32;
                    for r in 0..dk {
                        acc += kf[k_off + r] * phi_w[r * m + slot];
                    }
                    sl[j * m + slot] = acc;
                }
            }
            // softmax вдоль j для каждого слота
            let mut col = vec![0.0f32; s];
            for slot in 0..m {
                for j in 0..s {
                    col[j] = sl[j * m + slot];
                }
                softmax_inplace(&mut col);
                for j in 0..s {
                    sl[j * m + slot] = col[j];
                }
            }
            // Mem[slot,c] и Memk[slot,r]
            let mut mem = vec![0.0f32; m * dv];
            let mut memk = vec![0.0f32; m * dk];
            for j in 0..s {
                let k_off = (base + j) * dk;
                let v_off = (base + j) * dv;
                for slot in 0..m {
                    let p = sl[j * m + slot];
                    let mrow = slot * dv;
                    for c in 0..dv {
                        mem[mrow + c] += p * vf[v_off + c];
                    }
                    let mkrow = slot * dk;
                    for r in 0..dk {
                        memk[mkrow + r] += p * kf[k_off + r];
                    }
                }
            }
            // запрос по слотам
            let mut alpha = vec![0.0f32; m];
            for t in 0..s {
                let q_off = (base + t) * dk;
                let o_off = (base + t) * dv;
                for slot in 0..m {
                    let mkrow = slot * dk;
                    let mut acc = 0.0f32;
                    for r in 0..dk {
                        acc += qf[q_off + r] * memk[mkrow + r];
                    }
                    alpha[slot] = acc * scale;
                }
                softmax_inplace(&mut alpha);
                for c in 0..dv {
                    let mut acc = 0.0f32;
                    for slot in 0..m {
                        acc += alpha[slot] * mem[slot * dv + c];
                    }
                    out[o_off + c] = acc;
                }
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, h, s, dv], q.device())?.to_dtype(dtype_in)
}
