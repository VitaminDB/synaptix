use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

/// Валидирует q,k,v как rank-4 `[B,H,S,D]` (dk у q/k, dv у v) и возвращает
/// `(b, h, s, dk, dv)`. Общий валидатор для всего семейства linear-attention.
pub(super) fn linear_dims(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<(usize, usize, usize, usize, usize)> {
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Err(SynaptixError::Unsupported(
            "linear attention: requires rank-4 [B,H,S,D]",
        ));
    }
    let b = q.dims()[0];
    let h = q.dims()[1];
    let s = q.dims()[2];
    let dk = q.dims()[3];
    let dv = v.dims()[3];
    if k.dims() != [b, h, s, dk] {
        return Err(SynaptixError::shape_mismatch(q.dims(), k.dims()));
    }
    if v.dims()[0] != b || v.dims()[1] != h || v.dims()[2] != s {
        return Err(SynaptixError::shape_mismatch(q.dims(), v.dims()));
    }
    Ok((b, h, s, dk, dv))
}

/// Преобразует тензор в F32 и возвращает плоский (row-major) `Vec<f32>`.
pub(super) fn to_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Численно устойчивый softmax над срезом `xs` (in-place): max-subtract → exp → нормализация.
pub(super) fn softmax_inplace(xs: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &x in xs.iter() {
        if x > m {
            m = x;
        }
    }
    if !m.is_finite() {
        // все -inf → нулевой выход (как torch при полном маскировании строки)
        for x in xs.iter_mut() {
            *x = 0.0;
        }
        return;
    }
    let mut sum = 0.0f32;
    for x in xs.iter_mut() {
        *x = (*x - m).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in xs.iter_mut() {
            *x /= sum;
        }
    }
}

/// Наивная (без нормализации) глобальная linear-attention:
/// `S = Σ_t k_t v_tᵀ` (форма `[dk,dv]`), `o_t = q_tᵀ S`. Non-causal.
pub fn naive_linear_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
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
                let k_off = ((bi * h + hi) * s + t) * dk;
                let v_off = ((bi * h + hi) * s + t) * dv;
                for r in 0..dk {
                    let kk = kf[k_off + r];
                    for c in 0..dv {
                        state[r * dv + c] += kk * vf[v_off + c];
                    }
                }
            }
            for t in 0..s {
                let q_off = ((bi * h + hi) * s + t) * dk;
                let o_off = ((bi * h + hi) * s + t) * dv;
                for c in 0..dv {
                    let mut acc = 0.0f32;
                    for r in 0..dk {
                        acc += qf[q_off + r] * state[r * dv + c];
                    }
                    out[o_off + c] = acc;
                }
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, h, s, dv], q.device())?.to_dtype(dtype_in)
}
