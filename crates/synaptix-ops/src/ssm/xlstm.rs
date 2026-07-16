use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

#[inline]
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// sLSTM (xLSTM-семейство) — шаг скалярной памяти. `x:[B,4D]` — это
/// pre-activations гейтов, сконкатенированные как `[z, i, f, o]`; `h,c:[B,D]`.
/// (sigmoid-гейтированный вариант ячейки):
///   `c_new = σ(f)⊙c + σ(i)⊙tanh(z)`;  `h_new = σ(o)⊙tanh(c_new)`.
/// Возвращает `h_new:[B,D]`.
pub fn slstm_step(x: &Tensor, h: &Tensor, c: &Tensor) -> Result<Tensor> {
    if h.rank() != 2 {
        return Err(SynaptixError::Unsupported("slstm: h must be rank-2 [B,D]"));
    }
    let (bsz, d) = (h.dims()[0], h.dims()[1]);
    if c.dims() != [bsz, d] {
        return Err(SynaptixError::shape_mismatch(h.dims(), c.dims()));
    }
    if x.dims() != [bsz, 4 * d] {
        return Err(SynaptixError::Unsupported("slstm: x must be [B,4D] = [z,i,f,o]"));
    }
    let dtype_in = h.dtype();
    let xf = f32v(x)?;
    let cf = f32v(c)?;

    let mut out = vec![0.0f32; bsz * d];
    for bi in 0..bsz {
        let xb = bi * 4 * d;
        for di in 0..d {
            let z = xf[xb + di].tanh();
            let i = sigmoid(xf[xb + d + di]);
            let f = sigmoid(xf[xb + 2 * d + di]);
            let o = sigmoid(xf[xb + 3 * d + di]);
            let c_new = f * cf[bi * d + di] + i * z;
            out[bi * d + di] = o * c_new.tanh();
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![bsz, d], h.device())?.to_dtype(dtype_in)
}

/// mLSTM (xLSTM-семейство) — шаг матричной памяти. `x:[B,3D]` = `[q, k, v]`,
/// матричное состояние `c:[B,D·D]` (C[i,j]), нормализатор `h:[B,D]` (n[j]).
/// Ковариационное обновление + нормализация (forget=input=1):
///   `C_new[i,j] = C[i,j] + v[i]·k[j]`;  `n_new[j] = n[j] + k[j]`;
///   `out[i] = (Σ_j C_new[i,j]·q[j]) / max(|Σ_j n_new[j]·q[j]|, 1)`.
/// Возвращает извлечённый выход `out:[B,D]`.
pub fn mlstm_step(x: &Tensor, h: &Tensor, c: &Tensor) -> Result<Tensor> {
    if h.rank() != 2 {
        return Err(SynaptixError::Unsupported("mlstm: h (normalizer) must be rank-2 [B,D]"));
    }
    let (bsz, d) = (h.dims()[0], h.dims()[1]);
    if x.dims() != [bsz, 3 * d] {
        return Err(SynaptixError::Unsupported("mlstm: x must be [B,3D] = [q,k,v]"));
    }
    if c.dims() != [bsz, d * d] {
        return Err(SynaptixError::Unsupported("mlstm: c (matrix state) must be [B,D*D]"));
    }
    let dtype_in = h.dtype();
    let xf = f32v(x)?;
    let hf = f32v(h)?;
    let cf = f32v(c)?;

    let mut out = vec![0.0f32; bsz * d];
    let mut c_new = vec![0.0f32; d * d];
    let mut n_new = vec![0.0f32; d];
    for bi in 0..bsz {
        let xb = bi * 3 * d;
        // q = x[0..d], k = x[d..2d], v = x[2d..3d]
        let q = &xf[xb..xb + d];
        let k = &xf[xb + d..xb + 2 * d];
        let v = &xf[xb + 2 * d..xb + 3 * d];
        let cb = bi * d * d;
        for i in 0..d {
            let row = i * d;
            for j in 0..d {
                c_new[row + j] = cf[cb + row + j] + v[i] * k[j];
            }
        }
        for j in 0..d {
            n_new[j] = hf[bi * d + j] + k[j];
        }
        let mut den = 0.0f32;
        for j in 0..d {
            den += n_new[j] * q[j];
        }
        let denom = den.abs().max(1.0);
        for i in 0..d {
            let row = i * d;
            let mut num = 0.0f32;
            for j in 0..d {
                num += c_new[row + j] * q[j];
            }
            out[bi * d + i] = num / denom;
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![bsz, d], h.device())?.to_dtype(dtype_in)
}
