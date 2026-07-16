use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Monarch — структурное (block-structured) линейное смешение вместо плотной
/// матрицы. Канал `D = d1·d2`; каждый вектор `x[D]` решейпится в `X[d1,d2]`
/// (row-major) и преобразуется как `Y = M1 · X · M2ᵀ`, затем разворачивается:
///   `Y[i,j] = Σ_p Σ_q M1[i,p]·X[p,q]·M2[j,q]`,  `out[i·d2 + j] = Y[i,j]`.
/// `x:[B,L,D]`, `m1:[d1,d1]`, `m2:[d2,d2]`.
pub fn monarch_ssm(x: &Tensor, m1: &Tensor, m2: &Tensor) -> Result<Tensor> {
    if x.rank() != 3 {
        return Err(SynaptixError::Unsupported("monarch: x must be rank-3 [B,L,D]"));
    }
    let (bsz, l, d) = (x.dims()[0], x.dims()[1], x.dims()[2]);
    if m1.rank() != 2 || m1.dims()[0] != m1.dims()[1] {
        return Err(SynaptixError::Unsupported("monarch: m1 must be square [d1,d1]"));
    }
    if m2.rank() != 2 || m2.dims()[0] != m2.dims()[1] {
        return Err(SynaptixError::Unsupported("monarch: m2 must be square [d2,d2]"));
    }
    let d1 = m1.dims()[0];
    let d2 = m2.dims()[0];
    if d1 * d2 != d {
        return Err(SynaptixError::Unsupported("monarch: requires d1*d2 == D"));
    }
    let dtype_in = x.dtype();
    let xf = f32v(x)?;
    let m1f = f32v(m1)?;
    let m2f = f32v(m2)?;

    let mut out = vec![0.0f32; bsz * l * d];
    // tmp[p,j] = Σ_q X[p,q] M2[j,q]
    let mut tmp = vec![0.0f32; d1 * d2];
    for bl in 0..bsz * l {
        let off = bl * d;
        // tmp[p,j] = Σ_q X[p,q]·M2[j,q]
        for p in 0..d1 {
            let xrow = off + p * d2;
            for j in 0..d2 {
                let m2row = j * d2;
                let mut acc = 0.0f32;
                for q in 0..d2 {
                    acc += xf[xrow + q] * m2f[m2row + q];
                }
                tmp[p * d2 + j] = acc;
            }
        }
        // Y[i,j] = Σ_p M1[i,p]·tmp[p,j]
        for i in 0..d1 {
            let m1row = i * d1;
            for j in 0..d2 {
                let mut acc = 0.0f32;
                for p in 0..d1 {
                    acc += m1f[m1row + p] * tmp[p * d2 + j];
                }
                out[off + i * d2 + j] = acc;
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![bsz, l, d], x.device())?.to_dtype(dtype_in)
}
