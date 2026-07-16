use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Monarch mixer — структурное (block-structured) линейное смешение по последней
/// оси `D = d1·d2`: каждый вектор решейпится в `X[d1,d2]` (row-major) и
/// преобразуется как `Y = M1 · X · M2ᵀ`, затем разворачивается обратно:
///   `out[i·d2 + j] = Σ_p Σ_q M1[i,p]·X[p,q]·M2[j,q]`.
/// `x:[..,D]`, `m1:[d1,d1]`, `m2:[d2,d2]`.
pub fn monarch_mixer(x: &Tensor, m1: &Tensor, m2: &Tensor) -> Result<Tensor> {
    let d = *x.dims().last().ok_or(SynaptixError::Unsupported("monarch_mixer: пустая форма"))?;
    if m1.rank() != 2 || m1.dims()[0] != m1.dims()[1] {
        return Err(SynaptixError::Unsupported("monarch_mixer: m1 must be square [d1,d1]"));
    }
    if m2.rank() != 2 || m2.dims()[0] != m2.dims()[1] {
        return Err(SynaptixError::Unsupported("monarch_mixer: m2 must be square [d2,d2]"));
    }
    let d1 = m1.dims()[0];
    let d2 = m2.dims()[0];
    if d1 * d2 != d {
        return Err(SynaptixError::Unsupported("monarch_mixer: requires d1*d2 == D"));
    }
    let dtype_in = x.dtype();
    let xf = f32v(x)?;
    let m1f = f32v(m1)?;
    let m2f = f32v(m2)?;
    let rows = xf.len() / d;

    let mut out = vec![0.0f32; xf.len()];
    let mut tmp = vec![0.0f32; d1 * d2];
    for r in 0..rows {
        let off = r * d;
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
    Tensor::from_vec::<_, f32>(out, x.dims().to_vec(), x.device())?.to_dtype(dtype_in)
}
