use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// S4 (diagonal SSM): по каждому из D каналов независимый SISO state-space с
/// диагональным состоянием размера N и статическими (не data-dependent) `b`,`c`.
/// `x:[B,L,D]`, дискретные `a,b,c:[D,N]`. Causal scan:
///   `h_t[d,n] = a[d,n]·h_{t-1}[d,n] + b[d,n]·x_t[d]`,  `y_t[d] = Σ_n c[d,n]·h_t[d,n]`.
pub fn s4_forward(x: &Tensor, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    if x.rank() != 3 {
        return Err(SynaptixError::Unsupported("s4: x must be rank-3 [B,L,D]"));
    }
    let (bsz, l, d) = (x.dims()[0], x.dims()[1], x.dims()[2]);
    if a.rank() != 2 || a.dims()[0] != d {
        return Err(SynaptixError::Unsupported("s4: a must be [D,N]"));
    }
    let n = a.dims()[1];
    if b.dims() != [d, n] || c.dims() != [d, n] {
        return Err(SynaptixError::Unsupported("s4: b,c must be [D,N]"));
    }
    let dtype_in = x.dtype();
    let xf = f32v(x)?;
    let af = f32v(a)?;
    let bf = f32v(b)?;
    let cf = f32v(c)?;

    let mut out = vec![0.0f32; bsz * l * d];
    for bi in 0..bsz {
        let mut h = vec![0.0f32; d * n];
        for t in 0..l {
            let x_off = (bi * l + t) * d;
            for di in 0..d {
                let xt = xf[x_off + di];
                let row = di * n;
                let mut y = 0.0f32;
                for ni in 0..n {
                    let idx = row + ni;
                    h[idx] = af[idx] * h[idx] + bf[idx] * xt;
                    y += cf[idx] * h[idx];
                }
                out[x_off + di] = y;
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![bsz, l, d], x.device())?.to_dtype(dtype_in)
}

/// S5 (diagonal MIMO SSM): единый state-space с диагональным состоянием размера N,
/// вход/выход размерности H. `x:[B,L,H]`, дискретные `lambda:[N]`, `b:[N,H]`,
/// `c:[H,N]`, skip `d:[H]`. Causal scan:
///   `h_t[n] = lambda[n]·h_{t-1}[n] + Σ_h b[n,h]·x_t[h]`,
///   `y_t[h] = Σ_n c[h,n]·h_t[n] + d[h]·x_t[h]`.
pub fn s5_forward(
    x: &Tensor,
    lambda: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: &Tensor,
) -> Result<Tensor> {
    if x.rank() != 3 {
        return Err(SynaptixError::Unsupported("s5: x must be rank-3 [B,L,H]"));
    }
    let (bsz, l, hsz) = (x.dims()[0], x.dims()[1], x.dims()[2]);
    if lambda.rank() != 1 {
        return Err(SynaptixError::Unsupported("s5: lambda must be [N]"));
    }
    let n = lambda.dims()[0];
    if b.dims() != [n, hsz] || c.dims() != [hsz, n] || d.dims() != [hsz] {
        return Err(SynaptixError::Unsupported("s5: expected b[N,H], c[H,N], d[H]"));
    }
    let dtype_in = x.dtype();
    let xf = f32v(x)?;
    let lf = f32v(lambda)?;
    let bf = f32v(b)?;
    let cf = f32v(c)?;
    let df = f32v(d)?;

    let mut out = vec![0.0f32; bsz * l * hsz];
    for bi in 0..bsz {
        let mut h = vec![0.0f32; n];
        for t in 0..l {
            let x_off = (bi * l + t) * hsz;
            // h[n] = lambda[n] h[n] + Σ_h b[n,h] x_t[h]
            for ni in 0..n {
                let mut acc = 0.0f32;
                let brow = ni * hsz;
                for hh in 0..hsz {
                    acc += bf[brow + hh] * xf[x_off + hh];
                }
                h[ni] = lf[ni] * h[ni] + acc;
            }
            // y[h] = Σ_n c[h,n] h[n] + d[h] x_t[h]
            for hh in 0..hsz {
                let crow = hh * n;
                let mut acc = 0.0f32;
                for ni in 0..n {
                    acc += cf[crow + ni] * h[ni];
                }
                out[x_off + hh] = acc + df[hh] * xf[x_off + hh];
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![bsz, l, hsz], x.device())?.to_dtype(dtype_in)
}
