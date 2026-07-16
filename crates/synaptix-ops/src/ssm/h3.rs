use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// H3 (Hungry Hungry Hippos) — упрощённый two-stage diagonal SSM с
/// мультипликативным гейтированием: значение `x` гейтируется ключом `k`,
/// проходит per-channel диагональный SSM с decay `a[D]` (состояние N=1),
/// затем гейтируется запросом `q`:
///   `u_t = k_t ⊙ x_t`;  `s_t[d] = a[d]·s_{t-1}[d] + u_t[d]`;  `y_t = q_t ⊙ s_t`.
/// `x,k,q:[B,L,D]`, `a:[D]`.
pub fn h3_forward(x: &Tensor, k: &Tensor, q: &Tensor, a: &Tensor) -> Result<Tensor> {
    if x.rank() != 3 {
        return Err(SynaptixError::Unsupported("h3: x must be rank-3 [B,L,D]"));
    }
    let (bsz, l, d) = (x.dims()[0], x.dims()[1], x.dims()[2]);
    if k.dims() != x.dims() || q.dims() != x.dims() {
        return Err(SynaptixError::shape_mismatch(x.dims(), k.dims()));
    }
    if a.dims() != [d] {
        return Err(SynaptixError::Unsupported("h3: a must be [D]"));
    }
    let dtype_in = x.dtype();
    let xf = f32v(x)?;
    let kf = f32v(k)?;
    let qf = f32v(q)?;
    let af = f32v(a)?;

    let mut out = vec![0.0f32; bsz * l * d];
    for bi in 0..bsz {
        let mut s = vec![0.0f32; d];
        for t in 0..l {
            let off = (bi * l + t) * d;
            for di in 0..d {
                let u = kf[off + di] * xf[off + di];
                s[di] = af[di] * s[di] + u;
                out[off + di] = qf[off + di] * s[di];
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![bsz, l, d], x.device())?.to_dtype(dtype_in)
}
