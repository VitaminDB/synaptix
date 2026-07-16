use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Test-Time Training (TTT-Linear): линейная внутренняя модель `W[D,D]`,
/// которая на каждом токене делает один шаг online-градиентного спуска по
/// self-supervised потере реконструкции `½‖W·x_t − x_t‖²`:
///   `pred = W·x_t`;  `grad = (pred − x_t)·x_tᵀ`;  `W ← W − lr·grad`;
///   `y_t = W·x_t` (после обновления).
/// `W` сбрасывается к стартовому `w` для каждого элемента батча. `x:[B,L,D]`, `w:[D,D]`.
pub fn ttt_layer(x: &Tensor, w: &Tensor, lr: f32) -> Result<Tensor> {
    if x.rank() != 3 {
        return Err(SynaptixError::Unsupported("ttt: x must be rank-3 [B,L,D]"));
    }
    let (bsz, l, d) = (x.dims()[0], x.dims()[1], x.dims()[2]);
    if w.dims() != [d, d] {
        return Err(SynaptixError::Unsupported("ttt: w must be [D,D]"));
    }
    let dtype_in = x.dtype();
    let xf = f32v(x)?;
    let w0 = f32v(w)?;

    let mut out = vec![0.0f32; bsz * l * d];
    let mut pred = vec![0.0f32; d];
    for bi in 0..bsz {
        let mut wmat = w0.clone(); // [D,D], сбрасывается на каждый элемент батча
        for t in 0..l {
            let off = (bi * l + t) * d;
            // pred = W x_t
            for i in 0..d {
                let mut acc = 0.0f32;
                let row = i * d;
                for j in 0..d {
                    acc += wmat[row + j] * xf[off + j];
                }
                pred[i] = acc;
            }
            // W -= lr (pred - x_t) x_t^T
            for i in 0..d {
                let err = pred[i] - xf[off + i];
                let row = i * d;
                let step = lr * err;
                for j in 0..d {
                    wmat[row + j] -= step * xf[off + j];
                }
            }
            // y_t = W x_t (после обновления)
            for i in 0..d {
                let mut acc = 0.0f32;
                let row = i * d;
                for j in 0..d {
                    acc += wmat[row + j] * xf[off + j];
                }
                out[off + i] = acc;
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![bsz, l, d], x.device())?.to_dtype(dtype_in)
}
