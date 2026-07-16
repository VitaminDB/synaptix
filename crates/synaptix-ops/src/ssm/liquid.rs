use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Liquid (LTC-подобный) ODE-шаг: один шаг прямого Эйлера (Δt=1) для
/// `dh/dt = (−h + tanh(x)) / τ`:
///   `h_new = h + (1/τ) ⊙ (tanh(x) − h) = (1 − 1/τ)⊙h + (1/τ)⊙tanh(x)`.
/// `x,state:[B,D]`, `tau:[D]` (> 0). Возвращает новое состояние `[B,D]`.
pub fn liquid_step(x: &Tensor, state: &Tensor, tau: &Tensor) -> Result<Tensor> {
    if x.rank() != 2 {
        return Err(SynaptixError::Unsupported("liquid: x must be rank-2 [B,D]"));
    }
    let (bsz, d) = (x.dims()[0], x.dims()[1]);
    if state.dims() != x.dims() {
        return Err(SynaptixError::shape_mismatch(x.dims(), state.dims()));
    }
    if tau.dims() != [d] {
        return Err(SynaptixError::Unsupported("liquid: tau must be [D]"));
    }
    let dtype_in = x.dtype();
    let xf = f32v(x)?;
    let sf = f32v(state)?;
    let tf = f32v(tau)?;

    let mut out = vec![0.0f32; bsz * d];
    for bi in 0..bsz {
        for di in 0..d {
            let idx = bi * d + di;
            let inv_tau = 1.0 / tf[di];
            out[idx] = sf[idx] + inv_tau * (xf[idx].tanh() - sf[idx]);
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![bsz, d], x.device())?.to_dtype(dtype_in)
}
