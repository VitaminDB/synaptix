use synaptix_core::{error::Result, tensor::Tensor};

pub struct MambaState {
    pub h: Tensor,
    pub conv_buf: Tensor,
}

pub fn mamba_step(
    x: &Tensor,
    state: &mut MambaState,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    dt: &Tensor,
) -> Result<Tensor> {
    // x: [B, D], a: [D, N], b: [B, N], c: [B, N], dt: [B, D]
    // h: [B, D, N]
    // ZOH: delta_A = exp(dt * a), delta_B = dt * b
    let dt3 = dt.unsqueeze(2)?;               // [B, D, 1]
    let a0 = a.unsqueeze(0)?;                 // [1, D, N]
    let a_bar = dt3.broadcast_mul(&a0)?.exp()?; // [B, D, N]
    let b1 = b.unsqueeze(1)?;                 // [B, 1, N]
    let delta_b = dt3.broadcast_mul(&b1)?;    // [B, D, N]
    let x3 = x.unsqueeze(2)?;                 // [B, D, 1]
    let delta_b_u = delta_b.broadcast_mul(&x3)?; // [B, D, N]
    state.h = a_bar.mul(&state.h)?.add(&delta_b_u)?; // [B, D, N]
    let c1 = c.unsqueeze(1)?;                 // [B, 1, N]
    let y = c1.broadcast_mul(&state.h)?.sum_keepdim(2)?.squeeze(2)?; // [B, D]
    Ok(y)
}

pub fn mamba_scan(
    x: &Tensor,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: &Tensor,
) -> Result<Tensor> {
    // x: [B, L, D]  a: [D, N] (discrete)  b: [B, L, N]  c: [B, L, N]  d: [D]
    let dims = x.dims();
    let (b_sz, l, d_inner) = (dims[0], dims[1], dims[2]);
    let n = a.dims()[1];
    let mut h = Tensor::zeros(vec![b_sz, d_inner, n], x.dtype(), x.device())?;
    let a0 = a.unsqueeze(0)?;  // [1, D, N]
    let d0 = d.unsqueeze(0)?;  // [1, D]
    let mut ys: Vec<Tensor> = Vec::with_capacity(l);
    for t in 0..l {
        let xt = x.narrow(1, t, 1)?.squeeze(1)?;    // [B, D]
        let bt = b.narrow(1, t, 1)?.squeeze(1)?;    // [B, N]
        let ct = c.narrow(1, t, 1)?.squeeze(1)?;    // [B, N]
        let b1 = bt.unsqueeze(1)?;                  // [B, 1, N]
        let b_x = b1.broadcast_mul(&xt.unsqueeze(2)?)?; // [B, D, N] via [B,1,N]*[B,D,1]
        h = a0.broadcast_mul(&h)?.add(&b_x)?;       // [B, D, N]
        let c1 = ct.unsqueeze(1)?;                  // [B, 1, N]
        let y_ssm = c1.broadcast_mul(&h)?.sum_keepdim(2)?.squeeze(2)?; // [B, D]
        let y = y_ssm.add(&d0.broadcast_mul(&xt)?)?; // [B, D]
        ys.push(y.unsqueeze(1)?);                   // [B, 1, D]
    }
    let refs: Vec<&Tensor> = ys.iter().collect();
    Tensor::cat(&refs, 1)
}
