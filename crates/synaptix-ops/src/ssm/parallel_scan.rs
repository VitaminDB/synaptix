use synaptix_core::{error::Result, tensor::Tensor};

pub fn parallel_prefix_scan(x: &Tensor, gates: &Tensor) -> Result<Tensor> {
    // x, gates: [B, L, D] — sequential linear recurrence on CPU: y[t] = gates[t] * y[t-1] + x[t]
    let dims = x.dims();
    let (b, l, d) = (dims[0], dims[1], dims[2]);
    let mut h = x.narrow(1, 0, 1)?.squeeze(1)?; // [B, D]
    let mut ys = vec![h.unsqueeze(1)?];          // [[B, 1, D]]
    for t in 1..l {
        let xt = x.narrow(1, t, 1)?.squeeze(1)?;   // [B, D]
        let gt = gates.narrow(1, t, 1)?.squeeze(1)?; // [B, D]
        h = gt.mul(&h)?.add(&xt)?;                  // [B, D]
        ys.push(h.unsqueeze(1)?);
    }
    let refs: Vec<&Tensor> = ys.iter().collect();
    let _ = (b, l, d);
    Tensor::cat(&refs, 1)
}

pub fn associative_scan(gates: &Tensor, x: &Tensor) -> Result<Tensor> {
    // gates: [B, L, D], x: [B, L, D] — same as parallel_prefix_scan
    parallel_prefix_scan(x, gates)
}
