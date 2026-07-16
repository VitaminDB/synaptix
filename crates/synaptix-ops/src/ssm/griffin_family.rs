use synaptix_core::{error::Result, tensor::Tensor};

pub fn hawk_step(x: &Tensor, state: &Tensor, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Hawk RG-LRU step (Real-Gated Linear Recurrent Unit from Griffin paper).
    // h = a * h_prev + (1 - a²).sqrt() * b * x   (a ∈ (0,1) via sigmoid, b gating)
    // Returns (y, new_h) but we flatten: y = new_h (output equals state).
    let one_minus_a2 = a.sqr()?.affine(-1.0, 1.0)?.sqrt()?;
    let new_h = a.mul(state)?.add(&one_minus_a2.mul(b)?.mul(x)?)?;
    Ok(new_h)
}

pub fn griffin_step(x: &Tensor, state: &Tensor, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Griffin = Hawk RG-LRU (used for global mixing).
    // Local attention handled externally; here we only do the recurrent path.
    hawk_step(x, state, a, b)
}

pub fn hgrn2_step(x: &Tensor, state: &Tensor, forget: &Tensor) -> Result<Tensor> {
    // HGRN2: h = forget * h_prev + (1 - forget) * x   (forget gate ∈ (0,1))
    let retain = forget.affine(-1.0, 1.0)?; // 1 - forget
    forget.mul(state)?.add(&retain.mul(x)?)
}
