use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    if x.rank() == 0 {
        return Err(SynaptixError::Unsupported("rms_norm: scalar input"));
    }
    // Fused backend-путь (CUDA — один launch вместо ~10 decomposed-ops). На CPU
    // или при неподдержке backend падаем в decomposed реализацию ниже.
    match x.rms_norm_fused(weight, eps, false) {
        Ok(out) => return Ok(out),
        Err(SynaptixError::Unsupported(_)) => {}
        Err(SynaptixError::NonContiguous) => {}
        Err(e) => return Err(e),
    }
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let last = x_f32.rank() - 1;
    let var = x_f32.sqr()?.mean_keepdim(last)?;
    let inv = var.add_scalar(eps)?.sqrt()?.recip()?;
    let x_norm = x_f32.broadcast_mul(&inv)?;

    let w_f32 = weight.to_dtype(DType::F32)?;
    let out = x_norm.broadcast_mul(&w_f32)?;
    out.to_dtype(dtype_in)
}

pub fn rms_norm_qwen(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    if x.rank() == 0 {
        return Err(SynaptixError::Unsupported("rms_norm_qwen: scalar input"));
    }
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let last = x_f32.rank() - 1;
    let var = x_f32.sqr()?.mean_keepdim(last)?;
    let inv = var.add_scalar(eps)?.sqrt()?.recip()?;
    let x_norm = x_f32.broadcast_mul(&inv)?;

    let w_f32 = weight.to_dtype(DType::F32)?;
    let gain = w_f32.add_scalar(1.0)?;
    let out = x_norm.broadcast_mul(&gain)?;
    out.to_dtype(dtype_in)
}

/// RMSNorm gated с явной `silu(gate)` активацией внутри — эквивалент
/// `mamba_ssm.ops.triton.layer_norm.RMSNormGated` (gate активируется silu).
/// Используется в Mamba2/GatedDeltaNet/Qwen36 SSM. Принимает **raw** gate;
/// silu делается внутри. Для уже активированного gate использовать
/// `rms_norm_gated` (без silu).
pub fn rms_norm_silu_gated(x: &Tensor, gate: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    if x.rank() == 0 {
        return Err(SynaptixError::Unsupported("rms_norm_silu_gated: scalar input"));
    }
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let gate_f32 = gate.to_dtype(DType::F32)?;
    let gated = x_f32.mul(&gate_f32.silu()?)?;
    let last = gated.rank() - 1;
    let var = gated.sqr()?.mean_keepdim(last)?;
    let inv = var.add_scalar(eps)?.sqrt()?.recip()?;
    let gated_norm = gated.broadcast_mul(&inv)?;
    let w_f32 = weight.to_dtype(DType::F32)?;
    let out = gated_norm.broadcast_mul(&w_f32)?;
    out.to_dtype(dtype_in)
}

/// RMSNorm gated **без** активации gate: `gated = x * gate`, далее обычный
/// RMS-normalize и масштабирование на weight. Для случаев, где gate уже
/// активирован пользователем (например, явный `silu(z)` или `tanh(z)` снаружи).
pub fn rms_norm_gated(x: &Tensor, gate: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    if x.rank() == 0 {
        return Err(SynaptixError::Unsupported("rms_norm_gated: scalar input"));
    }
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let gate_f32 = gate.to_dtype(DType::F32)?;
    let gated = x_f32.mul(&gate_f32)?;
    let last = gated.rank() - 1;
    let var = gated.sqr()?.mean_keepdim(last)?;
    let inv = var.add_scalar(eps)?.sqrt()?.recip()?;
    let gated_norm = gated.broadcast_mul(&inv)?;
    let w_f32 = weight.to_dtype(DType::F32)?;
    let out = gated_norm.broadcast_mul(&w_f32)?;
    out.to_dtype(dtype_in)
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self { Self { weight, eps } }
    pub fn weight(&self) -> &Tensor { &self.weight }
    pub fn eps(&self) -> f32 { self.eps }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> { rms_norm(x, &self.weight, self.eps) }
}

#[derive(Debug, Clone)]
pub struct RmsNormQwen {
    weight: Tensor,
    eps: f32,
}

impl RmsNormQwen {
    pub fn new(weight: Tensor, eps: f32) -> Self { Self { weight, eps } }
    pub fn weight(&self) -> &Tensor { &self.weight }
    pub fn eps(&self) -> f32 { self.eps }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> { rms_norm_qwen(x, &self.weight, self.eps) }
}

/// `RmsNormGated`: RMSNorm с **уже активированным** gate (без silu внутри).
#[derive(Debug, Clone)]
pub struct RmsNormGated {
    weight: Tensor,
    eps: f32,
}

impl RmsNormGated {
    pub fn new(weight: Tensor, eps: f32) -> Self { Self { weight, eps } }
    pub fn weight(&self) -> &Tensor { &self.weight }
    pub fn eps(&self) -> f32 { self.eps }
    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        rms_norm_gated(x, gate, &self.weight, self.eps)
    }
}

/// `RmsNormSiluGated`: RMSNorm с явной `silu(gate)` активацией внутри
/// (mamba_ssm RMSNormGated layer). Принимает raw gate.
#[derive(Debug, Clone)]
pub struct RmsNormSiluGated {
    weight: Tensor,
    eps: f32,
}

impl RmsNormSiluGated {
    pub fn new(weight: Tensor, eps: f32) -> Self { Self { weight, eps } }
    pub fn weight(&self) -> &Tensor { &self.weight }
    pub fn eps(&self) -> f32 { self.eps }
    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        rms_norm_silu_gated(x, gate, &self.weight, self.eps)
    }
}
