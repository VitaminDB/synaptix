use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn layer_norm(
    x: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    // Fused backend-путь (CUDA: один kernel-launch на строку вместо ~12 decomposed-
    // ops + промежуточных аллокаций — критично для CUDA-graph decode). Требует
    // gamma (weight); inference-only (no-grad). На CPU/неподдержке/grad → decomposed.
    if !synaptix_core::grad::is_grad_enabled() {
        if let Some(w) = weight {
            match x.layer_norm_fused(w, bias, eps) {
                Ok(out) => return Ok(out),
                Err(synaptix_core::error::SynaptixError::Unsupported(_))
                | Err(synaptix_core::error::SynaptixError::NonContiguous) => {}
                Err(e) => return Err(e),
            }
        }
    }

    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let last = x_f32.rank() - 1;
    let mean = x_f32.mean_keepdim(last)?;
    let centered = x_f32.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(last)?;
    let inv = var.add_scalar(eps)?.sqrt()?.recip()?;
    let normed = centered.broadcast_mul(&inv)?;
    let scaled = match weight {
        Some(w) => {
            let w_f32 = w.to_dtype(DType::F32)?;
            normed.broadcast_mul(&w_f32)?
        }
        None => normed,
    };
    let out = match bias {
        Some(b) => {
            let b_f32 = b.to_dtype(DType::F32)?;
            scaled.broadcast_add(&b_f32)?
        }
        None => scaled,
    };
    out.to_dtype(dtype_in)
}

#[derive(Debug, Clone)]
pub struct LayerNorm {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Option<Tensor>, bias: Option<Tensor>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }
    pub fn weight(&self) -> Option<&Tensor> { self.weight.as_ref() }
    pub fn bias(&self) -> Option<&Tensor> { self.bias.as_ref() }
    pub fn eps(&self) -> f32 { self.eps }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        layer_norm(x, self.weight.as_ref(), self.bias.as_ref(), self.eps)
    }
}
