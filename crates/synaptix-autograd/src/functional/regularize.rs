use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn dropout(x: &Tensor, p: f32, training: bool, mask: Option<&Tensor>) -> Result<Tensor> {
    if !training || p <= 0.0 {
        return Ok(x.clone());
    }
    let mask = match mask {
        Some(m) => m.clone(),
        None => {
            return Err(synaptix_core::error::SynaptixError::Unsupported(
                "dropout requires precomputed mask (RNG owned by caller)",
            ));
        }
    };
    let scale = 1.0 / (1.0 - p);
    x.mul(&mask)?.affine(scale, 0.0)
}
