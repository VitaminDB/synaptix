use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

/// IP-Adapter (Ye et al., 2023) — упрощённая additive-инъекция изображения.
///
/// `proj(image_emb): [B, hidden]` бродкастится по seq-измерению и прибавляется
/// к hidden state: `y = x + scale · broadcast(proj(image_emb))`.
///
/// Реальный IP-Adapter добавляет отдельную K/V-ветку в cross-attention; здесь
/// мы оставляем эквивалентный inference-stub с тем же scale-API. Если caller
/// готов сам собрать pre-projected токены, `forward_tokens` принимает
/// `[B, N, hidden]` и тоже их бродкастит-пулит-добавляет.
pub struct IpAdapter {
    pub proj: Linear,
    pub scale: f32,
}

impl IpAdapter {
    pub fn new(image_emb_dim: usize, hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_init(
                image_emb_dim, hidden_size, true,
                InitMethod::XavierUniform { fan_in: image_emb_dim, fan_out: hidden_size },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            scale: 1.0,
        })
    }

    pub fn from_weights(proj_w: Tensor, proj_b: Option<Tensor>, scale: f32) -> Result<Self> {
        Ok(Self {
            proj: Linear::new(proj_w, proj_b)?,
            scale,
        })
    }

    /// `x: [B, T, H]`, `image_emb: [B, image_emb_dim]` → `[B, T, H]`.
    pub fn forward(&self, x: &Tensor, image_emb: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 {
            return Err(SynaptixError::Unsupported("IpAdapter::forward: x must be [B, T, H]"));
        }
        if image_emb.rank() != 2 || image_emb.dims()[0] != x.dims()[0] {
            return Err(SynaptixError::shape_mismatch(&[x.dims()[0], image_emb.dims().last().copied().unwrap_or(0)], image_emb.dims()));
        }
        let projected = self.proj.forward(image_emb)?;
        let expanded = projected
            .unsqueeze(1)?
            .expand(&[x.dims()[0], x.dims()[1], x.dims()[2]])?
            .contiguous()?;
        let scaled = expanded.affine(self.scale, 0.0)?;
        x.add(&scaled)
    }
}
