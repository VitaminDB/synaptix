pub mod apg;
pub mod cfg;
pub mod cfg_zero;
pub mod msg;
pub mod negative_prompt;
pub mod pag;
pub mod refit;

use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub trait Guidance: Send {
    fn prepare_latents(&self, latent: &Tensor) -> Result<Tensor>;
    fn apply(&self, cond: &Tensor, uncond: &Tensor, scale: f32) -> Result<Tensor>;
}
