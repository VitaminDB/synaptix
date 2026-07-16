use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

pub struct Patchify {
    pub patch_size: usize,
    pub proj: Linear,
    pub in_channels: usize,
    pub hidden_size: usize,
}

impl Patchify {
    pub fn new(patch_size: usize, in_channels: usize, hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        let patch_dim = patch_size * patch_size * in_channels;
        Ok(Self {
            patch_size,
            proj: Linear::from_init(patch_dim, hidden_size, true, InitMethod::XavierUniform { fan_in: patch_dim, fan_out: hidden_size }, InitMethod::Zeros, device, dtype, 0)?,
            in_channels,
            hidden_size,
        })
    }

    pub fn from_weights(
        patch_size: usize, in_channels: usize, weight: Tensor, bias: Option<Tensor>,
    ) -> Result<Self> {
        let proj = Linear::new(weight, bias)?;
        let hidden_size = proj.out_features();
        Ok(Self { patch_size, proj, in_channels, hidden_size })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 4 {
            return Err(SynaptixError::Unsupported(
                "patchify: input must be [B, C, H, W]",
            ));
        }
        let (b, c, h, w) = (x.dims()[0], x.dims()[1], x.dims()[2], x.dims()[3]);
        if h % self.patch_size != 0 || w % self.patch_size != 0 {
            return Err(SynaptixError::Unsupported(
                "patchify: H and W must divide by patch_size",
            ));
        }
        let p = self.patch_size;
        let nh = h / p;
        let nw = w / p;
        let reshaped = x.reshape(vec![b, c, nh, p, nw, p])?;
        let permuted = reshaped.permute(vec![0, 2, 4, 1, 3, 5])?.contiguous()?;
        let tokens = permuted.reshape(vec![b, nh * nw, c * p * p])?;
        self.proj.forward(&tokens)
    }

    pub fn unpatchify(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        if x.rank() != 3 {
            return Err(SynaptixError::Unsupported(
                "unpatchify: input must be [B, N, P*P*C]",
            ));
        }
        let p = self.patch_size;
        let c = self.in_channels;
        let b = x.dims()[0];
        let nh = h / p;
        let nw = w / p;
        if nh * nw != x.dims()[1] {
            return Err(SynaptixError::Unsupported(
                "unpatchify: N != (H/p) * (W/p)",
            ));
        }
        if x.dims()[2] != p * p * c {
            return Err(SynaptixError::Unsupported(
                "unpatchify: token dim != P*P*C",
            ));
        }
        let reshaped = x.reshape(vec![b, nh, nw, c, p, p])?;
        let permuted = reshaped.permute(vec![0, 3, 1, 4, 2, 5])?.contiguous()?;
        permuted.reshape(vec![b, c, h, w])
    }
}
