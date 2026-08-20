//! LTX-2.3 spatial upscaler ×2 (LatentUpsampler) — отдельный чекпойнт
//! `ltx-2.3-spatial-upscaler-x2`. Апскейл VAE-латентов ×2 по пространству в
//! латентном пространстве (для two-stage HQ). GroupNorm (не PixelNorm), plain
//! Conv3d (zero-pad), per-frame Conv2d + PixelShuffle2.
//!
//! Конфиг: in 128, mid 1024, 4 блока/стадию, spatial ×2.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};
use synaptix_io::weights::WeightLoader;
use synaptix_ops::conv::conv2d::conv2d;
use synaptix_ops::conv::conv3d::conv3d;
use synaptix_ops::norm::group_norm::group_norm;

use crate::LtxError;

type R<T> = Result<T, SynaptixError>;
const GN_EPS: f32 = 1e-5;

fn conv3d_p1(x: &Tensor, w: &Tensor, b: &Tensor) -> R<Tensor> {
    conv3d(x, w, Some(b), (1, 1, 1), (1, 1, 1), (1, 1, 1))
}
fn gn(x: &Tensor, w: &Tensor, b: &Tensor) -> R<Tensor> {
    group_norm(x, Some(w), Some(b), 32, GN_EPS)
}

/// PixelShuffle2 `b (c p1 p2) h w -> b c (h p1)(w p2)`, p1=p2=2.
fn pixel_shuffle2(x: &Tensor) -> R<Tensor> {
    let (n, cc, h, w) = (x.dims()[0], x.dims()[1], x.dims()[2], x.dims()[3]);
    let c = cc / 4;
    x.reshape(vec![n, c, 2, 2, h, w])?
        .permute(vec![0, 1, 4, 2, 5, 3])?
        .contiguous()?
        .reshape(vec![n, c, h * 2, w * 2])
}

struct Conv3 {
    w: Tensor,
    b: Tensor,
}
struct Norm {
    w: Tensor,
    b: Tensor,
}
/// ResBlock: conv1→GN→silu→conv2→GN→silu(x+residual). GroupNorm, plain Conv3d.
struct ResBlock {
    conv1: Conv3,
    norm1: Norm,
    conv2: Conv3,
    norm2: Norm,
}
impl ResBlock {
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let h = gn(&conv3d_p1(x, &self.conv1.w, &self.conv1.b)?, &self.norm1.w, &self.norm1.b)?.silu()?;
        let h = gn(&conv3d_p1(&h, &self.conv2.w, &self.conv2.b)?, &self.norm2.w, &self.norm2.b)?;
        x.add(&h)?.silu()
    }
}

pub struct Upsampler {
    initial_conv: Conv3,
    initial_norm: Norm,
    res_blocks: Vec<ResBlock>,
    up_w: Tensor, // Conv2d [4096,1024,3,3]
    up_b: Tensor,
    post_blocks: Vec<ResBlock>,
    final_conv: Conv3,
    // VAE per-channel statistics (для un/normalize латента)
    mean: Tensor, // [1,128,1,1,1]
    std: Tensor,
    device: Device,
}

impl Upsampler {
    /// `path` — чекпойнт апскейлера; `vae_mean`/`vae_std` — статистики VAE
    /// (`vae.per_channel_statistics.*` из основного чекпойнта) `[128]`.
    pub fn load(
        path: impl AsRef<Path>,
        vae_mean: &Tensor,
        vae_std: &Tensor,
        device: Device,
    ) -> Result<Self, LtxError> {
        // `.syn`-бандл или сырой safetensors — как у чекпойнта/LoRA.
        let ld = crate::loader::open_weights(path.as_ref())?.with_device(device);
        let g = |n: &str| -> Result<Tensor, LtxError> {
            ld.load_to(n, device, DType::F32).map_err(|e| LtxError::Load(format!("{n}: {e}")))
        };
        let conv3 = |p: &str| -> Result<Conv3, LtxError> {
            Ok(Conv3 { w: g(&format!("{p}.weight"))?, b: g(&format!("{p}.bias"))? })
        };
        let norm = |p: &str| -> Result<Norm, LtxError> {
            Ok(Norm { w: g(&format!("{p}.weight"))?, b: g(&format!("{p}.bias"))? })
        };
        let resblock = |p: &str| -> Result<ResBlock, LtxError> {
            Ok(ResBlock {
                conv1: conv3(&format!("{p}.conv1"))?,
                norm1: norm(&format!("{p}.norm1"))?,
                conv2: conv3(&format!("{p}.conv2"))?,
                norm2: norm(&format!("{p}.norm2"))?,
            })
        };
        let mut res_blocks = Vec::new();
        let mut post_blocks = Vec::new();
        for i in 0..4 {
            res_blocks.push(resblock(&format!("res_blocks.{i}"))?);
            post_blocks.push(resblock(&format!("post_upsample_res_blocks.{i}"))?);
        }
        Ok(Self {
            initial_conv: conv3("initial_conv")?,
            initial_norm: norm("initial_norm")?,
            res_blocks,
            up_w: g("upsampler.0.weight")?,
            up_b: g("upsampler.0.bias")?,
            post_blocks,
            final_conv: conv3("final_conv")?,
            mean: vae_mean.to_device(device)?.to_dtype(DType::F32)?.reshape(vec![1, 128, 1, 1, 1])?,
            std: vae_std.to_device(device)?.to_dtype(DType::F32)?.reshape(vec![1, 128, 1, 1, 1])?,
            device,
        })
    }

    /// Сеть LatentUpsampler (без un/normalize). `latent` `[B,128,F,H,W]` →
    /// `[B,128,F,H·2,W·2]`.
    fn forward_net(&self, latent: &Tensor) -> R<Tensor> {
        let (b, f) = (latent.dims()[0], latent.dims()[2]);
        let mut x = gn(
            &conv3d_p1(latent, &self.initial_conv.w, &self.initial_conv.b)?,
            &self.initial_norm.w, &self.initial_norm.b,
        )?.silu()?;
        for blk in &self.res_blocks {
            x = blk.forward(&x)?;
        }
        // per-frame upsampler: [B,C,F,H,W] -> [B*F,C,H,W] -> conv2d -> pixelshuffle2 -> back
        let (c, h, w) = (x.dims()[1], x.dims()[3], x.dims()[4]);
        let x2 = x.permute(vec![0, 2, 1, 3, 4])?.contiguous()?.reshape(vec![b * f, c, h, w])?;
        let x2 = conv2d(&x2, &self.up_w, Some(&self.up_b), (1, 1), (1, 1), (1, 1))?;
        let x2 = pixel_shuffle2(&x2)?; // [B*F, 1024, H*2, W*2]
        let (h2, w2) = (h * 2, w * 2);
        x = x2.reshape(vec![b, f, c, h2, w2])?.permute(vec![0, 2, 1, 3, 4])?.contiguous()?;
        for blk in &self.post_blocks {
            x = blk.forward(&x)?;
        }
        conv3d_p1(&x, &self.final_conv.w, &self.final_conv.b)
    }

    /// `upsample_video`: un_normalize (VAE stats) → сеть → normalize. Латент
    /// `[B,128,F,H,W]` (нормализованный) → `[B,128,F,H·2,W·2]` (нормализованный).
    pub fn upsample(&self, latent: &Tensor) -> Result<Tensor, LtxError> {
        let x = latent.to_device(self.device)?.to_dtype(DType::F32)?;
        let raw = x.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?; // un_normalize
        let up = self.forward_net(&raw)?;
        // normalize: (up - mean)/std
        Ok(up.broadcast_sub(&self.mean)?.broadcast_div(&self.std)?)
    }
}
