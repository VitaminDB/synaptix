//! VAE FLUX.2 (`AutoencoderKLFlux2`): обычный diffusers-энкодер/декодер на
//! 32 латентных канала с `quant_conv`/`post_quant_conv`, плюс то, что делает
//! пайплайн: пакет 2×2 → 128 каналов и нормировка статистикой BatchNorm VAE
//! (`bn.running_mean/var`). Считается в F32 — в F16/BF16 декодер
//! переполняется.

use synaptix_core::{device::Device, dtype::DType, error::Result, tensor::Tensor};
use synaptix_nn::vae::{AutoencoderKlConfig, AutoencoderKlDecoder, AutoencoderKlEncoder};

use crate::config::Flux2VaeConfig;
use crate::memory;
use crate::source::Weights;

fn kl_config(c: &Flux2VaeConfig) -> AutoencoderKlConfig {
    AutoencoderKlConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: c.latent_channels,
        block_out_channels: c.block_out_channels.clone(),
        layers_per_block: c.layers_per_block,
        norm_num_groups: c.norm_num_groups,
        norm_eps: 1e-6,
        scaling_factor: 1.0,
        shift_factor: None,
        use_quant_conv: c.use_quant_conv && c.use_post_quant_conv,
    }
}

/// `(mean, std)` BatchNorm-статистики `[1, 4·latent, 1, 1]` в F32.
fn bn_stats(w: &Weights, c: &Flux2VaeConfig, dev: Device) -> Result<(Tensor, Tensor)> {
    let ch = 4 * c.latent_channels;
    let mean = w.get("bn.running_mean", dev, DType::F32)?.reshape((1, ch, 1, 1))?;
    let var = w.get("bn.running_var", dev, DType::F32)?.reshape((1, ch, 1, 1))?;
    let std = var.add_scalar(c.batch_norm_eps as f32)?.sqrt()?;
    Ok((mean, std))
}

/// `[B, C, H, W]` → `[B, 4C, H/2, W/2]` (порядок `permute(0,1,3,5,2,4)`).
pub fn patchify(x: &Tensor) -> Result<Tensor> {
    let d = x.dims();
    let (b, c, h, w) = (d[0], d[1], d[2], d[3]);
    x.reshape(vec![b, c, h / 2, 2, w / 2, 2])?
        .permute([0, 1, 3, 5, 2, 4])?
        .contiguous()?
        .reshape((b, c * 4, h / 2, w / 2))
}

/// Обратное к [`patchify`] (`permute(0,1,4,2,5,3)`).
pub fn unpatchify(x: &Tensor) -> Result<Tensor> {
    let d = x.dims();
    let (b, c4, h, w) = (d[0], d[1], d[2], d[3]);
    let c = c4 / 4;
    x.reshape(vec![b, c, 2, 2, h, w])?
        .permute([0, 1, 4, 2, 5, 3])?
        .contiguous()?
        .reshape((b, c, h * 2, w * 2))
}

/// Нормированный упакованный латент `[1, 128, h, w]` → картинка `[3, H, W]`
/// в [0, 1] на CPU.
pub fn decode(w: &Weights, c: &Flux2VaeConfig, dev: Device, latent: &Tensor) -> Result<Tensor> {
    let (mean, std) = bn_stats(w, c, dev)?;
    let lat = latent.to_device(dev)?.to_dtype(DType::F32)?.broadcast_mul(&std)?.broadcast_add(&mean)?;
    let z = unpatchify(&lat)?;
    drop(lat);
    let image = {
        let vae = {
            let _g = memory::weights_guard(dev);
            AutoencoderKlDecoder::load(&kl_config(c), &|n| w.get(n, dev, DType::F32))?
        };
        vae.decode(&z)?
    };
    drop(z);
    let image = image.affine(0.5, 0.5)?.clamp(0.0, 1.0)?;
    let d = image.dims().to_vec();
    let out = image.reshape(vec![d[1], d[2], d[3]])?.contiguous()?.to_device(Device::Cpu)?;
    drop(image);
    memory::release_pools(dev);
    Ok(out)
}

/// Картинка `[3, H, W]` в [0, 1] (стороны кратны 16) → нормированный
/// упакованный латент `[1, 128, H/16, W/16]` (мода распределения, как
/// `sample_mode="argmax"` у пайплайна).
pub fn encode(w: &Weights, c: &Flux2VaeConfig, dev: Device, image: &Tensor) -> Result<Tensor> {
    let d = image.dims().to_vec();
    let x = image.to_device(dev)?.to_dtype(DType::F32)?.reshape(vec![1, d[0], d[1], d[2]])?.affine(2.0, -1.0)?;
    let enc = {
        let _g = memory::weights_guard(dev);
        AutoencoderKlEncoder::load(&kl_config(c), &|n| w.get(n, dev, DType::F32))?
    };
    let moments = enc.encode(&x)?;
    drop(x);
    let (mean_z, _logvar) = enc.split_moments(&moments)?;
    drop(enc);
    let p = patchify(&mean_z.contiguous()?)?;
    let (mean, std) = bn_stats(w, c, dev)?;
    let out = p.broadcast_sub(&mean)?.broadcast_div(&std)?;
    drop((moments, mean_z, p));
    memory::release_pools(dev);
    Ok(out)
}
