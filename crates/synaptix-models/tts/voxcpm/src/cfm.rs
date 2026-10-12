use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::locdit::LocDit;
use crate::VoxError;

/// Граница графа на NPU: каждый шаг CFM — один и тот же граф.
fn flush(device: Device) -> Result<(), VoxError> {
    Ok(synaptix_core::backend::registry::flush(device)?)
}

#[derive(Debug, Clone)]
pub struct CfmOptions {
    pub n_timesteps: usize,
    pub cfg_value: f32,
    pub temperature: f32,
    pub sway_coef: f32,
}

impl Default for CfmOptions {
    fn default() -> Self {
        Self { n_timesteps: 10, cfg_value: 2.0, temperature: 1.0, sway_coef: 1.0 }
    }
}

fn t_span(n: usize, sway_coef: f32) -> Vec<f32> {
    let mut span: Vec<f32> = (0..=n).map(|i| 1.0 - (i as f32) / (n as f32)).collect();
    for t in span.iter_mut() {
        let v = *t;
        *t = v + sway_coef * ((std::f32::consts::FRAC_PI_2 * v).cos() - 1.0 + v);
    }
    span
}

pub fn sample(
    dit: &LocDit,
    mu: &Tensor,
    cond: &Tensor,
    patch_size: usize,
    feat_dim: usize,
    opts: &CfmOptions,
    device: Device,
    compute: DType,
    seed: u64,
) -> Result<Tensor, VoxError> {
    let b = mu.dims()[0];
    let z = Tensor::randn_seeded((b, feat_dim, patch_size), seed, Device::Cpu)?
        .to_device(device)?
        .to_dtype(compute)?
        .mul_scalar(opts.temperature)?;

    let span = t_span(opts.n_timesteps, opts.sway_coef);
    let len = span.len();
    let zero_init_steps = ((len as f32) * 0.04) as usize;
    let zero_init_steps = zero_init_steps.max(1);

    let mu_zero = Tensor::zeros(mu.dims().to_vec(), compute, device)?;
    let mu_in = Tensor::cat(&[mu, &mu_zero], 0)?;
    let cond_in = Tensor::cat(&[cond, cond], 0)?;
    let dt_in = Tensor::zeros(vec![2 * b], DType::F32, device)?;

    let mut x = z;
    flush(device)?;
    for s in 1..len {
        let tval = span[s - 1];
        let dt = span[s - 1] - span[s];
        if s <= zero_init_steps {
            continue;
        }
        let x_in = Tensor::cat(&[&x, &x], 0)?;
        let t_in = Tensor::from_vec(vec![tval; 2 * b], 2 * b, device)?;
        let out = dit.forward(&x_in, &mu_in, &t_in, &cond_in, &dt_in)?;
        let cond_v = out.narrow(0, 0, b)?.contiguous()?;
        let uncond_v = out.narrow(0, b, b)?.contiguous()?;

        let u_s = if matches!(device, Device::Npu(_)) {
            // На NPU коэффициент считается на устройстве: чтение скаляра на
            // хост — сброс графа на каждом из шагов CFM.
            let (c32, u32_) = (cond_v.to_dtype(DType::F32)?, uncond_v.to_dtype(DType::F32)?);
            let num = c32.mul(&u32_)?.sum_all()?;
            let den = u32_.sqr()?.sum_all()?.add_scalar(1e-8)?;
            let st = num.div(&den)?.reshape((1usize, 1usize, 1usize))?;
            u32_.broadcast_mul(&st)?.to_dtype(compute)?
        } else {
            let num = cond_v.mul(&uncond_v)?.sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
            let den = uncond_v.sqr()?.sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()? + 1e-8;
            uncond_v.mul_scalar(num / den)?
        };
        let guided = u_s.add(&cond_v.sub(&u_s)?.mul_scalar(opts.cfg_value)?)?;
        x = x.sub(&guided.mul_scalar(dt)?)?;
        flush(device)?;
    }

    Ok(x.transpose(1, 2)?.contiguous()?)
}
