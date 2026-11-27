//! Превью по ходу денойза: линейная проекция латента (128 каналов) в RGB,
//! коэффициенты `[3 × 129]` подобраны по парам «латент — кадр» (пример
//! `latent_rgb_fit` в synaptix-gen). Картинка — в разрешении латента (1/32
//! кадра): видно, как проявляется сцена, без декодера.

use std::sync::OnceLock;

use synaptix_core::gen_status;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

const C: usize = 128;

/// Коэффициенты: `SYN_LTX_RGB_FACTORS` (файл f32 LE) или встроенные.
fn factors() -> Option<&'static Vec<f32>> {
    static F: OnceLock<Option<Vec<f32>>> = OnceLock::new();
    F.get_or_init(|| match std::env::var_os("SYN_LTX_RGB_FACTORS") {
        Some(p) => gen_status::parse_factors(&std::fs::read(p).ok()?, C),
        None => gen_status::parse_factors(include_bytes!("../assets/latent_rgb.f32"), C),
    })
    .as_ref()
}

/// Превью среднего кадра по оценке `x0 = x − σ·v`. `x`, `v` — токены
/// `[1, T, 128]` (кадр — `hp·wp` токенов подряд), `fp` — латентных кадров.
#[allow(clippy::too_many_arguments)]
pub fn emit(x: &Tensor, v: &Tensor, sigma: f64, fp: usize, hp: usize, wp: usize, step: usize, total: usize) {
    let (Some(f), n) = (factors(), gen_status::preview_every(2)) else { return };
    if n == 0 || step % n != 0 {
        return;
    }
    let _ = (|| -> Result<(), synaptix_core::error::SynaptixError> {
        let hw = hp * wp;
        let off = (fp / 2) * hw;
        let xs = x.narrow(1, off, hw)?.contiguous()?.to_dtype(DType::F32)?;
        let vs = v.narrow(1, off, hw)?.contiguous()?.to_dtype(DType::F32)?;
        let x0 = xs.sub(&vs.mul_scalar(sigma as f32)?)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        gen_status::linear_preview(&x0, C, wp, hp, f, step, total);
        Ok(())
    })();
}
