//! Bit-exact-проверка FLUX VAE-декодера (AutoencoderKL 16-ch) против diffusers
//! НА CUDA. Reference: `scripts/reference/gen_flux_vae.py` (CPU f32) →
//!   - decode: `z[1,16,16,16]` + `sample[1,3,128,128]` (raw decoder, без scale/shift)
//!   - pipe_decode: `z_pipe` + `image` (полный путь pipeline: z/scaling + shift → decode)
//!
//! Грузим bf16-веса как F32 на CUDA, собираем `AutoencoderKlDecoder::flux()`
//! и сверяем. Гейт — cos И per-element max-abs (не только cos). Тест
//! пропускается, если веса/reference отсутствуют. Требует feature `cuda`.


use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_image_flux::loader::ComponentWeights;
use synaptix_nn::vae::{AutoencoderKlConfig, AutoencoderKlDecoder};

fn flux_model() -> String {
    std::env::var("FLUX_MODEL")
        .unwrap_or_else(|_| "models/black-forest-labs/FLUX.1-dev".to_string())
}

/// cos + max-abs-error + диапазон reference.
fn cos_maxabs(a: &Tensor, b: &Tensor) -> (f64, f64, f64) {
    let n: usize = a.dims().iter().product();
    let av = a.reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let bv = b.reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let (mut dot, mut na, mut nb, mut maxerr, mut maxref) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (x, y) in av.iter().zip(bv.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxerr = maxerr.max((x - y).abs());
        maxref = maxref.max(y.abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), maxerr, maxref)
}

fn load_decoder() -> Option<(AutoencoderKlDecoder, Device)> {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    let dev = Device::Cuda(0);
    let vae_dir = format!("{}/vae", flux_model());
    if !Path::new(&vae_dir).is_dir() {
        eprintln!("SKIP flux vae: нет {vae_dir}");
        return None;
    }
    let w = ComponentWeights::open_dir(&vae_dir, dev, DType::F32).ok()?;
    let get = |name: &str| w.get(name);
    let dec = AutoencoderKlDecoder::load(&AutoencoderKlConfig::flux(), &get).unwrap();
    Some((dec, dev))
}

#[test]
fn flux_vae_decode_bit_exact_cuda() {
    let Some((dec, dev)) = load_decoder() else { return };
    let ref_path = synaptix_test_utils::reference_data_path("flux_vae", "decode.safetensors");
    if !ref_path.exists() {
        eprintln!("SKIP flux vae decode: нет reference {ref_path:?} (запусти gen_flux_vae.py)");
        return;
    }
    let refs = synaptix_test_utils::load_safetensors(&ref_path);
    let z = refs["z"].to_dtype(DType::F32).unwrap().to_device(dev).unwrap();
    let out = dec.decode(&z).unwrap().to_device(Device::Cpu).unwrap();
    assert_eq!(out.dims(), refs["sample"].dims());

    let (cos, err, maxref) = cos_maxabs(&out, &refs["sample"]);
    eprintln!("[flux vae decode CUDA] cos={cos:.8} max_err={err:.4e} max_ref={maxref:.3} dims={:?}", out.dims());
    assert!(cos > 0.9999, "cos {cos} < 0.9999 (max_err {err:.3e})");
    assert!(err < 0.02 * maxref + 0.02, "max_err {err:.4e} слишком велик (max_ref {maxref:.3})");
}

#[test]
fn flux_vae_pipe_decode_bit_exact_cuda() {
    let Some((dec, dev)) = load_decoder() else { return };
    let ref_path = synaptix_test_utils::reference_data_path("flux_vae", "pipe_decode.safetensors");
    if !ref_path.exists() {
        eprintln!("SKIP flux vae pipe_decode: нет reference {ref_path:?}");
        return;
    }
    let refs = synaptix_test_utils::load_safetensors(&ref_path);
    let cfg = AutoencoderKlConfig::flux();
    let z_pipe = refs["z_pipe"].to_dtype(DType::F32).unwrap().to_device(dev).unwrap();
    // pipeline-путь: latents = z/scaling + shift, затем decode.
    let lat = z_pipe
        .mul_scalar(1.0 / cfg.scaling_factor)
        .unwrap()
        .add_scalar(cfg.shift_factor.unwrap_or(0.0))
        .unwrap();
    let out = dec.decode(&lat).unwrap().to_device(Device::Cpu).unwrap();
    assert_eq!(out.dims(), refs["image"].dims());

    let (cos, err, maxref) = cos_maxabs(&out, &refs["image"]);
    eprintln!("[flux vae pipe_decode CUDA] cos={cos:.8} max_err={err:.4e} max_ref={maxref:.3}");
    assert!(cos > 0.9999, "cos {cos} < 0.9999 (max_err {err:.3e})");
    assert!(err < 0.02 * maxref + 0.02, "max_err {err:.4e} слишком велик (max_ref {maxref:.3})");
}
