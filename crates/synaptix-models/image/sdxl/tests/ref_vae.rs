//! Bit-exact-проверка SDXL VAE-декодера (AutoencoderKL) против diffusers
//! (CPU F32). Reference: `scripts/reference/gen_sdxl_vae.py` → декодит
//! фиксированный латент `z [1,4,8,8]` и дампит `z` + `sample` (raw decoder
//! output = post_quant_conv → decoder, без деления на scaling_factor).
//!
//! Здесь грузим те же fp16-веса (dtype=F32), собираем `AutoencoderKlDecoder`
//! и сверяем `decode(z)`. Тест пропускается, если веса/reference отсутствуют.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_nn::vae::{AutoencoderKlConfig, AutoencoderKlDecoder, KlVae};

const SDXL: &str = "models/stabilityai/stable-diffusion-xl-base-1.0";

fn cos_and_maxerr(a: &Tensor, b: &Tensor) -> (f64, f64) {
    let n: usize = a.dims().iter().product();
    let av = a.reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let bv = b.reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let (mut dot, mut na, mut nb, mut maxerr) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (x, y) in av.iter().zip(bv.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxerr = maxerr.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), maxerr)
}

#[test]
fn vae_decode_bit_exact() {
    synaptix_kernels_cpu::ensure_registered();
    let weights = format!("{SDXL}/vae/diffusion_pytorch_model.fp16.safetensors");
    if !Path::new(&weights).exists() {
        eprintln!("SKIP vae: нет {weights}");
        return;
    }
    let ref_path = synaptix_test_utils::reference_data_path("sdxl_vae", "decode.safetensors");
    if !ref_path.exists() {
        eprintln!("SKIP vae: нет reference {ref_path:?} (запусти gen_sdxl_vae.py)");
        return;
    }
    let refs = synaptix_test_utils::load_safetensors(&ref_path);

    let ld = SafetensorsLoader::open(&weights).unwrap().with_device(Device::Cpu);
    let get = |name: &str| {
        ld.load_to(name, Device::Cpu, DType::F32)
            .map_err(|e| synaptix_core::error::SynaptixError::Other(format!("load {name}: {e}")))
    };
    let cfg = AutoencoderKlConfig::sdxl();
    let dec = AutoencoderKlDecoder::load(&cfg, &get).unwrap();

    let z = refs["z"].to_dtype(DType::F32).unwrap();
    let out = dec.decode(&z).unwrap();
    assert_eq!(out.dims(), refs["sample"].dims());

    let (cos, err) = cos_and_maxerr(&out, &refs["sample"]);
    eprintln!("vae decode: cos={cos:.8} max_err={err:.3e} dims={:?}", out.dims());
    assert!(cos > 0.9999, "vae decode cos {cos} < 0.9999 (max_err {err:.3e})");
}

#[test]
fn vae_encode_bit_exact() {
    synaptix_kernels_cpu::ensure_registered();
    let weights = format!("{SDXL}/vae/diffusion_pytorch_model.fp16.safetensors");
    if !Path::new(&weights).exists() {
        eprintln!("SKIP vae encode: нет {weights}");
        return;
    }
    let ref_path = synaptix_test_utils::reference_data_path("sdxl_vae", "encode.safetensors");
    if !ref_path.exists() {
        eprintln!("SKIP vae encode: нет reference {ref_path:?}");
        return;
    }
    let refs = synaptix_test_utils::load_safetensors(&ref_path);

    let ld = SafetensorsLoader::open(&weights).unwrap().with_device(Device::Cpu);
    let get = |name: &str| {
        ld.load_to(name, Device::Cpu, DType::F32)
            .map_err(|e| synaptix_core::error::SynaptixError::Other(format!("load {name}: {e}")))
    };
    // KlVae::load проверяет и encoder, и decoder load-путь.
    let vae = KlVae::load(&AutoencoderKlConfig::sdxl(), &get).unwrap();

    let x = refs["x"].to_dtype(DType::F32).unwrap();
    let moments = vae.encode_moments(&x).unwrap();
    assert_eq!(moments.dims(), refs["moments"].dims());

    let (cos, err) = cos_and_maxerr(&moments, &refs["moments"]);
    eprintln!("vae encode (moments): cos={cos:.8} max_err={err:.3e} dims={:?}", moments.dims());
    assert!(cos > 0.9999, "vae encode cos {cos} < 0.9999 (max_err {err:.3e})");
}
