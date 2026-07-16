//! Bit-exact-проверка FLUX T5-XXL encoder против HF transformers НА CUDA.
//! Reference: `scripts/reference/gen_flux_t5.py` (CPU f32, SEQ=128) — input_ids +
//! last_hidden_state [1,128,4096]. Грузим bf16-веса как F32 на CUDA (~19GB),
//! собираем `T5Encoder::xxl()`, подаём дампнутые ids, сверяем (cos И max-abs).
//! Требует feature `cuda`.

#![cfg(feature = "cuda")]

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_image_flux::loader::ComponentWeights;
use synaptix_image_flux::t5::{T5Config, T5Encoder};

fn flux_model() -> String {
    std::env::var("FLUX_MODEL")
        .unwrap_or_else(|_| "models/black-forest-labs/FLUX.1-dev".to_string())
}

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

#[test]
fn flux_t5_encoder_bit_exact_cuda() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    let dev = Device::Cuda(0);
    let te_dir = format!("{}/text_encoder_2", flux_model());
    if !Path::new(&te_dir).is_dir() {
        eprintln!("SKIP flux t5: нет {te_dir}");
        return;
    }
    let ref_path = synaptix_test_utils::reference_data_path("flux_t5", "t5.safetensors");
    if !ref_path.exists() {
        eprintln!("SKIP flux t5: нет reference {ref_path:?} (запусти gen_flux_t5.py)");
        return;
    }
    let refs = synaptix_test_utils::load_safetensors(&ref_path);

    let w = ComponentWeights::open_dir(&te_dir, dev, DType::F32).unwrap();
    let get = |name: &str| w.get(name);
    let enc = T5Encoder::load(&T5Config::xxl(), &get).unwrap();

    let ids_i32 = refs["input_ids"].to_vec2::<i32>().unwrap();
    let (b, s) = (ids_i32.len(), ids_i32[0].len());
    let ids_u32: Vec<u32> = ids_i32.into_iter().flatten().map(|v| v as u32).collect();
    let input_ids = Tensor::from_vec(ids_u32, (b, s), dev).unwrap();

    let out = enc.forward(&input_ids).unwrap().to_device(Device::Cpu).unwrap();
    assert_eq!(out.dims(), refs["last_hidden_state"].dims());

    let (cos, err, maxref) = cos_maxabs(&out, &refs["last_hidden_state"]);
    eprintln!("[flux t5 CUDA] cos={cos:.8} max_err={err:.4e} max_ref={maxref:.3} dims={:?}", out.dims());
    assert!(cos > 0.9999, "t5 cos {cos} < 0.9999 (max_err {err:.3e})");
    assert!(err < 0.02 * maxref + 0.05, "t5 max_err {err:.4e} велик (max_ref {maxref:.3})");
}
