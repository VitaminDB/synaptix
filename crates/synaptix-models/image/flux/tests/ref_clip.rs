//! Bit-exact-проверка FLUX CLIP-L (pooled) против HF transformers НА CUDA.
//! FLUX берёт из CLIP-L только `pooler_output [B,768]` = last_hidden_state
//! (после final_layer_norm eps=1e-5) в позиции argmax(input_ids) (первый EOS).
//! Reference: `scripts/reference/gen_flux_clip.py` (CPU f32) — input_ids + pooled
//! + last_hidden_state. Грузим bf16-веса как F32 на CUDA, реюзаем
//! `ClipTextEncoder::clip_l()` (совпадает с FLUX CLIP-L) и сверяем на тех же ids.
//! Требует feature `cuda`.

#![cfg(feature = "cuda")]

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_image_flux::loader::ComponentWeights;
use synaptix_nn::text::{ClipTextConfig, ClipTextEncoder};

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
fn flux_clip_l_pooled_bit_exact_cuda() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    let dev = Device::Cuda(0);
    let te_dir = format!("{}/text_encoder", flux_model());
    if !Path::new(&te_dir).is_dir() {
        eprintln!("SKIP flux clip: нет {te_dir}");
        return;
    }
    let ref_path = synaptix_test_utils::reference_data_path("flux_clip", "clip_l.safetensors");
    if !ref_path.exists() {
        eprintln!("SKIP flux clip: нет reference {ref_path:?} (запусти gen_flux_clip.py)");
        return;
    }
    let refs = synaptix_test_utils::load_safetensors(&ref_path);

    let w = ComponentWeights::open_dir(&te_dir, dev, DType::F32).unwrap();
    let get = |name: &str| w.get(name);
    let enc = ClipTextEncoder::load(&ClipTextConfig::clip_l(), "text_model", &get).unwrap();

    let ids_i32 = refs["input_ids"].to_vec2::<i32>().unwrap();
    let (b, s) = (ids_i32.len(), ids_i32[0].len());
    let ids_u32: Vec<u32> = ids_i32.into_iter().flatten().map(|v| v as u32).collect();
    let input_ids = Tensor::from_vec(ids_u32, (b, s), dev).unwrap();

    let out = enc.forward(&input_ids).unwrap();
    let pooled = out.pooled_output.to_device(Device::Cpu).unwrap();
    let last = out.last_hidden_state.to_device(Device::Cpu).unwrap();
    assert_eq!(pooled.dims(), refs["pooled"].dims());

    let (cos_l, err_l, mr_l) = cos_maxabs(&last, &refs["last_hidden_state"]);
    let (cos_p, err_p, mr_p) = cos_maxabs(&pooled, &refs["pooled"]);
    eprintln!("[flux clip CUDA] last cos={cos_l:.8} max_err={err_l:.4e} | pooled cos={cos_p:.8} max_err={err_p:.4e} max_ref={mr_p:.3}");

    assert!(cos_l > 0.9999, "last_hidden cos {cos_l} < 0.9999 (max_err {err_l:.3e})");
    assert!(err_l < 0.01 * mr_l + 0.02, "last_hidden max_err {err_l:.4e} велик (max_ref {mr_l:.3})");
    assert!(cos_p > 0.9999, "pooled cos {cos_p} < 0.9999 (max_err {err_p:.3e})");
    assert!(err_p < 0.01 * mr_p + 0.02, "pooled max_err {err_p:.4e} велик (max_ref {mr_p:.3})");
}
