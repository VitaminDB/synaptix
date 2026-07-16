//! Сверка velocity forward_cap (CUDA) с Python vel0 — тем же путём, что
//! cpu_vs_cuda. Резолвит противоречие CLI(0.954)-vs-test(CPU≡CUDA). feature cuda.

#![cfg(feature = "cuda")]

use std::path::Path;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_image_flux::loader::ComponentWeights;
use synaptix_image_flux::transformer::{FluxConfig, FluxTransformer};

fn metrics(a: &Tensor, b: &Tensor) -> (f64, f64, f64) {
    let n: usize = a.dims().iter().product();
    let av = a.contiguous().unwrap().reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let bv = b.contiguous().unwrap().reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let (mut dot, mut na, mut nb, mut mx, mut mr) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (x, y) in av.iter().zip(bv.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y; na += x * x; nb += y * y; mx = mx.max((x - y).abs()); mr = mr.max(y.abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), mx, mr)
}

#[test]
fn flux_vel_check() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cuda(0);
    let model_dir = std::env::var("FLUX_MODEL").unwrap_or_else(|_| "models/black-forest-labs/FLUX.1-dev".into());
    let dir = format!("{model_dir}/transformer");
    let io_p = synaptix_test_utils::reference_data_path("flux_io", "io.safetensors");
    if !Path::new(&dir).is_dir() || !io_p.exists() { eprintln!("SKIP"); return; }
    let io = synaptix_test_utils::load_safetensors(&io_p);
    let to = |t: &Tensor| t.to_dtype(DType::BF16).unwrap().to_device(dev).unwrap();

    let w = ComponentWeights::open_dir(&dir, dev, DType::BF16).unwrap();
    let model = FluxTransformer::load(&FluxConfig::dev(), &|n| w.get(n)).unwrap();
    let guidance = Tensor::from_vec(vec![3.5f32], (1,), dev).unwrap();
    let sigma = Tensor::from_vec(vec![1.0f32], (1,), dev).unwrap();

    // (A) входы через load_safetensors (как тест), grad ВКЛ
    let vel_a = model.forward(&to(&io["init_latent"]), &to(&io["t5_seq"]), &to(&io["pooled"]),
                              &sigma, &guidance, 32, 32).unwrap();
    let (cos, mx, mr) = metrics(&vel_a.to_device(Device::Cpu).unwrap(), &io["vel0"]);
    eprintln!("  [A grad-ON] vs vel0: cos={cos:.6} max_abs={mx:.4} max_ref={mr:.2}");

    // (NG) как pipeline: под NoGradGuard (fused linear)
    {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let vel_ng = model.forward(&to(&io["init_latent"]), &to(&io["t5_seq"]), &to(&io["pooled"]),
                                   &sigma, &guidance, 32, 32).unwrap();
        let (cos, mx, mr) = metrics(&vel_ng.to_device(Device::Cpu).unwrap(), &io["vel0"]);
        eprintln!("  [NG fused] vs vel0: cos={cos:.6} max_abs={mx:.4} max_ref={mr:.2}");
    }
    // (NGU) NoGrad + force-unfused linear (matmul-путь) — в ТОМ ЖЕ процессе
    {
        let _ng = synaptix_core::grad::NoGradGuard::new();
        synaptix_core::tensor::ops::set_force_unfused_linear(true);
        let vel_ngu = model.forward(&to(&io["init_latent"]), &to(&io["t5_seq"]), &to(&io["pooled"]),
                                    &sigma, &guidance, 32, 32).unwrap();
        synaptix_core::tensor::ops::set_force_unfused_linear(false);
        let (cos, mx, mr) = metrics(&vel_ngu.to_device(Device::Cpu).unwrap(), &io["vel0"]);
        eprintln!("  [NGU unfused] vs vel0: cos={cos:.6} max_abs={mx:.4} max_ref={mr:.2}");
    }

    // (B) входы через open_file(BF16) (как CLI pipeline)
    let cw = ComponentWeights::open_file(&io_p, dev, DType::BF16).unwrap();
    let lat = cw.get("init_latent").unwrap();
    let t5 = cw.get("t5_seq").unwrap();
    let pl = cw.get("pooled").unwrap();
    eprintln!("  open_file: init_latent dims={:?} dtype={:?}, t5 dims={:?}, pooled dims={:?}",
              lat.dims(), lat.dtype(), t5.dims(), pl.dims());
    let lat = lat.to_dtype(DType::F32).unwrap().to_dtype(DType::BF16).unwrap(); // как pipeline f32→bf16
    let vel_b = model.forward(&lat, &t5, &pl, &sigma, &guidance, 32, 32).unwrap();
    let (cos, mx, mr) = metrics(&vel_b.to_device(Device::Cpu).unwrap(), &io["vel0"]);
    eprintln!("  [B open_file inputs]        vs vel0: cos={cos:.6} max_abs={mx:.4} max_ref={mr:.2}");

    // diff входов A-vs-B
    let (c1, m1, _) = metrics(&to(&io["init_latent"]), &lat);
    let (c2, m2, _) = metrics(&to(&io["t5_seq"]), &t5);
    let (c3, m3, _) = metrics(&to(&io["pooled"]), &pl);
    eprintln!("  input diff A-vs-B: init_latent cos={c1:.6} mx={m1:.4} | t5 cos={c2:.6} mx={m2:.4} | pooled cos={c3:.6} mx={m3:.4}");
}
