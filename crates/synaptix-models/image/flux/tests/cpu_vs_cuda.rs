//! Бисекция CUDA bf16-бага: CPU bf16 = golden (velocity cos 0.999986 к Python),
//! CUDA bf16 расходится (0.955 → сетка/зерно). Гоняем ОДИН forward_cap на CPU и
//! на CUDA с теми же bf16-входами, сравниваем КАЖДУЮ под-операцию CPU-vs-CUDA.
//! Под-операция с макс. расхождением = багованное CUDA-ядро. feature cuda.


use std::collections::HashMap;
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

fn run(dev: Device, io: &HashMap<String, Tensor>, dir: &str) -> Vec<(String, Tensor)> {
    let to = |t: &Tensor| t.to_dtype(DType::BF16).unwrap().to_device(dev).unwrap();
    let w = ComponentWeights::open_dir(dir, dev, DType::BF16).unwrap();
    let model = FluxTransformer::load(&FluxConfig::dev(), &|n| w.get(n)).unwrap();
    let guidance = Tensor::from_vec(vec![3.5f32], (1,), dev).unwrap();
    let sigma = Tensor::from_vec(vec![1.0f32], (1,), dev).unwrap();
    let mut cap = Some(Vec::new());
    let vel = model.forward_cap(&to(&io["init_latent"]), &to(&io["t5_seq"]), &to(&io["pooled"]),
                                &sigma, &guidance, 32, 32, &mut cap).unwrap();
    let mut out = cap.unwrap();
    out.push(("velocity".into(), vel));
    out.into_iter().map(|(n, t)| (n, t.to_device(Device::Cpu).unwrap())).collect()
}

#[test]
fn flux_cpu_vs_cuda() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    let model = std::env::var("FLUX_MODEL").unwrap_or_else(|_| "models/black-forest-labs/FLUX.1-dev".into());
    let dir = format!("{model}/transformer");
    let io_p = synaptix_test_utils::reference_data_path("flux_io", "io.safetensors");
    if !Path::new(&dir).is_dir() || !io_p.exists() {
        eprintln!("SKIP cpu_vs_cuda: нет данных"); return;
    }
    let io = synaptix_test_utils::load_safetensors(&io_p);

    eprintln!("--- CPU forward ---");
    let cpu = run(Device::Cpu, &io, &dir);
    eprintln!("--- CUDA forward ---");
    let cuda: HashMap<String, Tensor> = run(Device::Cuda(0), &io, &dir).into_iter().collect();

    eprintln!("=== CPU-vs-CUDA per stage (CPU=golden) ===");
    for (name, c) in &cpu {
        if let Some(g) = cuda.get(name) {
            let (cos, mx, mr) = metrics(g, c);
            eprintln!("  {name:<18} cos={cos:.6} max_abs={mx:.4} max_ref={mr:.2} rel={:.4}", mx / mr.max(1e-9));
        }
    }
}
