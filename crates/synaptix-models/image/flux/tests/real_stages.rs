//! Локализация bf16-дрейфа на РЕАЛЬНЫХ входах (real T5, sigma=1.0): сравнить
//! мои стадии трансформера с Python-промежуточными (inter_real). Reference:
//! gen_flux_io.py (io + inter_real). 512² → packed 32×32, txt=512. feature cuda.


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
        dot += x*y; na += x*x; nb += y*y; mx = mx.max((x-y).abs()); mr = mr.max(y.abs());
    }
    (dot/(na.sqrt()*nb.sqrt()), mx, mr)
}

#[test]
fn flux_real_stages() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cuda(0);
    let model = std::env::var("FLUX_MODEL").unwrap_or_else(|_| "models/black-forest-labs/FLUX.1-dev".into());
    let dir = format!("{model}/transformer");
    let io_p = synaptix_test_utils::reference_data_path("flux_io", "io.safetensors");
    let inter_p = synaptix_test_utils::reference_data_path("flux_io", "inter_real.safetensors");
    if !Path::new(&dir).is_dir() || !io_p.exists() || !inter_p.exists() {
        eprintln!("SKIP real_stages: нет данных"); return;
    }
    let io = synaptix_test_utils::load_safetensors(&io_p);
    let inter = synaptix_test_utils::load_safetensors(&inter_p);
    let cdt = if std::env::var("FLUX_TEST_F32").is_ok() { DType::F32 } else { DType::BF16 };
    eprintln!("compute dtype = {cdt:?}");
    let to = |t: &Tensor| t.to_dtype(cdt).unwrap().to_device(dev).unwrap();

    let w = ComponentWeights::open_dir(&dir, Device::Cpu, cdt).unwrap();
    let model = FluxTransformer::load(&FluxConfig::dev(), &|n| w.get(n)).unwrap().into_streaming(dev).unwrap();

    let guidance = Tensor::from_vec(vec![3.5f32], (1,), dev).unwrap();
    let sigma = Tensor::from_vec(vec![1.0f32], (1,), dev).unwrap();
    let mut cap = Some(Vec::new());
    let _ = model.forward_cap(&to(&io["init_latent"]), &to(&io["t5_seq"]), &to(&io["pooled"]),
                              &sigma, &guidance, 32, 32, &mut cap).unwrap();
    let refkey = |n: &str| match n {
        "db0_img" => "db0_1", "db0_txt" => "db0_0", "sb0_img" => "sb0_1", "sb0_txt" => "sb0_0",
        "db0sub_nh" => "db0_norm1_0", "db0sub_ne" => "db0_norm1ctx_0",
        "db0sub_img_attn" => "db0_attn_0", "db0sub_ctx_attn" => "db0_attn_1",
        "db0sub_ff" => "db0_ff", "db0sub_attn_raw" => "",
        "depthD9_img" => "depthD9_1", "depthD18_img" => "depthD18_1",
        "depthS9_img" => "depthS9_1", "depthS18_img" => "depthS18_1", "depthS37_img" => "depthS37_1",
        _ => "",
    };
    for (name, t) in cap.unwrap() {
        let rk = if refkey(&name).is_empty() { name.clone() } else { refkey(&name).into() };
        let Some(r) = inter.get(&rk) else { continue };
        let (cos, mx, mr) = metrics(&t.to_device(Device::Cpu).unwrap(), r);
        eprintln!("  REAL stage {name:<10} cos={cos:.6} max_abs={mx:.4} max_ref={mr:.2} rel={:.4}", mx/mr.max(1e-9));
    }
}
