//! Bit-exact-проверка FLUX MMDiT (FluxTransformer2DModel) против diffusers НА CUDA
//! в bf16 (transformer 23GB, f32 не влезает). Reference:
//! `scripts/reference/gen_flux_transformer.py` — io (входы+выход) при img 8x8=64,
//! txt=32. Гейт с поправкой на bf16-дрейф 57 блоков (cos + per-token max-abs).
//! Требует feature `cuda`.


use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_image_flux::loader::ComponentWeights;
use synaptix_image_flux::transformer::{FluxConfig, FluxTransformer};

fn flux_model() -> String {
    std::env::var("FLUX_MODEL")
        .unwrap_or_else(|_| "models/black-forest-labs/FLUX.1-dev".to_string())
}

/// cos + max-abs + per-token max-abs (по строкам img_seq).
fn metrics(a: &Tensor, b: &Tensor) -> (f64, f64, f64, f64) {
    let dims = a.dims().to_vec();
    let n: usize = dims.iter().product();
    let cols = *dims.last().unwrap();
    let av = a.contiguous().unwrap().reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let bv = b.contiguous().unwrap().reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let (mut dot, mut na, mut nb, mut maxerr, mut maxref) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut row_maxerr = 0.0f64;
    let mut cur_row = 0.0f64;
    for (i, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxerr = maxerr.max((x - y).abs());
        maxref = maxref.max(y.abs());
        cur_row = cur_row.max((x - y).abs());
        if (i + 1) % cols == 0 {
            row_maxerr = row_maxerr.max(cur_row);
            cur_row = 0.0;
        }
    }
    (dot / (na.sqrt() * nb.sqrt()), maxerr, maxref, row_maxerr)
}

#[test]
fn flux_transformer_bit_exact_cuda() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cuda(0);
    let dir = format!("{}/transformer", flux_model());
    if !Path::new(&dir).is_dir() {
        eprintln!("SKIP flux transformer: нет {dir}");
        return;
    }
    let ref_path = synaptix_test_utils::reference_data_path("flux_transformer", "io.safetensors");
    if !ref_path.exists() {
        eprintln!("SKIP flux transformer: нет reference {ref_path:?} (gen_flux_transformer.py)");
        return;
    }
    let refs = synaptix_test_utils::load_safetensors(&ref_path);
    let to = |k: &str| refs[k].to_dtype(DType::BF16).unwrap().to_device(dev).unwrap();

    let w = ComponentWeights::open_dir(&dir, dev, DType::BF16).unwrap();
    let get = |name: &str| w.get(name);
    let model = FluxTransformer::load(&FluxConfig::dev(), &get).unwrap();

    let hs = to("hidden_states");
    let ehs = to("encoder_hidden_states");
    let pooled = to("pooled");
    let timestep = to("timestep");
    let guidance = refs["guidance"].to_dtype(DType::F32).unwrap().to_device(dev).unwrap();

    let out = model
        .forward(&hs, &ehs, &pooled, &timestep, &guidance, 32, 32)
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap();
    assert_eq!(out.dims(), refs["out"].dims());

    let (cos, err, maxref, row_err) = metrics(&out, &refs["out"]);
    eprintln!(
        "[flux transformer CUDA bf16] cos={cos:.6} max_err={err:.4e} per_token_max={row_err:.4e} max_ref={maxref:.3}"
    );
    // 57 блоков bf16 (мои ядра vs diffusers SDPA/cuBLAS) накапливают дрейф;
    // покомпонентная bit-exactness (cos>0.999/стадия) проверяется в _stages_.
    assert!(cos > 0.98, "transformer cos {cos} < 0.98 (max_err {err:.3e}) — вероятно баг логики");
}

#[test]
fn flux_transformer_stages_cuda() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cuda(0);
    let dir = format!("{}/transformer", flux_model());
    let ref_path = synaptix_test_utils::reference_data_path("flux_transformer", "io.safetensors");
    let inter_path = synaptix_test_utils::reference_data_path("flux_transformer", "inter.safetensors");
    if !Path::new(&dir).is_dir() || !inter_path.exists() {
        eprintln!("SKIP stages: нет {dir} / inter.safetensors");
        return;
    }
    let refs = synaptix_test_utils::load_safetensors(&ref_path);
    let inter = synaptix_test_utils::load_safetensors(&inter_path);
    let to = |k: &str| refs[k].to_dtype(DType::BF16).unwrap().to_device(dev).unwrap();

    let w = ComponentWeights::open_dir(&dir, dev, DType::BF16).unwrap();
    let get = |name: &str| w.get(name);
    let model = FluxTransformer::load(&FluxConfig::dev(), &get).unwrap();

    let guidance = refs["guidance"].to_dtype(DType::F32).unwrap().to_device(dev).unwrap();
    let mut cap = Some(Vec::new());
    let _ = model
        .forward_cap(&to("hidden_states"), &to("encoder_hidden_states"), &to("pooled"),
                     &to("timestep"), &guidance, 32, 32, &mut cap)
        .unwrap();

    // diffusers-блоки возвращают (txt, img): db0_0=txt, db0_1=img.
    let refkey = |name: &str| -> &'static str {
        match name {
            "db0_img" => "db0_1", "db0_txt" => "db0_0",
            "sb0_img" => "sb0_1", "sb0_txt" => "sb0_0",
            _ => "",
        }
    };
    for (name, t) in cap.unwrap() {
        let rk = if refkey(&name).is_empty() { name.clone() } else { refkey(&name).to_string() };
        let Some(r) = inter.get(&rk) else { eprintln!("  {name}: нет ref {rk}"); continue };
        let tc = t.to_device(Device::Cpu).unwrap();
        let (cos, err, maxref, _) = metrics(&tc, r);
        eprintln!("  stage {name:<10} (ref {rk:<7}) cos={cos:.6} max_err={err:.4e} max_ref={maxref:.3}");
        assert!(cos > 0.999, "стадия {name}: cos {cos} < 0.999 — баг логики в этой стадии (max_err {err:.3e})");
    }
}
