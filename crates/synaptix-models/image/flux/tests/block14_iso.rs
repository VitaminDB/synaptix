//! ИЗОЛЯЦИЯ CUDA bf16-бага на блоке 14 (massive-активации). Скармливаем ЧИСТЫЙ
//! Python-вход блока 14 (b14in_img/txt из inter_real) в МОЙ блок 14 на CUDA,
//! сравниваем КАЖДУЮ под-операцию с Python. Расхождение = чистая ошибка CUDA-
//! ядра блока 14 (унаследованная отсутствует — вход bit-exact Python). feature cuda.


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
fn flux_block14_iso() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cuda(0);
    let model_dir = std::env::var("FLUX_MODEL").unwrap_or_else(|_| "models/black-forest-labs/FLUX.1-dev".into());
    let dir = format!("{model_dir}/transformer");
    let inter_p = synaptix_test_utils::reference_data_path("flux_io", "inter_real.safetensors");
    if !Path::new(&dir).is_dir() || !inter_p.exists() {
        eprintln!("SKIP block14_iso: нет данных"); return;
    }
    let inter = synaptix_test_utils::load_safetensors(&inter_p);
    let to = |t: &Tensor| t.to_dtype(DType::BF16).unwrap().to_device(dev).unwrap();

    let runon = |d: Device| -> Vec<(String, Tensor)> {
        let toN = |t: &Tensor| t.to_dtype(DType::BF16).unwrap().to_device(d).unwrap();
        let w = ComponentWeights::open_dir(&dir, d, DType::BF16).unwrap();
        let model = FluxTransformer::load(&FluxConfig::dev(), &|n| w.get(n)).unwrap();
        model.dbg_double_block(14, &toN(&inter["b14in_img"]), &toN(&inter["b14in_txt"]), &toN(&inter["temb"]), 32, 32)
            .unwrap().into_iter().map(|(n, t)| (n, t.to_device(Device::Cpu).unwrap())).collect()
    };
    let _ = to; // dev helper used via runon
    eprintln!("b14 input img max={:.1} txt max={:.1}",
        inter["b14in_img"].to_vec1_flat_abs_max(), inter["b14in_txt"].to_vec1_flat_abs_max());

    eprintln!("--- CPU block14 ---");
    let cpu: std::collections::HashMap<String, Tensor> = runon(Device::Cpu).into_iter().collect();
    eprintln!("--- CUDA block14 ---");
    let cuda = runon(dev);

    let refkey = |n: &str| match n {
        "nh" => "b14_norm1_0", "ne" => "b14_norm1ctx_0",
        "img_attn" => "b14_attn_0", "ctx_attn" => "b14_attn_1",
        "ff" => "b14_ff", "out_img" => "b14_out_1", "out_txt" => "b14_out_0",
        _ => "",
    };
    eprintln!("=== block14 ISOLATED: CUDA-vs-Python | CPU-vs-Python | CUDA-vs-CPU ===");
    for (name, cu) in &cuda {
        let rk = refkey(name);
        let cp = cpu.get(name);
        let py = if rk.is_empty() { None } else { inter.get(rk) };
        let s_cupy = py.map(|r| metrics(cu, r));
        let s_cppy = py.zip(cp).map(|(r, c)| metrics(c, r));
        let s_cucp = cp.map(|c| metrics(cu, c));
        let f = |s: Option<(f64,f64,f64)>| s.map(|(c,mx,mr)| format!("cos={c:.6} rel={:.4}", mx/mr.max(1e-9))).unwrap_or_else(|| "-".into());
        eprintln!("  {name:<10} CUDA/PY[{}]  CPU/PY[{}]  CUDA/CPU[{}]", f(s_cupy), f(s_cppy), f(s_cucp));
    }
}

trait FlatAbsMax { fn to_vec1_flat_abs_max(&self) -> f64; }
impl FlatAbsMax for Tensor {
    fn to_vec1_flat_abs_max(&self) -> f64 {
        let n: usize = self.dims().iter().product();
        self.contiguous().unwrap().reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap()
            .to_vec1::<f32>().unwrap().iter().fold(0.0f64, |m, &v| m.max((v as f64).abs()))
    }
}
