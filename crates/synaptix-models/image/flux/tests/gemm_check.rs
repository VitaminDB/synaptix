//! Проверка нашего bf16 GEMM (Linear, gemm_bf16) против torch на FLUX-формах
//! ПОЭЛЕМЕНТНО (max-abs, не cos — cos прячет per-row баги). Reference:
//! `scripts/reference/gen_gemm_ref.py`. Требует feature `cuda`.


use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_nn::linear::Linear;
use synaptix_nn::module::Module;

fn metrics(a: &Tensor, b: &Tensor) -> (f64, f64, f64) {
    let n: usize = a.dims().iter().product();
    let av = a.contiguous().unwrap().reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let bv = b.contiguous().unwrap().reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let (mut dot, mut na, mut nb, mut mx, mut mr) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (x, y) in av.iter().zip(bv.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y; na += x * x; nb += y * y;
        mx = mx.max((x - y).abs()); mr = mr.max(y.abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), mx, mr)
}

#[test]
fn gemm_bf16_vs_torch() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cuda(0);
    let ref_path = synaptix_test_utils::reference_data_path("gemm_bf16", "ref.safetensors");
    if !ref_path.exists() {
        eprintln!("SKIP gemm: нет {ref_path:?} (gen_gemm_ref.py)");
        return;
    }
    let refs = synaptix_test_utils::load_safetensors(&ref_path);
    let shapes = [
        (1024, 3072, 64), (1024, 64, 3072), (512, 3072, 4096),
        (1536, 12288, 3072), (1536, 3072, 12288), (1536, 3072, 15360), (1536, 18432, 3072),
    ];
    let mut worst = 0.0f64;
    for (m, n, k) in shapes {
        let tag = format!("{m}_{n}_{k}");
        let x = refs[&format!("{tag}.x")].to_device(dev).unwrap();
        let w = refs[&format!("{tag}.w")].to_device(dev).unwrap();
        let y = &refs[&format!("{tag}.y")];
        let lin = Linear::new(w, None).unwrap();
        let my = lin.forward(&x).unwrap().to_device(Device::Cpu).unwrap();
        let (cos, mx, mr) = metrics(&my, y);
        let rel = mx / mr.max(1e-9);
        eprintln!("[gemm {tag:<16}] cos={cos:.7} max_abs={mx:.4} max_ref={mr:.2} rel={rel:.4}");
        worst = worst.max(rel);
    }
    eprintln!("worst rel error = {worst:.4}");
    assert!(worst < 0.05, "gemm_bf16 расходится с torch (worst rel {worst:.4}) — БАГ ядра");
}
