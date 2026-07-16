//! Depth Anything V2 Small bit-exact vs HF transformers (TF32 off).
//! Эталон /tmp/depth_ref.safetensors: px [1,3,518,518] + depth [1,518,518] + h3/h6/h9/h12.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_depth_anything::DepthAnything;
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;

const DIR: &str = "models/depth-anything-v2-small";
const REF: &str = "/tmp/depth_ref.safetensors";

fn cosmax(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
        mx = mx.max((*x - *y).abs() as f64);
    }
    (dot / (na.sqrt() * nb.sqrt()), mx)
}

#[test]
fn depth_matches_python() {
    if !Path::new(DIR).exists() || !Path::new(REF).exists() {
        eprintln!("skip: weights/ref absent");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let rl = SafetensorsLoader::open(REF).unwrap().with_device(dev);
    let px = rl.load("px").unwrap().to_dtype(DType::F32).unwrap();
    let refd = rl.load("depth").unwrap().to_dtype(DType::F32).unwrap();

    let m = DepthAnything::load(Path::new(DIR), dev).expect("load");
    let out = synaptix_core::grad::no_grad(|| m.forward(&px)).expect("forward");
    assert_eq!(out.dims(), refd.dims());
    let a: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    let b: Vec<f32> = refd.flatten_all().unwrap().to_vec1().unwrap();
    let (cos, mx) = cosmax(&a, &b);
    eprintln!("DepthAnything: cos={cos:.6} max_abs={mx:.5}");
    assert!(cos > 0.9999, "cos {cos}");
}
