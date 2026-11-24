//! Сверка башни с эталоном: патчи `[n, feat]` f32 из файла → выход башни f32.
//! `cargo run --example tower_dump -- <bundle.syn> <patches.f32> <t> <h> <w> <out.f32> [cpu|cuda] [f32|bf16|f16]`
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_vlm_qwen3::{load_from_bundle, ImageGrid};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (bundle, patches, out) = (&a[1], &a[2], &a[6]);
    let grid = ImageGrid { t: a[3].parse().unwrap(), h: a[4].parse().unwrap(), w: a[5].parse().unwrap() };
    let device = match a.get(7).map(String::as_str) {
        Some("cuda") => { synaptix_kernels_cuda::ensure_registered(); Device::Cuda(0) }
        _ => { synaptix_kernels_cpu::ensure_registered(); Device::Cpu }
    };
    let dtype = match a.get(8).map(String::as_str) { Some("bf16") => DType::BF16, Some("f16") => DType::F16, _ => DType::F32 };
    synaptix_kernels_cpu::ensure_registered();
    let raw = std::fs::read(patches).unwrap();
    let v: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let tower = load_from_bundle(bundle, device, dtype).unwrap();
    let n = grid.t * grid.h * grid.w;
    let x = Tensor::from_vec(v, vec![n, tower.config.patch_features()], device).unwrap().to_dtype(dtype).unwrap();
    let t0 = std::time::Instant::now();
    let y = synaptix_core::grad::no_grad(|| tower.forward(&x, grid)).unwrap();
    eprintln!("forward {:.1} с, выход {:?}", t0.elapsed().as_secs_f32(), y.dims());
    let y: Vec<f32> = y.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    std::fs::write(out, y.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}
