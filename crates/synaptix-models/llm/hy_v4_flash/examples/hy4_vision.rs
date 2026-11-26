//! Башня зрения Hy4: `cargo run --example hy4_vision -- <бандл> <картинка> <каталог>` —
//! пишет `patches.f32` [gh·gw, 768], `grid.txt` (gh gw) и `features.f32` [N, 3072]
//! для сверки с `scripts/reference/hy4_vision_ref.py`.

use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::grad::no_grad;
use synaptix_llm_hy_v4_flash::loader::Hy4Weights;
use synaptix_llm_hy_v4_flash::vision::{prepare, Hy4Vision};

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let a: Vec<String> = std::env::args().collect();
    let (bundle, image, out) = (&a[1], &a[2], std::path::Path::new(&a[3]));
    std::fs::create_dir_all(out).unwrap();
    let w = Hy4Weights::open(bundle).expect("бандл");
    let dev = Device::Cuda(0);
    let v = no_grad(|| Hy4Vision::load(&w, dev, 27)).unwrap().expect("в бандле нет vit.*");
    let p = prepare(std::path::Path::new(image)).unwrap();
    eprintln!("сетка {}×{} патчей, токенов {}", p.gh, p.gw, p.tokens());
    let t0 = Instant::now();
    let f = no_grad(|| v.encode(&p)).unwrap();
    let f = f.to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap();
    eprintln!("башня {:.2} с, выход {:?}", t0.elapsed().as_secs_f64(), f.dims());
    let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    std::fs::write(out.join("patches.f32"), bytes(&p.patches)).unwrap();
    std::fs::write(out.join("grid.txt"), format!("{} {}", p.gh, p.gw)).unwrap();
    std::fs::write(out.join("features.f32"), bytes(&f.flatten_all().unwrap().to_vec1::<f32>().unwrap())).unwrap();
}
