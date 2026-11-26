//! Генерация LTX тем же путём, что служба (`synaptix_gen::video::ltx`).
//! `ltx_gen MODEL.syn ENC.syn [UPSCALER.syn|-] OUT.mp4 SECONDS W H SEED "промпт"`
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use synaptix_core::device::Device;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let paths = synaptix_gen::video::LtxPaths {
        model: PathBuf::from(&a[0]),
        gemma: PathBuf::from(&a[1]),
        upscaler: (a[2] != "-").then(|| PathBuf::from(&a[2])),
        ..Default::default()
    };
    let req = synaptix_gen::video::VideoRequest {
        prompt: a[8].clone(),
        seconds: Some(a[4].parse().unwrap()),
        width: Some(a[5].parse().unwrap()),
        height: Some(a[6].parse().unwrap()),
        seed: a[7].parse().unwrap(),
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let stop = AtomicBool::new(false);
    let r = synaptix_gen::video::ltx(&paths, Device::Cuda(0), &req, std::path::Path::new(&a[3]), &stop, &|s, n| eprint!("\rшаг {s}/{n}  "));
    match r {
        Ok(o) => println!("\nготово за {:.1} с: {}", t0.elapsed().as_secs_f32(), o.label),
        Err(e) => println!("\nошибка: {e}"),
    }
}
