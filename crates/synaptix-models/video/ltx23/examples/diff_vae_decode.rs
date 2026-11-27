//! Декод латента хвоста (`.tail.safetensors`, тензор `video`) диффузионным
//! декодером LTX-2.5 → PNG-кадры. `diff_vae_decode VAE.safetensors TAIL.safetensors OUT_DIR`
use std::path::Path;
use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader as _;
use synaptix_video_ltx23::diff_vae::DiffVaeDecoder;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let lat = SafetensorsLoader::open(Path::new(&a[1])).expect("tail").load_to("video", dev, DType::F32).expect("video");
    // DV_CROP=N — левый верхний угол N×N латента (быстрая сверка с эталоном).
    let lat = match std::env::var("DV_CROP").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(n) => lat.narrow(3, 0, n).unwrap().narrow(4, 0, n).unwrap().contiguous().unwrap(),
        None => lat,
    };
    println!("латент {:?}", lat.dims());
    let dec = DiffVaeDecoder::load(&a[0], dev).expect("декодер");
    let t0 = std::time::Instant::now();
    let rgb = synaptix_core::grad::no_grad(|| dec.decode(&lat, 1)).expect("decode");
    let _ = synaptix_core::device::cuda::synchronize(0);
    println!("декод за {:.1} с → {:?}", t0.elapsed().as_secs_f32(), rgb.dims());
    // Сырой выход [1,3,T,H,W] f32 (−1…1) — для численной сверки.
    let raw = rgb.to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    std::fs::create_dir_all(&a[2]).unwrap();
    let d = rgb.dims();
    std::fs::write(Path::new(&a[2]).join(format!("rgb_{}x{}x{}.f32", d[2], d[3], d[4])), raw.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let frames = synaptix_video_ltx23::pipeline::rgb_to_frames(&rgb).expect("frames");
    std::fs::create_dir_all(&a[2]).unwrap();
    for (i, f) in frames.iter().enumerate() {
        let (h, w) = (f.dims()[1], f.dims()[2]);
        let v = f.to_device(Device::Cpu).unwrap().permute(vec![1, 2, 0]).unwrap().contiguous().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
        out.extend(v.iter().map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8));
        std::fs::write(Path::new(&a[2]).join(format!("f{i:04}.ppm")), out).unwrap();
    }
}
