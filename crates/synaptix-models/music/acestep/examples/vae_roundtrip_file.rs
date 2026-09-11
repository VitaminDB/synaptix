//! Круговой прогон VAE ACE-Step на своём аудио: decode(encode_mean(x)) против
//! x. В отличие от `vae_roundtrip` (синус 440 Гц) берёт реальный трек — так
//! видно, не уводит ли энкодер латент исходника (extract/repaint/edit).
//! Вход/выход — сырой f32le моно 48 кГц (`ffmpeg -f f32le`), латент — тоже
//! f32le `[64, T]`.
//! cargo run --release -p synaptix-music-acestep --example vae_roundtrip_file -- \
//!     <vae.syn> <in.f32> <out.f32> [latent.f32]

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_acestep::vae::AceStepVae;

fn read_f32(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .expect("read input")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn write_f32(path: &str, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).expect("write output");
}

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let args: Vec<String> = std::env::args().collect();
    let mono = read_f32(&args[2]);
    let n = mono.len();
    let mut flat = Vec::with_capacity(2 * n);
    flat.extend_from_slice(&mono);
    flat.extend_from_slice(&mono);
    let dev = Device::Cuda(0);
    let x = Tensor::from_vec(flat, vec![1usize, 2, n], dev).expect("tensor");
    let vae = AceStepVae::open(std::path::Path::new(&args[1]), dev).expect("vae");
    let z = vae.encode_mean(&x).expect("encode");
    let zv: Vec<f32> = z.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let mean = zv.iter().sum::<f32>() / zv.len() as f32;
    let std = (zv.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / zv.len() as f32).sqrt();
    eprintln!("latent {:?} mean={mean:.4} std={std:.4}", z.dims());
    if let Some(p) = args.get(4) {
        write_f32(p, &zv);
    }
    let y = vae.decode(&z).expect("decode");
    let ch0: Vec<f32> =
        y.narrow(1, 0, 1).unwrap().contiguous().unwrap().flatten_all().unwrap().to_vec1().unwrap();
    write_f32(&args[3], &ch0[..ch0.len().min(n)]);
    eprintln!("decoded {} samples", ch0.len());
}
