use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_video_ltx23::loader::LtxCheckpoint;
use synaptix_video_ltx23::vae::VaeDecoder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let ckpt = LtxCheckpoint::open(
        std::path::Path::new("models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors"),
        Device::Cpu, DType::BF16,
    )?;
    let vae = VaeDecoder::load(&ckpt, dev)?;
    let fp: usize = std::env::var("FP").ok().and_then(|s| s.parse().ok()).unwrap_or(11);
    let lat = Tensor::zeros(vec![1, 128, fp, 22, 40], DType::BF16, dev)?;
    for i in 0..3 {
        let t = std::time::Instant::now();
        let out = vae.decode(&lat)?;
        let _ = synaptix_core::device::cuda::synchronize(0);
        eprintln!("iter{i}: {:.2}s out {:?}", t.elapsed().as_secs_f32(), out.dims());
    }
    Ok(())
}
