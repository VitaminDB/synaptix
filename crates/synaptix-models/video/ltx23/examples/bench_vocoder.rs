//! Изолированный бенч вокодера (BigVGAN base+BWE): mel [1,2,501,64] → wave.
//! SYN_LTX_VOC_PROF=1 — пер-уровневая разбивка. Запускать одним процессом.

use std::time::Instant;
use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_video_ltx23::vocoder::VocoderWithBwe;

fn main() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    let _ng = synaptix_core::grad::NoGradGuard::new();
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        format!("{}/models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors", std::env::var("HOME").unwrap())
    });
    let dev = Device::Cuda(0);
    let t = Instant::now();
    let voc = VocoderWithBwe::load(&path, dev).expect("vocoder load");
    println!("load: {:.1}s", t.elapsed().as_secs_f32());
    let mel = Tensor::randn(vec![1usize, 2, 501, 64], Device::Cpu)
        .unwrap()
        .to_device(dev)
        .unwrap()
        .mul_scalar(0.1)
        .unwrap();
    // warmup (NVRTC)
    let w = voc.forward(&mel).expect("vocoder fwd");
    synaptix_core::device::cuda::synchronize(0).unwrap();
    println!("warmup done, wave {:?}", w.dims());
    let t = Instant::now();
    let w = voc.forward(&mel).expect("vocoder fwd");
    synaptix_core::device::cuda::synchronize(0).unwrap();
    println!("vocoder forward: {:.2}s (wave {:?})", t.elapsed().as_secs_f32(), w.dims());
}
