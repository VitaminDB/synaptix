//! Хостовая перепаковка NVFP4 (`QuantWeight::nvfp4_repack_host`) обязана
//! совпадать байт в байт с ядром `nvfp4_w_repack`, которое строит
//! `ensure_shuffled`: на ней держится pinned-зеркало экспертов Qwen4Exp —
//! подкачанный из зеркала вес читают GEMV/GEMM без перепаковки на карте.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::Tensor;

fn ready() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

fn lcg(seed: &mut u64) -> u8 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 33) as u8
}

fn check(n: usize, k: usize) {
    let mut seed = (n * 1_000_003 + k) as u64;
    let packed: Vec<u8> = (0..n * k / 2).map(|_| lcg(&mut seed)).collect();
    let scales: Vec<u8> = (0..n * k / 16).map(|_| lcg(&mut seed)).collect();

    let mut host = vec![0u8; packed.len()];
    QuantWeight::nvfp4_repack_host(&packed, &mut host, n, k).expect("host repack");

    let dev = Device::Cuda(0);
    let p = Tensor::from_raw_slice(&packed, vec![packed.len()], DType::U8, dev).expect("packed");
    let s = Tensor::from_raw_slice(&scales, vec![scales.len()], DType::U8, dev).expect("scales");
    let w = QuantWeight::new(p.storage_arc(), s.storage_arc(), DType::NVFP4, n, k).expect("weight");
    w.ensure_shuffled().expect("ensure_shuffled");
    let shuf = w.shuffled().expect("shuffled");
    let stream = synaptix_core::device::cuda::default_stream(0).expect("stream");
    stream.synchronize().expect("sync");
    let device: Vec<u8> = stream.clone_dtoh(shuf.as_cuda().expect("cuda").slice()).expect("readback");
    // Буфер из пула может быть длиннее логического размера (гранулярность
    // аллокатора) — сравниваем ровно n·k/2 байт.
    assert!(device.len() >= host.len());
    let device = &device[..host.len()];
    let first_diff = device.iter().zip(&host).position(|(a, b)| a != b);
    assert_eq!(first_diff, None, "n={n} k={k}: раскладки расходятся");
}

#[test]
fn host_repack_matches_device_kernel() {
    if !ready() {
        eprintln!("CUDA недоступна — пропуск");
        return;
    }
    // Формы экспертов qwen3.8-flash-next: gate_up [1280, 2560], down [2560, 640].
    check(1280, 2560);
    check(2560, 640);
    check(64, 64);
}
