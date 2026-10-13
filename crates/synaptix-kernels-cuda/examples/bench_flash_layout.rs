//! Бенч flash_splitq на форме видео-DiT: BHSD (v5) против BSHD (v1/v5).
//! `S=4316 H=42 cargo run --profile fast-release --example bench_flash_layout -p synaptix-kernels-cuda`
//! Ядро BSHD выбирается переменной SYN_FLASH_V5_BSHD (1 — v5, иначе v1).

use cudarc::driver::CudaSlice;
use synaptix_core::dtype::DType;
use synaptix_kernels_cuda::attention::flash_fp8::{flash_fp8_bshd, FlashFp8Kernels};
use synaptix_kernels_cuda::attention::flash_splitq::{flash_splitq_u8, FlashSplitQKernels};

fn main() {
    synaptix_kernels_cuda::ensure_registered();
    let ctx = synaptix_core::device::cuda::get(0).unwrap();
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let k = FlashSplitQKernels::for_context(&ctx).unwrap();
    let env = |n: &str, d: usize| std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (s, h, d) = (env("S", 4316), env("H", 42), 128usize);
    let n = s * h * d;
    let mk = || -> CudaSlice<u8> {
        let v: Vec<u8> = (0..n * 2).map(|i| if i % 2 == 1 { 0x3c } else { (i * 37 % 251) as u8 }).collect();
        stream.clone_htod(&v).unwrap()
    };
    let (q, kk, v) = (mk(), mk(), mk());
    let mut out: CudaSlice<u8> = stream.alloc_zeros(n * 2).unwrap();
    let flops = 4.0 * (s * s * h * d) as f64;
    for bshd in [false, true] {
        let run = |o: &mut CudaSlice<u8>| {
            flash_splitq_u8(&k, &stream, &q, 0, &kk, 0, &v, 0, o, 0, 1, h as u32, h as u32, s as u32, s as u32, d as u32, 0.088, false, 0, DType::BF16, bshd).unwrap();
        };
        for _ in 0..5 {
            run(&mut out);
        }
        stream.synchronize().unwrap();
        let t = std::time::Instant::now();
        let iters = 30;
        for _ in 0..iters {
            run(&mut out);
        }
        stream.synchronize().unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!("{} S={s} H={h}: {ms:.3} мс, {:.1} TFLOP/s", if bshd { "BSHD" } else { "BHSD" }, flops / ms / 1e9);
    }
    let k8 = FlashFp8Kernels::for_context(&ctx).unwrap();
    for full in [false, true] {
        let run = |o: &mut CudaSlice<u8>| {
            flash_fp8_bshd(&k8, &stream, (&q, 0), (&kk, 0), (&v, 0), o, 1, h, s, s, 0.088, full).unwrap();
        };
        for _ in 0..5 {
            run(&mut out);
        }
        stream.synchronize().unwrap();
        let t = std::time::Instant::now();
        let iters = 30;
        for _ in 0..iters {
            run(&mut out);
        }
        stream.synchronize().unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!("FP8 {} S={s} H={h}: {ms:.3} мс, {:.1} TFLOP/s (с квантом)", if full { "QK+PV" } else { "QK" }, flops / ms / 1e9);
    }
}
