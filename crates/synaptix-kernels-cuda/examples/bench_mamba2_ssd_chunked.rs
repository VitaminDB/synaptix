#![cfg(feature = "cuda")]

//! Bench: Mamba2 SSD chunked-form vs recurrent baseline.
//!
//! Цель: проверить что chunked-форма быстрее на длинных L (≥ 4096), и
//! найти crossover-point. Размеры из Mamba2-2.7B: H=64, P=64, N=128.
//! Run: `cargo run --release --features cuda -p synaptix-kernels-cuda --example bench_mamba2_ssd_chunked`.

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::bf16;

use synaptix_core::dtype::DType;
use synaptix_kernels_cuda::ssm::mamba2_ssd::Mamba2SsdKernels;
use synaptix_kernels_cuda::ssm::mamba2_ssd_chunked::Mamba2SsdChunkedKernels;

fn det_bf16(seed: u64, n: usize, scale: f32, offset: f32) -> Vec<bf16> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            bf16::from_f32(f * scale + offset)
        })
        .collect()
}

fn upload_bf16(stream: &Arc<CudaStream>, host: &[bf16]) -> CudaSlice<bf16> {
    let mut d = unsafe { stream.alloc::<bf16>(host.len()).unwrap() };
    stream.memcpy_htod(host, &mut d).unwrap();
    d
}

fn time_ms<F: FnMut()>(stream: &Arc<CudaStream>, warmup: usize, iters: usize, mut f: F) -> f32 {
    for _ in 0..warmup {
        f();
    }
    stream.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    stream.synchronize().unwrap();
    let total_ms = t.elapsed().as_secs_f64() * 1000.0;
    (total_ms / iters as f64) as f32
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let recur = Mamba2SsdKernels::for_context(&ctx).expect("recurrent");
    let chunked = Mamba2SsdChunkedKernels::for_context(&ctx).expect("chunked");

    // Mamba2-2.7B base: H=64 heads, P=64 head_dim, N=128 d_state.
    // (Уменьшаем B до 1 для одного эксперимента; H=64 типично.)
    let b: u32 = 1;
    let h: u32 = 64;
    let p: u32 = 64;
    let n: u32 = 128;
    let q: u32 = 64;

    println!("Mamba2 SSD: chunked vs recurrent (BF16, B={b}, H={h}, P={p}, N={n}, Q={q})");
    println!(
        "{:>8} {:>12} {:>12} {:>10}",
        "L", "recur ms", "chunk ms", "speedup"
    );

    for l in [256u32, 512, 1024, 2048, 4096, 8192] {
        if l % q != 0 {
            continue;
        }
        let total_x = (b * l * h * p) as usize;
        let total_bc = (b * l * h * n) as usize;
        let total_dt = (b * l * h) as usize;
        let x_h = det_bf16(0x100, total_x, 0.5, 0.0);
        let dt_h = det_bf16(0x101, total_dt, 0.2, 0.5);
        let a_h = det_bf16(0x102, h as usize, 0.5, -1.5);
        let b_h_ = det_bf16(0x103, total_bc, 0.5, 0.0);
        let c_h = det_bf16(0x104, total_bc, 0.5, 0.0);

        let x = upload_bf16(&stream, &x_h);
        let dt = upload_bf16(&stream, &dt_h);
        let a = upload_bf16(&stream, &a_h);
        let bb = upload_bf16(&stream, &b_h_);
        let cc = upload_bf16(&stream, &c_h);

        let mut y_recur: CudaSlice<bf16> = stream.alloc_zeros(total_x).unwrap();
        let mut y_chunk: CudaSlice<bf16> = stream.alloc_zeros(total_x).unwrap();

        let warmup = 2;
        let iters = 5;

        let ms_recur = time_ms(&stream, warmup, iters, || {
            recur
                .ssd_bf16(
                    &stream,
                    &x,
                    &dt,
                    &a,
                    &bb,
                    &cc,
                    None,
                    &mut y_recur,
                    b,
                    l,
                    h,
                    p,
                    n,
                )
                .unwrap();
        });
        let ms_chunk = time_ms(&stream, warmup, iters, || {
            chunked
                .ssd(
                    &stream,
                    &x,
                    &dt,
                    &a,
                    &bb,
                    &cc,
                    None,
                    &mut y_chunk,
                    b,
                    l,
                    h,
                    p,
                    n,
                    q,
                    DType::BF16,
                )
                .unwrap();
        });
        let speedup = ms_recur / ms_chunk;
        println!(
            "{:>8} {:>12.3} {:>12.3} {:>10.2}x",
            l, ms_recur, ms_chunk, speedup
        );
    }
}
