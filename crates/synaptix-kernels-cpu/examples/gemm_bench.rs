//! Замер блочного SGEMM/linear: `cargo run --example gemm_bench`.
use std::time::Instant;

fn main() {
    let shapes = [(200usize, 4096usize, 1024usize), (200, 1024, 4096), (200, 1024, 1024)];
    for (m, n, k) in shapes {
        let x: Vec<f32> = (0..m * k).map(|i| (i % 97) as f32 * 0.01).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i % 89) as f32 * 0.01).collect();
        let mut out = vec![0f32; m * n];
        synaptix_kernels_cpu::gemm::blocked::linear(m, n, k, &x, &w, &mut out);
        let t = Instant::now();
        let reps = 10;
        for _ in 0..reps {
            synaptix_kernels_cpu::gemm::blocked::linear(m, n, k, &x, &w, &mut out);
        }
        let dt = t.elapsed().as_secs_f64() / reps as f64;
        println!("linear {m}x{n}x{k}: {:.2} мс, {:.0} GFLOP/s", dt * 1e3, 2.0 * (m * n * k) as f64 / dt / 1e9);
    }
}
