//! Замер Linear (`x·wᵀ`): `cargo run --profile fast-release --example gemm_bench`.
//! Старый блочный путь с упаковкой весов против [`xwt`] с весами F32/F16/BF16.
use std::time::Instant;
use synaptix_kernels_cpu::gemm::{blocked, xwt};

fn time(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() / reps as f64
}

fn main() {
    let shapes = [(300usize, 1024usize, 1024usize), (300, 3072, 1024), (300, 1024, 3072), (1, 3072, 1024), (8, 3072, 1024)];
    for (m, n, k) in shapes {
        let x: Vec<f32> = (0..m * k).map(|i| (i % 97) as f32 * 0.01).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i % 89) as f32 * 0.01).collect();
        let wh: Vec<half::f16> = w.iter().map(|&v| half::f16::from_f32(v)).collect();
        let mut out = vec![0f32; m * n];
        let gf = |dt: f64| 2.0 * (m * n * k) as f64 / dt / 1e9;
        let reps = if m < 16 { 200 } else { 20 };
        let d0 = time(reps, || blocked::linear(m, n, k, &x, &w, &mut out));
        let d1 = time(reps, || xwt::linear(m, n, k, &x, bytemuck::cast_slice(&w), xwt::WType::F32, &mut out));
        let d2 = time(reps, || xwt::linear(m, n, k, &x, bytemuck::cast_slice(&wh), xwt::WType::F16, &mut out));
        println!(
            "{m}x{n}x{k}: упаковка {:.2} мс ({:.0} GF/s) · xwt F32 {:.2} мс ({:.0}) · xwt F16 {:.2} мс ({:.0})",
            d0 * 1e3, gf(d0), d1 * 1e3, gf(d1), d2 * 1e3, gf(d2)
        );
    }
}
