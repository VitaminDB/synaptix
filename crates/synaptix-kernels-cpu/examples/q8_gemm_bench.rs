//! Замер q8row-Linear на префилле: `q8_gemm_bench [m…]` (по умолчанию 1 16 64 256), формы LFM3-1.7B.
use std::time::Instant;
use synaptix_core::dtype::DType;
use synaptix_kernels_cpu::quant::dequant::WeightView;
use synaptix_kernels_cpu::quant::gemv::gemm_quant_f32;

fn main() {
    let ms: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let ms = if ms.is_empty() { vec![1, 16, 64, 256] } else { ms };
    let shapes = [(2048usize, 2048usize), (7680, 2048), (2048, 7680), (6144, 2048)];
    for &m in &ms {
        let mut tot = 0f64;
        for &(n, k) in &shapes {
            let packed: Vec<u8> = (0..n * k).map(|i| ((i * 7919 % 251) as i32 - 125) as i8 as u8).collect();
            let scales: Vec<u8> = (0..n).flat_map(|j| (0.01f32 + j as f32 * 1e-6).to_le_bytes()).collect();
            let view = WeightView::new(DType::Q8Row, n, k, &packed, Some(&scales)).unwrap();
            let x: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
            let mut out = vec![0f32; m * n];
            let mut refo = vec![0f32; m * n];
            // Эталон — путь декода (dot4) по 4 строки.
            for i0 in (0..m).step_by(4) {
                let r = 4.min(m - i0);
                gemm_quant_f32(&view, &x[i0 * k..(i0 + r) * k], r, &mut refo[i0 * n..(i0 + r) * n]);
            }
            gemm_quant_f32(&view, &x, m, &mut out);
            let num: f32 = out.iter().zip(&refo).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f32 = refo.iter().map(|b| b * b).sum();
            if (num / den.max(1e-30)).sqrt() > 1e-5 {
                println!("  РАСХОЖДЕНИЕ: отн. {:.2e}", (num / den).sqrt());
            }
            let reps = (2_000_000_000usize / (m * n * k)).clamp(3, 200);
            let t = Instant::now();
            for _ in 0..reps {
                gemm_quant_f32(&view, &x, m, &mut out);
            }
            let dt = t.elapsed().as_secs_f64() / reps as f64;
            tot += dt;
            println!("m={m:4} {n}x{k}: {:.3} мс, {:.0} GOP/s", dt * 1e3, 2.0 * (m * n * k) as f64 / dt / 1e9);
        }
        println!("m={m}: сумма {:.2} мс", tot * 1e3);
    }
}
