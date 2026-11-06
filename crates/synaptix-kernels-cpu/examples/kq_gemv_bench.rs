//! Декод (m=1): q8row против Q4_K/Q5_K на формах LFM3 — веса больше кэша (16 матриц подряд).
use std::time::Instant;
use synaptix_core::dtype::DType;
use synaptix_core::quant::ggml::{quantize_q4_0, quantize_q4_k, quantize_q5_k, GgmlType};
use synaptix_kernels_cpu::quant::dequant::WeightView;
use synaptix_kernels_cpu::quant::gemv::gemm_quant_f32;

fn main() {
    let (n, k, copies) = (7680usize, 2048usize, 16usize);
    let w: Vec<f32> = (0..n * k).map(|i| ((i * 7919 % 251) as f32 - 125.0) * 1e-4).collect();
    let x: Vec<f32> = (0..k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let q8: Vec<u8> = (0..n * k).map(|i| ((i * 7919 % 251) as i32 - 125) as i8 as u8).collect();
    let sc: Vec<u8> = (0..n).flat_map(|_| 1e-4f32.to_le_bytes()).collect();
    let q4 = quantize_q4_k(&w, n, k, None);
    let q5 = quantize_q5_k(&w, n, k, None);
    let q40 = quantize_q4_0(&w, n, k);
    let sets: Vec<(&str, DType, Vec<Vec<u8>>)> = vec![
        ("q8row", DType::Q8Row, vec![q8; copies]),
        ("q4_k", DType::Ggml(GgmlType::Q4K), vec![q4; copies]),
        ("q5_k", DType::Ggml(GgmlType::Q5K), vec![q5; copies]),
        ("q4_0", DType::Ggml(GgmlType::Q4_0), vec![q40; copies]),
    ];
    let mut out = vec![0f32; n];
    for (name, dt, ws) in &sets {
        let views: Vec<WeightView> = ws.iter().map(|p| WeightView::new(*dt, n, k, p, if *dt == DType::Q8Row { Some(&sc) } else { None }).unwrap()).collect();
        for v in &views { gemm_quant_f32(v, &x, 1, &mut out); }
        let reps = 20;
        let t = Instant::now();
        for _ in 0..reps { for v in &views { gemm_quant_f32(v, &x, 1, &mut out); } }
        let dt_s = t.elapsed().as_secs_f64() / (reps * copies) as f64;
        let bytes = ws[0].len() as f64;
        println!("{name}: {:.3} мс на матрицу {n}×{k}, {:.1} ГБ/с", dt_s * 1e3, bytes / dt_s / 1e9);
    }
}
