//! Охота на баг best_gemm: TN-ядро (gemm_bf16, backend.linear, no-grad) vs
//! NN-ядро (gemm_f16, backend.matmul, grad). NN матчит Python; TN расходится в
//! хаотичном FLUX-стэке → сетка. Сравниваем на РЕАЛИСТИЧНЫХ (post-LayerNorm-like)
//! данных по всем FLUX-формам, ищем где TN отходит от NN. feature cuda.

#![cfg(feature = "cuda")]

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_nn::linear::Linear;
use synaptix_nn::module::Module;

fn metrics(a: &Tensor, b: &Tensor) -> (f64, f64, f64) {
    let n: usize = a.dims().iter().product();
    let av = a.contiguous().unwrap().reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let bv = b.contiguous().unwrap().reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let (mut dot, mut na, mut nb, mut mx, mut mr) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (x, y) in av.iter().zip(bv.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y; na += x * x; nb += y * y; mx = mx.max((x - y).abs()); mr = mr.max(y.abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), mx, mr)
}

// детерминированный нормальный шум (Box-Muller на LCG) — реалистичные post-LN активации
fn fill_normal(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    let mut nxt = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) as f32
    };
    (0..n).map(|_| {
        let (u1, u2) = (nxt().max(1e-7), nxt());
        ((-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()) * scale
    }).collect()
}

#[test]
fn flux_gemm_tn_vs_nn() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cuda(0);
    // реальные FLUX per-block формы (M, N, K)
    let shapes = [
        (512usize, 3072usize, 3072usize), (1024, 3072, 3072), (1536, 3072, 3072), // qkv/to_out
        (512, 12288, 3072), (1024, 12288, 3072), (1536, 12288, 3072),             // ff.0/proj_mlp
        (512, 3072, 12288), (1024, 3072, 12288),                                   // ff.2
        (1536, 3072, 15360),                                                       // single proj_out
        (1024, 3072, 64), (512, 3072, 4096),                                       // x_emb / ctx_emb
    ];
    eprintln!("=== best_gemm TN (no-grad) vs matmul NN (grad), realistic data ===");
    let mut worst = (1.0f64, String::new());
    for (m, n, k) in shapes {
        let xv = fill_normal(m * k, 0x1111 + (m as u64) * 7 + (k as u64), 1.0);
        let wv = fill_normal(n * k, 0x2222 + (n as u64) * 7 + (k as u64), 0.02);
        let mk = |v: &[f32], r: usize, c: usize| {
            Tensor::from_vec(v.to_vec(), (r, c), dev).unwrap().to_dtype(DType::BF16).unwrap()
        };
        let x = mk(&xv, m, k);
        let w = mk(&wv, n, k);
        let lin = Linear::new(w, None).unwrap();
        let tn = { let _ng = synaptix_core::grad::NoGradGuard::new(); lin.forward(&x).unwrap() }; // backend.linear TN
        let nn = lin.forward(&x).unwrap(); // grad → run_matmul → backend.matmul NN
        let (cos, mx, mr) = metrics(&tn.to_device(Device::Cpu).unwrap(), &nn.to_device(Device::Cpu).unwrap());
        eprintln!("  [{m:>4}x{n:>5}x{k:>5}] TN-vs-NN cos={cos:.7} max_abs={mx:.3} max_ref={mr:.2} rel={:.5}", mx / mr.max(1e-9));
        if cos < worst.0 { worst = (cos, format!("{m}x{n}x{k}")); }
    }
    eprintln!("worst: {} cos={:.7}", worst.1, worst.0);
}
