#![cfg(feature = "cuda")]

//! Durable correctness-гейт для Backend::linear (callsite #2, prefill M>1):
//! out[m,n] = sum_k x[m,k] * w[n,k], weight = [N,K] row-major.
//! Тестирует ТОТ бэкенд, куда роутится linear — сейчас (cutlass on) выровненный
//! BF16 идёт в best_cu, остальное (F16, non-aligned BF16) в CUTLASS; после
//! decutlass-миграции весь путь должен уйти в best_cu, а тест остаётся гейтом.

use half::{bf16, f16};

use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;

fn have_gpu() -> bool {
    synaptix_core::device::cuda::get(0).is_ok()
}

fn det(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            ((u as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

/// CPU f32-эталон первых `r` строк: ref[i,j] = sum_k x[i,k]*w[j,k].
fn cpu_ref(x: &[f32], w: &[f32], m: usize, k: usize, n: usize, r: usize) -> Vec<f32> {
    let r = r.min(m);
    let mut out = vec![0.0f32; r * n];
    for i in 0..r {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += x[i * k + kk] as f64 * w[j * k + kk] as f64;
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

fn read_f16(t: &Tensor, n: usize) -> Vec<f32> {
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    stream.synchronize().unwrap();
    let bytes: Vec<u8> = stream
        .clone_dtoh(t.storage().as_cuda().unwrap().slice())
        .unwrap();
    bytemuck::cast_slice::<u8, f16>(&bytes)[..n]
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn read_bf16(t: &Tensor, n: usize) -> Vec<f32> {
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    stream.synchronize().unwrap();
    let bytes: Vec<u8> = stream
        .clone_dtoh(t.storage().as_cuda().unwrap().slice())
        .unwrap();
    bytemuck::cast_slice::<u8, bf16>(&bytes)[..n]
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

// best_cu TN-ядро (bf16 и f16) тянет ЛЮБЫЕ M/N (part bounds-checked) + K (K-tail
// zero-pad). Полный набор форм — один список на оба dtype.
const CASES: &[(usize, usize, usize, &str)] = &[
    (128, 256, 512, "aligned"),
    (100, 256, 512, "M-unaligned"),
    (128, 256, 500, "N-unaligned"),
    (96, 160, 320, "MNK-unaligned"),
    (601, 512, 512, "prefill-odd"),
    (128, 200, 256, "K-tail"),
    (96, 250, 320, "K-tail+MN-unaligned"),
];

#[test]
fn linear_bf16_vs_cpu() {
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    // no-grad → run_linear идёт в Backend::linear (инференс-путь), а не matmul-fallback.
    let _nograd = synaptix_core::grad::NoGradGuard::new();
    for &(m, k, n, label) in CASES {
        let x_host = det(0x1111 + (m * k) as u64, m * k, 0.3);
        let w_host = det(0x2222 + (n * k) as u64, n * k, 0.3);
        let x: Vec<bf16> = x_host.iter().map(|&v| bf16::from_f32(v)).collect();
        let w: Vec<bf16> = w_host.iter().map(|&v| bf16::from_f32(v)).collect();
        let xt = Tensor::from_vec(x, (m, k), Device::Cuda(0)).unwrap();
        let wt = Tensor::from_vec(w, (n, k), Device::Cuda(0)).unwrap();
        let out = xt.linear(&wt).unwrap();
        assert_eq!(out.dims(), &[m, n]);

        let r = 8usize;
        let got = read_bf16(&out, r.min(m) * n);
        let want = cpu_ref(&x_host, &w_host, m, k, n, r);
        let cos = cos_sim(&got, &want);
        eprintln!("[linear bf16 {label} {m}x{k}x{n}] vs CPU cos={cos:.6}");
        assert!(cos >= 0.99, "linear bf16 {label} {m}x{k}x{n} cos={cos} < 0.99");
    }
}

#[test]
fn linear_f16_vs_cpu() {
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    // no-grad → run_linear идёт в Backend::linear (инференс-путь), а не matmul-fallback.
    let _nograd = synaptix_core::grad::NoGradGuard::new();
    for &(m, k, n, label) in CASES {
        let x_host = det(0x3333 + (m * k) as u64, m * k, 0.3);
        let w_host = det(0x4444 + (n * k) as u64, n * k, 0.3);
        let x: Vec<f16> = x_host.iter().map(|&v| f16::from_f32(v)).collect();
        let w: Vec<f16> = w_host.iter().map(|&v| f16::from_f32(v)).collect();
        let xt = Tensor::from_vec(x, (m, k), Device::Cuda(0)).unwrap();
        let wt = Tensor::from_vec(w, (n, k), Device::Cuda(0)).unwrap();
        let out = xt.linear(&wt).unwrap();
        assert_eq!(out.dims(), &[m, n]);

        let r = 8usize;
        let got = read_f16(&out, r.min(m) * n);
        let want = cpu_ref(&x_host, &w_host, m, k, n, r);
        let cos = cos_sim(&got, &want);
        eprintln!("[linear f16 {label} {m}x{k}x{n}] vs CPU cos={cos:.6}");
        assert!(cos >= 0.99, "linear f16 {label} {m}x{k}x{n} cos={cos} < 0.99");
    }
}
