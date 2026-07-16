#![cfg(feature = "cuda")]

//! best_cu NN matmul (Backend::matmul → gemm_nn_u8): C[M,N]=A[M,K]@B[K,N],
//! F16/BF16 float-acc vs CPU-f32. Non-batched, K%16==0, N%8==0, M/N любые.

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

// NN-эталон первых r строк: ref[i,j] = sum_k a[i,k]*b[k,j], B = [K,N].
fn cpu_ref_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, r: usize) -> Vec<f32> {
    let r = r.min(m);
    let mut out = vec![0.0f32; r * n];
    for i in 0..r {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += a[i * k + kk] as f64 * b[kk * n + j] as f64;
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

// N%8==0; K любой (K-tail pad), M/N любые (partial-тайл).
const CASES: &[(usize, usize, usize, &str)] = &[
    (64, 64, 64, "aligned-small"),
    (128, 256, 512, "aligned"),
    (100, 256, 512, "M-part"),
    (128, 256, 504, "N-part"),
    (513, 1024, 256, "large-K-part"),
    (128, 27, 256, "K-tail-27"),
    (100, 36, 512, "K-tail-36"),
    (128, 250, 256, "K-tail-250"),
    (128, 256, 500, "N-pad-500"),
    (100, 36, 50, "N-pad+K-tail"),
    (64, 64, 2, "N-pad-2"),
];

#[test]
fn matmul_nn_f16_vs_cpu() {
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    let _nograd = synaptix_core::grad::NoGradGuard::new();
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    for &(m, k, n, label) in CASES {
        let a_host = det(0x7171 + (m * k) as u64, m * k, 0.3);
        let b_host = det(0x8282 + (k * n) as u64, k * n, 0.3);
        let a: Vec<f16> = a_host.iter().map(|&v| f16::from_f32(v)).collect();
        let b: Vec<f16> = b_host.iter().map(|&v| f16::from_f32(v)).collect();
        let at = Tensor::from_vec(a, (m, k), Device::Cuda(0)).unwrap();
        let bt = Tensor::from_vec(b, (k, n), Device::Cuda(0)).unwrap();
        let c = at.matmul(&bt).unwrap();
        assert_eq!(c.dims(), &[m, n]);
        stream.synchronize().unwrap();
        let r = 8usize.min(m);
        let bytes: Vec<u8> = stream
            .clone_dtoh(c.storage().as_cuda().unwrap().slice())
            .unwrap();
        let got: Vec<f32> = bytemuck::cast_slice::<u8, f16>(&bytes)[..r * n]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let want = cpu_ref_nn(&a_host, &b_host, m, k, n, r);
        let cos = cos_sim(&got, &want);
        eprintln!("[matmul_nn f16 {label} {m}x{k}x{n}] vs CPU cos={cos:.6}");
        assert!(cos >= 0.99, "matmul_nn f16 {label} {m}x{k}x{n} cos={cos} < 0.99");
    }
}

#[test]
fn matmul_nn_batched_f16_vs_cpu() {
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    let _nograd = synaptix_core::grad::NoGradGuard::new();
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    // (batch, m, k, n, broadcast_B, label)
    for &(bt, m, k, n, bcast, label) in &[
        (4usize, 64usize, 64usize, 64usize, false, "batched"),
        (3, 100, 256, 512, false, "batched-part"),
        (3, 64, 27, 256, false, "batched-K-tail"),
        (3, 32, 32, 64, true, "broadcast-B"),
    ] {
        let a_host = det(0xB5B5 + (bt * m * k) as u64, bt * m * k, 0.3);
        let bb = if bcast { 1 } else { bt };
        let b_host = det(0xC6C6 + (k * n) as u64, bb * k * n, 0.3);
        let a: Vec<f16> = a_host.iter().map(|&v| f16::from_f32(v)).collect();
        let b: Vec<f16> = b_host.iter().map(|&v| f16::from_f32(v)).collect();
        let at = Tensor::from_vec(a, (bt, m, k), Device::Cuda(0)).unwrap();
        let btn = if bcast {
            Tensor::from_vec(b, (k, n), Device::Cuda(0)).unwrap()
        } else {
            Tensor::from_vec(b, (bt, k, n), Device::Cuda(0)).unwrap()
        };
        let c = at.matmul(&btn).unwrap();
        assert_eq!(c.dims(), &[bt, m, n]);
        stream.synchronize().unwrap();
        let bytes: Vec<u8> = stream
            .clone_dtoh(c.storage().as_cuda().unwrap().slice())
            .unwrap();
        let got: Vec<f32> = bytemuck::cast_slice::<u8, f16>(&bytes)[..bt * m * n]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let mut want = vec![0.0f32; bt * m * n];
        for bi in 0..bt {
            let b_bi = if bcast { 0 } else { bi };
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for kk in 0..k {
                        acc += a_host[bi * m * k + i * k + kk] as f64
                            * b_host[b_bi * k * n + kk * n + j] as f64;
                    }
                    want[bi * m * n + i * n + j] = acc as f32;
                }
            }
        }
        let cos = cos_sim(&got, &want);
        eprintln!("[matmul_nn f16 {label} b={bt} {m}x{k}x{n}] vs CPU cos={cos:.6}");
        assert!(cos >= 0.99, "matmul_nn batched {label} cos={cos} < 0.99");
    }
}

#[test]
fn matmul_nn_f32_vs_cpu() {
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    let _nograd = synaptix_core::grad::NoGradGuard::new();
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    // (batch, m, k, n, broadcast, label) — истинный f32 SIMT тянет ЛЮБЫЕ M/N/K.
    for &(bt, m, k, n, bcast, label) in &[
        (1usize, 128usize, 256usize, 512usize, false, "aligned"),
        (1, 100, 250, 500, false, "all-unaligned"),
        (4, 64, 64, 64, false, "batched"),
        (3, 32, 48, 80, true, "broadcast-B"),
    ] {
        let a_host = det(0xD7D7 + (bt * m * k) as u64, bt * m * k, 0.3);
        let bb = if bcast { 1 } else { bt };
        let b_host = det(0xE8E8 + (k * n) as u64, bb * k * n, 0.3);
        let at = Tensor::from_vec(a_host.clone(), (bt, m, k), Device::Cuda(0)).unwrap();
        let btn = if bcast {
            Tensor::from_vec(b_host.clone(), (k, n), Device::Cuda(0)).unwrap()
        } else {
            Tensor::from_vec(b_host.clone(), (bt, k, n), Device::Cuda(0)).unwrap()
        };
        let c = at.matmul(&btn).unwrap();
        assert_eq!(c.dims(), &[bt, m, n]);
        stream.synchronize().unwrap();
        let bytes: Vec<u8> = stream
            .clone_dtoh(c.storage().as_cuda().unwrap().slice())
            .unwrap();
        let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&bytes)[..bt * m * n].to_vec();
        let mut want = vec![0.0f32; bt * m * n];
        for bi in 0..bt {
            let b_bi = if bcast { 0 } else { bi };
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for kk in 0..k {
                        acc += a_host[bi * m * k + i * k + kk] as f64
                            * b_host[b_bi * k * n + kk * n + j] as f64;
                    }
                    want[bi * m * n + i * n + j] = acc as f32;
                }
            }
        }
        let cos = cos_sim(&got, &want);
        eprintln!("[matmul_nn f32 {label} b={bt} {m}x{k}x{n}] vs CPU cos={cos:.6}");
        assert!(cos >= 0.999, "matmul_nn f32 {label} cos={cos} < 0.999");
    }
}

#[test]
fn matmul_nn_bf16_vs_cpu() {
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    let _nograd = synaptix_core::grad::NoGradGuard::new();
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    for &(m, k, n, label) in CASES {
        let a_host = det(0x9393 + (m * k) as u64, m * k, 0.3);
        let b_host = det(0xA4A4 + (k * n) as u64, k * n, 0.3);
        let a: Vec<bf16> = a_host.iter().map(|&v| bf16::from_f32(v)).collect();
        let b: Vec<bf16> = b_host.iter().map(|&v| bf16::from_f32(v)).collect();
        let at = Tensor::from_vec(a, (m, k), Device::Cuda(0)).unwrap();
        let bt = Tensor::from_vec(b, (k, n), Device::Cuda(0)).unwrap();
        let c = at.matmul(&bt).unwrap();
        assert_eq!(c.dims(), &[m, n]);
        stream.synchronize().unwrap();
        let r = 8usize.min(m);
        let bytes: Vec<u8> = stream
            .clone_dtoh(c.storage().as_cuda().unwrap().slice())
            .unwrap();
        let got: Vec<f32> = bytemuck::cast_slice::<u8, bf16>(&bytes)[..r * n]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let want = cpu_ref_nn(&a_host, &b_host, m, k, n, r);
        let cos = cos_sim(&got, &want);
        eprintln!("[matmul_nn bf16 {label} {m}x{k}x{n}] vs CPU cos={cos:.6}");
        assert!(cos >= 0.99, "matmul_nn bf16 {label} {m}x{k}x{n} cos={cos} < 0.99");
    }
}
