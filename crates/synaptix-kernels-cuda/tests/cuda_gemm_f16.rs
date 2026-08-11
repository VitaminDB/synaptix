
//! Корректность best_cu gemm_f16 (NN: C[M,N] = A[M,K] @ B[K,N], B row-major [K,N])
//! с f32-аккумулятором против CPU-f32. Покрывает выровненные/партиал-тайлы и
//! large-K (где half-аккумулятор деградировал бы — этот тест его и ловит).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::f16;

use synaptix_kernels_cuda::best_cu::gemm::gemm_f16::{gemm_f16, GemmF16Kernels};

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
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

#[test]
fn gemm_f16_nn_vs_cpu_f32() {
    synaptix_kernels_cuda::ensure_registered();
    let Some((ctx, stream)) = setup() else {
        return;
    };
    let kernels = GemmF16Kernels::for_context(&ctx).expect("compile gemm_f16");

    // (m, k, n, label) — ограничения ядра: K%16==0, N%8==0 (векторный cp.async
    // B по 16 байт); M любое (partial-тайл по строкам).
    for &(m, k, n, label) in &[
        (128usize, 256usize, 512usize, "aligned"),
        (100, 256, 512, "M-part"),
        (128, 256, 504, "N-part"),
        (96, 160, 320, "MN-part"),
        (513, 1024, 256, "large-K-part"),
    ] {
        let a_host = det(0x5151 + (m * k) as u64, m * k, 0.3);
        let b_host = det(0x6262 + (k * n) as u64, k * n, 0.3);
        let a: Vec<f16> = a_host.iter().map(|&v| f16::from_f32(v)).collect();
        let b: Vec<f16> = b_host.iter().map(|&v| f16::from_f32(v)).collect();

        let da: CudaSlice<f16> = stream.clone_htod(&a).unwrap();
        let db: CudaSlice<f16> = stream.clone_htod(&b).unwrap();
        let mut dc: CudaSlice<f16> = stream.alloc_zeros(m * n).unwrap();
        gemm_f16(
            &kernels, &stream, &da, &db, &mut dc, m as u32, n as u32, k as u32,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let c_h: Vec<f16> = stream.clone_dtoh(&dc).unwrap();

        let r = 8usize.min(m);
        let mut want = vec![0.0f32; r * n];
        for i in 0..r {
            for j in 0..n {
                let mut acc = 0.0f64;
                for kk in 0..k {
                    acc += a_host[i * k + kk] as f64 * b_host[kk * n + j] as f64;
                }
                want[i * n + j] = acc as f32;
            }
        }
        let got: Vec<f32> = c_h[..r * n].iter().map(|v| v.to_f32()).collect();
        let cos = cos_sim(&got, &want);
        eprintln!("[gemm_f16 NN {label} {m}x{k}x{n}] f32-acc vs CPU cos={cos:.6}");
        assert!(cos >= 0.99, "gemm_f16 {label} {m}x{k}x{n} cos={cos} < 0.99");
    }
}
