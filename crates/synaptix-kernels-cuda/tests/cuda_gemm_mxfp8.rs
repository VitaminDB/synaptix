#![cfg(feature = "cuda")]

//! Корректность best_cu MXFP8 GEMM (block-scale, TMA + warp-spec) end-to-end:
//! gemm_mxfp8_linear (GPU MXFP8-квант x+w + TMA + gemm) vs CPU-f32 по исходным
//! f16. Device-quant путь — без host round-trip.

use cudarc::driver::CudaSlice;
use half::f16;

use synaptix_kernels_cuda::best_cu::gemm::gemm_mxfp8::gemm_mxfp8_linear;
use synaptix_kernels_cuda::elementwise::quant::Mxfp8QuantKernels;

fn det_f16(seed: u64, n: usize, scale: f32) -> Vec<f16> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            f16::from_f32((((x >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale)
        })
        .collect()
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..a.len() {
        d += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    (d / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

#[test]
fn gemm_mxfp8_linear_device_vs_cpu() {
    synaptix_kernels_cuda::ensure_registered();
    let Some(ctx) = synaptix_core::device::cuda::get(0).ok() else {
        return;
    };
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let qk = Mxfp8QuantKernels::for_context(&ctx).expect("mxfp8 quant compile");

    for &(m, n, k) in &[(256u32, 256u32, 512u32), (256, 512, 512), (512, 256, 256)] {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let x = det_f16(0xC0DE_BA5E, mu * ku, 0.5);
        let w = det_f16(0xA110_C8E1, nu * ku, 0.5);
        let xd: CudaSlice<f16> = stream.clone_htod(&x).unwrap();
        let wd: CudaSlice<f16> = stream.clone_htod(&w).unwrap();
        let mut y: CudaSlice<f16> = stream.alloc_zeros(mu * nu).unwrap();
        gemm_mxfp8_linear(&qk, &stream, &xd, &wd, &mut y.slice_mut(0..), m, n, k).unwrap();
        stream.synchronize().unwrap();

        let r = 8usize.min(mu);
        let mut yref = vec![0.0f32; r * nu];
        for i in 0..r {
            for c in 0..nu {
                let mut acc = 0.0f64;
                for kk in 0..ku {
                    acc += x[i * ku + kk].to_f32() as f64 * w[c * ku + kk].to_f32() as f64;
                }
                yref[i * nu + c] = acc as f32;
            }
        }
        let y_h: Vec<f16> = stream.clone_dtoh(&y).unwrap();
        let got: Vec<f32> = y_h[..r * nu].iter().map(|v| v.to_f32()).collect();
        let cos = cos_sim(&got, &yref);
        eprintln!("[gemm_mxfp8_linear device-quant {m}x{k}x{n}] vs CPU cos={cos:.6}");
        assert!(cos >= 0.97, "gemm_mxfp8_linear {m}x{k}x{n} cos={cos} < 0.97");
    }
}
