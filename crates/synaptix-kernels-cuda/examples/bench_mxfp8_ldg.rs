//! MXFP8: cp.async-ядро против rot (SF_LDG при K%512≠0). `bench_mxfp8_ldg M N K…`
//! (тройки). TFLOP/s по медиане 20 запусков.

use cudarc::driver::CudaSlice;
use half::f16;
use std::time::Instant;
use synaptix_kernels_cuda::best_cu::gemm::gemm_mxfp8::{gemm_mxfp8, gemm_mxfp8_rot, GemmMxFp8Kernels};

fn main() {
    let ctx = synaptix_core::device::cuda::get(0).unwrap();
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let gk = GemmMxFp8Kernels::for_context(&ctx).unwrap();
    let a: Vec<u32> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    for t in a.chunks(3) {
        let (m, n, k) = (t[0].div_ceil(128) * 128, t[1], t[2]);
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        // e4m3 0x38 = 1.0, E8M0 127 = 1
        let xq: CudaSlice<u8> = stream.clone_htod(&vec![0x38u8; mu * ku]).unwrap();
        let wq: CudaSlice<u8> = stream.clone_htod(&vec![0x30u8; nu * ku]).unwrap();
        let sa: CudaSlice<u8> = stream.clone_htod(&vec![127u8; mu * ku / 32]).unwrap();
        let sb: CudaSlice<u8> = stream.clone_htod(&vec![120u8; nu * ku / 32]).unwrap();
        let mut y: CudaSlice<f16> = stream.alloc_zeros(mu * nu).unwrap();
        let flops = 2.0 * (mu * nu * ku) as f64;
        for (name, rot) in [("cp.async", false), ("rot", true)] {
            let mut ts = Vec::new();
            for i in 0..25 {
                stream.synchronize().unwrap();
                let t0 = Instant::now();
                let mut yv = y.slice_mut(..);
                if rot {
                    gemm_mxfp8_rot(&gk, &stream, &xq, &wq, &sa, &sb, &mut yv, m, n, k, true).unwrap();
                } else {
                    gemm_mxfp8(&gk, &stream, &xq, &wq, &sa, &sb, &mut yv, m, n, k).unwrap();
                }
                stream.synchronize().unwrap();
                if i >= 5 {
                    ts.push(t0.elapsed().as_secs_f64());
                }
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let dt = ts[ts.len() / 2];
            println!("{m}x{n}x{k} {name:8}: {:.3} мс {:6.1} TFLOP/s", dt * 1e3, flops / dt / 1e12);
        }
    }
}
