
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};
use synaptix_kernels_cuda::fused::moe_dispatch::{
    moe_gather_bf16, moe_gather_f32, moe_scatter_f16, moe_scatter_f32, MoeDispatchKernels,
};

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det_f32(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
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

// Случайная перестановка [0, n) (Fisher–Yates на детерминированном rng).
fn perm(seed: u64, n: usize) -> Vec<u32> {
    let mut p: Vec<u32> = (0..n as u32).collect();
    let mut x = seed.wrapping_add(0xDEAD_BEEF_CAFE_0001);
    for i in (1..n).rev() {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = ((x >> 33) as usize) % (i + 1);
        p.swap(i, j);
    }
    p
}

#[test]
fn moe_scatter_f32_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = MoeDispatchKernels::for_context(&ctx).expect("compile moe_dispatch");
    let (n, d) = (96usize, 128usize);
    let x = det_f32(0xA1, n * d, 1.0);
    let idx = perm(0xB1, n);
    // scatter: out[i] = x[idx[i]]
    let mut expected = vec![0.0_f32; n * d];
    for i in 0..n {
        let s = idx[i] as usize;
        expected[i * d..i * d + d].copy_from_slice(&x[s * d..s * d + d]);
    }
    let dx: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let didx: CudaSlice<u32> = stream.clone_htod(&idx).unwrap();
    let mut dout: CudaSlice<f32> = stream.alloc_zeros(n * d).unwrap();
    moe_scatter_f32(&kernels, &stream, &dx, &didx, &mut dout, n as u32, d as u32).unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
    assert_eq!(got, expected, "moe_scatter f32 mismatch");
}

#[test]
fn moe_gather_f32_inverse_of_scatter() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = MoeDispatchKernels::for_context(&ctx).expect("compile moe_dispatch");
    let (n, d) = (96usize, 128usize);
    let x = det_f32(0xA2, n * d, 1.0);
    let idx = perm(0xB2, n);
    // gather: out[idx[i]] = x[i]
    let mut expected = vec![0.0_f32; n * d];
    for i in 0..n {
        let dst = idx[i] as usize;
        expected[dst * d..dst * d + d].copy_from_slice(&x[i * d..i * d + d]);
    }
    let dx: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let didx: CudaSlice<u32> = stream.clone_htod(&idx).unwrap();
    let mut dout: CudaSlice<f32> = stream.alloc_zeros(n * d).unwrap();
    moe_gather_f32(&kernels, &stream, &dx, &didx, &mut dout, n as u32, d as u32).unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
    assert_eq!(got, expected, "moe_gather f32 mismatch");

    // scatter(gather(x)) == x для перестановки idx.
    let dx2: CudaSlice<f32> = stream.clone_htod(&got).unwrap();
    let mut dround: CudaSlice<f32> = stream.alloc_zeros(n * d).unwrap();
    moe_scatter_f32(
        &kernels,
        &stream,
        &dx2,
        &didx,
        &mut dround,
        n as u32,
        d as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let round: Vec<f32> = stream.clone_dtoh(&dround).unwrap();
    assert_eq!(
        round, x,
        "scatter∘gather должно быть identity для перестановки"
    );
}

#[test]
fn moe_scatter_f16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = MoeDispatchKernels::for_context(&ctx).expect("compile moe_dispatch");
    let (n, d) = (64usize, 96usize);
    let xf = det_f32(0xA3, n * d, 1.0);
    let x: Vec<f16> = xf.iter().map(|v| f16::from_f32(*v)).collect();
    let idx = perm(0xB3, n);
    let mut expected = vec![f16::from_f32(0.0); n * d];
    for i in 0..n {
        let s = idx[i] as usize;
        expected[i * d..i * d + d].copy_from_slice(&x[s * d..s * d + d]);
    }
    let dx: CudaSlice<f16> = stream.clone_htod(&x).unwrap();
    let didx: CudaSlice<u32> = stream.clone_htod(&idx).unwrap();
    let mut dout: CudaSlice<f16> = stream.alloc_zeros(n * d).unwrap();
    moe_scatter_f16(&kernels, &stream, &dx, &didx, &mut dout, n as u32, d as u32).unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f16> = stream.clone_dtoh(&dout).unwrap();
    assert_eq!(got, expected, "moe_scatter f16 mismatch (pure copy)");
}

#[test]
fn moe_gather_bf16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = MoeDispatchKernels::for_context(&ctx).expect("compile moe_dispatch");
    let (n, d) = (64usize, 80usize);
    let xf = det_f32(0xA4, n * d, 1.0);
    let x: Vec<bf16> = xf.iter().map(|v| bf16::from_f32(*v)).collect();
    let idx = perm(0xB4, n);
    let mut expected = vec![bf16::from_f32(0.0); n * d];
    for i in 0..n {
        let dst = idx[i] as usize;
        expected[dst * d..dst * d + d].copy_from_slice(&x[i * d..i * d + d]);
    }
    let dx: CudaSlice<bf16> = stream.clone_htod(&x).unwrap();
    let didx: CudaSlice<u32> = stream.clone_htod(&idx).unwrap();
    let mut dout: CudaSlice<bf16> = stream.alloc_zeros(n * d).unwrap();
    moe_gather_bf16(&kernels, &stream, &dx, &didx, &mut dout, n as u32, d as u32).unwrap();
    stream.synchronize().unwrap();
    let got: Vec<bf16> = stream.clone_dtoh(&dout).unwrap();
    assert_eq!(got, expected, "moe_gather bf16 mismatch (pure copy)");
}
