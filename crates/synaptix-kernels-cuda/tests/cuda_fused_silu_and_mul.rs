
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};

use synaptix_kernels_cuda::fused::silu_and_mul::{
    silu_and_mul_bf16, silu_and_mul_f16, silu_and_mul_f32, SiluAndMulKernels,
};

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            f * scale
        })
        .collect()
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

#[test]
fn cuda_silu_and_mul_f32_matches_host() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = SiluAndMulKernels::for_context(&ctx).expect("kernels");
    let total = 4096usize;
    let g_h = det(0xABCD, total, 1.5);
    let u_h = det(0x1234, total, 0.7);
    let g: CudaSlice<f32> = stream.clone_htod(&g_h).unwrap();
    let u: CudaSlice<f32> = stream.clone_htod(&u_h).unwrap();
    let mut out: CudaSlice<f32> = stream.alloc_zeros(total).unwrap();
    {
        let gv = g.as_view();
        let uv = u.as_view();
        let mut ov = out.as_view_mut();
        silu_and_mul_f32(&kernels, &stream, &gv, &uv, &mut ov, total as u32).unwrap();
    }
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&out).unwrap();
    let mut max_err = 0.0_f32;
    for i in 0..total {
        let ref_v = silu(g_h[i]) * u_h[i];
        let err = (got[i] - ref_v).abs();
        if err > max_err {
            max_err = err;
        }
    }
    assert!(max_err < 5e-6, "f32 silu_and_mul max_err={max_err}");
}

#[test]
fn cuda_silu_and_mul_f16_matches_host() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = SiluAndMulKernels::for_context(&ctx).expect("kernels");
    let total = 4096usize;
    let g_f = det(0xCAFE, total, 2.0);
    let u_f = det(0xBEEF, total, 1.0);
    let g_h: Vec<f16> = g_f.iter().copied().map(f16::from_f32).collect();
    let u_h: Vec<f16> = u_f.iter().copied().map(f16::from_f32).collect();
    let g: CudaSlice<f16> = stream.clone_htod(&g_h).unwrap();
    let u: CudaSlice<f16> = stream.clone_htod(&u_h).unwrap();
    let mut out: CudaSlice<f16> = stream.alloc_zeros(total).unwrap();
    {
        let gv = g.as_view();
        let uv = u.as_view();
        let mut ov = out.as_view_mut();
        silu_and_mul_f16(&kernels, &stream, &gv, &uv, &mut ov, total as u32).unwrap();
    }
    stream.synchronize().unwrap();
    let got: Vec<f16> = stream.clone_dtoh(&out).unwrap();
    let mut max_err = 0.0_f32;
    for i in 0..total {
        let ref_v = silu(g_h[i].to_f32()) * u_h[i].to_f32();
        let err = (got[i].to_f32() - ref_v).abs();
        if err > max_err {
            max_err = err;
        }
    }
    assert!(max_err < 2e-3, "f16 silu_and_mul max_err={max_err}");
}

#[test]
fn cuda_silu_and_mul_bf16_matches_host() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = SiluAndMulKernels::for_context(&ctx).expect("kernels");
    let total = 4096usize;
    let g_f = det(0xDEAD, total, 2.0);
    let u_f = det(0xFACE, total, 1.0);
    let g_h: Vec<bf16> = g_f.iter().copied().map(bf16::from_f32).collect();
    let u_h: Vec<bf16> = u_f.iter().copied().map(bf16::from_f32).collect();
    let g: CudaSlice<bf16> = stream.clone_htod(&g_h).unwrap();
    let u: CudaSlice<bf16> = stream.clone_htod(&u_h).unwrap();
    let mut out: CudaSlice<bf16> = stream.alloc_zeros(total).unwrap();
    {
        let gv = g.as_view();
        let uv = u.as_view();
        let mut ov = out.as_view_mut();
        silu_and_mul_bf16(&kernels, &stream, &gv, &uv, &mut ov, total as u32).unwrap();
    }
    stream.synchronize().unwrap();
    let got: Vec<bf16> = stream.clone_dtoh(&out).unwrap();
    let mut max_err = 0.0_f32;
    for i in 0..total {
        let ref_v = silu(g_h[i].to_f32()) * u_h[i].to_f32();
        let err = (got[i].to_f32() - ref_v).abs();
        if err > max_err {
            max_err = err;
        }
    }
    assert!(max_err < 1.5e-2, "bf16 silu_and_mul max_err={max_err}");
}
