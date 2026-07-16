#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};
use synaptix_kernels_cuda::elementwise::logit_cap::{
    logit_cap_bf16, logit_cap_f16, logit_cap_f32, LogitCapKernels,
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
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            f * scale
        })
        .collect()
}

fn cpu_logit_cap(x: &[f32], cap: f32) -> Vec<f32> {
    x.iter().map(|v| cap * (v / cap).tanh()).collect()
}

#[test]
fn logit_cap_f32_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = LogitCapKernels::for_context(&ctx).expect("compile logit_cap");
    let n = 4096usize;
    let cap = 30.0f32;
    // scale 80 >> cap, чтобы значительная часть значений уходила в насыщение tanh.
    let x = det_f32(0x10617, n, 80.0);
    let expected = cpu_logit_cap(&x, cap);

    let dev_x: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let mut dev_out: CudaSlice<f32> = stream.alloc_zeros(n).unwrap();
    logit_cap_f32(&kernels, &stream, &dev_x, &mut dev_out, cap, n as u32).unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dev_out).unwrap();

    let mut max_abs = 0.0_f32;
    for i in 0..n {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[logit_cap_f32 n={n} cap={cap}] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-4, "max_abs={max_abs}");
}

#[test]
fn logit_cap_f16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = LogitCapKernels::for_context(&ctx).expect("compile logit_cap");
    let n = 2048usize;
    let cap = 50.0f32;
    let x_f = det_f32(0x20617, n, 80.0);
    let x: Vec<f16> = x_f.iter().map(|v| f16::from_f32(*v)).collect();
    let x_back: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_logit_cap(&x_back, cap);

    let dev_x: CudaSlice<f16> = stream.clone_htod(&x).unwrap();
    let mut dev_out: CudaSlice<f16> = stream.alloc_zeros(n).unwrap();
    logit_cap_f16(&kernels, &stream, &dev_x, &mut dev_out, cap, n as u32).unwrap();
    stream.synchronize().unwrap();
    let got_h: Vec<f16> = stream.clone_dtoh(&dev_out).unwrap();
    let got: Vec<f32> = got_h.iter().map(|v| v.to_f32()).collect();

    let mut max_abs = 0.0_f32;
    for i in 0..n {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[logit_cap_f16 n={n} cap={cap}] max_abs={max_abs:.4}");
    assert!(max_abs < 0.05, "max_abs={max_abs}");
}

#[test]
fn logit_cap_bf16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = LogitCapKernels::for_context(&ctx).expect("compile logit_cap");
    let n = 2048usize;
    let cap = 50.0f32;
    let x_f = det_f32(0x30617, n, 80.0);
    let x: Vec<bf16> = x_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let x_back: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_logit_cap(&x_back, cap);

    let dev_x: CudaSlice<bf16> = stream.clone_htod(&x).unwrap();
    let mut dev_out: CudaSlice<bf16> = stream.alloc_zeros(n).unwrap();
    logit_cap_bf16(&kernels, &stream, &dev_x, &mut dev_out, cap, n as u32).unwrap();
    stream.synchronize().unwrap();
    let got_b: Vec<bf16> = stream.clone_dtoh(&dev_out).unwrap();
    let got: Vec<f32> = got_b.iter().map(|v| v.to_f32()).collect();

    let mut max_abs = 0.0_f32;
    for i in 0..n {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[logit_cap_bf16 n={n} cap={cap}] max_abs={max_abs:.4}");
    assert!(max_abs < 0.4, "max_abs={max_abs}");
}
