#![cfg(feature = "cuda")]

use half::{bf16, f16};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use synaptix_core::dtype::DType;
use synaptix_kernels_cuda::reduction::softmax::{run_softmax, SoftmaxKernels};

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

fn cpu_softmax(x: &[f32], batch: usize, hidden: usize) -> Vec<f32> {
    let mut y = vec![0.0_f32; x.len()];
    for b in 0..batch {
        let row = &x[b * hidden..(b + 1) * hidden];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for &v in row {
            sum += (v - mx).exp();
        }
        for i in 0..hidden {
            y[b * hidden + i] = (row[i] - mx).exp() / sum;
        }
    }
    y
}

fn run<T: DeviceRepr + ValidAsZeroBits + bytemuck::Pod + Copy>(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    dtype: DType,
    batch: usize,
    hidden: usize,
    tol: f32,
    name: &str,
    to_t: fn(f32) -> T,
    to_f32: fn(T) -> f32,
) {
    let x_f32 = det_f32(0xA110_C8E1, batch * hidden, 3.0);
    let x_t: Vec<T> = x_f32.iter().map(|v| to_t(*v)).collect();
    let x_back: Vec<f32> = x_t.iter().map(|v| to_f32(*v)).collect();
    let expected = cpu_softmax(&x_back, batch, hidden);

    let kernels = SoftmaxKernels::for_context(ctx).expect("compile softmax");
    let dev_x: CudaSlice<T> = stream.clone_htod(&x_t).unwrap();
    let mut dev_y: CudaSlice<T> = stream.alloc_zeros(batch * hidden).unwrap();
    run_softmax(
        &kernels,
        stream,
        &dev_x,
        &mut dev_y,
        batch as u32,
        hidden as u32,
        dtype,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got_t: Vec<T> = stream.clone_dtoh(&dev_y).unwrap();
    let got: Vec<f32> = got_t.iter().map(|v| to_f32(*v)).collect();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    // Каждая строка должна суммироваться в ~1.0.
    for b in 0..batch {
        let sum: f32 = got[b * hidden..(b + 1) * hidden].iter().sum();
        assert!((sum - 1.0).abs() < 1e-2, "{name}: row {b} sum={sum} (≠ 1)");
    }
    eprintln!("[{name} batch={batch} hidden={hidden}] max_abs={max_abs:.6}");
    assert!(max_abs < tol, "{name}: max_abs={max_abs} > tol {tol}");
}

#[test]
fn softmax_f32_small() {
    let Some((ctx, stream)) = setup() else { return };
    run::<f32>(
        &ctx,
        &stream,
        DType::F32,
        4,
        64,
        1e-6,
        "f32_4x64",
        |v| v,
        |v| v,
    );
}

#[test]
fn softmax_f32_large_hidden() {
    let Some((ctx, stream)) = setup() else { return };
    run::<f32>(
        &ctx,
        &stream,
        DType::F32,
        2,
        4096,
        1e-6,
        "f32_2x4096",
        |v| v,
        |v| v,
    );
}

#[test]
fn softmax_f16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    run::<f16>(
        &ctx,
        &stream,
        DType::F16,
        4,
        128,
        5e-3,
        "f16_4x128",
        f16::from_f32,
        |v| v.to_f32(),
    );
}

#[test]
fn softmax_bf16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    run::<bf16>(
        &ctx,
        &stream,
        DType::BF16,
        4,
        128,
        5e-2,
        "bf16_4x128",
        bf16::from_f32,
        |v| v.to_f32(),
    );
}
