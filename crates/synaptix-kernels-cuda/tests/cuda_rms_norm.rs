#![cfg(feature = "cuda")]

use half::{bf16, f16};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use synaptix_core::dtype::DType;
use synaptix_kernels_cuda::reduction::rmsnorm::{
    run_rms_norm, run_rms_norm_gated, RmsNormKernels, RmsVariant,
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

fn cpu_rms_norm_ref(
    x: &[f32],
    w: &[f32],
    eps: f32,
    batch: usize,
    hidden: usize,
    variant: RmsVariant,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; batch * hidden];
    for b in 0..batch {
        let row = &x[b * hidden..(b + 1) * hidden];
        let mut acc = 0.0_f64;
        for v in row {
            acc += (*v as f64) * (*v as f64);
        }
        let mean = (acc / hidden as f64) as f32;
        let rms = 1.0_f32 / (mean + eps).sqrt();
        for i in 0..hidden {
            let scale = match variant {
                RmsVariant::Plain => w[i],
                RmsVariant::Qwen => 1.0 + w[i],
            };
            out[b * hidden + i] = scale * row[i] * rms;
        }
    }
    out
}

fn cpu_rms_norm_gated_ref(
    x: &[f32],
    gate: &[f32],
    w: &[f32],
    eps: f32,
    batch: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; batch * hidden];
    for b in 0..batch {
        let row = &x[b * hidden..(b + 1) * hidden];
        let g_row = &gate[b * hidden..(b + 1) * hidden];
        let mut acc = 0.0_f64;
        for v in row {
            acc += (*v as f64) * (*v as f64);
        }
        let mean = (acc / hidden as f64) as f32;
        let rms = 1.0_f32 / (mean + eps).sqrt();
        for i in 0..hidden {
            let sig = 1.0_f32 / (1.0 + (-g_row[i]).exp());
            let silu_g = g_row[i] * sig;
            out[b * hidden + i] = w[i] * silu_g * row[i] * rms;
        }
    }
    out
}

fn assert_close(name: &str, got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len(), "{name}: length mismatch");
    let mut max_err: f32 = 0.0;
    let mut idx_worst = 0usize;
    for i in 0..got.len() {
        let e = (got[i] - expected[i]).abs();
        if e > max_err {
            max_err = e;
            idx_worst = i;
        }
    }
    assert!(
        max_err < tol,
        "{name}: max_abs_err={max_err} >= {tol}, worst idx={idx_worst} got={} expected={}",
        got[idx_worst],
        expected[idx_worst]
    );
}

fn run_plain_test<T: DeviceRepr + ValidAsZeroBits + bytemuck::Pod>(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    dtype: DType,
    variant: RmsVariant,
    batch: usize,
    hidden: usize,
    eps: f32,
    tol: f32,
    name: &str,
    to_t: fn(f32) -> T,
    to_f32: fn(T) -> f32,
) {
    let x_f32 = det_f32(101, batch * hidden, 0.7);
    let w_f32 = det_f32(202, hidden, 0.5);
    let x_t: Vec<T> = x_f32.iter().map(|v| to_t(*v)).collect();
    let w_t: Vec<T> = w_f32.iter().map(|v| to_t(*v)).collect();
    let x_back: Vec<f32> = x_t.iter().map(|v| to_f32(*v)).collect();
    let w_back: Vec<f32> = w_t.iter().map(|v| to_f32(*v)).collect();
    let cpu = cpu_rms_norm_ref(&x_back, &w_back, eps, batch, hidden, variant);

    let kernels = RmsNormKernels::for_context(ctx).expect("compile rms_norm");
    let dev_x: CudaSlice<T> = stream.clone_htod(&x_t).unwrap();
    let dev_w: CudaSlice<T> = stream.clone_htod(&w_t).unwrap();
    let mut dev_y: CudaSlice<T> = stream.alloc_zeros(batch * hidden).unwrap();

    run_rms_norm(
        &kernels,
        stream,
        &dev_x,
        &dev_w,
        &mut dev_y,
        batch as u32,
        hidden as u32,
        eps,
        variant,
        dtype,
    )
    .unwrap();
    stream.synchronize().unwrap();

    let got_t: Vec<T> = stream.clone_dtoh(&dev_y).unwrap();
    let got: Vec<f32> = got_t.iter().map(|v| to_f32(*v)).collect();
    assert_close(name, &got, &cpu, tol);
}

fn run_gated_test<T: DeviceRepr + ValidAsZeroBits + bytemuck::Pod>(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    dtype: DType,
    batch: usize,
    hidden: usize,
    eps: f32,
    tol: f32,
    name: &str,
    to_t: fn(f32) -> T,
    to_f32: fn(T) -> f32,
) {
    let x_f32 = det_f32(11, batch * hidden, 0.7);
    let g_f32 = det_f32(22, batch * hidden, 0.5);
    let w_f32 = det_f32(33, hidden, 0.5);
    let x_t: Vec<T> = x_f32.iter().map(|v| to_t(*v)).collect();
    let g_t: Vec<T> = g_f32.iter().map(|v| to_t(*v)).collect();
    let w_t: Vec<T> = w_f32.iter().map(|v| to_t(*v)).collect();
    let x_back: Vec<f32> = x_t.iter().map(|v| to_f32(*v)).collect();
    let g_back: Vec<f32> = g_t.iter().map(|v| to_f32(*v)).collect();
    let w_back: Vec<f32> = w_t.iter().map(|v| to_f32(*v)).collect();
    let cpu = cpu_rms_norm_gated_ref(&x_back, &g_back, &w_back, eps, batch, hidden);

    let kernels = RmsNormKernels::for_context(ctx).expect("compile rms_norm");
    let dev_x: CudaSlice<T> = stream.clone_htod(&x_t).unwrap();
    let dev_g: CudaSlice<T> = stream.clone_htod(&g_t).unwrap();
    let dev_w: CudaSlice<T> = stream.clone_htod(&w_t).unwrap();
    let mut dev_y: CudaSlice<T> = stream.alloc_zeros(batch * hidden).unwrap();

    run_rms_norm_gated(
        &kernels,
        stream,
        &dev_x,
        &dev_g,
        &dev_w,
        &mut dev_y,
        batch as u32,
        hidden as u32,
        eps,
        dtype,
    )
    .unwrap();
    stream.synchronize().unwrap();

    let got_t: Vec<T> = stream.clone_dtoh(&dev_y).unwrap();
    let got: Vec<f32> = got_t.iter().map(|v| to_f32(*v)).collect();
    assert_close(name, &got, &cpu, tol);
}

#[test]
fn rms_norm_fused_f32_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    run_plain_test::<f32>(
        &ctx,
        &stream,
        DType::F32,
        RmsVariant::Plain,
        6,
        128,
        1e-6,
        1e-5,
        "rms_norm_f32_6x128",
        |v| v,
        |v| v,
    );
}

#[test]
fn rms_norm_fused_f32_large_hidden() {
    let Some((ctx, stream)) = setup() else { return };
    run_plain_test::<f32>(
        &ctx,
        &stream,
        DType::F32,
        RmsVariant::Plain,
        2,
        4096,
        1e-6,
        2e-5,
        "rms_norm_f32_2x4096",
        |v| v,
        |v| v,
    );
}

#[test]
fn rms_norm_fused_f16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    run_plain_test::<f16>(
        &ctx,
        &stream,
        DType::F16,
        RmsVariant::Plain,
        4,
        128,
        1e-6,
        5e-3,
        "rms_norm_f16_4x128",
        f16::from_f32,
        |v| v.to_f32(),
    );
}

#[test]
fn rms_norm_fused_bf16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    run_plain_test::<bf16>(
        &ctx,
        &stream,
        DType::BF16,
        RmsVariant::Plain,
        4,
        128,
        1e-6,
        5e-2,
        "rms_norm_bf16_4x128",
        bf16::from_f32,
        |v| v.to_f32(),
    );
}

#[test]
fn rms_norm_qwen_fused_f32() {
    let Some((ctx, stream)) = setup() else { return };
    run_plain_test::<f32>(
        &ctx,
        &stream,
        DType::F32,
        RmsVariant::Qwen,
        3,
        64,
        1e-6,
        1e-5,
        "rms_norm_qwen_f32_3x64",
        |v| v,
        |v| v,
    );
}

#[test]
fn rms_norm_qwen_fused_bf16() {
    let Some((ctx, stream)) = setup() else { return };
    run_plain_test::<bf16>(
        &ctx,
        &stream,
        DType::BF16,
        RmsVariant::Qwen,
        3,
        64,
        1e-6,
        5e-2,
        "rms_norm_qwen_bf16_3x64",
        bf16::from_f32,
        |v| v.to_f32(),
    );
}

#[test]
fn rms_norm_gated_fused_f32() {
    let Some((ctx, stream)) = setup() else { return };
    run_gated_test::<f32>(
        &ctx,
        &stream,
        DType::F32,
        3,
        64,
        1e-6,
        1e-5,
        "rms_norm_gated_f32_3x64",
        |v| v,
        |v| v,
    );
}

#[test]
fn rms_norm_gated_fused_bf16() {
    let Some((ctx, stream)) = setup() else { return };
    run_gated_test::<bf16>(
        &ctx,
        &stream,
        DType::BF16,
        3,
        64,
        1e-6,
        5e-2,
        "rms_norm_gated_bf16_3x64",
        bf16::from_f32,
        |v| v.to_f32(),
    );
}
