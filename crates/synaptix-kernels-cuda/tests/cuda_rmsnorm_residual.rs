
use half::{bf16, f16};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use synaptix_core::dtype::DType;
use synaptix_kernels_cuda::fused::rmsnorm_residual::{run, RmsNormResidualKernels};
use synaptix_kernels_cuda::reduction::rmsnorm::RmsVariant;

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

fn cpu_rmsnorm_residual(
    x: &[f32],
    residual_in: &[f32],
    weight: &[f32],
    batch: usize,
    hidden: usize,
    eps: f32,
    variant: RmsVariant,
) -> (Vec<f32>, Vec<f32>) {
    let mut residual_out = vec![0.0_f32; batch * hidden];
    let mut y = vec![0.0_f32; batch * hidden];
    for b in 0..batch {
        let mut sum_sq = 0.0_f64;
        for i in 0..hidden {
            let v = x[b * hidden + i] + residual_in[b * hidden + i];
            residual_out[b * hidden + i] = v;
            sum_sq += (v as f64) * (v as f64);
        }
        let mean = (sum_sq / hidden as f64) as f32;
        let rms = 1.0_f32 / (mean + eps).sqrt();
        for i in 0..hidden {
            let scale = match variant {
                RmsVariant::Plain => weight[i],
                RmsVariant::Qwen => 1.0 + weight[i],
            };
            y[b * hidden + i] = scale * residual_out[b * hidden + i] * rms;
        }
    }
    (residual_out, y)
}

fn run_test<T: DeviceRepr + ValidAsZeroBits + bytemuck::Pod + Copy>(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    dtype: DType,
    variant: RmsVariant,
    batch: usize,
    hidden: usize,
    tol: f32,
    name: &str,
    to_t: fn(f32) -> T,
    to_f32: fn(T) -> f32,
) {
    let x_f32 = det_f32(0xA110, batch * hidden, 0.5);
    let r_f32 = det_f32(0xB220, batch * hidden, 0.4);
    let w_f32 = det_f32(0xC330, hidden, 0.3);
    let x_t: Vec<T> = x_f32.iter().map(|v| to_t(*v)).collect();
    let r_t: Vec<T> = r_f32.iter().map(|v| to_t(*v)).collect();
    let w_t: Vec<T> = w_f32.iter().map(|v| to_t(*v)).collect();
    let x_back: Vec<f32> = x_t.iter().map(|v| to_f32(*v)).collect();
    let r_back: Vec<f32> = r_t.iter().map(|v| to_f32(*v)).collect();
    let w_back: Vec<f32> = w_t.iter().map(|v| to_f32(*v)).collect();
    let (res_ref, y_ref) =
        cpu_rmsnorm_residual(&x_back, &r_back, &w_back, batch, hidden, 1e-6, variant);

    let kernels = RmsNormResidualKernels::for_context(ctx).expect("compile");
    let dev_x: CudaSlice<T> = stream.clone_htod(&x_t).unwrap();
    let mut dev_r: CudaSlice<T> = stream.clone_htod(&r_t).unwrap();
    let dev_w: CudaSlice<T> = stream.clone_htod(&w_t).unwrap();
    let mut dev_y: CudaSlice<T> = stream.alloc_zeros(batch * hidden).unwrap();
    run::<T>(
        &kernels,
        stream,
        &dev_x,
        &mut dev_r,
        &dev_w,
        &mut dev_y,
        batch as u32,
        hidden as u32,
        1e-6,
        variant,
        dtype,
    )
    .unwrap();
    stream.synchronize().unwrap();

    let r_got_t: Vec<T> = stream.clone_dtoh(&dev_r).unwrap();
    let y_got_t: Vec<T> = stream.clone_dtoh(&dev_y).unwrap();
    let r_got: Vec<f32> = r_got_t.iter().map(|v| to_f32(*v)).collect();
    let y_got: Vec<f32> = y_got_t.iter().map(|v| to_f32(*v)).collect();
    let mut r_max = 0.0_f32;
    let mut y_max = 0.0_f32;
    for i in 0..batch * hidden {
        r_max = r_max.max((r_got[i] - res_ref[i]).abs());
        y_max = y_max.max((y_got[i] - y_ref[i]).abs());
    }
    eprintln!("[{name} {batch}x{hidden}] r_max={r_max:.6} y_max={y_max:.6}");
    assert!(r_max < tol, "{name}: residual max_err={r_max}");
    assert!(y_max < tol, "{name}: y max_err={y_max}");
}

#[test]
fn fused_rmsnorm_residual_f32_plain() {
    let Some((ctx, stream)) = setup() else { return };
    run_test::<f32>(
        &ctx,
        &stream,
        DType::F32,
        RmsVariant::Plain,
        4,
        128,
        1e-5,
        "f32_plain_4x128",
        |v| v,
        |v| v,
    );
}

#[test]
fn fused_rmsnorm_residual_f32_qwen() {
    let Some((ctx, stream)) = setup() else { return };
    run_test::<f32>(
        &ctx,
        &stream,
        DType::F32,
        RmsVariant::Qwen,
        3,
        64,
        1e-5,
        "f32_qwen_3x64",
        |v| v,
        |v| v,
    );
}

#[test]
fn fused_rmsnorm_residual_f16() {
    let Some((ctx, stream)) = setup() else { return };
    run_test::<f16>(
        &ctx,
        &stream,
        DType::F16,
        RmsVariant::Plain,
        4,
        128,
        5e-3,
        "f16_plain_4x128",
        f16::from_f32,
        |v| v.to_f32(),
    );
}

#[test]
fn fused_rmsnorm_residual_bf16() {
    let Some((ctx, stream)) = setup() else { return };
    run_test::<bf16>(
        &ctx,
        &stream,
        DType::BF16,
        RmsVariant::Plain,
        4,
        128,
        5e-2,
        "bf16_plain_4x128",
        bf16::from_f32,
        |v| v.to_f32(),
    );
}
