
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};
use synaptix_kernels_cuda::fused::layernorm_residual::{
    layernorm_residual_bf16, layernorm_residual_f16, layernorm_residual_f32,
    LayerNormResidualKernels,
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

/// Возвращает (residual_out = x + residual, y = LN(residual_out)*gamma + beta).
fn cpu_ln_residual(
    x: &[f32],
    residual: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    batch: usize,
    hidden: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut r_out = vec![0.0_f32; batch * hidden];
    let mut y = vec![0.0_f32; batch * hidden];
    for b in 0..batch {
        let off = b * hidden;
        let mut sum = 0.0_f32;
        let mut sumsq = 0.0_f32;
        for t in 0..hidden {
            let v = x[off + t] + residual[off + t];
            r_out[off + t] = v;
            sum += v;
            sumsq += v * v;
        }
        let mean = sum / hidden as f32;
        let var = (sumsq / hidden as f32 - mean * mean).max(0.0);
        let inv = 1.0 / (var + eps).sqrt();
        for t in 0..hidden {
            let bt = beta.map(|b| b[t]).unwrap_or(0.0);
            y[off + t] = ((r_out[off + t] - mean) * inv) * gamma[t] + bt;
        }
    }
    (r_out, y)
}

#[test]
fn layernorm_residual_f32_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = LayerNormResidualKernels::for_context(&ctx).expect("compile ln_residual");
    let batch = 8usize;
    let hidden = 1024usize;
    let eps = 1e-5f32;
    let x = det_f32(0x5A1, batch * hidden, 0.5);
    let residual = det_f32(0x5B2, batch * hidden, 0.5);
    let gamma = det_f32(0x5C3, hidden, 0.3);
    let beta = det_f32(0x5D4, hidden, 0.2);
    let (r_exp, y_exp) = cpu_ln_residual(&x, &residual, &gamma, Some(&beta), batch, hidden, eps);

    let dev_x: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let mut dev_r: CudaSlice<f32> = stream.clone_htod(&residual).unwrap();
    let dev_g: CudaSlice<f32> = stream.clone_htod(&gamma).unwrap();
    let dev_b: CudaSlice<f32> = stream.clone_htod(&beta).unwrap();
    let mut dev_y: CudaSlice<f32> = stream.alloc_zeros(batch * hidden).unwrap();
    layernorm_residual_f32(
        &kernels,
        &stream,
        &dev_x,
        &mut dev_r,
        &dev_g,
        Some(&dev_b),
        &mut dev_y,
        batch as u32,
        hidden as u32,
        eps,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let r_got: Vec<f32> = stream.clone_dtoh(&dev_r).unwrap();
    let y_got: Vec<f32> = stream.clone_dtoh(&dev_y).unwrap();

    let mut max_r = 0.0_f32;
    let mut max_y = 0.0_f32;
    for i in 0..batch * hidden {
        max_r = max_r.max((r_got[i] - r_exp[i]).abs());
        max_y = max_y.max((y_got[i] - y_exp[i]).abs());
    }
    eprintln!("[ln_residual_f32 b={batch} h={hidden}] max_r={max_r:.6} max_y={max_y:.6}");
    assert!(max_r < 1e-5, "residual max_abs={max_r}");
    assert!(max_y < 1e-4, "y max_abs={max_y}");
}

#[test]
fn layernorm_residual_f32_no_beta() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = LayerNormResidualKernels::for_context(&ctx).expect("compile ln_residual");
    let batch = 4usize;
    let hidden = 768usize;
    let eps = 1e-5f32;
    let x = det_f32(0x6A1, batch * hidden, 0.5);
    let residual = det_f32(0x6B2, batch * hidden, 0.5);
    let gamma = det_f32(0x6C3, hidden, 0.3);
    let (_r_exp, y_exp) = cpu_ln_residual(&x, &residual, &gamma, None, batch, hidden, eps);

    let dev_x: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let mut dev_r: CudaSlice<f32> = stream.clone_htod(&residual).unwrap();
    let dev_g: CudaSlice<f32> = stream.clone_htod(&gamma).unwrap();
    let mut dev_y: CudaSlice<f32> = stream.alloc_zeros(batch * hidden).unwrap();
    layernorm_residual_f32(
        &kernels,
        &stream,
        &dev_x,
        &mut dev_r,
        &dev_g,
        None,
        &mut dev_y,
        batch as u32,
        hidden as u32,
        eps,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let y_got: Vec<f32> = stream.clone_dtoh(&dev_y).unwrap();
    let mut max_y = 0.0_f32;
    for i in 0..batch * hidden {
        max_y = max_y.max((y_got[i] - y_exp[i]).abs());
    }
    eprintln!("[ln_residual_f32 no_beta] max_y={max_y:.6}");
    assert!(max_y < 1e-4);
}

#[test]
fn layernorm_residual_f16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = LayerNormResidualKernels::for_context(&ctx).expect("compile ln_residual");
    let batch = 4usize;
    let hidden = 1024usize;
    let eps = 1e-5f32;
    let x_f = det_f32(0x7A1, batch * hidden, 0.5);
    let r_f = det_f32(0x7B2, batch * hidden, 0.5);
    let g_f = det_f32(0x7C3, hidden, 0.3);
    let b_f = det_f32(0x7D4, hidden, 0.2);
    let x: Vec<f16> = x_f.iter().map(|v| f16::from_f32(*v)).collect();
    let r: Vec<f16> = r_f.iter().map(|v| f16::from_f32(*v)).collect();
    let g: Vec<f16> = g_f.iter().map(|v| f16::from_f32(*v)).collect();
    let bt: Vec<f16> = b_f.iter().map(|v| f16::from_f32(*v)).collect();
    let x_b: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let r_b: Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    let g_b: Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bt_b: Vec<f32> = bt.iter().map(|v| v.to_f32()).collect();
    let (_r_exp, y_exp) = cpu_ln_residual(&x_b, &r_b, &g_b, Some(&bt_b), batch, hidden, eps);

    let dev_x: CudaSlice<f16> = stream.clone_htod(&x).unwrap();
    let mut dev_r: CudaSlice<f16> = stream.clone_htod(&r).unwrap();
    let dev_g: CudaSlice<f16> = stream.clone_htod(&g).unwrap();
    let dev_b: CudaSlice<f16> = stream.clone_htod(&bt).unwrap();
    let mut dev_y: CudaSlice<f16> = stream.alloc_zeros(batch * hidden).unwrap();
    layernorm_residual_f16(
        &kernels,
        &stream,
        &dev_x,
        &mut dev_r,
        &dev_g,
        Some(&dev_b),
        &mut dev_y,
        batch as u32,
        hidden as u32,
        eps,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let y_h: Vec<f16> = stream.clone_dtoh(&dev_y).unwrap();
    let y_got: Vec<f32> = y_h.iter().map(|v| v.to_f32()).collect();
    let mut max_y = 0.0_f32;
    for i in 0..batch * hidden {
        max_y = max_y.max((y_got[i] - y_exp[i]).abs());
    }
    eprintln!("[ln_residual_f16] max_y={max_y:.4}");
    assert!(max_y < 0.05, "y max_abs={max_y}");
}

#[test]
fn layernorm_residual_bf16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = LayerNormResidualKernels::for_context(&ctx).expect("compile ln_residual");
    let batch = 4usize;
    let hidden = 1024usize;
    let eps = 1e-5f32;
    let x_f = det_f32(0x8A1, batch * hidden, 0.5);
    let r_f = det_f32(0x8B2, batch * hidden, 0.5);
    let g_f = det_f32(0x8C3, hidden, 0.3);
    let b_f = det_f32(0x8D4, hidden, 0.2);
    let x: Vec<bf16> = x_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let r: Vec<bf16> = r_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let g: Vec<bf16> = g_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let bt: Vec<bf16> = b_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let x_b: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let r_b: Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
    let g_b: Vec<f32> = g.iter().map(|v| v.to_f32()).collect();
    let bt_b: Vec<f32> = bt.iter().map(|v| v.to_f32()).collect();
    let (_r_exp, y_exp) = cpu_ln_residual(&x_b, &r_b, &g_b, Some(&bt_b), batch, hidden, eps);

    let dev_x: CudaSlice<bf16> = stream.clone_htod(&x).unwrap();
    let mut dev_r: CudaSlice<bf16> = stream.clone_htod(&r).unwrap();
    let dev_g: CudaSlice<bf16> = stream.clone_htod(&g).unwrap();
    let dev_b: CudaSlice<bf16> = stream.clone_htod(&bt).unwrap();
    let mut dev_y: CudaSlice<bf16> = stream.alloc_zeros(batch * hidden).unwrap();
    layernorm_residual_bf16(
        &kernels,
        &stream,
        &dev_x,
        &mut dev_r,
        &dev_g,
        Some(&dev_b),
        &mut dev_y,
        batch as u32,
        hidden as u32,
        eps,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let y_b: Vec<bf16> = stream.clone_dtoh(&dev_y).unwrap();
    let y_got: Vec<f32> = y_b.iter().map(|v| v.to_f32()).collect();
    let mut max_y = 0.0_f32;
    for i in 0..batch * hidden {
        max_y = max_y.max((y_got[i] - y_exp[i]).abs());
    }
    eprintln!("[ln_residual_bf16] max_y={max_y:.4}");
    assert!(max_y < 0.3, "y max_abs={max_y}");
}
