#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};

use synaptix_core::dtype::DType;
use synaptix_kernels_cuda::reduction::layernorm::{
    run_bf16 as ln_bf16, run_f16 as ln_f16, run_f32 as ln_f32, run_u8 as ln_u8, LayerNormKernels,
};

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det_f32(seed: u64, n: usize, scale: f32, mean_shift: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            f * scale + mean_shift
        })
        .collect()
}

fn cpu_ref(
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    batch: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; batch * hidden];
    for bi in 0..batch {
        let row = &x[bi * hidden..(bi + 1) * hidden];
        let n = hidden as f64;
        let mut sum = 0.0_f64;
        let mut sumsq = 0.0_f64;
        for &v in row {
            sum += v as f64;
            sumsq += (v as f64) * (v as f64);
        }
        let mean = sum / n;
        let var = (sumsq / n - mean * mean).max(0.0);
        let inv_std = 1.0 / (var + eps as f64).sqrt();
        for t in 0..hidden {
            let n_v = ((row[t] as f64 - mean) * inv_std) as f32 * w[t];
            out[bi * hidden + t] = match b {
                Some(bv) => n_v + bv[t],
                None => n_v,
            };
        }
    }
    out
}

#[test]
fn ln_f32_with_beta() {
    let Some((ctx, stream)) = setup() else { return };
    let k = LayerNormKernels::for_context(&ctx).expect("compile");
    let batch = 4_u32;
    let hidden = 1024_u32;
    let x = det_f32(0x11, (batch * hidden) as usize, 1.0, 0.5);
    let w = det_f32(0x22, hidden as usize, 0.3, 1.0);
    let b = det_f32(0x33, hidden as usize, 0.1, 0.0);

    let dev_x: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let dev_w: CudaSlice<f32> = stream.clone_htod(&w).unwrap();
    let dev_b: CudaSlice<f32> = stream.clone_htod(&b).unwrap();
    let mut dev_y: CudaSlice<f32> = stream.alloc_zeros((batch * hidden) as usize).unwrap();

    ln_f32(
        &k,
        &stream,
        &dev_x,
        &dev_w,
        Some(&dev_b),
        &mut dev_y,
        batch,
        hidden,
        1e-5,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gpu = stream.clone_dtoh(&dev_y).unwrap();
    let cpu = cpu_ref(&x, &w, Some(&b), batch as usize, hidden as usize, 1e-5);
    for i in 0..gpu.len() {
        let d = (gpu[i] - cpu[i]).abs();
        assert!(d < 1e-4, "[{i}] gpu={} cpu={} diff={d}", gpu[i], cpu[i]);
    }
}

#[test]
fn ln_f32_no_beta() {
    let Some((ctx, stream)) = setup() else { return };
    let k = LayerNormKernels::for_context(&ctx).expect("compile");
    let batch = 2_u32;
    let hidden = 512_u32;
    let x = det_f32(0x44, (batch * hidden) as usize, 2.0, -0.3);
    let w = det_f32(0x55, hidden as usize, 0.5, 1.0);

    let dev_x: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let dev_w: CudaSlice<f32> = stream.clone_htod(&w).unwrap();
    let mut dev_y: CudaSlice<f32> = stream.alloc_zeros((batch * hidden) as usize).unwrap();

    ln_f32(
        &k, &stream, &dev_x, &dev_w, None, &mut dev_y, batch, hidden, 1e-5,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gpu = stream.clone_dtoh(&dev_y).unwrap();
    let cpu = cpu_ref(&x, &w, None, batch as usize, hidden as usize, 1e-5);
    for i in 0..gpu.len() {
        let d = (gpu[i] - cpu[i]).abs();
        assert!(d < 1e-4, "[{i}] gpu={} cpu={}", gpu[i], cpu[i]);
    }
}

#[test]
fn ln_f16_dit_shape() {
    let Some((ctx, stream)) = setup() else { return };
    let k = LayerNormKernels::for_context(&ctx).expect("compile");
    let batch = 4_u32;
    let hidden = 2048_u32;
    let x_f32 = det_f32(0x66, (batch * hidden) as usize, 0.5, 0.0);
    let w_f32 = det_f32(0x77, hidden as usize, 0.2, 1.0);
    let b_f32 = det_f32(0x88, hidden as usize, 0.05, 0.0);
    let x: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let w: Vec<f16> = w_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let b: Vec<f16> = b_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_x: CudaSlice<f16> = stream.clone_htod(&x).unwrap();
    let dev_w: CudaSlice<f16> = stream.clone_htod(&w).unwrap();
    let dev_b: CudaSlice<f16> = stream.clone_htod(&b).unwrap();
    let mut dev_y: CudaSlice<f16> = stream.alloc_zeros((batch * hidden) as usize).unwrap();

    ln_f16(
        &k,
        &stream,
        &dev_x,
        &dev_w,
        Some(&dev_b),
        &mut dev_y,
        batch,
        hidden,
        1e-5,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gpu_f16 = stream.clone_dtoh(&dev_y).unwrap();
    let gpu: Vec<f32> = gpu_f16.iter().map(|v| v.to_f32()).collect();
    let x_back: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let w_back: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
    let b_back: Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let cpu = cpu_ref(
        &x_back,
        &w_back,
        Some(&b_back),
        batch as usize,
        hidden as usize,
        1e-5,
    );
    for i in 0..gpu.len() {
        let d = (gpu[i] - cpu[i]).abs();
        assert!(d < 5e-3, "[{i}] gpu={} cpu={}", gpu[i], cpu[i]);
    }
}

// run_u8 (путь Backend::layer_norm): byte-offset по x + bias. Skip первой строки
// через x_off, сравнение с cpu_ref на rows[1..].
#[test]
fn ln_u8_f16_offset() {
    let Some((ctx, stream)) = setup() else { return };
    let k = LayerNormKernels::for_context(&ctx).expect("compile");
    let (rows_total, batch, hidden) = (3usize, 2u32, 1280u32);
    let x_f32 = det_f32(0xC0, rows_total * hidden as usize, 0.7, 0.2);
    let w_f32 = det_f32(0xD0, hidden as usize, 0.3, 1.0);
    let b_f32 = det_f32(0xE0, hidden as usize, 0.1, 0.0);
    let to_bytes = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|&f| f16::from_f32(f).to_le_bytes())
            .collect()
    };
    let dev_x: CudaSlice<u8> = stream.clone_htod(&to_bytes(&x_f32)).unwrap();
    let dev_w: CudaSlice<u8> = stream.clone_htod(&to_bytes(&w_f32)).unwrap();
    let dev_b: CudaSlice<u8> = stream.clone_htod(&to_bytes(&b_f32)).unwrap();
    let mut dev_y: CudaSlice<u8> = stream.alloc_zeros((batch * hidden) as usize * 2).unwrap();

    let x_off = hidden as usize * 2; // skip первую строку (f16 = 2 байта)
    ln_u8(
        &k,
        &stream,
        &dev_x,
        x_off,
        &dev_w,
        0,
        Some((&dev_b, 0)),
        &mut dev_y,
        0,
        batch,
        hidden,
        1e-5,
        DType::F16,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let y_bytes = stream.clone_dtoh(&dev_y).unwrap();
    let gpu: Vec<f32> = y_bytes
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();
    let x_skip: Vec<f32> = x_f32[hidden as usize..]
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    let w_b: Vec<f32> = w_f32.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let b_b: Vec<f32> = b_f32.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let cpu = cpu_ref(
        &x_skip,
        &w_b,
        Some(&b_b),
        batch as usize,
        hidden as usize,
        1e-5,
    );
    for i in 0..gpu.len() {
        let d = (gpu[i] - cpu[i]).abs();
        assert!(d < 5e-3, "[{i}] gpu={} cpu={}", gpu[i], cpu[i]);
    }
}

#[test]
fn ln_bf16_dit_shape() {
    let Some((ctx, stream)) = setup() else { return };
    let k = LayerNormKernels::for_context(&ctx).expect("compile");
    let batch = 3_u32;
    let hidden = 2048_u32;
    let x_f32 = det_f32(0x99, (batch * hidden) as usize, 0.5, 0.0);
    let w_f32 = det_f32(0xAA, hidden as usize, 0.2, 1.0);
    let b_f32 = det_f32(0xBB, hidden as usize, 0.05, 0.0);
    let x: Vec<bf16> = x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let w: Vec<bf16> = w_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let b: Vec<bf16> = b_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_x: CudaSlice<bf16> = stream.clone_htod(&x).unwrap();
    let dev_w: CudaSlice<bf16> = stream.clone_htod(&w).unwrap();
    let dev_b: CudaSlice<bf16> = stream.clone_htod(&b).unwrap();
    let mut dev_y: CudaSlice<bf16> = stream.alloc_zeros((batch * hidden) as usize).unwrap();

    ln_bf16(
        &k,
        &stream,
        &dev_x,
        &dev_w,
        Some(&dev_b),
        &mut dev_y,
        batch,
        hidden,
        1e-5,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gpu_bf16 = stream.clone_dtoh(&dev_y).unwrap();
    let gpu: Vec<f32> = gpu_bf16.iter().map(|v| v.to_f32()).collect();
    let x_back: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let w_back: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
    let b_back: Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    let cpu = cpu_ref(
        &x_back,
        &w_back,
        Some(&b_back),
        batch as usize,
        hidden as usize,
        1e-5,
    );
    for i in 0..gpu.len() {
        let d = (gpu[i] - cpu[i]).abs();
        assert!(d < 5e-2, "[{i}] gpu={} cpu={}", gpu[i], cpu[i]);
    }
}
