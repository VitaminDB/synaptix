
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};

use synaptix_kernels_cuda::fused::cross_entropy::{
    run_bf16 as ce_bf16, run_f16 as ce_f16, run_f32 as ce_f32, CrossEntropyKernels,
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

fn cpu_ref(logits: &[f32], targets: &[i32], b: usize, v: usize, ignore: i32) -> Vec<f32> {
    let mut out = vec![0.0_f32; b];
    for bi in 0..b {
        let row = &logits[bi * v..(bi + 1) * v];
        let tgt = targets[bi];
        if tgt == ignore {
            out[bi] = 0.0;
            continue;
        }
        let mut m = f32::NEG_INFINITY;
        for &x in row {
            if x > m {
                m = x;
            }
        }
        let mut s = 0.0_f64;
        for &x in row {
            s += ((x - m) as f64).exp();
        }
        let lse = m + (s.ln() as f32);
        out[bi] = lse - row[tgt as usize];
    }
    out
}

#[test]
fn ce_f32_basic() {
    let Some((ctx, stream)) = setup() else { return };
    let ce = CrossEntropyKernels::for_context(&ctx).expect("compile");
    let b = 4_u32;
    let v = 512_u32;
    let logits = det_f32(0x1234, (b * v) as usize, 3.0);
    let targets: Vec<i32> = (0..b as i32).map(|i| (i * 73 + 5) % (v as i32)).collect();

    let dev_logits: CudaSlice<f32> = stream.clone_htod(&logits).unwrap();
    let dev_targets: CudaSlice<i32> = stream.clone_htod(&targets).unwrap();
    let mut losses: CudaSlice<f32> = stream.alloc_zeros(b as usize).unwrap();

    ce_f32(
        &ce,
        &stream,
        &dev_logits,
        &dev_targets,
        &mut losses,
        b,
        v,
        -100,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gpu = stream.clone_dtoh(&losses).unwrap();
    let cpu = cpu_ref(&logits, &targets, b as usize, v as usize, -100);
    for i in 0..b as usize {
        let d = (gpu[i] - cpu[i]).abs();
        assert!(d < 1e-4, "[{i}] gpu={} cpu={} diff={d}", gpu[i], cpu[i]);
    }
}

#[test]
fn ce_f32_large_vocab() {
    let Some((ctx, stream)) = setup() else { return };
    let ce = CrossEntropyKernels::for_context(&ctx).expect("compile");
    let b = 2_u32;
    let v = 32000_u32;
    let logits = det_f32(0xC0FE, (b * v) as usize, 4.0);
    let targets: Vec<i32> = (0..b as i32).map(|i| (i * 12345) % (v as i32)).collect();

    let dev_logits: CudaSlice<f32> = stream.clone_htod(&logits).unwrap();
    let dev_targets: CudaSlice<i32> = stream.clone_htod(&targets).unwrap();
    let mut losses: CudaSlice<f32> = stream.alloc_zeros(b as usize).unwrap();

    ce_f32(
        &ce,
        &stream,
        &dev_logits,
        &dev_targets,
        &mut losses,
        b,
        v,
        -100,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gpu = stream.clone_dtoh(&losses).unwrap();
    let cpu = cpu_ref(&logits, &targets, b as usize, v as usize, -100);
    for i in 0..b as usize {
        let d = (gpu[i] - cpu[i]).abs();
        let rel = d / cpu[i].abs().max(1e-6);
        assert!(rel < 1e-4, "[{i}] gpu={} cpu={} rel={rel}", gpu[i], cpu[i]);
    }
}

#[test]
fn ce_f16_basic() {
    let Some((ctx, stream)) = setup() else { return };
    let ce = CrossEntropyKernels::for_context(&ctx).expect("compile");
    let b = 3_u32;
    let v = 1024_u32;
    let logits_f32 = det_f32(0xBEEF, (b * v) as usize, 2.0);
    let logits_f16: Vec<f16> = logits_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let targets: Vec<i32> = (0..b as i32).map(|i| (i * 211 + 3) % (v as i32)).collect();

    let dev_logits: CudaSlice<f16> = stream.clone_htod(&logits_f16).unwrap();
    let dev_targets: CudaSlice<i32> = stream.clone_htod(&targets).unwrap();
    let mut losses: CudaSlice<f32> = stream.alloc_zeros(b as usize).unwrap();

    ce_f16(
        &ce,
        &stream,
        &dev_logits,
        &dev_targets,
        &mut losses,
        b,
        v,
        -100,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gpu = stream.clone_dtoh(&losses).unwrap();
    let logits_back: Vec<f32> = logits_f16.iter().map(|v| v.to_f32()).collect();
    let cpu = cpu_ref(&logits_back, &targets, b as usize, v as usize, -100);
    for i in 0..b as usize {
        let d = (gpu[i] - cpu[i]).abs();
        let rel = d / cpu[i].abs().max(1e-3);
        assert!(rel < 5e-3, "[{i}] gpu={} cpu={} rel={rel}", gpu[i], cpu[i]);
    }
}

#[test]
fn ce_bf16_basic() {
    let Some((ctx, stream)) = setup() else { return };
    let ce = CrossEntropyKernels::for_context(&ctx).expect("compile");
    let b = 3_u32;
    let v = 1024_u32;
    let logits_f32 = det_f32(0xCAFE, (b * v) as usize, 2.0);
    let logits_bf16: Vec<bf16> = logits_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let targets: Vec<i32> = (0..b as i32).map(|i| (i * 89 + 17) % (v as i32)).collect();

    let dev_logits: CudaSlice<bf16> = stream.clone_htod(&logits_bf16).unwrap();
    let dev_targets: CudaSlice<i32> = stream.clone_htod(&targets).unwrap();
    let mut losses: CudaSlice<f32> = stream.alloc_zeros(b as usize).unwrap();

    ce_bf16(
        &ce,
        &stream,
        &dev_logits,
        &dev_targets,
        &mut losses,
        b,
        v,
        -100,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gpu = stream.clone_dtoh(&losses).unwrap();
    let logits_back: Vec<f32> = logits_bf16.iter().map(|v| v.to_f32()).collect();
    let cpu = cpu_ref(&logits_back, &targets, b as usize, v as usize, -100);
    for i in 0..b as usize {
        let d = (gpu[i] - cpu[i]).abs();
        let rel = d / cpu[i].abs().max(1e-3);
        assert!(rel < 2e-2, "[{i}] gpu={} cpu={} rel={rel}", gpu[i], cpu[i]);
    }
}

#[test]
fn ce_ignore_index() {
    let Some((ctx, stream)) = setup() else { return };
    let ce = CrossEntropyKernels::for_context(&ctx).expect("compile");
    let b = 4_u32;
    let v = 128_u32;
    let logits = det_f32(0xCEDE, (b * v) as usize, 2.0);
    let targets: Vec<i32> = vec![3, -100, 7, 0];

    let dev_logits: CudaSlice<f32> = stream.clone_htod(&logits).unwrap();
    let dev_targets: CudaSlice<i32> = stream.clone_htod(&targets).unwrap();
    let mut losses: CudaSlice<f32> = stream.alloc_zeros(b as usize).unwrap();

    ce_f32(
        &ce,
        &stream,
        &dev_logits,
        &dev_targets,
        &mut losses,
        b,
        v,
        -100,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gpu = stream.clone_dtoh(&losses).unwrap();
    assert_eq!(gpu[1], 0.0, "ignored sample should be 0");
    let cpu = cpu_ref(&logits, &targets, b as usize, v as usize, -100);
    for i in [0_usize, 2, 3] {
        let d = (gpu[i] - cpu[i]).abs();
        assert!(d < 1e-4, "[{i}] gpu={} cpu={}", gpu[i], cpu[i]);
    }
}
