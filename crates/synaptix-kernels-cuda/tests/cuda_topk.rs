#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};

use synaptix_kernels_cuda::reduction::topk::{
    run_bf16 as topk_bf16, run_f16 as topk_f16, run_f32 as topk_f32, TopkKernels,
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

fn cpu_topk(row: &[f32], k: usize) -> (Vec<f32>, Vec<i32>) {
    let mut paired: Vec<(f32, i32)> = row
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as i32))
        .collect();
    paired.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let vals: Vec<f32> = paired.iter().take(k).map(|&(v, _)| v).collect();
    let idx: Vec<i32> = paired.iter().take(k).map(|&(_, i)| i).collect();
    (vals, idx)
}

#[test]
fn topk_f32_small() {
    let Some((ctx, stream)) = setup() else { return };
    let k = TopkKernels::for_context(&ctx).expect("compile");
    let b = 2_u32;
    let v = 512_u32;
    let topk = 10_u32;
    let logits = det_f32(0xA0, (b * v) as usize, 3.0);

    let dev_l: CudaSlice<f32> = stream.clone_htod(&logits).unwrap();
    let mut out_v: CudaSlice<f32> = stream.alloc_zeros((b * topk) as usize).unwrap();
    let mut out_i: CudaSlice<i32> = stream.alloc_zeros((b * topk) as usize).unwrap();

    topk_f32(&k, &stream, &dev_l, &mut out_v, &mut out_i, b, v, topk).unwrap();
    stream.synchronize().unwrap();
    let gpu_v = stream.clone_dtoh(&out_v).unwrap();
    let gpu_i = stream.clone_dtoh(&out_i).unwrap();

    for bi in 0..b as usize {
        let row = &logits[bi * v as usize..(bi + 1) * v as usize];
        let (cpu_v, _) = cpu_topk(row, topk as usize);
        for j in 0..topk as usize {
            let off = bi * topk as usize + j;
            assert!(
                (gpu_v[off] - cpu_v[j]).abs() < 1e-5,
                "[{bi},{j}] val gpu={} cpu={}",
                gpu_v[off],
                cpu_v[j]
            );
            let gi = gpu_i[off] as usize;
            assert!(
                (row[gi] - gpu_v[off]).abs() < 1e-5,
                "[{bi},{j}] idx points wrong"
            );
        }
    }
}

#[test]
fn topk_f32_large_vocab() {
    let Some((ctx, stream)) = setup() else { return };
    let k = TopkKernels::for_context(&ctx).expect("compile");
    let b = 1_u32;
    let v = 152064_u32; // Qwen3 vocab ~152K
    let topk = 50_u32;
    let logits = det_f32(0xB0, (b * v) as usize, 5.0);

    let dev_l: CudaSlice<f32> = stream.clone_htod(&logits).unwrap();
    let mut out_v: CudaSlice<f32> = stream.alloc_zeros((b * topk) as usize).unwrap();
    let mut out_i: CudaSlice<i32> = stream.alloc_zeros((b * topk) as usize).unwrap();

    topk_f32(&k, &stream, &dev_l, &mut out_v, &mut out_i, b, v, topk).unwrap();
    stream.synchronize().unwrap();
    let gpu_v = stream.clone_dtoh(&out_v).unwrap();
    let gpu_i = stream.clone_dtoh(&out_i).unwrap();

    let row = &logits[..v as usize];
    let (cpu_v, _) = cpu_topk(row, topk as usize);
    for j in 0..topk as usize {
        assert!(
            (gpu_v[j] - cpu_v[j]).abs() < 1e-5,
            "[{j}] val gpu={} cpu={}",
            gpu_v[j],
            cpu_v[j]
        );
        let gi = gpu_i[j] as usize;
        assert!(gi < v as usize, "[{j}] idx out of range");
        assert!(
            (row[gi] - gpu_v[j]).abs() < 1e-5,
            "[{j}] idx points to wrong value"
        );
    }
    for j in 1..topk as usize {
        assert!(gpu_v[j] <= gpu_v[j - 1], "[{j}] not sorted descending");
    }
}

#[test]
fn topk_f16_vocab() {
    let Some((ctx, stream)) = setup() else { return };
    let k = TopkKernels::for_context(&ctx).expect("compile");
    let b = 2_u32;
    let v = 4096_u32;
    let topk = 32_u32;
    let logits_f32 = det_f32(0xC0, (b * v) as usize, 2.0);
    let logits: Vec<f16> = logits_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_l: CudaSlice<f16> = stream.clone_htod(&logits).unwrap();
    let mut out_v: CudaSlice<f32> = stream.alloc_zeros((b * topk) as usize).unwrap();
    let mut out_i: CudaSlice<i32> = stream.alloc_zeros((b * topk) as usize).unwrap();

    topk_f16(&k, &stream, &dev_l, &mut out_v, &mut out_i, b, v, topk).unwrap();
    stream.synchronize().unwrap();
    let gpu_v = stream.clone_dtoh(&out_v).unwrap();
    let gpu_i = stream.clone_dtoh(&out_i).unwrap();

    let logits_back: Vec<f32> = logits.iter().map(|v| v.to_f32()).collect();
    for bi in 0..b as usize {
        let row = &logits_back[bi * v as usize..(bi + 1) * v as usize];
        let (cpu_v, _) = cpu_topk(row, topk as usize);
        for j in 0..topk as usize {
            let off = bi * topk as usize + j;
            assert!(
                (gpu_v[off] - cpu_v[j]).abs() < 1e-3,
                "[{bi},{j}] val gpu={} cpu={}",
                gpu_v[off],
                cpu_v[j]
            );
            let gi = gpu_i[off] as usize;
            assert!(
                (row[gi] - gpu_v[off]).abs() < 1e-3,
                "[{bi},{j}] idx points wrong"
            );
        }
    }
}

#[test]
fn topk_bf16_vocab() {
    let Some((ctx, stream)) = setup() else { return };
    let k = TopkKernels::for_context(&ctx).expect("compile");
    let b = 1_u32;
    let v = 8192_u32;
    let topk = 16_u32;
    let logits_f32 = det_f32(0xD0, (b * v) as usize, 2.0);
    let logits: Vec<bf16> = logits_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_l: CudaSlice<bf16> = stream.clone_htod(&logits).unwrap();
    let mut out_v: CudaSlice<f32> = stream.alloc_zeros((b * topk) as usize).unwrap();
    let mut out_i: CudaSlice<i32> = stream.alloc_zeros((b * topk) as usize).unwrap();

    topk_bf16(&k, &stream, &dev_l, &mut out_v, &mut out_i, b, v, topk).unwrap();
    stream.synchronize().unwrap();
    let gpu_v = stream.clone_dtoh(&out_v).unwrap();
    let gpu_i = stream.clone_dtoh(&out_i).unwrap();

    let logits_back: Vec<f32> = logits.iter().map(|v| v.to_f32()).collect();
    let row = &logits_back[..v as usize];
    let (cpu_v, _) = cpu_topk(row, topk as usize);
    for j in 0..topk as usize {
        assert!(
            (gpu_v[j] - cpu_v[j]).abs() < 1e-2,
            "[{j}] val gpu={} cpu={}",
            gpu_v[j],
            cpu_v[j]
        );
        let gi = gpu_i[j] as usize;
        assert!((row[gi] - gpu_v[j]).abs() < 1e-2, "[{j}] idx points wrong");
    }
}
