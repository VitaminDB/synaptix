
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};

use synaptix_kernels_cuda::scan::parallel_scan::{
    run_bf16 as scan_bf16, run_f16 as scan_f16, run_f32 as scan_f32, ParallelScanKernels,
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

fn cpu_scan(x: &[f32], inclusive: bool) -> Vec<f32> {
    let mut out = vec![0.0_f32; x.len()];
    let mut s = 0.0_f64;
    for i in 0..x.len() {
        if inclusive {
            s += x[i] as f64;
            out[i] = s as f32;
        } else {
            out[i] = s as f32;
            s += x[i] as f64;
        }
    }
    out
}

fn run_test(n: u32, b: u32, inclusive: bool, dtype_label: &str) {
    let Some((ctx, stream)) = setup() else { return };
    let k = ParallelScanKernels::for_context(&ctx).expect("compile");
    let x_f32 = det_f32(0x1100 + n as u64, (b * n) as usize, 0.5);

    match dtype_label {
        "f32" => {
            let dev_x: CudaSlice<f32> = stream.clone_htod(&x_f32).unwrap();
            let mut dev_y: CudaSlice<f32> = stream.alloc_zeros((b * n) as usize).unwrap();
            scan_f32(&k, &stream, &dev_x, &mut dev_y, b, n, inclusive).unwrap();
            stream.synchronize().unwrap();
            let gpu = stream.clone_dtoh(&dev_y).unwrap();
            for bi in 0..b as usize {
                let row = &x_f32[bi * n as usize..(bi + 1) * n as usize];
                let cpu = cpu_scan(row, inclusive);
                for i in 0..n as usize {
                    let off = bi * n as usize + i;
                    let d = (gpu[off] - cpu[i]).abs();
                    let tol = (cpu[i].abs() * 1e-5).max(1e-4);
                    assert!(
                        d < tol,
                        "[{bi},{i}] gpu={} cpu={} diff={d}",
                        gpu[off],
                        cpu[i]
                    );
                }
            }
        }
        "f16" => {
            let x: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
            let dev_x: CudaSlice<f16> = stream.clone_htod(&x).unwrap();
            let mut dev_y: CudaSlice<f16> = stream.alloc_zeros((b * n) as usize).unwrap();
            scan_f16(&k, &stream, &dev_x, &mut dev_y, b, n, inclusive).unwrap();
            stream.synchronize().unwrap();
            let gpu = stream.clone_dtoh(&dev_y).unwrap();
            let x_back: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
            for bi in 0..b as usize {
                let row = &x_back[bi * n as usize..(bi + 1) * n as usize];
                let cpu = cpu_scan(row, inclusive);
                for i in 0..n as usize {
                    let off = bi * n as usize + i;
                    let g = gpu[off].to_f32();
                    let d = (g - cpu[i]).abs();
                    let tol = (cpu[i].abs() * 5e-3).max(5e-2);
                    assert!(d < tol, "[{bi},{i}] gpu={g} cpu={} diff={d}", cpu[i]);
                }
            }
        }
        "bf16" => {
            let x: Vec<bf16> = x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
            let dev_x: CudaSlice<bf16> = stream.clone_htod(&x).unwrap();
            let mut dev_y: CudaSlice<bf16> = stream.alloc_zeros((b * n) as usize).unwrap();
            scan_bf16(&k, &stream, &dev_x, &mut dev_y, b, n, inclusive).unwrap();
            stream.synchronize().unwrap();
            let gpu = stream.clone_dtoh(&dev_y).unwrap();
            let x_back: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
            for bi in 0..b as usize {
                let row = &x_back[bi * n as usize..(bi + 1) * n as usize];
                let cpu = cpu_scan(row, inclusive);
                for i in 0..n as usize {
                    let off = bi * n as usize + i;
                    let g = gpu[off].to_f32();
                    let d = (g - cpu[i]).abs();
                    let tol = (cpu[i].abs() * 2e-2).max(0.1);
                    assert!(d < tol, "[{bi},{i}] gpu={g} cpu={} diff={d}", cpu[i]);
                }
            }
        }
        _ => unreachable!(),
    }
}

#[test]
fn scan_f32_inclusive_small() {
    run_test(128, 4, true, "f32");
}
#[test]
fn scan_f32_exclusive_small() {
    run_test(128, 4, false, "f32");
}
#[test]
fn scan_f32_inclusive_512() {
    run_test(512, 2, true, "f32");
}
#[test]
fn scan_f32_inclusive_2048() {
    run_test(2048, 2, true, "f32");
}
#[test]
fn scan_f32_inclusive_8192() {
    run_test(8192, 1, true, "f32");
}
#[test]
fn scan_f16_inclusive_1024() {
    run_test(1024, 2, true, "f16");
}
#[test]
fn scan_bf16_exclusive_1024() {
    run_test(1024, 2, false, "bf16");
}
