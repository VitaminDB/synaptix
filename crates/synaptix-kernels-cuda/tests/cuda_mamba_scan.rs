#![cfg(feature = "cuda")]

use half::{bf16, f16};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use synaptix_kernels_cuda::ssm::mamba_scan::{scan_bf16, scan_f16, scan_f32, MambaScanKernels};

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det_f32(seed: u64, n: usize, scale: f32, offset: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            f * scale + offset
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn cpu_mamba_scan(
    u: &[f32],
    delta: &[f32],
    a: &[f32],
    b_in: &[f32],
    c_in: &[f32],
    d_skip: Option<&[f32]>,
    b: usize,
    l: usize,
    d: usize,
    n: usize,
) -> Vec<f32> {
    let mut y = vec![0.0_f32; b * l * d];
    let mut h = vec![0.0_f32; b * d * n];
    for bi in 0..b {
        for t in 0..l {
            for di in 0..d {
                let u_off = (bi * l + t) * d + di;
                let u_t = u[u_off];
                let delta_t = delta[u_off];
                let mut y_val = 0.0_f32;
                for ni in 0..n {
                    let bc_off = (bi * l + t) * n + ni;
                    let b_tn = b_in[bc_off];
                    let c_tn = c_in[bc_off];
                    let a_dn = a[di * n + ni];
                    let delta_a = (a_dn * delta_t).exp();
                    let delta_b = delta_t * b_tn;
                    let h_idx = (bi * d + di) * n + ni;
                    h[h_idx] = delta_a * h[h_idx] + delta_b * u_t;
                    y_val += c_tn * h[h_idx];
                }
                if let Some(ds) = d_skip {
                    y_val += ds[di] * u_t;
                }
                y[u_off] = y_val;
            }
        }
    }
    y
}

#[test]
fn mamba_scan_f32_basic() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = MambaScanKernels::for_context(&ctx).expect("compile mamba_scan");
    let b = 1usize;
    let l = 8usize;
    let d = 4usize;
    let n = 16usize;
    let u_host = det_f32(0xA110, b * l * d, 0.5, 0.0);
    // delta > 0 (softplus output)
    let delta_host: Vec<f32> = det_f32(0xB220, b * l * d, 0.3, 0.5);
    // A < 0 (initialized as -1 .. -2)
    let a_host: Vec<f32> = det_f32(0xCC33, d * n, 0.5, -1.5);
    let b_in_host = det_f32(0xD444, b * l * n, 0.5, 0.0);
    let c_in_host = det_f32(0xE555, b * l * n, 0.5, 0.0);
    let d_skip_host = det_f32(0xF666, d, 0.3, 0.0);
    let expected = cpu_mamba_scan(
        &u_host,
        &delta_host,
        &a_host,
        &b_in_host,
        &c_in_host,
        Some(&d_skip_host),
        b,
        l,
        d,
        n,
    );

    let dev_u: CudaSlice<f32> = stream.clone_htod(&u_host).unwrap();
    let dev_delta: CudaSlice<f32> = stream.clone_htod(&delta_host).unwrap();
    let dev_a: CudaSlice<f32> = stream.clone_htod(&a_host).unwrap();
    let dev_b_in: CudaSlice<f32> = stream.clone_htod(&b_in_host).unwrap();
    let dev_c_in: CudaSlice<f32> = stream.clone_htod(&c_in_host).unwrap();
    let dev_d_skip: CudaSlice<f32> = stream.clone_htod(&d_skip_host).unwrap();
    let mut dev_y: CudaSlice<f32> = stream.alloc_zeros(b * l * d).unwrap();
    scan_f32(
        &kernels,
        &stream,
        &dev_u,
        &dev_delta,
        &dev_a,
        &dev_b_in,
        &dev_c_in,
        Some(&dev_d_skip),
        &mut dev_y,
        b as u32,
        l as u32,
        d as u32,
        n as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dev_y).unwrap();
    let mut max_abs = 0.0_f32;
    let mut worst = 0usize;
    for i in 0..got.len() {
        let diff = (got[i] - expected[i]).abs();
        if diff > max_abs {
            max_abs = diff;
            worst = i;
        }
    }
    eprintln!("[mamba_scan_f32 B={b} L={l} D={d} N={n}] max_abs={max_abs:.6} (worst i={worst} got={} ref={})",
        got[worst], expected[worst]);
    assert!(max_abs < 1e-4, "max_abs={max_abs}");
}

#[test]
fn mamba_scan_f32_no_skip() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = MambaScanKernels::for_context(&ctx).expect("compile mamba_scan");
    let b = 2usize;
    let l = 16usize;
    let d = 8usize;
    let n = 8usize;
    let u_host = det_f32(0x1A11, b * l * d, 0.5, 0.0);
    let delta_host: Vec<f32> = det_f32(0x2B22, b * l * d, 0.3, 0.5);
    let a_host: Vec<f32> = det_f32(0x3C33, d * n, 0.5, -1.5);
    let b_in_host = det_f32(0x4D44, b * l * n, 0.5, 0.0);
    let c_in_host = det_f32(0x5E55, b * l * n, 0.5, 0.0);
    let expected = cpu_mamba_scan(
        &u_host,
        &delta_host,
        &a_host,
        &b_in_host,
        &c_in_host,
        None,
        b,
        l,
        d,
        n,
    );

    let dev_u: CudaSlice<f32> = stream.clone_htod(&u_host).unwrap();
    let dev_delta: CudaSlice<f32> = stream.clone_htod(&delta_host).unwrap();
    let dev_a: CudaSlice<f32> = stream.clone_htod(&a_host).unwrap();
    let dev_b_in: CudaSlice<f32> = stream.clone_htod(&b_in_host).unwrap();
    let dev_c_in: CudaSlice<f32> = stream.clone_htod(&c_in_host).unwrap();
    let mut dev_y: CudaSlice<f32> = stream.alloc_zeros(b * l * d).unwrap();
    scan_f32(
        &kernels, &stream, &dev_u, &dev_delta, &dev_a, &dev_b_in, &dev_c_in, None, &mut dev_y,
        b as u32, l as u32, d as u32, n as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dev_y).unwrap();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[mamba_scan_f32 no_skip] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-4);
}

#[test]
fn mamba_scan_f16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = MambaScanKernels::for_context(&ctx).expect("compile mamba_scan");
    let b = 1usize;
    let l = 8usize;
    let d = 4usize;
    let n = 16usize;
    let u_f = det_f32(0xA110, b * l * d, 0.4, 0.0);
    let delta_f = det_f32(0xB220, b * l * d, 0.2, 0.5);
    let a_f = det_f32(0xCC33, d * n, 0.3, -1.0);
    let b_in_f = det_f32(0xD444, b * l * n, 0.4, 0.0);
    let c_in_f = det_f32(0xE555, b * l * n, 0.4, 0.0);

    let u_h: Vec<f16> = u_f.iter().map(|v| f16::from_f32(*v)).collect();
    let delta_h: Vec<f16> = delta_f.iter().map(|v| f16::from_f32(*v)).collect();
    let a_h: Vec<f16> = a_f.iter().map(|v| f16::from_f32(*v)).collect();
    let b_in_h: Vec<f16> = b_in_f.iter().map(|v| f16::from_f32(*v)).collect();
    let c_in_h: Vec<f16> = c_in_f.iter().map(|v| f16::from_f32(*v)).collect();

    let u_back: Vec<f32> = u_h.iter().map(|v| v.to_f32()).collect();
    let delta_back: Vec<f32> = delta_h.iter().map(|v| v.to_f32()).collect();
    let a_back: Vec<f32> = a_h.iter().map(|v| v.to_f32()).collect();
    let b_back: Vec<f32> = b_in_h.iter().map(|v| v.to_f32()).collect();
    let c_back: Vec<f32> = c_in_h.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_mamba_scan(
        &u_back,
        &delta_back,
        &a_back,
        &b_back,
        &c_back,
        None,
        b,
        l,
        d,
        n,
    );

    let dev_u: CudaSlice<f16> = stream.clone_htod(&u_h).unwrap();
    let dev_delta: CudaSlice<f16> = stream.clone_htod(&delta_h).unwrap();
    let dev_a: CudaSlice<f16> = stream.clone_htod(&a_h).unwrap();
    let dev_b_in: CudaSlice<f16> = stream.clone_htod(&b_in_h).unwrap();
    let dev_c_in: CudaSlice<f16> = stream.clone_htod(&c_in_h).unwrap();
    let mut dev_y: CudaSlice<f16> = stream.alloc_zeros(b * l * d).unwrap();
    scan_f16(
        &kernels, &stream, &dev_u, &dev_delta, &dev_a, &dev_b_in, &dev_c_in, None, &mut dev_y,
        b as u32, l as u32, d as u32, n as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got_h: Vec<f16> = stream.clone_dtoh(&dev_y).unwrap();
    let got: Vec<f32> = got_h.iter().map(|v| v.to_f32()).collect();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[mamba_scan_f16] max_abs={max_abs:.4}");
    assert!(max_abs < 0.1, "max_abs={max_abs}");
}

#[test]
fn mamba_scan_bf16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = MambaScanKernels::for_context(&ctx).expect("compile mamba_scan");
    let b = 1usize;
    let l = 8usize;
    let d = 4usize;
    let n = 16usize;
    let u_f = det_f32(0xA110, b * l * d, 0.4, 0.0);
    let delta_f = det_f32(0xB220, b * l * d, 0.2, 0.5);
    let a_f = det_f32(0xCC33, d * n, 0.3, -1.0);
    let b_in_f = det_f32(0xD444, b * l * n, 0.4, 0.0);
    let c_in_f = det_f32(0xE555, b * l * n, 0.4, 0.0);

    let u_h: Vec<bf16> = u_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let delta_h: Vec<bf16> = delta_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let a_h: Vec<bf16> = a_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let b_in_h: Vec<bf16> = b_in_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let c_in_h: Vec<bf16> = c_in_f.iter().map(|v| bf16::from_f32(*v)).collect();

    let u_back: Vec<f32> = u_h.iter().map(|v| v.to_f32()).collect();
    let delta_back: Vec<f32> = delta_h.iter().map(|v| v.to_f32()).collect();
    let a_back: Vec<f32> = a_h.iter().map(|v| v.to_f32()).collect();
    let b_back: Vec<f32> = b_in_h.iter().map(|v| v.to_f32()).collect();
    let c_back: Vec<f32> = c_in_h.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_mamba_scan(
        &u_back,
        &delta_back,
        &a_back,
        &b_back,
        &c_back,
        None,
        b,
        l,
        d,
        n,
    );

    let dev_u: CudaSlice<bf16> = stream.clone_htod(&u_h).unwrap();
    let dev_delta: CudaSlice<bf16> = stream.clone_htod(&delta_h).unwrap();
    let dev_a: CudaSlice<bf16> = stream.clone_htod(&a_h).unwrap();
    let dev_b_in: CudaSlice<bf16> = stream.clone_htod(&b_in_h).unwrap();
    let dev_c_in: CudaSlice<bf16> = stream.clone_htod(&c_in_h).unwrap();
    let mut dev_y: CudaSlice<bf16> = stream.alloc_zeros(b * l * d).unwrap();
    scan_bf16(
        &kernels, &stream, &dev_u, &dev_delta, &dev_a, &dev_b_in, &dev_c_in, None, &mut dev_y,
        b as u32, l as u32, d as u32, n as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got_b: Vec<bf16> = stream.clone_dtoh(&dev_y).unwrap();
    let got: Vec<f32> = got_b.iter().map(|v| v.to_f32()).collect();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[mamba_scan_bf16] max_abs={max_abs:.4}");
    assert!(max_abs < 0.5);
}
