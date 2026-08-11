
//! Bit-exact (F32-эталон) тесты для Mamba2 SSD рекуррентного forward.

use half::{bf16, f16};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use synaptix_kernels_cuda::ssm::mamba2_ssd::Mamba2SsdKernels;

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
fn cpu_ssd(
    x: &[f32],
    dt: &[f32],
    a: &[f32],
    b_in: &[f32],
    c_in: &[f32],
    d_skip: Option<&[f32]>,
    b: usize,
    l: usize,
    h: usize,
    p: usize,
    n: usize,
) -> Vec<f32> {
    let mut y = vec![0.0_f32; b * l * h * p];
    // state[(b,h,p,n)]
    let mut state = vec![0.0_f32; b * h * p * n];
    for bi in 0..b {
        for t in 0..l {
            for hi in 0..h {
                let dt_t = dt[(bi * l + t) * h + hi];
                let a_h = a[hi];
                let a_t = (dt_t * a_h).exp();
                for pi in 0..p {
                    let x_off = ((bi * l + t) * h + hi) * p + pi;
                    let x_t = x[x_off];
                    let mut y_val = 0.0_f32;
                    for ni in 0..n {
                        let bc_off = ((bi * l + t) * h + hi) * n + ni;
                        let st_idx = ((bi * h + hi) * p + pi) * n + ni;
                        state[st_idx] = a_t * state[st_idx] + dt_t * x_t * b_in[bc_off];
                        y_val += c_in[bc_off] * state[st_idx];
                    }
                    if let Some(ds) = d_skip {
                        y_val += ds[hi] * x_t;
                    }
                    y[x_off] = y_val;
                }
            }
        }
    }
    y
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn mamba2_ssd_f32_with_skip() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2SsdKernels::for_context(&ctx).expect("compile");
    let (b, l, h, p, n) = (2usize, 8usize, 3usize, 4usize, 16usize);
    let x = det_f32(0x1, b * l * h * p, 0.5, 0.0);
    let dt = det_f32(0x2, b * l * h, 0.2, 0.5); // dt > 0
    let a = det_f32(0x3, h, 0.5, -1.5); // A < 0
    let b_in = det_f32(0x4, b * l * h * n, 0.5, 0.0);
    let c_in = det_f32(0x5, b * l * h * n, 0.5, 0.0);
    let d_skip = det_f32(0x6, h, 0.3, 0.0);
    let expected = cpu_ssd(&x, &dt, &a, &b_in, &c_in, Some(&d_skip), b, l, h, p, n);

    let dx: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let ddt: CudaSlice<f32> = stream.clone_htod(&dt).unwrap();
    let da: CudaSlice<f32> = stream.clone_htod(&a).unwrap();
    let dbi: CudaSlice<f32> = stream.clone_htod(&b_in).unwrap();
    let dci: CudaSlice<f32> = stream.clone_htod(&c_in).unwrap();
    let dds: CudaSlice<f32> = stream.clone_htod(&d_skip).unwrap();
    let mut dy: CudaSlice<f32> = stream.alloc_zeros(b * l * h * p).unwrap();
    kern.ssd_f32(
        &stream,
        &dx,
        &ddt,
        &da,
        &dbi,
        &dci,
        Some(&dds),
        &mut dy,
        b as u32,
        l as u32,
        h as u32,
        p as u32,
        n as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dy).unwrap();
    let m = max_abs(&got, &expected);
    eprintln!("[mamba2_ssd_f32 skip] max_abs={m:.6}");
    assert!(m < 1e-4, "max_abs={m}");
}

#[test]
fn mamba2_ssd_f32_no_skip_n64() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2SsdKernels::for_context(&ctx).expect("compile");
    let (b, l, h, p, n) = (1usize, 12usize, 2usize, 8usize, 64usize);
    let x = det_f32(0x11, b * l * h * p, 0.5, 0.0);
    let dt = det_f32(0x22, b * l * h, 0.15, 0.4);
    let a = det_f32(0x33, h, 0.5, -1.0);
    let b_in = det_f32(0x44, b * l * h * n, 0.3, 0.0);
    let c_in = det_f32(0x55, b * l * h * n, 0.3, 0.0);
    let expected = cpu_ssd(&x, &dt, &a, &b_in, &c_in, None, b, l, h, p, n);

    let dx: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let ddt: CudaSlice<f32> = stream.clone_htod(&dt).unwrap();
    let da: CudaSlice<f32> = stream.clone_htod(&a).unwrap();
    let dbi: CudaSlice<f32> = stream.clone_htod(&b_in).unwrap();
    let dci: CudaSlice<f32> = stream.clone_htod(&c_in).unwrap();
    let mut dy: CudaSlice<f32> = stream.alloc_zeros(b * l * h * p).unwrap();
    kern.ssd_f32(
        &stream, &dx, &ddt, &da, &dbi, &dci, None, &mut dy, b as u32, l as u32, h as u32, p as u32,
        n as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dy).unwrap();
    let m = max_abs(&got, &expected);
    eprintln!("[mamba2_ssd_f32 no_skip N=64] max_abs={m:.6}");
    assert!(m < 1e-4, "max_abs={m}");
}

#[test]
fn mamba2_ssd_f16_and_bf16() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2SsdKernels::for_context(&ctx).expect("compile");
    let (b, l, h, p, n) = (1usize, 8usize, 2usize, 4usize, 16usize);
    let xf = det_f32(0x1, b * l * h * p, 0.4, 0.0);
    let dtf = det_f32(0x2, b * l * h, 0.1, 0.4);
    let af = det_f32(0x3, h, 0.3, -1.0);
    let bf = det_f32(0x4, b * l * h * n, 0.4, 0.0);
    let cf = det_f32(0x5, b * l * h * n, 0.4, 0.0);

    // f16
    {
        let conv = |v: &[f32]| -> Vec<f16> { v.iter().map(|x| f16::from_f32(*x)).collect() };
        let back = |v: &[f16]| -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() };
        let (xh, dth, ah, bh, ch) = (conv(&xf), conv(&dtf), conv(&af), conv(&bf), conv(&cf));
        let expected = cpu_ssd(
            &back(&xh),
            &back(&dth),
            &back(&ah),
            &back(&bh),
            &back(&ch),
            None,
            b,
            l,
            h,
            p,
            n,
        );
        let dx: CudaSlice<f16> = stream.clone_htod(&xh).unwrap();
        let ddt: CudaSlice<f16> = stream.clone_htod(&dth).unwrap();
        let da: CudaSlice<f16> = stream.clone_htod(&ah).unwrap();
        let dbi: CudaSlice<f16> = stream.clone_htod(&bh).unwrap();
        let dci: CudaSlice<f16> = stream.clone_htod(&ch).unwrap();
        let mut dy: CudaSlice<f16> = stream.alloc_zeros(b * l * h * p).unwrap();
        kern.ssd_f16(
            &stream, &dx, &ddt, &da, &dbi, &dci, None, &mut dy, b as u32, l as u32, h as u32,
            p as u32, n as u32,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let got: Vec<f32> = back(&stream.clone_dtoh(&dy).unwrap());
        let m = max_abs(&got, &expected);
        eprintln!("[mamba2_ssd_f16] max_abs={m:.4}");
        assert!(m < 0.1, "f16 max_abs={m}");
    }
    // bf16
    {
        let conv = |v: &[f32]| -> Vec<bf16> { v.iter().map(|x| bf16::from_f32(*x)).collect() };
        let back = |v: &[bf16]| -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() };
        let (xh, dth, ah, bh, ch) = (conv(&xf), conv(&dtf), conv(&af), conv(&bf), conv(&cf));
        let expected = cpu_ssd(
            &back(&xh),
            &back(&dth),
            &back(&ah),
            &back(&bh),
            &back(&ch),
            None,
            b,
            l,
            h,
            p,
            n,
        );
        let dx: CudaSlice<bf16> = stream.clone_htod(&xh).unwrap();
        let ddt: CudaSlice<bf16> = stream.clone_htod(&dth).unwrap();
        let da: CudaSlice<bf16> = stream.clone_htod(&ah).unwrap();
        let dbi: CudaSlice<bf16> = stream.clone_htod(&bh).unwrap();
        let dci: CudaSlice<bf16> = stream.clone_htod(&ch).unwrap();
        let mut dy: CudaSlice<bf16> = stream.alloc_zeros(b * l * h * p).unwrap();
        kern.ssd_bf16(
            &stream, &dx, &ddt, &da, &dbi, &dci, None, &mut dy, b as u32, l as u32, h as u32,
            p as u32, n as u32,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let got: Vec<f32> = back(&stream.clone_dtoh(&dy).unwrap());
        let m = max_abs(&got, &expected);
        eprintln!("[mamba2_ssd_bf16] max_abs={m:.4}");
        assert!(m < 0.5, "bf16 max_abs={m}");
    }
}
