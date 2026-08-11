
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};
use synaptix_kernels_cuda::conv::causal_conv1d::{
    causal_conv1d_bf16, causal_conv1d_f16, causal_conv1d_f32, out_len, CausalConv1dKernels,
};

// Host-эталон stateful update (decode T=1) == s=1 случай causal_conv1d_stateful + silu.
// state [km1, conv_dim] обновляется in-place, возвращает out [conv_dim].
fn cpu_update(
    state: &mut [f32],
    x: &[f32],
    w: &[f32],
    conv_dim: usize,
    k: usize,
    silu: bool,
) -> Vec<f32> {
    let km1 = k - 1;
    let mut out = vec![0.0_f32; conv_dim];
    for c in 0..conv_dim {
        let mut acc = 0.0_f32;
        for j in 0..km1 {
            acc += state[j * conv_dim + c] * w[c * k + j];
        }
        acc += x[c] * w[c * k + km1];
        if silu {
            acc /= 1.0 + (-acc).exp();
        }
        out[c] = acc;
    }
    for j in 0..km1.saturating_sub(1) {
        for c in 0..conv_dim {
            state[j * conv_dim + c] = state[(j + 1) * conv_dim + c];
        }
    }
    if km1 > 0 {
        let dst = (km1 - 1) * conv_dim;
        state[dst..dst + conv_dim].copy_from_slice(&x[..conv_dim]);
    }
    out
}

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
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .map(|f| f * scale)
        .collect()
}

// out[b,c,i] = sum_ki w[c,ki] * x[b,c, i*stride - (K-1) + ki] (OOB 0) + bias[c]
fn cpu_causal(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    b: usize,
    c: usize,
    l: usize,
    k: usize,
    stride: usize,
) -> Vec<f32> {
    let o = out_len(l, stride);
    let mut out = vec![0.0_f32; b * c * o];
    for bi in 0..b {
        for ci in 0..c {
            for i in 0..o {
                let mut acc = 0.0_f32;
                let base = i as isize * stride as isize - (k as isize - 1);
                for ki in 0..k {
                    let l_in = base + ki as isize;
                    if l_in < 0 || l_in >= l as isize {
                        continue;
                    }
                    acc += x[(bi * c + ci) * l + l_in as usize] * w[ci * k + ki];
                }
                if let Some(bt) = bias {
                    acc += bt[ci];
                }
                out[(bi * c + ci) * o + i] = acc;
            }
        }
    }
    out
}

#[test]
fn causal_conv1d_f32_stride1_bias() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = CausalConv1dKernels::for_context(&ctx).expect("compile causal_conv1d");
    let (b, c, l, k, stride) = (2usize, 5usize, 32usize, 4usize, 1usize);
    let o = out_len(l, stride);
    let x = det_f32(0xC10, b * c * l, 0.5);
    let w = det_f32(0xC20, c * k, 0.4);
    let bias = det_f32(0xC30, c, 0.2);
    let exp = cpu_causal(&x, &w, Some(&bias), b, c, l, k, stride);

    let dx: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let dw: CudaSlice<f32> = stream.clone_htod(&w).unwrap();
    let db: CudaSlice<f32> = stream.clone_htod(&bias).unwrap();
    let mut dout: CudaSlice<f32> = stream.alloc_zeros(b * c * o).unwrap();
    causal_conv1d_f32(
        &kernels,
        &stream,
        &dx,
        &dw,
        Some(&db),
        &mut dout,
        b as u32,
        c as u32,
        l as u32,
        k as u32,
        stride as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
    let max = got
        .iter()
        .zip(&exp)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max);
    eprintln!("[causal_conv1d_f32 s1] max_abs={max:.6}");
    assert!(max < 1e-5, "max={max}");
}

#[test]
fn causal_conv1d_f32_stride2_no_bias() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = CausalConv1dKernels::for_context(&ctx).expect("compile causal_conv1d");
    let (b, c, l, k, stride) = (1usize, 3usize, 17usize, 3usize, 2usize);
    let o = out_len(l, stride);
    let x = det_f32(0xC40, b * c * l, 0.5);
    let w = det_f32(0xC50, c * k, 0.4);
    let exp = cpu_causal(&x, &w, None, b, c, l, k, stride);

    let dx: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let dw: CudaSlice<f32> = stream.clone_htod(&w).unwrap();
    let mut dout: CudaSlice<f32> = stream.alloc_zeros(b * c * o).unwrap();
    causal_conv1d_f32(
        &kernels,
        &stream,
        &dx,
        &dw,
        None,
        &mut dout,
        b as u32,
        c as u32,
        l as u32,
        k as u32,
        stride as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
    let max = got
        .iter()
        .zip(&exp)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max);
    eprintln!("[causal_conv1d_f32 s2] max_abs={max:.6}");
    assert!(max < 1e-5, "max={max}");
}

#[test]
fn causal_conv1d_f16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = CausalConv1dKernels::for_context(&ctx).expect("compile causal_conv1d");
    let (b, c, l, k, stride) = (2usize, 4usize, 24usize, 4usize, 1usize);
    let o = out_len(l, stride);
    let xf = det_f32(0xC60, b * c * l, 0.5);
    let wf = det_f32(0xC70, c * k, 0.4);
    let x: Vec<f16> = xf.iter().map(|v| f16::from_f32(*v)).collect();
    let w: Vec<f16> = wf.iter().map(|v| f16::from_f32(*v)).collect();
    let xb: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let wb: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
    let exp = cpu_causal(&xb, &wb, None, b, c, l, k, stride);

    let dx: CudaSlice<f16> = stream.clone_htod(&x).unwrap();
    let dw: CudaSlice<f16> = stream.clone_htod(&w).unwrap();
    let mut dout: CudaSlice<f16> = stream.alloc_zeros(b * c * o).unwrap();
    causal_conv1d_f16(
        &kernels,
        &stream,
        &dx,
        &dw,
        None,
        &mut dout,
        b as u32,
        c as u32,
        l as u32,
        k as u32,
        stride as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gh: Vec<f16> = stream.clone_dtoh(&dout).unwrap();
    let got: Vec<f32> = gh.iter().map(|v| v.to_f32()).collect();
    let max = got
        .iter()
        .zip(&exp)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max);
    eprintln!("[causal_conv1d_f16] max_abs={max:.4}");
    assert!(max < 0.05, "max={max}");
}

#[test]
fn causal_conv1d_update_f32_multistep() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = CausalConv1dKernels::for_context(&ctx).expect("compile causal_conv1d");
    let (conv_dim, k) = (37usize, 4usize);
    let km1 = k - 1;
    let w = det_f32(0xD10, conv_dim * k, 0.4);

    let mut cpu_state = vec![0.0_f32; km1 * conv_dim];
    let dw: CudaSlice<f32> = stream.clone_htod(&w).unwrap();
    let mut dev_state: CudaSlice<f32> = stream.alloc_zeros(km1 * conv_dim).unwrap();

    let mut worst = 0.0_f32;
    for t in 0..6usize {
        let x = det_f32(0xD20 + t as u64, conv_dim, 0.7);
        let exp = cpu_update(&mut cpu_state, &x, &w, conv_dim, k, true);

        let dx: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
        let mut dout: CudaSlice<f32> = stream.alloc_zeros(conv_dim).unwrap();
        kernels
            .causal_conv1d_update_f32(
                &stream,
                &dx,
                &mut dev_state,
                &dw,
                &mut dout,
                conv_dim as u32,
                k as u32,
                true,
            )
            .unwrap();
        stream.synchronize().unwrap();
        let got: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
        let m = got
            .iter()
            .zip(&exp)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0, f32::max);
        worst = worst.max(m);
        assert!(m < 1e-5, "step {t}: out max_abs={m}");
    }
    let got_state: Vec<f32> = stream.clone_dtoh(&dev_state).unwrap();
    let ms = got_state
        .iter()
        .zip(&cpu_state)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max);
    eprintln!("[causal_conv1d_update_f32] worst_out={worst:.6} state_max_abs={ms:.6}");
    assert!(ms < 1e-5, "state max_abs={ms}");
}

#[test]
fn causal_conv1d_bf16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = CausalConv1dKernels::for_context(&ctx).expect("compile causal_conv1d");
    let (b, c, l, k, stride) = (2usize, 4usize, 24usize, 4usize, 1usize);
    let o = out_len(l, stride);
    let xf = det_f32(0xC80, b * c * l, 0.5);
    let wf = det_f32(0xC90, c * k, 0.4);
    let x: Vec<bf16> = xf.iter().map(|v| bf16::from_f32(*v)).collect();
    let w: Vec<bf16> = wf.iter().map(|v| bf16::from_f32(*v)).collect();
    let xb: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let wb: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
    let exp = cpu_causal(&xb, &wb, None, b, c, l, k, stride);

    let dx: CudaSlice<bf16> = stream.clone_htod(&x).unwrap();
    let dw: CudaSlice<bf16> = stream.clone_htod(&w).unwrap();
    let mut dout: CudaSlice<bf16> = stream.alloc_zeros(b * c * o).unwrap();
    causal_conv1d_bf16(
        &kernels,
        &stream,
        &dx,
        &dw,
        None,
        &mut dout,
        b as u32,
        c as u32,
        l as u32,
        k as u32,
        stride as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gb: Vec<bf16> = stream.clone_dtoh(&dout).unwrap();
    let got: Vec<f32> = gb.iter().map(|v| v.to_f32()).collect();
    let max = got
        .iter()
        .zip(&exp)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max);
    eprintln!("[causal_conv1d_bf16] max_abs={max:.4}");
    assert!(max < 0.3, "max={max}");
}
