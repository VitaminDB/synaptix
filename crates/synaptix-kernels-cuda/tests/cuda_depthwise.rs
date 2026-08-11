
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};
use synaptix_kernels_cuda::conv::conv1d::l_out;
use synaptix_kernels_cuda::conv::depthwise::{
    depthwise_conv1d_bf16, depthwise_conv1d_f16, depthwise_conv1d_f32, DepthwiseConv1dKernels,
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
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .map(|f| f * scale)
        .collect()
}

// out[b,c,i] = sum_ki w[c,ki] * x[b,c, i*stride - padding + ki] (OOB 0) + bias[c]
fn cpu_depthwise(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    b: usize,
    c: usize,
    l: usize,
    k: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let lo = l_out(l, k, stride, padding);
    let mut out = vec![0.0_f32; b * c * lo];
    for bi in 0..b {
        for ci in 0..c {
            for i in 0..lo {
                let mut acc = 0.0_f32;
                let base = i as isize * stride as isize - padding as isize;
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
                out[(bi * c + ci) * lo + i] = acc;
            }
        }
    }
    out
}

#[test]
fn depthwise_f32_pad_bias() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = DepthwiseConv1dKernels::for_context(&ctx).expect("compile depthwise");
    let (b, c, l, k, stride, pad) = (2usize, 6usize, 32usize, 3usize, 1usize, 1usize);
    let lo = l_out(l, k, stride, pad);
    let x = det_f32(0xD10, b * c * l, 0.5);
    let w = det_f32(0xD20, c * k, 0.4);
    let bias = det_f32(0xD30, c, 0.2);
    let exp = cpu_depthwise(&x, &w, Some(&bias), b, c, l, k, stride, pad);

    let dx: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let dw: CudaSlice<f32> = stream.clone_htod(&w).unwrap();
    let db: CudaSlice<f32> = stream.clone_htod(&bias).unwrap();
    let mut dout: CudaSlice<f32> = stream.alloc_zeros(b * c * lo).unwrap();
    depthwise_conv1d_f32(
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
        pad as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
    let max = got
        .iter()
        .zip(&exp)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max);
    eprintln!("[depthwise_f32 pad1] max_abs={max:.6}");
    assert!(max < 1e-5, "max={max}");
}

#[test]
fn depthwise_f32_stride2_pad2() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = DepthwiseConv1dKernels::for_context(&ctx).expect("compile depthwise");
    let (b, c, l, k, stride, pad) = (1usize, 4usize, 20usize, 5usize, 2usize, 2usize);
    let lo = l_out(l, k, stride, pad);
    let x = det_f32(0xD40, b * c * l, 0.5);
    let w = det_f32(0xD50, c * k, 0.4);
    let exp = cpu_depthwise(&x, &w, None, b, c, l, k, stride, pad);

    let dx: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let dw: CudaSlice<f32> = stream.clone_htod(&w).unwrap();
    let mut dout: CudaSlice<f32> = stream.alloc_zeros(b * c * lo).unwrap();
    depthwise_conv1d_f32(
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
        pad as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
    let max = got
        .iter()
        .zip(&exp)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max);
    eprintln!("[depthwise_f32 s2 p2] max_abs={max:.6}");
    assert!(max < 1e-5, "max={max}");
}

#[test]
fn depthwise_f16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = DepthwiseConv1dKernels::for_context(&ctx).expect("compile depthwise");
    let (b, c, l, k, stride, pad) = (2usize, 4usize, 24usize, 3usize, 1usize, 1usize);
    let lo = l_out(l, k, stride, pad);
    let xf = det_f32(0xD60, b * c * l, 0.5);
    let wf = det_f32(0xD70, c * k, 0.4);
    let x: Vec<f16> = xf.iter().map(|v| f16::from_f32(*v)).collect();
    let w: Vec<f16> = wf.iter().map(|v| f16::from_f32(*v)).collect();
    let xb: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let wb: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
    let exp = cpu_depthwise(&xb, &wb, None, b, c, l, k, stride, pad);

    let dx: CudaSlice<f16> = stream.clone_htod(&x).unwrap();
    let dw: CudaSlice<f16> = stream.clone_htod(&w).unwrap();
    let mut dout: CudaSlice<f16> = stream.alloc_zeros(b * c * lo).unwrap();
    depthwise_conv1d_f16(
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
        pad as u32,
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
    eprintln!("[depthwise_f16] max_abs={max:.4}");
    assert!(max < 0.05, "max={max}");
}

#[test]
fn depthwise_bf16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = DepthwiseConv1dKernels::for_context(&ctx).expect("compile depthwise");
    let (b, c, l, k, stride, pad) = (2usize, 4usize, 24usize, 3usize, 1usize, 1usize);
    let lo = l_out(l, k, stride, pad);
    let xf = det_f32(0xD80, b * c * l, 0.5);
    let wf = det_f32(0xD90, c * k, 0.4);
    let x: Vec<bf16> = xf.iter().map(|v| bf16::from_f32(*v)).collect();
    let w: Vec<bf16> = wf.iter().map(|v| bf16::from_f32(*v)).collect();
    let xb: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let wb: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
    let exp = cpu_depthwise(&xb, &wb, None, b, c, l, k, stride, pad);

    let dx: CudaSlice<bf16> = stream.clone_htod(&x).unwrap();
    let dw: CudaSlice<bf16> = stream.clone_htod(&w).unwrap();
    let mut dout: CudaSlice<bf16> = stream.alloc_zeros(b * c * lo).unwrap();
    depthwise_conv1d_bf16(
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
        pad as u32,
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
    eprintln!("[depthwise_bf16] max_abs={max:.4}");
    assert!(max < 0.3, "max={max}");
}
