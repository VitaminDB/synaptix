
use half::{bf16, f16};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use synaptix_kernels_cuda::conv::conv1d::{
    conv1d_bf16, conv1d_f16, conv1d_f32, l_out, Conv1dKernels,
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

fn cpu_conv1d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    b: usize,
    c_in: usize,
    l: usize,
    c_out: usize,
    k: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let l_o = l_out(l, k, stride, padding);
    let mut out = vec![0.0_f32; b * c_out * l_o];
    for bi in 0..b {
        for co in 0..c_out {
            for lo in 0..l_o {
                let mut acc = 0.0_f32;
                let l_in_base = lo as isize * stride as isize - padding as isize;
                for ci in 0..c_in {
                    for kk in 0..k {
                        let l_in = l_in_base + kk as isize;
                        if l_in < 0 || l_in >= l as isize {
                            continue;
                        }
                        let x = input[(bi * c_in + ci) * l + l_in as usize];
                        let w = weight[(co * c_in + ci) * k + kk];
                        acc += x * w;
                    }
                }
                if let Some(b_t) = bias {
                    acc += b_t[co];
                }
                out[(bi * c_out + co) * l_o + lo] = acc;
            }
        }
    }
    out
}

#[test]
fn conv1d_f32_no_pad() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv1dKernels::for_context(&ctx).expect("compile conv1d");
    let b = 2usize;
    let c_in = 3usize;
    let l = 16usize;
    let c_out = 4usize;
    let k = 3usize;
    let stride = 1usize;
    let padding = 0usize;
    let l_o = l_out(l, k, stride, padding);

    let inp = det_f32(0xA110, b * c_in * l, 0.5);
    let w = det_f32(0xB220, c_out * c_in * k, 0.3);
    let bs = det_f32(0xCC33, c_out, 0.2);
    let expected = cpu_conv1d(&inp, &w, Some(&bs), b, c_in, l, c_out, k, stride, padding);

    let dev_in: CudaSlice<f32> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<f32> = stream.clone_htod(&w).unwrap();
    let dev_b: CudaSlice<f32> = stream.clone_htod(&bs).unwrap();
    let mut dev_out: CudaSlice<f32> = stream.alloc_zeros(b * c_out * l_o).unwrap();
    conv1d_f32(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        Some(&dev_b),
        &mut dev_out,
        b as u32,
        c_in as u32,
        l as u32,
        c_out as u32,
        k as u32,
        stride as u32,
        padding as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dev_out).unwrap();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[conv1d_f32 no_pad b={b} cin={c_in} l={l} cout={c_out} k={k}] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-5, "max_abs={max_abs}");
}

#[test]
fn conv1d_f32_pad_stride() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv1dKernels::for_context(&ctx).expect("compile conv1d");
    let b = 1usize;
    let c_in = 2usize;
    let l = 12usize;
    let c_out = 3usize;
    let k = 5usize;
    let stride = 2usize;
    let padding = 2usize;
    let l_o = l_out(l, k, stride, padding);

    let inp = det_f32(0xD414, b * c_in * l, 0.5);
    let w = det_f32(0xE525, c_out * c_in * k, 0.3);
    let expected = cpu_conv1d(&inp, &w, None, b, c_in, l, c_out, k, stride, padding);

    let dev_in: CudaSlice<f32> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<f32> = stream.clone_htod(&w).unwrap();
    let mut dev_out: CudaSlice<f32> = stream.alloc_zeros(b * c_out * l_o).unwrap();
    conv1d_f32(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        None,
        &mut dev_out,
        b as u32,
        c_in as u32,
        l as u32,
        c_out as u32,
        k as u32,
        stride as u32,
        padding as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dev_out).unwrap();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[conv1d_f32 pad_stride] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-5);
}

#[test]
fn conv1d_f16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv1dKernels::for_context(&ctx).expect("compile conv1d");
    let b = 1usize;
    let c_in = 4usize;
    let l = 8usize;
    let c_out = 8usize;
    let k = 3usize;
    let stride = 1usize;
    let padding = 1usize;
    let l_o = l_out(l, k, stride, padding);

    let inp_f = det_f32(0xA1A2, b * c_in * l, 0.5);
    let w_f = det_f32(0xB1B2, c_out * c_in * k, 0.3);
    let inp: Vec<f16> = inp_f.iter().map(|v| f16::from_f32(*v)).collect();
    let w: Vec<f16> = w_f.iter().map(|v| f16::from_f32(*v)).collect();
    let inp_back: Vec<f32> = inp.iter().map(|v| v.to_f32()).collect();
    let w_back: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_conv1d(
        &inp_back, &w_back, None, b, c_in, l, c_out, k, stride, padding,
    );

    let dev_in: CudaSlice<f16> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<f16> = stream.clone_htod(&w).unwrap();
    let mut dev_out: CudaSlice<f16> = stream.alloc_zeros(b * c_out * l_o).unwrap();
    conv1d_f16(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        None,
        &mut dev_out,
        b as u32,
        c_in as u32,
        l as u32,
        c_out as u32,
        k as u32,
        stride as u32,
        padding as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got_h: Vec<f16> = stream.clone_dtoh(&dev_out).unwrap();
    let got: Vec<f32> = got_h.iter().map(|v| v.to_f32()).collect();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[conv1d_f16] max_abs={max_abs:.4}");
    assert!(max_abs < 0.05);
}

#[test]
fn conv1d_bf16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv1dKernels::for_context(&ctx).expect("compile conv1d");
    let b = 1usize;
    let c_in = 4usize;
    let l = 8usize;
    let c_out = 8usize;
    let k = 3usize;
    let stride = 1usize;
    let padding = 1usize;
    let l_o = l_out(l, k, stride, padding);

    let inp_f = det_f32(0xA1A2, b * c_in * l, 0.5);
    let w_f = det_f32(0xB1B2, c_out * c_in * k, 0.3);
    let inp: Vec<bf16> = inp_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let w: Vec<bf16> = w_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let inp_back: Vec<f32> = inp.iter().map(|v| v.to_f32()).collect();
    let w_back: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_conv1d(
        &inp_back, &w_back, None, b, c_in, l, c_out, k, stride, padding,
    );

    let dev_in: CudaSlice<bf16> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<bf16> = stream.clone_htod(&w).unwrap();
    let mut dev_out: CudaSlice<bf16> = stream.alloc_zeros(b * c_out * l_o).unwrap();
    conv1d_bf16(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        None,
        &mut dev_out,
        b as u32,
        c_in as u32,
        l as u32,
        c_out as u32,
        k as u32,
        stride as u32,
        padding as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got_b: Vec<bf16> = stream.clone_dtoh(&dev_out).unwrap();
    let got: Vec<f32> = got_b.iter().map(|v| v.to_f32()).collect();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[conv1d_bf16] max_abs={max_abs:.4}");
    assert!(max_abs < 0.5);
}
