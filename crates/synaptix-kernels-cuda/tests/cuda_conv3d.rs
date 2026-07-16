#![cfg(feature = "cuda")]

use half::{bf16, f16};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use synaptix_kernels_cuda::conv::conv3d::{
    conv3d_bf16, conv3d_f16, conv3d_f32, out_dim, Conv3dKernels,
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

#[allow(clippy::too_many_arguments)]
fn cpu_conv3d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    b: usize,
    c_in: usize,
    d: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    sd: usize,
    sh: usize,
    sw: usize,
    pd: usize,
    ph: usize,
    pw: usize,
) -> Vec<f32> {
    let d_out = out_dim(d, kd, sd, pd);
    let h_out = out_dim(h, kh, sh, ph);
    let w_out = out_dim(w, kw, sw, pw);
    let mut out = vec![0.0_f32; b * c_out * d_out * h_out * w_out];
    for bi in 0..b {
        for co in 0..c_out {
            for d_o in 0..d_out {
                for h_o in 0..h_out {
                    for w_o in 0..w_out {
                        let mut acc = 0.0_f32;
                        let d_in_base = d_o as isize * sd as isize - pd as isize;
                        let h_in_base = h_o as isize * sh as isize - ph as isize;
                        let w_in_base = w_o as isize * sw as isize - pw as isize;
                        for ci in 0..c_in {
                            for ki in 0..kd {
                                let d_in = d_in_base + ki as isize;
                                if d_in < 0 || d_in >= d as isize {
                                    continue;
                                }
                                for kj in 0..kh {
                                    let h_in = h_in_base + kj as isize;
                                    if h_in < 0 || h_in >= h as isize {
                                        continue;
                                    }
                                    for kk in 0..kw {
                                        let w_in = w_in_base + kk as isize;
                                        if w_in < 0 || w_in >= w as isize {
                                            continue;
                                        }
                                        let i_off = ((((bi * c_in + ci) * d + d_in as usize) * h
                                            + h_in as usize)
                                            * w)
                                            + w_in as usize;
                                        let w_off =
                                            ((((co * c_in + ci) * kd + ki) * kh + kj) * kw) + kk;
                                        acc += input[i_off] * weight[w_off];
                                    }
                                }
                            }
                        }
                        if let Some(bs) = bias {
                            acc += bs[co];
                        }
                        let o_off =
                            ((((bi * c_out + co) * d_out + d_o) * h_out + h_o) * w_out) + w_o;
                        out[o_off] = acc;
                    }
                }
            }
        }
    }
    out
}

#[test]
fn conv3d_f32_no_pad() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv3dKernels::for_context(&ctx).expect("compile conv3d");
    let b = 1usize;
    let c_in = 2usize;
    let d = 6usize;
    let h = 6usize;
    let w = 6usize;
    let c_out = 3usize;
    let kd = 3usize;
    let kh = 3usize;
    let kw = 3usize;
    let sd = 1usize;
    let sh = 1usize;
    let sw = 1usize;
    let pd = 0usize;
    let ph = 0usize;
    let pw = 0usize;
    let d_o = out_dim(d, kd, sd, pd);
    let h_o = out_dim(h, kh, sh, ph);
    let w_o = out_dim(w, kw, sw, pw);

    let inp = det_f32(0xA110, b * c_in * d * h * w, 0.5);
    let we = det_f32(0xB220, c_out * c_in * kd * kh * kw, 0.3);
    let bs = det_f32(0xCC33, c_out, 0.2);
    let expected = cpu_conv3d(
        &inp,
        &we,
        Some(&bs),
        b,
        c_in,
        d,
        h,
        w,
        c_out,
        kd,
        kh,
        kw,
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
    );

    let dev_in: CudaSlice<f32> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<f32> = stream.clone_htod(&we).unwrap();
    let dev_b: CudaSlice<f32> = stream.clone_htod(&bs).unwrap();
    let mut dev_out: CudaSlice<f32> = stream.alloc_zeros(b * c_out * d_o * h_o * w_o).unwrap();
    conv3d_f32(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        Some(&dev_b),
        &mut dev_out,
        b as u32,
        c_in as u32,
        d as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kd as u32,
        kh as u32,
        kw as u32,
        sd as u32,
        sh as u32,
        sw as u32,
        pd as u32,
        ph as u32,
        pw as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dev_out).unwrap();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[conv3d_f32 no_pad] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-5);
}

#[test]
fn conv3d_f32_pad_stride() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv3dKernels::for_context(&ctx).expect("compile conv3d");
    let b = 1usize;
    let c_in = 3usize;
    let d = 8usize;
    let h = 8usize;
    let w = 8usize;
    let c_out = 4usize;
    let kd = 3usize;
    let kh = 3usize;
    let kw = 3usize;
    let sd = 2usize;
    let sh = 2usize;
    let sw = 2usize;
    let pd = 1usize;
    let ph = 1usize;
    let pw = 1usize;
    let d_o = out_dim(d, kd, sd, pd);
    let h_o = out_dim(h, kh, sh, ph);
    let w_o = out_dim(w, kw, sw, pw);

    let inp = det_f32(0xD414, b * c_in * d * h * w, 0.5);
    let we = det_f32(0xE525, c_out * c_in * kd * kh * kw, 0.3);
    let expected = cpu_conv3d(
        &inp, &we, None, b, c_in, d, h, w, c_out, kd, kh, kw, sd, sh, sw, pd, ph, pw,
    );

    let dev_in: CudaSlice<f32> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<f32> = stream.clone_htod(&we).unwrap();
    let mut dev_out: CudaSlice<f32> = stream.alloc_zeros(b * c_out * d_o * h_o * w_o).unwrap();
    conv3d_f32(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        None,
        &mut dev_out,
        b as u32,
        c_in as u32,
        d as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kd as u32,
        kh as u32,
        kw as u32,
        sd as u32,
        sh as u32,
        sw as u32,
        pd as u32,
        ph as u32,
        pw as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dev_out).unwrap();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[conv3d_f32 pad_stride] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-5);
}

#[test]
fn conv3d_f16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv3dKernels::for_context(&ctx).expect("compile conv3d");
    let b = 1usize;
    let c_in = 2usize;
    let d = 6usize;
    let h = 6usize;
    let w = 6usize;
    let c_out = 4usize;
    let kd = 3usize;
    let kh = 3usize;
    let kw = 3usize;
    let sd = 1usize;
    let sh = 1usize;
    let sw = 1usize;
    let pd = 1usize;
    let ph = 1usize;
    let pw = 1usize;
    let d_o = out_dim(d, kd, sd, pd);
    let h_o = out_dim(h, kh, sh, ph);
    let w_o = out_dim(w, kw, sw, pw);
    let inp_f = det_f32(0xA1A2, b * c_in * d * h * w, 0.5);
    let w_f = det_f32(0xB1B2, c_out * c_in * kd * kh * kw, 0.3);
    let inp: Vec<f16> = inp_f.iter().map(|v| f16::from_f32(*v)).collect();
    let we: Vec<f16> = w_f.iter().map(|v| f16::from_f32(*v)).collect();
    let inp_back: Vec<f32> = inp.iter().map(|v| v.to_f32()).collect();
    let w_back: Vec<f32> = we.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_conv3d(
        &inp_back, &w_back, None, b, c_in, d, h, w, c_out, kd, kh, kw, sd, sh, sw, pd, ph, pw,
    );

    let dev_in: CudaSlice<f16> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<f16> = stream.clone_htod(&we).unwrap();
    let mut dev_out: CudaSlice<f16> = stream.alloc_zeros(b * c_out * d_o * h_o * w_o).unwrap();
    conv3d_f16(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        None,
        &mut dev_out,
        b as u32,
        c_in as u32,
        d as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kd as u32,
        kh as u32,
        kw as u32,
        sd as u32,
        sh as u32,
        sw as u32,
        pd as u32,
        ph as u32,
        pw as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got_h: Vec<f16> = stream.clone_dtoh(&dev_out).unwrap();
    let got: Vec<f32> = got_h.iter().map(|v| v.to_f32()).collect();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[conv3d_f16] max_abs={max_abs:.4}");
    assert!(max_abs < 0.1);
}

#[test]
fn conv3d_bf16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv3dKernels::for_context(&ctx).expect("compile conv3d");
    let b = 1usize;
    let c_in = 2usize;
    let d = 6usize;
    let h = 6usize;
    let w = 6usize;
    let c_out = 4usize;
    let kd = 3usize;
    let kh = 3usize;
    let kw = 3usize;
    let sd = 1usize;
    let sh = 1usize;
    let sw = 1usize;
    let pd = 1usize;
    let ph = 1usize;
    let pw = 1usize;
    let d_o = out_dim(d, kd, sd, pd);
    let h_o = out_dim(h, kh, sh, ph);
    let w_o = out_dim(w, kw, sw, pw);
    let inp_f = det_f32(0xA1A2, b * c_in * d * h * w, 0.5);
    let w_f = det_f32(0xB1B2, c_out * c_in * kd * kh * kw, 0.3);
    let inp: Vec<bf16> = inp_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let we: Vec<bf16> = w_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let inp_back: Vec<f32> = inp.iter().map(|v| v.to_f32()).collect();
    let w_back: Vec<f32> = we.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_conv3d(
        &inp_back, &w_back, None, b, c_in, d, h, w, c_out, kd, kh, kw, sd, sh, sw, pd, ph, pw,
    );

    let dev_in: CudaSlice<bf16> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<bf16> = stream.clone_htod(&we).unwrap();
    let mut dev_out: CudaSlice<bf16> = stream.alloc_zeros(b * c_out * d_o * h_o * w_o).unwrap();
    conv3d_bf16(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        None,
        &mut dev_out,
        b as u32,
        c_in as u32,
        d as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kd as u32,
        kh as u32,
        kw as u32,
        sd as u32,
        sh as u32,
        sw as u32,
        pd as u32,
        ph as u32,
        pw as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got_b: Vec<bf16> = stream.clone_dtoh(&dev_out).unwrap();
    let got: Vec<f32> = got_b.iter().map(|v| v.to_f32()).collect();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[conv3d_bf16] max_abs={max_abs:.4}");
    assert!(max_abs < 0.5);
}
