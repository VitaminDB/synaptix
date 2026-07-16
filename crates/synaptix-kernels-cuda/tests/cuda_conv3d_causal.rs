#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};
use synaptix_kernels_cuda::conv::conv3d_causal::{
    conv3d_causal_bf16, conv3d_causal_f16, conv3d_causal_f32, spatial_out, t_out,
    Conv3dCausalKernels,
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
fn cpu_conv3d_causal(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    b: usize,
    c_in: usize,
    t: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kt: usize,
    kh: usize,
    kw: usize,
    stride: usize,
) -> Vec<f32> {
    let s = stride.max(1);
    let to = t_out(t, s);
    let ho = spatial_out(h, kh, s);
    let wo = spatial_out(w, kw, s);
    let mut out = vec![0.0_f32; b * c_out * to * ho * wo];
    for bi in 0..b {
        for co in 0..c_out {
            for t_o in 0..to {
                for h_o in 0..ho {
                    for w_o in 0..wo {
                        let mut acc = bias.map_or(0.0, |bv| bv[co]);
                        for ci in 0..c_in {
                            for kti in 0..kt {
                                let tp = t_o * s + kti;
                                if tp < kt - 1 {
                                    continue;
                                }
                                let ti = tp - (kt - 1);
                                if ti >= t {
                                    continue;
                                }
                                for khi in 0..kh {
                                    let hi = h_o * s + khi;
                                    if hi >= h {
                                        continue;
                                    }
                                    for kwi in 0..kw {
                                        let wi = w_o * s + kwi;
                                        if wi >= w {
                                            continue;
                                        }
                                        let i_off =
                                            ((((bi * c_in + ci) * t + ti) * h + hi) * w) + wi;
                                        let w_off =
                                            ((((co * c_in + ci) * kt + kti) * kh + khi) * kw) + kwi;
                                        acc += input[i_off] * weight[w_off];
                                    }
                                }
                            }
                        }
                        let o_off = ((((bi * c_out + co) * to + t_o) * ho + h_o) * wo) + w_o;
                        out[o_off] = acc;
                    }
                }
            }
        }
    }
    out
}

#[test]
fn causal_conv3d_f32_stride1_with_bias() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv3dCausalKernels::for_context(&ctx).expect("compile conv3d_causal");
    let (b, c_in, t, h, w) = (1, 2, 6, 5, 5);
    let (c_out, kt, kh, kw, stride) = (3, 3, 3, 3, 1);
    let to = t_out(t, stride);
    let ho = spatial_out(h, kh, stride);
    let wo = spatial_out(w, kw, stride);

    let inp = det_f32(0xA110, b * c_in * t * h * w, 0.5);
    let we = det_f32(0xB220, c_out * c_in * kt * kh * kw, 0.3);
    let bs = det_f32(0xCC33, c_out, 0.2);
    let expected = cpu_conv3d_causal(
        &inp,
        &we,
        Some(&bs),
        b,
        c_in,
        t,
        h,
        w,
        c_out,
        kt,
        kh,
        kw,
        stride,
    );

    let dev_in: CudaSlice<f32> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<f32> = stream.clone_htod(&we).unwrap();
    let dev_b: CudaSlice<f32> = stream.clone_htod(&bs).unwrap();
    let mut dev_out: CudaSlice<f32> = stream.alloc_zeros(b * c_out * to * ho * wo).unwrap();
    conv3d_causal_f32(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        Some(&dev_b),
        &mut dev_out,
        b as u32,
        c_in as u32,
        t as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kt as u32,
        kh as u32,
        kw as u32,
        stride as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dev_out).unwrap();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[conv3d_causal_f32 s1] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-5, "max_abs={max_abs} too high");
}

#[test]
fn causal_conv3d_f32_stride2_no_bias() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv3dCausalKernels::for_context(&ctx).expect("compile conv3d_causal");
    let (b, c_in, t, h, w) = (2, 3, 9, 8, 8);
    let (c_out, kt, kh, kw, stride) = (4, 3, 3, 3, 2);
    let to = t_out(t, stride);
    let ho = spatial_out(h, kh, stride);
    let wo = spatial_out(w, kw, stride);

    let inp = det_f32(0xD414, b * c_in * t * h * w, 0.5);
    let we = det_f32(0xE525, c_out * c_in * kt * kh * kw, 0.3);
    let expected = cpu_conv3d_causal(&inp, &we, None, b, c_in, t, h, w, c_out, kt, kh, kw, stride);

    let dev_in: CudaSlice<f32> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<f32> = stream.clone_htod(&we).unwrap();
    let mut dev_out: CudaSlice<f32> = stream.alloc_zeros(b * c_out * to * ho * wo).unwrap();
    conv3d_causal_f32(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        None,
        &mut dev_out,
        b as u32,
        c_in as u32,
        t as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kt as u32,
        kh as u32,
        kw as u32,
        stride as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dev_out).unwrap();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[conv3d_causal_f32 s2] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-5, "max_abs={max_abs} too high");
}

#[test]
fn causal_conv3d_f16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv3dCausalKernels::for_context(&ctx).expect("compile conv3d_causal");
    let (b, c_in, t, h, w) = (1, 2, 6, 5, 5);
    let (c_out, kt, kh, kw, stride) = (4, 3, 3, 3, 1);
    let to = t_out(t, stride);
    let ho = spatial_out(h, kh, stride);
    let wo = spatial_out(w, kw, stride);

    let inp_f = det_f32(0xA1A2, b * c_in * t * h * w, 0.5);
    let w_f = det_f32(0xB1B2, c_out * c_in * kt * kh * kw, 0.3);
    let inp: Vec<f16> = inp_f.iter().map(|v| f16::from_f32(*v)).collect();
    let we: Vec<f16> = w_f.iter().map(|v| f16::from_f32(*v)).collect();
    let inp_back: Vec<f32> = inp.iter().map(|v| v.to_f32()).collect();
    let w_back: Vec<f32> = we.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_conv3d_causal(
        &inp_back, &w_back, None, b, c_in, t, h, w, c_out, kt, kh, kw, stride,
    );

    let dev_in: CudaSlice<f16> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<f16> = stream.clone_htod(&we).unwrap();
    let mut dev_out: CudaSlice<f16> = stream.alloc_zeros(b * c_out * to * ho * wo).unwrap();
    conv3d_causal_f16(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        None,
        &mut dev_out,
        b as u32,
        c_in as u32,
        t as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kt as u32,
        kh as u32,
        kw as u32,
        stride as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got_h: Vec<f16> = stream.clone_dtoh(&dev_out).unwrap();
    let got: Vec<f32> = got_h.iter().map(|v| v.to_f32()).collect();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[conv3d_causal_f16] max_abs={max_abs:.4}");
    assert!(max_abs < 0.1, "max_abs={max_abs} above F16 tolerance");
}

#[test]
fn causal_conv3d_bf16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv3dCausalKernels::for_context(&ctx).expect("compile conv3d_causal");
    let (b, c_in, t, h, w) = (1, 2, 6, 5, 5);
    let (c_out, kt, kh, kw, stride) = (4, 3, 3, 3, 1);
    let to = t_out(t, stride);
    let ho = spatial_out(h, kh, stride);
    let wo = spatial_out(w, kw, stride);

    let inp_f = det_f32(0xA1A2, b * c_in * t * h * w, 0.5);
    let w_f = det_f32(0xB1B2, c_out * c_in * kt * kh * kw, 0.3);
    let inp: Vec<bf16> = inp_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let we: Vec<bf16> = w_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let inp_back: Vec<f32> = inp.iter().map(|v| v.to_f32()).collect();
    let w_back: Vec<f32> = we.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_conv3d_causal(
        &inp_back, &w_back, None, b, c_in, t, h, w, c_out, kt, kh, kw, stride,
    );

    let dev_in: CudaSlice<bf16> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<bf16> = stream.clone_htod(&we).unwrap();
    let mut dev_out: CudaSlice<bf16> = stream.alloc_zeros(b * c_out * to * ho * wo).unwrap();
    conv3d_causal_bf16(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        None,
        &mut dev_out,
        b as u32,
        c_in as u32,
        t as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kt as u32,
        kh as u32,
        kw as u32,
        stride as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got_b: Vec<bf16> = stream.clone_dtoh(&dev_out).unwrap();
    let got: Vec<f32> = got_b.iter().map(|v| v.to_f32()).collect();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[conv3d_causal_bf16] max_abs={max_abs:.4}");
    assert!(max_abs < 0.5, "max_abs={max_abs} above BF16 tolerance");
}

#[test]
fn causal_conv3d_zero_pads_left() {
    // Verify causal padding: при kt=3 первый временной выход видит только x[0]
    // (через kt=2-й kernel-tap), второй — x[0..2], начиная с третьего — полные kt=3.
    // Чтобы убедиться что pad-нули слева — а не справа — собираем kernel который
    // активирует только первый kernel-tap (dt=0) и сравниваем head выходов.
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv3dCausalKernels::for_context(&ctx).expect("compile conv3d_causal");
    let (b, c_in, t, h, w) = (1, 1, 5, 1, 1);
    let (c_out, kt, kh, kw, stride) = (1, 3, 1, 1, 1);
    let to = t_out(t, stride);
    let ho = spatial_out(h, kh, stride);
    let wo = spatial_out(w, kw, stride);

    let inp: Vec<f32> = (0..t).map(|i| (i + 1) as f32).collect();
    let mut we = vec![0.0f32; kt];
    we[0] = 1.0;
    let expected = cpu_conv3d_causal(&inp, &we, None, b, c_in, t, h, w, c_out, kt, kh, kw, stride);

    let dev_in: CudaSlice<f32> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<f32> = stream.clone_htod(&we).unwrap();
    let mut dev_out: CudaSlice<f32> = stream.alloc_zeros(b * c_out * to * ho * wo).unwrap();
    conv3d_causal_f32(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        None,
        &mut dev_out,
        b as u32,
        c_in as u32,
        t as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kt as u32,
        kh as u32,
        kw as u32,
        stride as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dev_out).unwrap();

    // dt=0 + causal-pad → first 2 outputs must be zeros (tp=0,1 < kt-1=2)
    assert_eq!(got[0], 0.0, "first causal output must be zero (left-pad)");
    assert_eq!(got[1], 0.0, "second causal output must be zero (left-pad)");
    assert_eq!(got[2], inp[0], "third output equals x[0] with dt=0 tap");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1e-6, "mismatch at {i}: got={g} exp={e}");
    }
}
