
use half::{bf16, f16};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use synaptix_kernels_cuda::conv::conv2d::{
    conv2d_bf16, conv2d_f16, conv2d_f32, out_dim, Conv2dKernels,
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
fn cpu_conv2d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    b: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> Vec<f32> {
    let h_out = out_dim(h, kh, sh, ph);
    let w_out = out_dim(w, kw, sw, pw);
    let mut out = vec![0.0_f32; b * c_out * h_out * w_out];
    for bi in 0..b {
        for co in 0..c_out {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut acc = 0.0_f32;
                    let h_in_base = ho as isize * sh as isize - ph as isize;
                    let w_in_base = wo as isize * sw as isize - pw as isize;
                    for ci in 0..c_in {
                        for ki in 0..kh {
                            let h_in = h_in_base + ki as isize;
                            if h_in < 0 || h_in >= h as isize {
                                continue;
                            }
                            for kj in 0..kw {
                                let w_in = w_in_base + kj as isize;
                                if w_in < 0 || w_in >= w as isize {
                                    continue;
                                }
                                let x = input
                                    [((bi * c_in + ci) * h + h_in as usize) * w + w_in as usize];
                                let we = weight[((co * c_in + ci) * kh + ki) * kw + kj];
                                acc += x * we;
                            }
                        }
                    }
                    if let Some(bs) = bias {
                        acc += bs[co];
                    }
                    out[((bi * c_out + co) * h_out + ho) * w_out + wo] = acc;
                }
            }
        }
    }
    out
}

#[test]
fn conv2d_f32_no_pad() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv2dKernels::for_context(&ctx).expect("compile conv2d");
    let b = 1usize;
    let c_in = 3usize;
    let h = 8usize;
    let w = 8usize;
    let c_out = 4usize;
    let kh = 3usize;
    let kw = 3usize;
    let sh = 1usize;
    let sw = 1usize;
    let ph = 0usize;
    let pw = 0usize;
    let h_out = out_dim(h, kh, sh, ph);
    let w_out = out_dim(w, kw, sw, pw);

    let inp = det_f32(0xA110, b * c_in * h * w, 0.5);
    let we = det_f32(0xB220, c_out * c_in * kh * kw, 0.3);
    let bs = det_f32(0xCC33, c_out, 0.2);
    let expected = cpu_conv2d(
        &inp,
        &we,
        Some(&bs),
        b,
        c_in,
        h,
        w,
        c_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
    );

    let dev_in: CudaSlice<f32> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<f32> = stream.clone_htod(&we).unwrap();
    let dev_b: CudaSlice<f32> = stream.clone_htod(&bs).unwrap();
    let mut dev_out: CudaSlice<f32> = stream.alloc_zeros(b * c_out * h_out * w_out).unwrap();
    conv2d_f32(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        Some(&dev_b),
        &mut dev_out,
        b as u32,
        c_in as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kh as u32,
        kw as u32,
        sh as u32,
        sw as u32,
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
    eprintln!("[conv2d_f32 no_pad] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-5);
}

#[test]
fn conv2d_f32_pad_stride() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv2dKernels::for_context(&ctx).expect("compile conv2d");
    let b = 2usize;
    let c_in = 4usize;
    let h = 16usize;
    let w = 16usize;
    let c_out = 8usize;
    let kh = 5usize;
    let kw = 5usize;
    let sh = 2usize;
    let sw = 2usize;
    let ph = 2usize;
    let pw = 2usize;
    let h_out = out_dim(h, kh, sh, ph);
    let w_out = out_dim(w, kw, sw, pw);

    let inp = det_f32(0xD414, b * c_in * h * w, 0.4);
    let we = det_f32(0xE525, c_out * c_in * kh * kw, 0.2);
    let expected = cpu_conv2d(
        &inp, &we, None, b, c_in, h, w, c_out, kh, kw, sh, sw, ph, pw,
    );

    let dev_in: CudaSlice<f32> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<f32> = stream.clone_htod(&we).unwrap();
    let mut dev_out: CudaSlice<f32> = stream.alloc_zeros(b * c_out * h_out * w_out).unwrap();
    conv2d_f32(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        None,
        &mut dev_out,
        b as u32,
        c_in as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kh as u32,
        kw as u32,
        sh as u32,
        sw as u32,
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
    eprintln!("[conv2d_f32 pad_stride] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-5);
}

#[allow(clippy::too_many_arguments)]
fn cpu_group_norm(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    b: usize,
    c: usize,
    hw: usize,
    g: usize,
    eps: f32,
) -> Vec<f32> {
    let pg = c / g;
    let mut out = vec![0.0_f32; b * c * hw];
    for bi in 0..b {
        for gi in 0..g {
            let (mut sum, mut sq) = (0.0_f64, 0.0_f64);
            for pc in 0..pg {
                let ch = gi * pg + pc;
                for s in 0..hw {
                    let v = x[(bi * c + ch) * hw + s] as f64;
                    sum += v;
                    sq += v * v;
                }
            }
            let n = (pg * hw) as f64;
            let mean = sum / n;
            let var = (sq / n - mean * mean).max(0.0);
            let inv = (1.0 / (var + eps as f64).sqrt()) as f32;
            let mean = mean as f32;
            for pc in 0..pg {
                let ch = gi * pg + pc;
                for s in 0..hw {
                    let idx = (bi * c + ch) * hw + s;
                    out[idx] = (x[idx] - mean) * inv * w[ch] + bias[ch];
                }
            }
        }
    }
    out
}

#[test]
fn geglu_split_tensor_f32() {
    use synaptix_core::device::Device;
    use synaptix_core::tensor::Tensor;
    let Some(_) = setup() else { return };
    synaptix_kernels_cuda::cuda_backend::ensure_registered();

    let t = 64usize;
    let inner = 320usize;
    let p = det_f32(0x9E91, t * 2 * inner, 1.2);
    let mut expected = vec![0.0_f32; t * inner];
    let ge = |v: f32| 0.5 * v * (1.0 + libm_erf(v * 0.70710678118654752_f32));
    for r in 0..t {
        for i in 0..inner {
            let val = p[r * 2 * inner + i];
            let gate = p[r * 2 * inner + inner + i];
            expected[r * inner + i] = val * ge(gate);
        }
    }
    let pt = Tensor::from_vec(p, vec![t, 2 * inner], Device::Cuda(0)).unwrap();
    let out = pt.geglu_split().unwrap();
    assert_eq!(out.dims(), &[t, inner]);
    let got: Vec<f32> = out
        .reshape(vec![t * inner])
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[geglu_split f32] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-4, "max_abs={max_abs}");
}

// erf через ряд/прибл. (для теста, без libm-зависимости).
fn libm_erf(x: f32) -> f32 {
    // Abramowitz-Stegun 7.1.26, точности ~1e-7 достаточно для tol 1e-4.
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    if x < 0.0 {
        -y
    } else {
        y
    }
}

#[test]
fn group_norm_tensor_f32() {
    use synaptix_core::device::Device;
    use synaptix_core::tensor::Tensor;
    let Some(_) = setup() else { return };
    synaptix_kernels_cuda::cuda_backend::ensure_registered();

    let b = 2usize;
    let c = 256usize;
    let h = 24usize;
    let w = 24usize;
    let g = 32usize;
    let hw = h * w;
    let eps = 1e-6_f32;
    let x = det_f32(0x6611, b * c * hw, 1.5);
    let wt = det_f32(0x6622, c, 0.7);
    let bs = det_f32(0x6633, c, 0.3);
    let expected = cpu_group_norm(&x, &wt, &bs, b, c, hw, g, eps);

    let xt = Tensor::from_vec(x, vec![b, c, h, w], Device::Cuda(0)).unwrap();
    let wtt = Tensor::from_vec(wt, vec![c], Device::Cuda(0)).unwrap();
    let bst = Tensor::from_vec(bs, vec![c], Device::Cuda(0)).unwrap();
    let out = xt
        .group_norm_fused(Some(&wtt), Some(&bst), g, eps, false)
        .unwrap();
    assert_eq!(out.dims(), &[b, c, h, w]);
    let got: Vec<f32> = out
        .reshape(vec![b * c * hw])
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[group_norm tensor f32] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-3, "max_abs={max_abs}");

    // С fused SiLU: ожидаем silu(group_norm(x)) = norm * sigmoid(norm).
    let xt2 = Tensor::from_vec(
        det_f32(0x6611, b * c * hw, 1.5),
        vec![b, c, h, w],
        Device::Cuda(0),
    )
    .unwrap();
    let out_s = xt2
        .group_norm_fused(Some(&wtt), Some(&bst), g, eps, true)
        .unwrap();
    let got_s: Vec<f32> = out_s
        .reshape(vec![b * c * hw])
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut max_abs_s = 0.0_f32;
    for i in 0..got_s.len() {
        let n = expected[i];
        let silu = n / (1.0 + (-n).exp());
        max_abs_s = max_abs_s.max((got_s[i] - silu).abs());
    }
    eprintln!("[group_norm+silu] max_abs={max_abs_s:.6}");
    assert!(max_abs_s < 1e-3, "max_abs={max_abs_s}");
}

// ─── Tensor-уровень: Backend::conv2d через ops-dispatch (Tensor::conv2d) ───

#[test]
fn im2col_tensor_f32() {
    use synaptix_core::device::Device;
    use synaptix_core::tensor::Tensor;
    let Some(_) = setup() else { return };
    synaptix_kernels_cuda::cuda_backend::ensure_registered();

    let b = 2usize;
    let c_in = 3usize;
    let h = 8usize;
    let w = 8usize;
    let kh = 3usize;
    let kw = 3usize;
    let sh = 1usize;
    let sw = 1usize;
    let ph = 1usize;
    let pw = 1usize;
    let h_out = out_dim(h, kh, sh, ph);
    let w_out = out_dim(w, kw, sw, pw);
    let m = b * h_out * w_out;
    let k = c_in * kh * kw;
    let inp = det_f32(0x77AA, b * c_in * h * w, 0.5);

    // CPU-эталон im2col → col[M,K], k=(c,kh,kw), m=(b,ho,wo).
    let mut expected = vec![0.0_f32; m * k];
    for bi in 0..b {
        for ho in 0..h_out {
            for wo in 0..w_out {
                let mrow = (bi * h_out + ho) * w_out + wo;
                for c in 0..c_in {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let kcol = (c * kh + ki) * kw + kj;
                            let h_in = ho as isize * sh as isize - ph as isize + ki as isize;
                            let w_in = wo as isize * sw as isize - pw as isize + kj as isize;
                            let v =
                                if h_in >= 0 && h_in < h as isize && w_in >= 0 && w_in < w as isize
                                {
                                    inp[((bi * c_in + c) * h + h_in as usize) * w + w_in as usize]
                                } else {
                                    0.0
                                };
                            expected[mrow * k + kcol] = v;
                        }
                    }
                }
            }
        }
    }

    let x = Tensor::from_vec(inp, vec![b, c_in, h, w], Device::Cuda(0)).unwrap();
    let col = x
        .im2col(kh, kw, (sh, sw), (ph, pw), h_out, w_out, 0, m)
        .unwrap();
    assert_eq!(col.dims(), &[m, k]);
    let got: Vec<f32> = col.reshape(vec![m * k]).unwrap().to_vec1::<f32>().unwrap();
    let mut maxd = 0.0_f32;
    for i in 0..got.len() {
        maxd = maxd.max((got[i] - expected[i]).abs());
    }
    eprintln!("[im2col f32] max_abs={maxd:.6}");
    assert_eq!(maxd, 0.0);

    // Row-range (tiling): rows [m0, m0+mc) должны совпасть со срезом полного col.
    let m0 = 7usize;
    let mc = 11usize;
    let x2 = Tensor::from_vec(
        det_f32(0x77AA, b * c_in * h * w, 0.5),
        vec![b, c_in, h, w],
        Device::Cuda(0),
    )
    .unwrap();
    let sub = x2
        .im2col(kh, kw, (sh, sw), (ph, pw), h_out, w_out, m0, mc)
        .unwrap();
    assert_eq!(sub.dims(), &[mc, k]);
    let sub_got: Vec<f32> = sub.reshape(vec![mc * k]).unwrap().to_vec1::<f32>().unwrap();
    let mut sd = 0.0_f32;
    for r in 0..mc {
        for kk in 0..k {
            sd = sd.max((sub_got[r * k + kk] - expected[(m0 + r) * k + kk]).abs());
        }
    }
    eprintln!("[im2col range] max_abs={sd:.6}");
    assert_eq!(sd, 0.0);
}

#[test]
fn conv2d_tensor_dispatch_f32() {
    use synaptix_core::device::Device;
    use synaptix_core::tensor::Tensor;
    let Some(_) = setup() else { return };
    synaptix_kernels_cuda::cuda_backend::ensure_registered();

    let b = 2usize;
    let c_in = 4usize;
    let h = 16usize;
    let w = 16usize;
    let c_out = 8usize;
    let kh = 3usize;
    let kw = 3usize;
    let sh = 1usize;
    let sw = 1usize;
    let ph = 1usize;
    let pw = 1usize;
    let h_out = out_dim(h, kh, sh, ph);
    let w_out = out_dim(w, kw, sw, pw);
    let inp = det_f32(0x1234, b * c_in * h * w, 0.5);
    let we = det_f32(0x5678, c_out * c_in * kh * kw, 0.3);
    let bs = det_f32(0x9ABC, c_out, 0.2);

    let x = Tensor::from_vec(inp.clone(), vec![b, c_in, h, w], Device::Cuda(0)).unwrap();
    let wt = Tensor::from_vec(we.clone(), vec![c_out, c_in, kh, kw], Device::Cuda(0)).unwrap();
    let bt = Tensor::from_vec(bs.clone(), vec![c_out], Device::Cuda(0)).unwrap();

    // С bias.
    let expected = cpu_conv2d(
        &inp,
        &we,
        Some(&bs),
        b,
        c_in,
        h,
        w,
        c_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
    );
    let out = x.conv2d(&wt, Some(&bt), (sh, sw), (ph, pw)).unwrap();
    assert_eq!(out.dims(), &[b, c_out, h_out, w_out]);
    let got: Vec<f32> = out
        .reshape(vec![b * c_out * h_out * w_out])
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[conv2d tensor f32 +bias] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-5);

    // Без bias.
    let expected_nb = cpu_conv2d(
        &inp, &we, None, b, c_in, h, w, c_out, kh, kw, sh, sw, ph, pw,
    );
    let out_nb = x.conv2d(&wt, None, (sh, sw), (ph, pw)).unwrap();
    let got_nb: Vec<f32> = out_nb
        .reshape(vec![b * c_out * h_out * w_out])
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut max_abs_nb = 0.0_f32;
    for i in 0..got_nb.len() {
        max_abs_nb = max_abs_nb.max((got_nb[i] - expected_nb[i]).abs());
    }
    eprintln!("[conv2d tensor f32 -bias] max_abs={max_abs_nb:.6}");
    assert!(max_abs_nb < 1e-5);
}

#[test]
fn conv2d_tensor_dispatch_stride_f32() {
    use synaptix_core::device::Device;
    use synaptix_core::tensor::Tensor;
    let Some(_) = setup() else { return };
    synaptix_kernels_cuda::cuda_backend::ensure_registered();

    let b = 1usize;
    let c_in = 3usize;
    let h = 32usize;
    let w = 32usize;
    let c_out = 6usize;
    let kh = 3usize;
    let kw = 3usize;
    let sh = 2usize;
    let sw = 2usize;
    let ph = 1usize;
    let pw = 1usize;
    let h_out = out_dim(h, kh, sh, ph);
    let w_out = out_dim(w, kw, sw, pw);
    let inp = det_f32(0x2468, b * c_in * h * w, 0.5);
    let we = det_f32(0x1357, c_out * c_in * kh * kw, 0.3);
    let expected = cpu_conv2d(
        &inp, &we, None, b, c_in, h, w, c_out, kh, kw, sh, sw, ph, pw,
    );

    let x = Tensor::from_vec(inp, vec![b, c_in, h, w], Device::Cuda(0)).unwrap();
    let wt = Tensor::from_vec(we, vec![c_out, c_in, kh, kw], Device::Cuda(0)).unwrap();
    let out = x.conv2d(&wt, None, (sh, sw), (ph, pw)).unwrap();
    assert_eq!(out.dims(), &[b, c_out, h_out, w_out]);
    let got: Vec<f32> = out
        .reshape(vec![b * c_out * h_out * w_out])
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[conv2d tensor f32 stride2] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-5);
}

#[test]
fn group_norm_nhwc_matches_cpu() {
    use synaptix_core::device::Device;
    use synaptix_core::tensor::Tensor;
    let Some(_) = setup() else { return };
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    let (b, h, w, c, g) = (2usize, 16usize, 16usize, 320usize, 32usize);
    let hw = h * w;
    let pg = c / g;
    let eps = 1e-6_f32;
    let x = det_f32(0x5151, b * hw * c, 1.4);
    let wt = det_f32(0x5252, c, 0.8);
    let bs = det_f32(0x5353, c, 0.3);
    // CPU NHWC GN ref: x[b,s,c]; group=channels; reduce over s×pg.
    let mut expected = vec![0.0_f32; b * hw * c];
    for bi in 0..b {
        for gi in 0..g {
            let (mut sum, mut sq) = (0.0_f64, 0.0_f64);
            for s in 0..hw {
                for cc in 0..pg {
                    let v = x[(bi * hw + s) * c + gi * pg + cc] as f64;
                    sum += v;
                    sq += v * v;
                }
            }
            let n = (hw * pg) as f64;
            let mean = sum / n;
            let var = (sq / n - mean * mean).max(0.0);
            let inv = (1.0 / (var + eps as f64).sqrt()) as f32;
            let mean = mean as f32;
            for s in 0..hw {
                for cc in 0..pg {
                    let ch = gi * pg + cc;
                    let idx = (bi * hw + s) * c + ch;
                    expected[idx] = (x[idx] - mean) * inv * wt[ch] + bs[ch];
                }
            }
        }
    }
    let xt = Tensor::from_vec(x, vec![b, h, w, c], Device::Cuda(0)).unwrap();
    let wtt = Tensor::from_vec(wt, vec![c], Device::Cuda(0)).unwrap();
    let bst = Tensor::from_vec(bs, vec![c], Device::Cuda(0)).unwrap();
    let out = xt
        .group_norm_fused_layout(Some(&wtt), Some(&bst), g, eps, false, true)
        .unwrap();
    let got: Vec<f32> = out.reshape(vec![b * hw * c]).unwrap().to_vec1::<f32>().unwrap();
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[group_norm NHWC] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-3, "max_abs={max_abs}");
}

#[test]
fn conv2d_implicit_bf16_matches_ref() {
    use synaptix_core::device::Device;
    use synaptix_core::tensor::Tensor;
    let Some(_) = setup() else { return };
    synaptix_kernels_cuda::cuda_backend::ensure_registered();

    let cases = [
        (2usize, 32usize, 16usize, 16usize, 64usize, 3usize, 1usize, 1usize),
        (2, 64, 16, 16, 128, 3, 1, 1),
        (1, 64, 16, 16, 64, 3, 2, 1),
        (2, 64, 16, 16, 128, 1, 1, 0),
    ];
    for (b, c_in, h, w, c_out, kk, sh, ph) in cases {
        let (kh, kw, sw, pw) = (kk, kk, sh, ph);
        let h_out = out_dim(h, kh, sh, ph);
        let w_out = out_dim(w, kw, sw, pw);
        let inp_f = det_f32(0xA1A2 ^ (c_out as u64), b * c_in * h * w, 0.5);
        let w_f = det_f32(0xB1B2 ^ (kk as u64), c_out * c_in * kh * kw, 0.3);
        let inp: Vec<bf16> = inp_f.iter().map(|v| bf16::from_f32(*v)).collect();
        let we: Vec<bf16> = w_f.iter().map(|v| bf16::from_f32(*v)).collect();
        let inp_back: Vec<f32> = inp.iter().map(|v| v.to_f32()).collect();
        let w_back: Vec<f32> = we.iter().map(|v| v.to_f32()).collect();
        let expected = cpu_conv2d(
            &inp_back, &w_back, None, b, c_in, h, w, c_out, kh, kw, sh, sw, ph, pw,
        );

        let x = Tensor::from_vec(inp, vec![b, c_in, h, w], Device::Cuda(0)).unwrap();
        let wt = Tensor::from_vec(we, vec![c_out, c_in, kh, kw], Device::Cuda(0)).unwrap();
        let out = x.conv2d(&wt, None, (sh, sw), (ph, pw)).unwrap();
        assert_eq!(out.dims(), &[b, c_out, h_out, w_out]);
        let got_b: Vec<bf16> = out
            .reshape(vec![b * c_out * h_out * w_out])
            .unwrap()
            .to_vec1::<bf16>()
            .unwrap();
        let got: Vec<f32> = got_b.iter().map(|v| v.to_f32()).collect();
        let mut max_abs = 0.0_f32;
        let mut max_ref = 0.0_f32;
        for i in 0..got.len() {
            max_abs = max_abs.max((got[i] - expected[i]).abs());
            max_ref = max_ref.max(expected[i].abs());
        }
        eprintln!(
            "[implicit conv bf16 Cin={c_in} Cout={c_out} k={kk} s={sh}] max_abs={max_abs:.4} max_ref={max_ref:.3}"
        );
        assert!(
            max_abs < 0.05 * max_ref + 0.1,
            "Cin={c_in} Cout={c_out} k={kk} s={sh}: max_abs={max_abs} max_ref={max_ref}"
        );
    }
}

#[test]
fn conv2d_nhwc_io_matches_ref() {
    use synaptix_core::device::Device;
    use synaptix_core::tensor::Tensor;
    let Some(_) = setup() else { return };
    synaptix_kernels_cuda::cuda_backend::ensure_registered();

    let cases = [
        (2usize, 32usize, 16usize, 16usize, 64usize, 3usize, 1usize, 1usize),
        (1, 64, 16, 16, 64, 3, 2, 1),
        (2, 64, 16, 16, 128, 1, 1, 0),
    ];
    for (b, c_in, h, w, c_out, kk, sh, ph) in cases {
        let (kh, kw, sw, pw) = (kk, kk, sh, ph);
        let h_out = out_dim(h, kh, sh, ph);
        let w_out = out_dim(w, kw, sw, pw);
        let inp_f = det_f32(0xC1C2 ^ (c_out as u64), b * c_in * h * w, 0.5);
        let w_f = det_f32(0xD1D2 ^ (kk as u64), c_out * c_in * kh * kw, 0.3);
        let res_f = det_f32(0xE1E2 ^ (c_out as u64), b * c_out * h_out * w_out, 0.4);
        let inp: Vec<bf16> = inp_f.iter().map(|v| bf16::from_f32(*v)).collect();
        let we: Vec<bf16> = w_f.iter().map(|v| bf16::from_f32(*v)).collect();
        let inp_back: Vec<f32> = inp.iter().map(|v| v.to_f32()).collect();
        let w_back: Vec<f32> = we.iter().map(|v| v.to_f32()).collect();
        let mut expected = cpu_conv2d(
            &inp_back, &w_back, None, b, c_in, h, w, c_out, kh, kw, sh, sw, ph, pw,
        );
        for bi in 0..b {
            for co in 0..c_out {
                for s in 0..h_out * w_out {
                    let nchw = (bi * c_out + co) * (h_out * w_out) + s;
                    let nhwc = (bi * h_out * w_out + s) * c_out + co;
                    expected[nchw] += res_f[nhwc];
                }
            }
        }

        let x = Tensor::from_vec(inp, vec![b, c_in, h, w], Device::Cuda(0)).unwrap();
        let wt = Tensor::from_vec(we, vec![c_out, c_in, kh, kw], Device::Cuda(0)).unwrap();
        let res_b: Vec<bf16> = res_f.iter().map(|v| bf16::from_f32(*v)).collect();
        let res = Tensor::from_vec(res_b, vec![b, h_out, w_out, c_out], Device::Cuda(0)).unwrap();
        let x_nhwc = x.permute(vec![0, 2, 3, 1]).unwrap().contiguous().unwrap();
        let out = x_nhwc
            .conv2d_nhwc_io(&wt, None, Some(&res), None, (sh, sw), (ph, pw), h_out, w_out)
            .unwrap();
        assert_eq!(out.dims(), &[b, h_out, w_out, c_out]);
        let out_nchw = out.permute(vec![0, 3, 1, 2]).unwrap().contiguous().unwrap();
        let got_b: Vec<bf16> = out_nchw
            .reshape(vec![b * c_out * h_out * w_out])
            .unwrap()
            .to_vec1::<bf16>()
            .unwrap();
        let got: Vec<f32> = got_b.iter().map(|v| v.to_f32()).collect();
        let mut max_abs = 0.0_f32;
        let mut max_ref = 0.0_f32;
        for i in 0..got.len() {
            max_abs = max_abs.max((got[i] - expected[i]).abs());
            max_ref = max_ref.max(expected[i].abs());
        }
        eprintln!(
            "[conv nhwc_io bf16 Cin={c_in} Cout={c_out} k={kk} s={sh}] max_abs={max_abs:.4} max_ref={max_ref:.3}"
        );
        assert!(
            max_abs < 0.05 * max_ref + 0.1,
            "Cin={c_in} Cout={c_out} k={kk} s={sh}: max_abs={max_abs} max_ref={max_ref}"
        );
    }
}

#[test]
fn conv2d_f16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv2dKernels::for_context(&ctx).expect("compile conv2d");
    let b = 1usize;
    let c_in = 4usize;
    let h = 8usize;
    let w = 8usize;
    let c_out = 8usize;
    let kh = 3usize;
    let kw = 3usize;
    let sh = 1usize;
    let sw = 1usize;
    let ph = 1usize;
    let pw = 1usize;
    let h_out = out_dim(h, kh, sh, ph);
    let w_out = out_dim(w, kw, sw, pw);
    let inp_f = det_f32(0xA1A2, b * c_in * h * w, 0.5);
    let w_f = det_f32(0xB1B2, c_out * c_in * kh * kw, 0.3);
    let inp: Vec<f16> = inp_f.iter().map(|v| f16::from_f32(*v)).collect();
    let we: Vec<f16> = w_f.iter().map(|v| f16::from_f32(*v)).collect();
    let inp_back: Vec<f32> = inp.iter().map(|v| v.to_f32()).collect();
    let w_back: Vec<f32> = we.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_conv2d(
        &inp_back, &w_back, None, b, c_in, h, w, c_out, kh, kw, sh, sw, ph, pw,
    );

    let dev_in: CudaSlice<f16> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<f16> = stream.clone_htod(&we).unwrap();
    let mut dev_out: CudaSlice<f16> = stream.alloc_zeros(b * c_out * h_out * w_out).unwrap();
    conv2d_f16(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        None,
        &mut dev_out,
        b as u32,
        c_in as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kh as u32,
        kw as u32,
        sh as u32,
        sw as u32,
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
    eprintln!("[conv2d_f16] max_abs={max_abs:.4}");
    assert!(max_abs < 0.1);
}

#[test]
fn conv2d_bf16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = Conv2dKernels::for_context(&ctx).expect("compile conv2d");
    let b = 1usize;
    let c_in = 4usize;
    let h = 8usize;
    let w = 8usize;
    let c_out = 8usize;
    let kh = 3usize;
    let kw = 3usize;
    let sh = 1usize;
    let sw = 1usize;
    let ph = 1usize;
    let pw = 1usize;
    let h_out = out_dim(h, kh, sh, ph);
    let w_out = out_dim(w, kw, sw, pw);
    let inp_f = det_f32(0xA1A2, b * c_in * h * w, 0.5);
    let w_f = det_f32(0xB1B2, c_out * c_in * kh * kw, 0.3);
    let inp: Vec<bf16> = inp_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let we: Vec<bf16> = w_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let inp_back: Vec<f32> = inp.iter().map(|v| v.to_f32()).collect();
    let w_back: Vec<f32> = we.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_conv2d(
        &inp_back, &w_back, None, b, c_in, h, w, c_out, kh, kw, sh, sw, ph, pw,
    );

    let dev_in: CudaSlice<bf16> = stream.clone_htod(&inp).unwrap();
    let dev_w: CudaSlice<bf16> = stream.clone_htod(&we).unwrap();
    let mut dev_out: CudaSlice<bf16> = stream.alloc_zeros(b * c_out * h_out * w_out).unwrap();
    conv2d_bf16(
        &kernels,
        &stream,
        &dev_in,
        &dev_w,
        None,
        &mut dev_out,
        b as u32,
        c_in as u32,
        h as u32,
        w as u32,
        c_out as u32,
        kh as u32,
        kw as u32,
        sh as u32,
        sw as u32,
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
    eprintln!("[conv2d_bf16] max_abs={max_abs:.4}");
    assert!(max_abs < 0.5);
}
