
//! Smoke + bit-exact тесты для всех 11 Mamba2 chunked-SSD helpers.
//! Каждый kernel сверяется с CPU-эталоном на маленьких размерах.

use half::bf16;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use synaptix_core::dtype::DType;
use synaptix_kernels_cuda::ssm::mamba2_chunked_helpers::Mamba2ChunkedHelpersKernels;

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

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
}

fn upload_f32(stream: &Arc<CudaStream>, host: &[f32]) -> CudaSlice<f32> {
    let mut d = unsafe { stream.alloc::<f32>(host.len()).expect("alloc") };
    stream.memcpy_htod(host, &mut d).expect("memcpy");
    d
}

fn upload_bf16_from_f32(stream: &Arc<CudaStream>, host: &[f32]) -> CudaSlice<bf16> {
    let h: Vec<bf16> = host.iter().map(|&x| bf16::from_f32(x)).collect();
    let mut d = unsafe { stream.alloc::<bf16>(h.len()).expect("alloc") };
    stream.memcpy_htod(&h, &mut d).expect("memcpy");
    d
}

fn alloc_f32_zeros(stream: &Arc<CudaStream>, n: usize) -> CudaSlice<f32> {
    let h = vec![0.0_f32; n];
    let mut d = unsafe { stream.alloc::<f32>(n).expect("alloc f32") };
    stream.memcpy_htod(&h, &mut d).expect("memcpy");
    d
}

fn alloc_bf16_zeros(stream: &Arc<CudaStream>, n: usize) -> CudaSlice<bf16> {
    let h = vec![bf16::ZERO; n];
    let mut d = unsafe { stream.alloc::<bf16>(n).expect("alloc bf16") };
    stream.memcpy_htod(&h, &mut d).expect("memcpy");
    d
}

fn download_f32(stream: &Arc<CudaStream>, d: &CudaSlice<f32>) -> Vec<f32> {
    let mut h = vec![0.0_f32; d.len()];
    stream.memcpy_dtoh(d, &mut h).expect("memcpy dtoh");
    h
}

fn download_bf16_as_f32(stream: &Arc<CudaStream>, d: &CudaSlice<bf16>) -> Vec<f32> {
    let mut h = vec![bf16::ZERO; d.len()];
    stream.memcpy_dtoh(d, &mut h).expect("memcpy dtoh");
    h.iter().map(|x| x.to_f32()).collect()
}

// ── 1. alpha_cum ────────────────────────────────────────────────────────────
#[test]
fn helper_alpha_cum_f32() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2ChunkedHelpersKernels::for_context(&ctx).expect("compile helpers");

    let (b, h, t, q) = (2u32, 3u32, 4u32, 8u32);
    let l = t * q;
    let bh = b * h;
    let dt = det_f32(0x10, (b * l * h) as usize, 0.3, 0.5);
    let a = det_f32(0x11, h as usize, 0.5, -1.0);

    // Expected: layout (T, BH, Q) — chunk outermost.
    let mut expect = vec![0.0_f32; (bh * t * q) as usize];
    for bi in 0..b {
        for hi in 0..h {
            for ti in 0..t {
                let mut acc = 0.0_f32;
                for j in 0..q {
                    let ll = ti * q + j;
                    let dt_v = dt[((bi * l + ll) * h + hi) as usize];
                    acc += a[hi as usize] * dt_v;
                    let bh_i = bi * h + hi;
                    expect[((ti * bh + bh_i) * q + j) as usize] = acc;
                }
            }
        }
    }

    let dt_d = upload_f32(&stream, &dt);
    let a_d = upload_f32(&stream, &a);
    let mut out = alloc_f32_zeros(&stream, (bh * t * q) as usize);
    kern.alpha_cum(&stream, &dt_d, &a_d, &mut out, b, h, t, q, DType::F32)
        .unwrap();
    stream.synchronize().unwrap();

    let got = download_f32(&stream, &out);
    let err = max_abs(&got, &expect);
    assert!(err < 1e-4, "alpha_cum max_abs={err:.3e}");
}

// ── 2. permute_blhx_to_bhtqx ────────────────────────────────────────────────
#[test]
fn helper_permute_blhx_f32_to_bf16() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2ChunkedHelpersKernels::for_context(&ctx).unwrap();

    let (b, h, t, q, x) = (2u32, 3u32, 4u32, 8u32, 16u32);
    let l = t * q;
    let bh = b * h;
    let src = det_f32(0x20, (b * l * h * x) as usize, 0.5, 0.0);

    // Expected: dst[t, bh, q, x] = bf16(src[b, l, h, x]) where l = t*q + q_idx.
    let mut expect = vec![0.0_f32; (bh * t * q * x) as usize];
    for bi in 0..b {
        for hi in 0..h {
            for ti in 0..t {
                for qi in 0..q {
                    for xi in 0..x {
                        let ll = ti * q + qi;
                        let bh_i = bi * h + hi;
                        let v = src[((bi * l * h + ll * h + hi) * x + xi) as usize];
                        expect[(((ti * bh + bh_i) * q + qi) * x + xi) as usize] =
                            bf16::from_f32(v).to_f32();
                    }
                }
            }
        }
    }

    let src_d = upload_f32(&stream, &src);
    let mut dst_d = alloc_bf16_zeros(&stream, (bh * t * q * x) as usize);
    kern.permute_blhx_to_bhtqx(&stream, &src_d, &mut dst_d, b, l, h, x, q, DType::F32)
        .unwrap();
    stream.synchronize().unwrap();

    let got = download_bf16_as_f32(&stream, &dst_d);
    let err = max_abs(&got, &expect);
    assert!(err < 1e-3, "permute max_abs={err:.3e}");
}

// ── 3. compute_dt_x ─────────────────────────────────────────────────────────
#[test]
fn helper_compute_dt_x_f32_to_bf16() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2ChunkedHelpersKernels::for_context(&ctx).unwrap();

    let (b, h, t, q, p) = (2u32, 3u32, 4u32, 8u32, 16u32);
    let l = t * q;
    let bh = b * h;
    let dt = det_f32(0x30, (b * l * h) as usize, 0.3, 0.5);
    let x = det_f32(0x31, (b * l * h * p) as usize, 0.5, 0.0);

    // Expected: dt_x[t, bh, q, p] (chunk outermost).
    let mut expect = vec![0.0_f32; (bh * t * q * p) as usize];
    for bi in 0..b {
        for hi in 0..h {
            for ti in 0..t {
                for qi in 0..q {
                    let ll = ti * q + qi;
                    let dt_v = dt[((bi * l + ll) * h + hi) as usize];
                    for pi in 0..p {
                        let x_v = x[((bi * l * h + ll * h + hi) * p + pi) as usize];
                        let bh_i = bi * h + hi;
                        expect[(((ti * bh + bh_i) * q + qi) * p + pi) as usize] =
                            bf16::from_f32(dt_v * x_v).to_f32();
                    }
                }
            }
        }
    }

    let dt_d = upload_f32(&stream, &dt);
    let x_d = upload_f32(&stream, &x);
    let mut out = alloc_bf16_zeros(&stream, (bh * t * q * p) as usize);
    kern.compute_dt_x(&stream, &dt_d, &x_d, &mut out, b, l, h, p, q, DType::F32)
        .unwrap();
    stream.synchronize().unwrap();

    let got = download_bf16_as_f32(&stream, &out);
    let err = max_abs(&got, &expect);
    assert!(err < 1e-2, "dt_x max_abs={err:.3e}");
}

// ── 4. transpose_bf16 ───────────────────────────────────────────────────────
#[test]
fn helper_transpose_bf16() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2ChunkedHelpersKernels::for_context(&ctx).unwrap();

    let (bat, r, c) = (4u32, 16u32, 32u32);
    let src = det_f32(0x40, (bat * r * c) as usize, 1.0, 0.0);
    let mut expect = vec![0.0_f32; (bat * r * c) as usize];
    for bi in 0..bat {
        for ri in 0..r {
            for ci in 0..c {
                let v = bf16::from_f32(src[((bi * r + ri) * c + ci) as usize]).to_f32();
                expect[((bi * c + ci) * r + ri) as usize] = v;
            }
        }
    }

    let src_d = upload_bf16_from_f32(&stream, &src);
    let mut dst_d = alloc_bf16_zeros(&stream, (bat * r * c) as usize);
    kern.transpose_bf16(&stream, &src_d, &mut dst_d, bat, r, c)
        .unwrap();
    stream.synchronize().unwrap();

    let got = download_bf16_as_f32(&stream, &dst_d);
    let err = max_abs(&got, &expect);
    assert!(err < 1e-6, "transpose max_abs={err:.3e}");
}

// ── 5. apply_decay_mask ─────────────────────────────────────────────────────
#[test]
fn helper_apply_decay_mask() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2ChunkedHelpersKernels::for_context(&ctx).unwrap();

    let (bht, q) = (4u32, 16u32);
    let a_intra = det_f32(0x50, (bht * q * q) as usize, 0.5, 0.0);
    let alpha_cum = det_f32(0x51, (bht * q) as usize, 0.3, -1.0);

    let mut expect = vec![0.0_f32; (bht * q * q) as usize];
    for b in 0..bht {
        for i in 0..q {
            for j in 0..q {
                let off = ((b * q + i) * q + j) as usize;
                let v = if j > i {
                    0.0
                } else {
                    let ai = alpha_cum[(b * q + i) as usize];
                    let aj = alpha_cum[(b * q + j) as usize];
                    a_intra[off] * (ai - aj).exp()
                };
                expect[off] = bf16::from_f32(v).to_f32();
            }
        }
    }

    let a_d = upload_f32(&stream, &a_intra);
    let ac_d = upload_f32(&stream, &alpha_cum);
    let mut out = alloc_bf16_zeros(&stream, (bht * q * q) as usize);
    kern.apply_decay_mask(&stream, &a_d, &ac_d, &mut out, bht, q)
        .unwrap();
    stream.synchronize().unwrap();

    let got = download_bf16_as_f32(&stream, &out);
    let err = max_abs(&got, &expect);
    assert!(err < 5e-3, "decay_mask max_abs={err:.3e}");
}

// ── 6a. col_broadcast_exp_mul ───────────────────────────────────────────────
#[test]
fn helper_col_broadcast_exp_mul_from_end_false() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2ChunkedHelpersKernels::for_context(&ctx).unwrap();

    let (bat, r, c) = (4u32, 16u32, 32u32);
    let src = det_f32(0x60, (bat * r * c) as usize, 0.5, 0.0);
    let vec_f = det_f32(0x61, (bat * r) as usize, 0.2, -0.5);

    let mut expect = vec![0.0_f32; (bat * r * c) as usize];
    for b in 0..bat {
        for ri in 0..r {
            for ci in 0..c {
                let off = ((b * r + ri) * c + ci) as usize;
                let v = bf16::from_f32(src[off]).to_f32();
                let a = vec_f[(b * r + ri) as usize].exp();
                expect[off] = bf16::from_f32(v * a).to_f32();
            }
        }
    }

    let src_d = upload_bf16_from_f32(&stream, &src);
    let vec_d = upload_f32(&stream, &vec_f);
    let mut out = alloc_bf16_zeros(&stream, (bat * r * c) as usize);
    kern.col_broadcast_exp_mul(&stream, &src_d, &vec_d, &mut out, bat, r, c, false)
        .unwrap();
    stream.synchronize().unwrap();

    let got = download_bf16_as_f32(&stream, &out);
    let err = max_abs(&got, &expect);
    assert!(err < 5e-3, "col_broadcast_exp_mul max_abs={err:.3e}");
}

// ── 6b. row_broadcast_exp_mul ───────────────────────────────────────────────
#[test]
fn helper_row_broadcast_exp_mul_from_end_true() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2ChunkedHelpersKernels::for_context(&ctx).unwrap();

    let (bat, r, c) = (4u32, 32u32, 16u32);
    let q_vec = c;
    let src = det_f32(0x70, (bat * r * c) as usize, 0.5, 0.0);
    let vec_f = det_f32(0x71, (bat * q_vec) as usize, 0.2, -0.5);

    let mut expect = vec![0.0_f32; (bat * r * c) as usize];
    for b in 0..bat {
        for ri in 0..r {
            for ci in 0..c {
                let off = ((b * r + ri) * c + ci) as usize;
                let v = bf16::from_f32(src[off]).to_f32();
                let a_end = vec_f[(b * q_vec + (q_vec - 1)) as usize];
                let a_c = vec_f[(b * q_vec + ci) as usize];
                let a = (a_end - a_c).exp();
                expect[off] = bf16::from_f32(v * a).to_f32();
            }
        }
    }

    let src_d = upload_bf16_from_f32(&stream, &src);
    let vec_d = upload_f32(&stream, &vec_f);
    let mut out = alloc_bf16_zeros(&stream, (bat * r * c) as usize);
    kern.row_broadcast_exp_mul(&stream, &src_d, &vec_d, &mut out, bat, r, c, q_vec, true)
        .unwrap();
    stream.synchronize().unwrap();

    let got = download_bf16_as_f32(&stream, &out);
    let err = max_abs(&got, &expect);
    assert!(err < 5e-3, "row_broadcast_exp_mul max_abs={err:.3e}");
}

// ── 7. state_linear_decay ───────────────────────────────────────────────────
#[test]
fn helper_state_linear_decay() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2ChunkedHelpersKernels::for_context(&ctx).unwrap();

    let (bh, p, n, t, q) = (4u32, 8u32, 16u32, 4u32, 8u32);
    let chunk = 2u32;
    let mut state = det_f32(0x80, (bh * p * n) as usize, 0.5, 0.0);
    let alpha_cum = det_f32(0x81, (bh * t * q) as usize, 0.1, -0.5);

    // Layout alpha_cum (T, BH, Q): index = (chunk * BH + bh) * Q + (Q-1).
    let mut expect = state.clone();
    for b in 0..bh {
        let a_end = alpha_cum[((chunk * bh + b) * q + (q - 1)) as usize];
        let decay = a_end.exp();
        for pi in 0..p {
            for ni in 0..n {
                let off = ((b * p + pi) * n + ni) as usize;
                expect[off] *= decay;
            }
        }
    }

    let mut state_d = upload_f32(&stream, &state);
    let ac_d = upload_f32(&stream, &alpha_cum);
    kern.state_linear_decay(&stream, &mut state_d, &ac_d, bh, p, n, t, q, chunk)
        .unwrap();
    stream.synchronize().unwrap();
    let got = download_f32(&stream, &state_d);
    let err = max_abs(&got, &expect);
    let _ = &mut state;
    assert!(err < 1e-5, "state_linear_decay max_abs={err:.3e}");
}

// ── 8. add_inplace_f32 ──────────────────────────────────────────────────────
#[test]
fn helper_add_inplace_f32() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2ChunkedHelpersKernels::for_context(&ctx).unwrap();

    let n = 1024u64;
    let dst_h = det_f32(0x90, n as usize, 1.0, 0.0);
    let src_h = det_f32(0x91, n as usize, 0.5, 0.0);
    let expect: Vec<f32> = dst_h.iter().zip(src_h.iter()).map(|(a, b)| a + b).collect();

    let mut dst_d = upload_f32(&stream, &dst_h);
    let src_d = upload_f32(&stream, &src_h);
    kern.add_inplace_f32(&stream, &mut dst_d, &src_d, n)
        .unwrap();
    stream.synchronize().unwrap();
    let got = download_f32(&stream, &dst_d);
    let err = max_abs(&got, &expect);
    assert!(err < 1e-6, "add_inplace max_abs={err:.3e}");
}

// ── 9. add_yoff_chunk ───────────────────────────────────────────────────────
#[test]
fn helper_add_yoff_chunk() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2ChunkedHelpersKernels::for_context(&ctx).unwrap();

    let (bh, t, q, p) = (4u32, 4u32, 8u32, 16u32);
    let chunk = 1u32;
    let y_intra = det_f32(0xA0, (bh * t * q * p) as usize, 1.0, 0.0);
    let y_off = det_f32(0xA1, (bh * q * p) as usize, 0.5, 0.0);

    // Layout Y_intra (T, BH, Q, P): index = ((chunk * BH + bh) * Q + q) * P + p.
    let mut expect = y_intra.clone();
    for b in 0..bh {
        for qi in 0..q {
            for pi in 0..p {
                let yi_off = (((chunk * bh + b) * q + qi) * p + pi) as usize;
                let yoff_off = ((b * q + qi) * p + pi) as usize;
                expect[yi_off] += y_off[yoff_off];
            }
        }
    }

    let mut yi_d = upload_f32(&stream, &y_intra);
    let yoff_d = upload_f32(&stream, &y_off);
    kern.add_yoff_chunk(&stream, &mut yi_d, &yoff_d, bh, t, q, p, chunk)
        .unwrap();
    stream.synchronize().unwrap();
    let got = download_f32(&stream, &yi_d);
    let err = max_abs(&got, &expect);
    assert!(err < 1e-6, "add_yoff_chunk max_abs={err:.3e}");
}

// ── 10. state_cast_f32_to_bf16 ──────────────────────────────────────────────
#[test]
fn helper_state_cast_f32_to_bf16() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2ChunkedHelpersKernels::for_context(&ctx).unwrap();

    let n = 1024u64;
    let src_h = det_f32(0xB0, n as usize, 0.5, 0.0);
    let expect: Vec<f32> = src_h.iter().map(|&x| bf16::from_f32(x).to_f32()).collect();

    let src_d = upload_f32(&stream, &src_h);
    let mut dst_d = alloc_bf16_zeros(&stream, n as usize);
    kern.state_cast_f32_to_bf16(&stream, &src_d, &mut dst_d, n)
        .unwrap();
    stream.synchronize().unwrap();
    let got = download_bf16_as_f32(&stream, &dst_d);
    let err = max_abs(&got, &expect);
    assert!(err < 1e-6, "state_cast max_abs={err:.3e}");
}

// ── 11. post ────────────────────────────────────────────────────────────────
#[test]
fn helper_post_f32_with_skip() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2ChunkedHelpersKernels::for_context(&ctx).unwrap();

    let (b, h, t, q, p) = (2u32, 3u32, 4u32, 8u32, 16u32);
    let l = t * q;
    let bh = b * h;
    let y_intra = det_f32(0xC0, (bh * t * q * p) as usize, 1.0, 0.0);
    let x = det_f32(0xC1, (b * l * h * p) as usize, 0.5, 0.0);
    let d = det_f32(0xC2, h as usize, 0.5, 0.0);

    let mut expect = vec![0.0_f32; (b * l * h * p) as usize];
    for bi in 0..b {
        for hi in 0..h {
            for ti in 0..t {
                for qi in 0..q {
                    let ll = ti * q + qi;
                    let bh_i = bi * h + hi;
                    for pi in 0..p {
                        let yi_off = (((ti * bh + bh_i) * q + qi) * p + pi) as usize;
                        let yo_off = ((bi * l * h + ll * h + hi) * p + pi) as usize;
                        expect[yo_off] = y_intra[yi_off] + d[hi as usize] * x[yo_off];
                    }
                }
            }
        }
    }

    let yi_d = upload_f32(&stream, &y_intra);
    let x_d = upload_f32(&stream, &x);
    let d_d = upload_f32(&stream, &d);
    let mut out = alloc_f32_zeros(&stream, (b * l * h * p) as usize);
    kern.post(
        &stream,
        &yi_d,
        &x_d,
        Some(&d_d),
        &mut out,
        b,
        l,
        h,
        p,
        q,
        DType::F32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got = download_f32(&stream, &out);
    let err = max_abs(&got, &expect);
    assert!(err < 1e-5, "post max_abs={err:.3e}");
}

#[test]
fn helper_post_f32_no_skip() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2ChunkedHelpersKernels::for_context(&ctx).unwrap();

    let (b, h, t, q, p) = (2u32, 3u32, 4u32, 8u32, 16u32);
    let l = t * q;
    let bh = b * h;
    let y_intra = det_f32(0xD0, (bh * t * q * p) as usize, 1.0, 0.0);
    let x = det_f32(0xD1, (b * l * h * p) as usize, 0.5, 0.0);

    let mut expect = vec![0.0_f32; (b * l * h * p) as usize];
    for bi in 0..b {
        for hi in 0..h {
            for ti in 0..t {
                for qi in 0..q {
                    let ll = ti * q + qi;
                    let bh_i = bi * h + hi;
                    for pi in 0..p {
                        let yi_off = (((ti * bh + bh_i) * q + qi) * p + pi) as usize;
                        let yo_off = ((bi * l * h + ll * h + hi) * p + pi) as usize;
                        expect[yo_off] = y_intra[yi_off];
                    }
                }
            }
        }
    }

    let yi_d = upload_f32(&stream, &y_intra);
    let x_d = upload_f32(&stream, &x);
    let mut out = alloc_f32_zeros(&stream, (b * l * h * p) as usize);
    kern.post::<f32>(
        &stream,
        &yi_d,
        &x_d,
        None,
        &mut out,
        b,
        l,
        h,
        p,
        q,
        DType::F32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got = download_f32(&stream, &out);
    let err = max_abs(&got, &expect);
    assert!(err < 1e-6, "post no-skip max_abs={err:.3e}");
}
