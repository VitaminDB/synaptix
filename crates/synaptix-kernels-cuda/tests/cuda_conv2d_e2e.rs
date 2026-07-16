#![cfg(feature = "cuda")]

//! Полная цепочка Tensor::conv2d на CUDA vs ручной CPU-эталон. Без cutlass это
//! проверяет decutlass-fallback: Cout%8==0 → im2col + best_cu NN (K-tail для
//! K=Cin·KH·KW), Cout%8≠0 → direct conv2d (native). С cutlass — implicit-GEMM.

use half::f16;

use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;

fn have_gpu() -> bool {
    synaptix_core::device::cuda::get(0).is_ok()
}

fn det(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            ((u as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

// Прямая свёртка NCHW, stride=1, pad=1, f64-аккум.
#[allow(clippy::too_many_arguments)]
fn cpu_conv(
    x: &[f32],
    wt: &[f32],
    bn: usize,
    cin: usize,
    h: usize,
    w: usize,
    cout: usize,
    kh: usize,
    kw: usize,
) -> Vec<f32> {
    let (ph, pw) = (kh / 2, kw / 2);
    let mut out = vec![0.0f32; bn * cout * h * w];
    for n in 0..bn {
        for oc in 0..cout {
            for oy in 0..h {
                for ox in 0..w {
                    let mut acc = 0.0f64;
                    for ic in 0..cin {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy as isize + ky as isize - ph as isize;
                                let ix = ox as isize + kx as isize - pw as isize;
                                if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                    let xv = x[((n * cin + ic) * h + iy as usize) * w + ix as usize];
                                    let wv = wt[((oc * cin + ic) * kh + ky) * kw + kx];
                                    acc += xv as f64 * wv as f64;
                                }
                            }
                        }
                    }
                    out[((n * cout + oc) * h + oy) * w + ox] = acc as f32;
                }
            }
        }
    }
    out
}

#[test]
fn conv2d_e2e_cuda_vs_cpu() {
    synaptix_kernels_cuda::ensure_registered();
    if !have_gpu() {
        return;
    }
    let _nograd = synaptix_core::grad::NoGradGuard::new();
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    // (B, Cin, H, W, Cout, KH, KW, label) — 3x3 stride1 pad1.
    for &(bn, cin, h, w, cout, kh, kw, label) in &[
        (1usize, 8usize, 16usize, 16usize, 16usize, 3usize, 3usize, "Cout%8==0 (im2col+best_cu)"),
        (1, 8, 16, 16, 3, 3, 3, "Cout=3 (direct)"),
        (1, 16, 8, 8, 4, 3, 3, "Cout=4 (direct)"),
    ] {
        let x_host = det(0x1234 + (cin * h * w) as u64, bn * cin * h * w, 0.3);
        let w_host = det(0x5678 + (cout * cin) as u64, cout * cin * kh * kw, 0.3);
        let want = cpu_conv(&x_host, &w_host, bn, cin, h, w, cout, kh, kw);

        let xf: Vec<f16> = x_host.iter().map(|&v| f16::from_f32(v)).collect();
        let wf: Vec<f16> = w_host.iter().map(|&v| f16::from_f32(v)).collect();
        let xt = Tensor::from_vec(xf, (bn, cin, h, w), Device::Cuda(0)).unwrap();
        let wtt = Tensor::from_vec(wf, (cout, cin, kh, kw), Device::Cuda(0)).unwrap();
        let out = xt.conv2d(&wtt, None, (1, 1), (1, 1)).unwrap();
        assert_eq!(out.dims(), &[bn, cout, h, w]);
        stream.synchronize().unwrap();
        let bytes: Vec<u8> = stream
            .clone_dtoh(out.storage().as_cuda().unwrap().slice())
            .unwrap();
        let got: Vec<f32> = bytemuck::cast_slice::<u8, f16>(&bytes)
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let cos = cos_sim(&got, &want);
        eprintln!("[conv2d {label} B{bn} {cin}->{cout} {h}x{w} k{kh}] vs CPU cos={cos:.6}");
        assert!(cos >= 0.99, "conv2d {label} cos={cos} < 0.99");
    }
}
