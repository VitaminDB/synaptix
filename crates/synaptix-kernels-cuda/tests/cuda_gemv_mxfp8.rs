
//! Корректность best_cu gemv_mxfp8 (MXFP8 decode GEMV, SIMT-dequant):
//! y[N] = W[N,K] @ x[K], W/x = E4M3 + E8M0 per-32-block scales (natural) vs CPU-f32.
//! Host-квант (E8M0 + e4m3-encode) → gemv_mxfp8.

use cudarc::driver::CudaSlice;
use half::f16;

use synaptix_kernels_cuda::best_cu::gemv::gemv_mxfp8::{gemv_mxfp8, GemvMxfp8Kernels};

fn det_f16(seed: u64, n: usize, scale: f32) -> Vec<f16> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            f16::from_f32((((x >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale)
        })
        .collect()
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..a.len() {
        d += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    (d / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

// F32 → FP8 E4M3 byte (manual encode, как было в fp8_quant.cu).
fn e4m3_encode(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let v = x.clamp(-448.0, 448.0);
    let sign: u32 = if v.is_sign_negative() { 1 } else { 0 };
    let abs_v = v.abs();
    if abs_v == 0.0 {
        return (sign << 7) as u8;
    }
    let exp_raw = abs_v.log2().floor() as i32;
    let mut exp_biased = exp_raw + 7;
    if exp_biased < 1 {
        let m = (abs_v * 512.0).round().clamp(0.0, 7.0) as i32;
        return ((sign << 7) | m as u32) as u8;
    }
    if exp_biased > 15 {
        return ((sign << 7) | 0x7E) as u8;
    }
    let pow2 = (exp_raw as f32).exp2();
    let mut m = (((abs_v / pow2) - 1.0) * 8.0).round() as i32;
    if m == 8 {
        m = 0;
        exp_biased += 1;
        if exp_biased > 15 {
            return ((sign << 7) | 0x7E) as u8;
        }
    }
    if exp_biased == 15 && m == 7 {
        m = 6;
    }
    ((sign << 7) | ((exp_biased as u32) << 3) | (m as u32 & 7)) as u8
}

// MXFP8 natural квант: per-32-block E8M0 scale + e4m3 байты. → (fp8[rows*k], scales[rows*k/32]).
fn mxfp8_quant_natural(x: &[f16], rows: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let kb = k / 32;
    let mut fp8 = vec![0u8; rows * k];
    let mut scales = vec![0u8; rows * kb];
    for r in 0..rows {
        for b in 0..kb {
            let mut amax = 0.0f32;
            for i in 0..32 {
                amax = amax.max(x[r * k + b * 32 + i].to_f32().abs());
            }
            let sbyte = ((f32::from_bits(amax.to_bits() & 0x7F80_0000) / 256.0).to_bits() >> 23) as u8;
            scales[r * kb + b] = sbyte;
            let sv = f32::from_bits((sbyte as u32) << 23).max(1e-12);
            for i in 0..32 {
                fp8[r * k + b * 32 + i] = e4m3_encode(x[r * k + b * 32 + i].to_f32() / sv);
            }
        }
    }
    (fp8, scales)
}

#[test]
fn gemv_mxfp8_vs_cpu_f32() {
    synaptix_kernels_cuda::ensure_registered();
    let Some(ctx) = synaptix_core::device::cuda::get(0).ok() else {
        return;
    };
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let gk = GemvMxfp8Kernels::for_context(&ctx).expect("gemv_mxfp8 compile");

    // (N, K) — decode M=1, K%32==0.
    for &(n, k) in &[(256u32, 512u32), (512, 256), (5120, 5120)] {
        let (nu, ku) = (n as usize, k as usize);
        let w = det_f16(0xA110_C8E1, nu * ku, 0.5);
        let x = det_f16(0xC0DE_BA5E, ku, 0.5);
        let (wq, sw) = mxfp8_quant_natural(&w, nu, ku);
        let (xq, sx) = mxfp8_quant_natural(&x, 1, ku);

        let dwq: CudaSlice<u8> = stream.clone_htod(&wq).unwrap();
        let dsw: CudaSlice<u8> = stream.clone_htod(&sw).unwrap();
        let dxq: CudaSlice<u8> = stream.clone_htod(&xq).unwrap();
        let dsx: CudaSlice<u8> = stream.clone_htod(&sx).unwrap();
        let mut y: CudaSlice<f16> = stream.alloc_zeros(nu).unwrap();
        gemv_mxfp8(&gk, &stream, &dwq, &dsw, &dxq, &dsx, &mut y.as_view_mut(), n, k).unwrap();
        stream.synchronize().unwrap();

        // CPU-f32 reference по исходным f16.
        let mut yref = vec![0.0f32; nu];
        for r in 0..nu {
            let mut acc = 0.0f64;
            for kk in 0..ku {
                acc += w[r * ku + kk].to_f32() as f64 * x[kk].to_f32() as f64;
            }
            yref[r] = acc as f32;
        }
        let y_h: Vec<f16> = stream.clone_dtoh(&y).unwrap();
        let got: Vec<f32> = y_h.iter().map(|v| v.to_f32()).collect();
        let cos = cos_sim(&got, &yref);
        eprintln!("[gemv_mxfp8 N={n} K={k}] vs CPU cos={cos:.6}");
        assert!(cos >= 0.95, "gemv_mxfp8 N={n} K={k} cos={cos} < 0.95");
    }
}
