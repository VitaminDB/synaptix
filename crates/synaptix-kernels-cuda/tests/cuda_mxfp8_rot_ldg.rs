//! MXFP8 rot для K, кратного 128 (не 512): шкалы через cp.async producer-варпов
//! (SF_LDG) — бит-в-бит с cp.async-ядром gn_mxfp8_128x128 (тот же k-порядок
//! mma), в том числе при хвосте M.

use cudarc::driver::CudaSlice;
use half::f16;

use synaptix_kernels_cuda::best_cu::gemm::gemm_mxfp8::{gemm_mxfp8, gemm_mxfp8_rot, GemmMxFp8Kernels};
use synaptix_kernels_cuda::elementwise::quant::{mxfp8_quant_natural, Mxfp8QuantKernels};

fn det_f16(seed: u64, n: usize, scale: f32) -> Vec<f16> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            // разброс порядков по блокам — разные E8M0-шкалы
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let e = ((u >> 3) % 7) as i32 - 3;
            f16::from_f32(f * scale * 2f32.powi(e))
        })
        .collect()
}

fn check(m: u32, n: u32, k: u32) {
    check_dt(m, n, k, false);
    check_dt(m, n, k, true);
}

fn check_dt(m: u32, n: u32, k: u32, bf16: bool) {
    let Ok(ctx) = synaptix_core::device::cuda::get(0) else { return };
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let Ok(qk) = Mxfp8QuantKernels::for_context(&ctx) else { return };
    let Ok(gk) = (if bf16 { GemmMxFp8Kernels::for_context_bf16(&ctx) } else { GemmMxFp8Kernels::for_context(&ctx) }) else { return };
    let mp = m.div_ceil(128) * 128;
    let (mu, mpu, nu, ku) = (m as usize, mp as usize, n as usize, k as usize);
    let w: CudaSlice<f16> = stream.clone_htod(&det_f16(7 + k as u64, nu * ku, 0.5)).unwrap();
    let mut xh = det_f16(11 + n as u64, mpu * ku, 0.5);
    for v in &mut xh[mu * ku..] {
        *v = f16::ZERO;
    }
    let x: CudaSlice<f16> = stream.clone_htod(&xh).unwrap();
    let mut wq: CudaSlice<u8> = stream.alloc_zeros(nu * ku).unwrap();
    let mut sb: CudaSlice<u8> = stream.alloc_zeros(nu * ku / 32).unwrap();
    let mut xq: CudaSlice<u8> = stream.alloc_zeros(mpu * ku).unwrap();
    let mut sa: CudaSlice<u8> = stream.alloc_zeros(mpu * ku / 32).unwrap();
    mxfp8_quant_natural(&qk, &stream, &w.as_view(), &mut wq, &mut sb, n, k).unwrap();
    mxfp8_quant_natural(&qk, &stream, &x.as_view(), &mut xq, &mut sa, mp, k).unwrap();

    let mut base: CudaSlice<f16> = stream.alloc_zeros(mpu * nu).unwrap();
    gemm_mxfp8(&gk, &stream, &xq, &wq, &sa, &sb, &mut base.slice_mut(..), mp, n, k).unwrap();
    // rot на ровно m строк: хвост A TMA читает OOB-нулями, шкалы — cp.async с нулём
    let xq_m = xq.slice(0..mu * ku);
    let sa_m = sa.slice(0..mu * ku / 32);
    let mut xq_c: CudaSlice<u8> = stream.alloc_zeros(mu * ku).unwrap();
    let mut sa_c: CudaSlice<u8> = stream.alloc_zeros(mu * ku / 32).unwrap();
    stream.memcpy_dtod(&xq_m, &mut xq_c).unwrap();
    stream.memcpy_dtod(&sa_m, &mut sa_c).unwrap();
    let mut rot: CudaSlice<f16> = stream.alloc_zeros(mu * nu).unwrap();
    let reps: usize = std::env::var("LDG_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let b = stream.clone_dtoh(&base).unwrap();
    for rep in 0..reps {
    gemm_mxfp8_rot(&gk, &stream, &xq_c, &wq, &sa_c, &sb, &mut rot.slice_mut(..), m, n, k, true).unwrap();
    let r = stream.clone_dtoh(&rot).unwrap();
    if reps > 1 { eprintln!("rep {rep}"); }
    let mut bad = 0usize;
    for i in 0..mu * nu {
        if b[i].to_bits() != r[i].to_bits() {
            if bad < 6 {
                let cv = |v: f16| if bf16 { f32::from_bits((v.to_bits() as u32) << 16) } else { v.to_f32() };
                eprintln!("{m}x{n}x{k} bf16={bf16} rep={rep}: [{}/{}] base {} rot {}", i / nu, i % nu, cv(b[i]), cv(r[i]));
            }
            bad += 1;
        }
    }
    let nz = r.iter().filter(|v| v.to_f32() != 0.0).count();
    assert!(nz > mu * nu / 2, "{m}x{n}x{k}: выход почти нулевой");
    assert_eq!(bad, 0, "{m}x{n}x{k} bf16={bf16}: {bad} расхождений rot(ldg) vs base");
    }
}

#[test]
fn mxfp8_rot_ldg_k5376() {
    check(256, 384, 5376);
    check(200, 384, 5376);
}

#[test]
fn mxfp8_rot_ldg_k2816_small_m() {
    check(64, 256, 2816);
    check(37, 512, 2816);
    check(150, 256, 2816);
}

#[test]
fn mxfp8_rot_ldg_k640() {
    check(384, 128, 640);
}

#[test]
fn mxfp8_rot_ldg_h3_shapes() {
    check(4229, 5376, 5376);
    check(4229, 21504, 5376);
}

#[test]
fn mxfp8_rot_tma_k4096_stress() {
    check(4229, 4096, 4096);
    check(4229, 16384, 4096);
}
