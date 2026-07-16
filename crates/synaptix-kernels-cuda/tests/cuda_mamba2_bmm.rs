#![cfg(feature = "cuda")]

//! Bit-exact (BF16 F32-acc) тесты для Mamba2 chunked-SSD helper `mamba2_bmm`.
//!
//! Эталон CPU: naive `f32`-acc матрицу-произведение из `bf16`-входов
//! (`f32(bf16(a)) * f32(bf16(b))` accumulate в `f32`, без финального cast).
//! Это совпадает с tensor-core BF16-in F32-acc по точности матрично; единственный
//! источник расхождения — порядок суммирования (tree vs sequential), что на
//! K≤128 даёт абсолютную ошибку ≲ 1e-3 для значений ≈ U(-1, 1).

use half::bf16;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use synaptix_kernels_cuda::ssm::mamba2_bmm::Mamba2BmmKernels;

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

fn to_bf16(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f32(x)).collect()
}

fn cpu_bmm_bf16(
    a_bf: &[bf16],
    b_bf: &[bf16],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut c = vec![0.0_f32; batch * m * n];
    for bi in 0..batch {
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0_f32;
                for ki in 0..k {
                    let av = a_bf[(bi * m + mi) * k + ki].to_f32();
                    let bv = b_bf[(bi * n + ni) * k + ki].to_f32();
                    acc += av * bv;
                }
                c[(bi * m + mi) * n + ni] = acc;
            }
        }
    }
    c
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
}

fn upload_bf16(stream: &Arc<CudaStream>, host: &[bf16]) -> CudaSlice<bf16> {
    let mut d = unsafe { stream.alloc::<bf16>(host.len()).expect("alloc") };
    stream.memcpy_htod(host, &mut d).expect("memcpy bf16");
    d
}

fn alloc_f32_zeros(stream: &Arc<CudaStream>, n: usize) -> CudaSlice<f32> {
    let host = vec![0.0_f32; n];
    let mut d = unsafe { stream.alloc::<f32>(n).expect("alloc f32") };
    stream.memcpy_htod(&host, &mut d).expect("memcpy f32");
    d
}

fn download_f32(stream: &Arc<CudaStream>, d: &CudaSlice<f32>) -> Vec<f32> {
    let mut h = vec![0.0_f32; d.len()];
    stream.memcpy_dtoh(d, &mut h).expect("memcpy dtoh");
    h
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    label: &str,
    seed_a: u64,
    seed_b: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    tol: f32,
) {
    let Some((ctx, stream)) = setup() else { return };
    let kern = Mamba2BmmKernels::for_context(&ctx).expect("compile mamba2_bmm");

    let a_f = det_f32(seed_a, batch * m * k, 0.5);
    let b_f = det_f32(seed_b, batch * n * k, 0.5);
    let a_bf = to_bf16(&a_f);
    let b_bf = to_bf16(&b_f);
    let expect = cpu_bmm_bf16(&a_bf, &b_bf, batch, m, n, k);

    let a_dev = upload_bf16(&stream, &a_bf);
    let b_dev = upload_bf16(&stream, &b_bf);
    let mut c_dev = alloc_f32_zeros(&stream, batch * m * n);

    kern.bmm(
        &stream,
        &a_dev,
        &b_dev,
        &mut c_dev,
        m as u32,
        n as u32,
        k as u32,
        batch as u32,
    )
    .expect("bmm launch");
    stream.synchronize().expect("sync");

    let got = download_f32(&stream, &c_dev);
    let err = max_abs(&got, &expect);
    assert!(
        err <= tol,
        "[{label}] bmm max_abs={err:.3e} > tol={tol:.1e} \
         (batch={batch}, M={m}, N={n}, K={k})"
    );
}

// ── Smoke: минимальные размеры, 1 warp per output tile. ─────────────
#[test]
fn bmm_smoke_16x16x8() {
    run_case("smoke", 0x100, 0x101, 1, 16, 8, 16, 5e-3);
}

#[test]
fn bmm_batched_smoke() {
    run_case("batched_smoke", 0x110, 0x111, 4, 16, 16, 32, 5e-3);
}

// ── Mamba2-2.7B chunked shapes (Q=64, P=64, N_state=128). ─────────────

/// A_intra: M=Q, K=N_state, N_out=Q. (BH·T батч)
#[test]
fn bmm_mamba2_a_intra_q64_n128() {
    run_case("a_intra_q64_n128", 0x200, 0x201, 8, 64, 64, 128, 5e-3);
}

/// Y_intra: M=Q, K=Q, N_out=P. (BH·T батч)
#[test]
fn bmm_mamba2_y_intra_q64_p64() {
    run_case("y_intra_q64_p64", 0x210, 0x211, 8, 64, 64, 64, 5e-3);
}

/// Y_off: M=Q, K=N_state, N_out=P. (BH батч, per chunk)
#[test]
fn bmm_mamba2_y_off_q64_p64_n128() {
    run_case("y_off_q64_p64_n128", 0x220, 0x221, 4, 64, 64, 128, 5e-3);
}

/// state_update: M=P, K=Q, N_out=N_state. (BH батч)
#[test]
fn bmm_mamba2_state_update_p64_n128_q64() {
    run_case("state_update", 0x230, 0x231, 4, 64, 128, 64, 5e-3);
}

// ── GQA chunked shapes: Q=16, 32. ─────────────────────────────────────

#[test]
fn bmm_mamba2_gqa_q16_n128() {
    run_case("gqa_q16_n128", 0x300, 0x301, 16, 16, 16, 128, 5e-3);
}

#[test]
fn bmm_mamba2_gqa_q32_n128() {
    run_case("gqa_q32_n128", 0x310, 0x311, 16, 32, 32, 128, 5e-3);
}

// ── Asymmetric / больший K. ──────────────────────────────────────────

#[test]
fn bmm_large_k_256() {
    run_case("large_k_256", 0x400, 0x401, 2, 32, 16, 256, 1e-2);
}

#[test]
fn bmm_rect_m_big_n_small() {
    run_case("rect_m_big_n_small", 0x410, 0x411, 2, 64, 8, 64, 5e-3);
}

#[test]
fn bmm_rect_m_small_n_big() {
    run_case("rect_m_small_n_big", 0x420, 0x421, 2, 16, 64, 64, 5e-3);
}
