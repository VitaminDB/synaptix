//! Численная проверка SIMD SGEMM против скалярной reference.
//!
//! Все три варианта (avx2, avx512, neon) на платформах без соответствующего CPU-feature'a
//! падают в fallback (avx512 → avx2 → naive; neon → naive вне aarch64). Поэтому тесты
//! проходят везде; на x86_64 с AVX2/AVX-512 они дополнительно проверяют SIMD-путь.

use synaptix_kernels_cpu::gemm::{avx2, avx512, neon};

fn det(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            (u as f32 / u32::MAX as f32) * 2.0 * scale - scale
        })
        .collect()
}

/// Reference: чистая трёхвложенная петля без accumulate (C = A·B).
fn reference_sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, ctx: &str) {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (a, e) in actual.iter().zip(expected) {
        let abs = (a - e).abs();
        let rel = abs / e.abs().max(1.0);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_rel < tol,
        "{ctx}: max_abs={max_abs}, max_rel={max_rel}, tol={tol}"
    );
}

fn run_one<F: Fn(usize, usize, usize, &[f32], usize, &[f32], usize, &mut [f32], usize)>(
    f: F, m: usize, n: usize, k: usize, name: &str,
) {
    let a = det(1, m * k, 1.0);
    let b = det(2, k * n, 1.0);
    let expected = reference_sgemm(m, n, k, &a, &b);

    // C-buffer пред-обнулён (acc-семантика: C += A·B).
    let mut c = vec![0.0f32; m * n];
    f(m, n, k, &a, k, &b, n, &mut c, n);

    assert_close(&c, &expected, 1e-4, &format!("{name} {m}x{n}x{k}"));
}

#[test]
fn avx2_matches_reference_on_aligned_sizes() {
    // Размеры кратны MR×NR = 8×16 — main loop без хвостов.
    run_one(avx2::sgemm_avx2, 16, 32, 64, "avx2");
    run_one(avx2::sgemm_avx2, 8, 16, 1, "avx2");
    run_one(avx2::sgemm_avx2, 24, 48, 17, "avx2");
}

#[test]
fn avx2_matches_reference_on_unaligned_sizes() {
    // Хвосты M/N — должны попасть в naive fallback.
    run_one(avx2::sgemm_avx2, 7, 15, 13, "avx2-unaligned");
    run_one(avx2::sgemm_avx2, 17, 33, 23, "avx2-unaligned");
    run_one(avx2::sgemm_avx2, 1, 1, 5, "avx2-unaligned");
}

#[test]
fn avx512_matches_reference() {
    // На системе без AVX-512 fallback → avx2, тоже корректно.
    run_one(avx512::sgemm_avx512, 16, 16, 16, "avx512");
    run_one(avx512::sgemm_avx512, 32, 32, 32, "avx512");
    run_one(avx512::sgemm_avx512, 17, 31, 19, "avx512-unaligned");
    run_one(avx512::sgemm_avx512, 1, 1, 8, "avx512-tiny");
}

#[test]
fn neon_matches_reference() {
    // На x86_64 fallback на naive.
    run_one(neon::sgemm_neon, 8, 8, 8, "neon");
    run_one(neon::sgemm_neon, 16, 16, 32, "neon");
    run_one(neon::sgemm_neon, 5, 9, 11, "neon-unaligned");
    run_one(neon::sgemm_neon, 3, 3, 3, "neon-tiny");
}

#[test]
fn avx2_accumulates_into_c() {
    // Проверка accumulate-семантики (C += A·B, не overwrite).
    let m = 8;
    let n = 16;
    let k = 4;
    let a = det(3, m * k, 0.5);
    let b = det(4, k * n, 0.5);
    let initial = det(5, m * n, 0.1);
    let mut c = initial.clone();
    avx2::sgemm_avx2(m, n, k, &a, k, &b, n, &mut c, n);

    let expected_ab = reference_sgemm(m, n, k, &a, &b);
    let expected: Vec<f32> = initial
        .iter()
        .zip(expected_ab.iter())
        .map(|(i, ab)| i + ab)
        .collect();
    assert_close(&c, &expected, 1e-4, "avx2-acc");
}

#[test]
fn zero_k_no_change() {
    // K=0 — A·B пустой, C не должен изменяться.
    let m = 8;
    let n = 16;
    let initial = det(7, m * n, 0.2);
    let mut c = initial.clone();
    avx2::sgemm_avx2(m, n, 0, &[], 0, &[], n, &mut c, n);
    assert_eq!(c, initial);
}
