//! AVX2 + FMA SGEMM micro-kernel.
//!
//! Параметры тайлинга: MR=6, NR=16 — 12 ymm-аккумуляторов C + 2 загрузки B + 1
//! broadcast A = 15 регистров из 16 (раньше 8×16 требовало 19 и компилятор
//! выгружал аккумуляторы в память на каждой итерации K). Outer-product по K.
//! Хвосты M/N покрываются скалярным naive fallback.
//!
//! `sgemm_avx2` — runtime-dispatch: проверяет `is_x86_feature_detected!("avx2","fma")` и
//! либо вызывает unsafe-implementation, либо `sgemm_naive_fallback`. Acc-семантика:
//! `C += A·B` (BLAS sgemm без β-scaling).

pub(crate) const MR: usize = 6;
pub(crate) const NR: usize = 16;

pub fn sgemm_avx2(
    m: usize, n: usize, k: usize,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { sgemm_avx2_impl(m, n, k, a, lda, b, ldb, c, ldc) };
            return;
        }
    }
    sgemm_naive_fallback(m, n, k, a, lda, b, ldb, c, ldc);
}

/// Scalar fallback. Acc-семантика: `C[i,j] += Σ A[i,p]·B[p,j]`.
pub(crate) fn sgemm_naive_fallback(
    m: usize, n: usize, k: usize,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k { acc += a[i * lda + p] * b[p * ldb + j]; }
            c[i * ldc + j] += acc;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sgemm_avx2_impl(
    m: usize, n: usize, k: usize,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
) {
    use std::arch::x86_64::*;

    let m_main = (m / MR) * MR;
    let n_main = (n / NR) * NR;
    let (ap, bp, cp) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());

    for i0 in (0..m_main).step_by(MR) {
        for j0 in (0..n_main).step_by(NR) {
            let mut acc = [[_mm256_setzero_ps(); 2]; MR];
            for (r, row) in acc.iter_mut().enumerate() {
                row[0] = _mm256_loadu_ps(cp.add((i0 + r) * ldc + j0));
                row[1] = _mm256_loadu_ps(cp.add((i0 + r) * ldc + j0 + 8));
            }
            for p in 0..k {
                let b0 = _mm256_loadu_ps(bp.add(p * ldb + j0));
                let b1 = _mm256_loadu_ps(bp.add(p * ldb + j0 + 8));
                for (r, row) in acc.iter_mut().enumerate() {
                    let av = _mm256_broadcast_ss(&*ap.add((i0 + r) * lda + p));
                    row[0] = _mm256_fmadd_ps(av, b0, row[0]);
                    row[1] = _mm256_fmadd_ps(av, b1, row[1]);
                }
            }
            for (r, row) in acc.iter().enumerate() {
                _mm256_storeu_ps(cp.add((i0 + r) * ldc + j0), row[0]);
                _mm256_storeu_ps(cp.add((i0 + r) * ldc + j0 + 8), row[1]);
            }
        }
    }

    // Хвосты M/N — наивный fallback.
    tail_naive(m, n, k, a, lda, b, ldb, c, ldc, m_main, n_main);
}

/// Скалярная доработка хвостов: правая полоса (i ∈ [0, m_main), j ∈ [n_main, n)) и
/// нижняя полоса (i ∈ [m_main, m), j ∈ [0, n)).
pub(crate) fn tail_naive(
    m: usize, n: usize, k: usize,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
    m_main: usize, n_main: usize,
) {
    if n_main < n {
        for i in 0..m_main {
            for j in n_main..n {
                let mut acc = 0.0f32;
                for p in 0..k { acc += a[i * lda + p] * b[p * ldb + j]; }
                c[i * ldc + j] += acc;
            }
        }
    }
    if m_main < m {
        for i in m_main..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k { acc += a[i * lda + p] * b[p * ldb + j]; }
                c[i * ldc + j] += acc;
            }
        }
    }
}
