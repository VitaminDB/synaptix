//! AVX2 + FMA SGEMM micro-kernel.
//!
//! Параметры тайлинга: MR=8, NR=16 (16 регистров ymm для C — занимает весь регистровый
//! файл x86_64). Outer-product по K с broadcast A и двумя ymm-load'ами B на K-итерацию.
//! Хвосты M/N покрываются скалярным naive fallback.
//!
//! `sgemm_avx2` — runtime-dispatch: проверяет `is_x86_feature_detected!("avx2","fma")` и
//! либо вызывает unsafe-implementation, либо `sgemm_naive_fallback`. Acc-семантика:
//! `C += A·B` (BLAS sgemm без β-scaling).

const MR: usize = 8;
const NR: usize = 16;

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

    // Главный 8×16 outer-product loop.
    for i0 in (0..m_main).step_by(MR) {
        for j0 in (0..n_main).step_by(NR) {
            // 16 ymm регистров под C[8×16] = 8 строк × 2 ymm (left[0..8], right[8..16]).
            let mut c00 = _mm256_loadu_ps(c.as_ptr().add(i0 * ldc + j0));
            let mut c01 = _mm256_loadu_ps(c.as_ptr().add(i0 * ldc + j0 + 8));
            let mut c10 = _mm256_loadu_ps(c.as_ptr().add((i0 + 1) * ldc + j0));
            let mut c11 = _mm256_loadu_ps(c.as_ptr().add((i0 + 1) * ldc + j0 + 8));
            let mut c20 = _mm256_loadu_ps(c.as_ptr().add((i0 + 2) * ldc + j0));
            let mut c21 = _mm256_loadu_ps(c.as_ptr().add((i0 + 2) * ldc + j0 + 8));
            let mut c30 = _mm256_loadu_ps(c.as_ptr().add((i0 + 3) * ldc + j0));
            let mut c31 = _mm256_loadu_ps(c.as_ptr().add((i0 + 3) * ldc + j0 + 8));
            let mut c40 = _mm256_loadu_ps(c.as_ptr().add((i0 + 4) * ldc + j0));
            let mut c41 = _mm256_loadu_ps(c.as_ptr().add((i0 + 4) * ldc + j0 + 8));
            let mut c50 = _mm256_loadu_ps(c.as_ptr().add((i0 + 5) * ldc + j0));
            let mut c51 = _mm256_loadu_ps(c.as_ptr().add((i0 + 5) * ldc + j0 + 8));
            let mut c60 = _mm256_loadu_ps(c.as_ptr().add((i0 + 6) * ldc + j0));
            let mut c61 = _mm256_loadu_ps(c.as_ptr().add((i0 + 6) * ldc + j0 + 8));
            let mut c70 = _mm256_loadu_ps(c.as_ptr().add((i0 + 7) * ldc + j0));
            let mut c71 = _mm256_loadu_ps(c.as_ptr().add((i0 + 7) * ldc + j0 + 8));

            for p in 0..k {
                let b0 = _mm256_loadu_ps(b.as_ptr().add(p * ldb + j0));
                let b1 = _mm256_loadu_ps(b.as_ptr().add(p * ldb + j0 + 8));

                let a0 = _mm256_broadcast_ss(&*a.as_ptr().add(i0 * lda + p));
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c01 = _mm256_fmadd_ps(a0, b1, c01);
                let a1 = _mm256_broadcast_ss(&*a.as_ptr().add((i0 + 1) * lda + p));
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c11 = _mm256_fmadd_ps(a1, b1, c11);
                let a2 = _mm256_broadcast_ss(&*a.as_ptr().add((i0 + 2) * lda + p));
                c20 = _mm256_fmadd_ps(a2, b0, c20);
                c21 = _mm256_fmadd_ps(a2, b1, c21);
                let a3 = _mm256_broadcast_ss(&*a.as_ptr().add((i0 + 3) * lda + p));
                c30 = _mm256_fmadd_ps(a3, b0, c30);
                c31 = _mm256_fmadd_ps(a3, b1, c31);
                let a4 = _mm256_broadcast_ss(&*a.as_ptr().add((i0 + 4) * lda + p));
                c40 = _mm256_fmadd_ps(a4, b0, c40);
                c41 = _mm256_fmadd_ps(a4, b1, c41);
                let a5 = _mm256_broadcast_ss(&*a.as_ptr().add((i0 + 5) * lda + p));
                c50 = _mm256_fmadd_ps(a5, b0, c50);
                c51 = _mm256_fmadd_ps(a5, b1, c51);
                let a6 = _mm256_broadcast_ss(&*a.as_ptr().add((i0 + 6) * lda + p));
                c60 = _mm256_fmadd_ps(a6, b0, c60);
                c61 = _mm256_fmadd_ps(a6, b1, c61);
                let a7 = _mm256_broadcast_ss(&*a.as_ptr().add((i0 + 7) * lda + p));
                c70 = _mm256_fmadd_ps(a7, b0, c70);
                c71 = _mm256_fmadd_ps(a7, b1, c71);
            }

            _mm256_storeu_ps(c.as_mut_ptr().add(i0 * ldc + j0), c00);
            _mm256_storeu_ps(c.as_mut_ptr().add(i0 * ldc + j0 + 8), c01);
            _mm256_storeu_ps(c.as_mut_ptr().add((i0 + 1) * ldc + j0), c10);
            _mm256_storeu_ps(c.as_mut_ptr().add((i0 + 1) * ldc + j0 + 8), c11);
            _mm256_storeu_ps(c.as_mut_ptr().add((i0 + 2) * ldc + j0), c20);
            _mm256_storeu_ps(c.as_mut_ptr().add((i0 + 2) * ldc + j0 + 8), c21);
            _mm256_storeu_ps(c.as_mut_ptr().add((i0 + 3) * ldc + j0), c30);
            _mm256_storeu_ps(c.as_mut_ptr().add((i0 + 3) * ldc + j0 + 8), c31);
            _mm256_storeu_ps(c.as_mut_ptr().add((i0 + 4) * ldc + j0), c40);
            _mm256_storeu_ps(c.as_mut_ptr().add((i0 + 4) * ldc + j0 + 8), c41);
            _mm256_storeu_ps(c.as_mut_ptr().add((i0 + 5) * ldc + j0), c50);
            _mm256_storeu_ps(c.as_mut_ptr().add((i0 + 5) * ldc + j0 + 8), c51);
            _mm256_storeu_ps(c.as_mut_ptr().add((i0 + 6) * ldc + j0), c60);
            _mm256_storeu_ps(c.as_mut_ptr().add((i0 + 6) * ldc + j0 + 8), c61);
            _mm256_storeu_ps(c.as_mut_ptr().add((i0 + 7) * ldc + j0), c70);
            _mm256_storeu_ps(c.as_mut_ptr().add((i0 + 7) * ldc + j0 + 8), c71);
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
