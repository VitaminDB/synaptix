//! AVX-512 SGEMM micro-kernel.
//!
//! Параметры тайлинга: MR=16, NR=16 (1 zmm = 16 f32). 16 регистров zmm под C[16×16] —
//! половина регистрового файла zmm0..zmm31. Outer-product по K: 1 zmm B + 16 broadcast A.
//!
//! AVX-512 — nightly-only intrinsic в Rust на момент написания (`#![feature(stdarch_x86_avx512)]`
//! формально, но `_mm512_*` есть в stable начиная с 1.89). Для stable-сборки достаточно
//! `target_feature(enable = "avx512f")`. Runtime-detect через
//! `is_x86_feature_detected!("avx512f")`. Без поддержки — fallback на AVX2 (если есть) или
//! наивный скаляр.

const MR: usize = 16;
const NR: usize = 16;

pub fn sgemm_avx512(
    m: usize, n: usize, k: usize,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            unsafe { sgemm_avx512_impl(m, n, k, a, lda, b, ldb, c, ldc) };
            return;
        }
    }
    // Без AVX-512 — пробуем AVX2 (тоже SIMD, лучше скаляра), иначе чистый naive.
    crate::gemm::avx2::sgemm_avx2(m, n, k, a, lda, b, ldb, c, ldc);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sgemm_avx512_impl(
    m: usize, n: usize, k: usize,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
) {
    use std::arch::x86_64::*;

    let m_main = (m / MR) * MR;
    let n_main = (n / NR) * NR;

    for i0 in (0..m_main).step_by(MR) {
        for j0 in (0..n_main).step_by(NR) {
            // 16 zmm регистров под C[16×16] = 16 строк × 1 zmm.
            let mut c0  = _mm512_loadu_ps(c.as_ptr().add((i0 + 0) * ldc + j0));
            let mut c1  = _mm512_loadu_ps(c.as_ptr().add((i0 + 1) * ldc + j0));
            let mut c2  = _mm512_loadu_ps(c.as_ptr().add((i0 + 2) * ldc + j0));
            let mut c3  = _mm512_loadu_ps(c.as_ptr().add((i0 + 3) * ldc + j0));
            let mut c4  = _mm512_loadu_ps(c.as_ptr().add((i0 + 4) * ldc + j0));
            let mut c5  = _mm512_loadu_ps(c.as_ptr().add((i0 + 5) * ldc + j0));
            let mut c6  = _mm512_loadu_ps(c.as_ptr().add((i0 + 6) * ldc + j0));
            let mut c7  = _mm512_loadu_ps(c.as_ptr().add((i0 + 7) * ldc + j0));
            let mut c8  = _mm512_loadu_ps(c.as_ptr().add((i0 + 8) * ldc + j0));
            let mut c9  = _mm512_loadu_ps(c.as_ptr().add((i0 + 9) * ldc + j0));
            let mut c10 = _mm512_loadu_ps(c.as_ptr().add((i0 + 10) * ldc + j0));
            let mut c11 = _mm512_loadu_ps(c.as_ptr().add((i0 + 11) * ldc + j0));
            let mut c12 = _mm512_loadu_ps(c.as_ptr().add((i0 + 12) * ldc + j0));
            let mut c13 = _mm512_loadu_ps(c.as_ptr().add((i0 + 13) * ldc + j0));
            let mut c14 = _mm512_loadu_ps(c.as_ptr().add((i0 + 14) * ldc + j0));
            let mut c15 = _mm512_loadu_ps(c.as_ptr().add((i0 + 15) * ldc + j0));

            for p in 0..k {
                let b0 = _mm512_loadu_ps(b.as_ptr().add(p * ldb + j0));

                // 16 broadcasts + 16 FMA на одну K-итерацию.
                let a0  = _mm512_set1_ps(*a.as_ptr().add((i0 + 0)  * lda + p));
                c0  = _mm512_fmadd_ps(a0,  b0, c0);
                let a1  = _mm512_set1_ps(*a.as_ptr().add((i0 + 1)  * lda + p));
                c1  = _mm512_fmadd_ps(a1,  b0, c1);
                let a2  = _mm512_set1_ps(*a.as_ptr().add((i0 + 2)  * lda + p));
                c2  = _mm512_fmadd_ps(a2,  b0, c2);
                let a3  = _mm512_set1_ps(*a.as_ptr().add((i0 + 3)  * lda + p));
                c3  = _mm512_fmadd_ps(a3,  b0, c3);
                let a4  = _mm512_set1_ps(*a.as_ptr().add((i0 + 4)  * lda + p));
                c4  = _mm512_fmadd_ps(a4,  b0, c4);
                let a5  = _mm512_set1_ps(*a.as_ptr().add((i0 + 5)  * lda + p));
                c5  = _mm512_fmadd_ps(a5,  b0, c5);
                let a6  = _mm512_set1_ps(*a.as_ptr().add((i0 + 6)  * lda + p));
                c6  = _mm512_fmadd_ps(a6,  b0, c6);
                let a7  = _mm512_set1_ps(*a.as_ptr().add((i0 + 7)  * lda + p));
                c7  = _mm512_fmadd_ps(a7,  b0, c7);
                let a8  = _mm512_set1_ps(*a.as_ptr().add((i0 + 8)  * lda + p));
                c8  = _mm512_fmadd_ps(a8,  b0, c8);
                let a9  = _mm512_set1_ps(*a.as_ptr().add((i0 + 9)  * lda + p));
                c9  = _mm512_fmadd_ps(a9,  b0, c9);
                let a10 = _mm512_set1_ps(*a.as_ptr().add((i0 + 10) * lda + p));
                c10 = _mm512_fmadd_ps(a10, b0, c10);
                let a11 = _mm512_set1_ps(*a.as_ptr().add((i0 + 11) * lda + p));
                c11 = _mm512_fmadd_ps(a11, b0, c11);
                let a12 = _mm512_set1_ps(*a.as_ptr().add((i0 + 12) * lda + p));
                c12 = _mm512_fmadd_ps(a12, b0, c12);
                let a13 = _mm512_set1_ps(*a.as_ptr().add((i0 + 13) * lda + p));
                c13 = _mm512_fmadd_ps(a13, b0, c13);
                let a14 = _mm512_set1_ps(*a.as_ptr().add((i0 + 14) * lda + p));
                c14 = _mm512_fmadd_ps(a14, b0, c14);
                let a15 = _mm512_set1_ps(*a.as_ptr().add((i0 + 15) * lda + p));
                c15 = _mm512_fmadd_ps(a15, b0, c15);
            }

            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 0)  * ldc + j0), c0);
            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 1)  * ldc + j0), c1);
            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 2)  * ldc + j0), c2);
            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 3)  * ldc + j0), c3);
            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 4)  * ldc + j0), c4);
            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 5)  * ldc + j0), c5);
            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 6)  * ldc + j0), c6);
            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 7)  * ldc + j0), c7);
            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 8)  * ldc + j0), c8);
            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 9)  * ldc + j0), c9);
            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 10) * ldc + j0), c10);
            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 11) * ldc + j0), c11);
            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 12) * ldc + j0), c12);
            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 13) * ldc + j0), c13);
            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 14) * ldc + j0), c14);
            _mm512_storeu_ps(c.as_mut_ptr().add((i0 + 15) * ldc + j0), c15);
        }
    }

    crate::gemm::avx2::tail_naive(m, n, k, a, lda, b, ldb, c, ldc, m_main, n_main);
}
