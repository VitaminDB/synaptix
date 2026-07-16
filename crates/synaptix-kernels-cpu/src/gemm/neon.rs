//! ARM NEON SGEMM micro-kernel.
//!
//! Параметры тайлинга: MR=8, NR=8 (8 строк × 2 float32x4_t = 16 q-регистров для C).
//! NEON всегда доступен на aarch64 — compile-time feature `target_feature = "neon"`,
//! без runtime detection (в отличие от x86 SSE/AVX).
//!
//! На не-aarch64 платформах функция fallback на наивный скаляр.

#[cfg(target_arch = "aarch64")]
const MR: usize = 8;
#[cfg(target_arch = "aarch64")]
const NR: usize = 8;

pub fn sgemm_neon(
    m: usize, n: usize, k: usize,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { sgemm_neon_impl(m, n, k, a, lda, b, ldb, c, ldc) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        crate::gemm::avx2::sgemm_naive_fallback(m, n, k, a, lda, b, ldb, c, ldc);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sgemm_neon_impl(
    m: usize, n: usize, k: usize,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
) {
    use std::arch::aarch64::*;

    let m_main = (m / MR) * MR;
    let n_main = (n / NR) * NR;

    for i0 in (0..m_main).step_by(MR) {
        for j0 in (0..n_main).step_by(NR) {
            // C[8×8] = 8 строк × 2 q-регистра (left[0..4], right[4..8]).
            let mut c00 = vld1q_f32(c.as_ptr().add((i0 + 0) * ldc + j0));
            let mut c01 = vld1q_f32(c.as_ptr().add((i0 + 0) * ldc + j0 + 4));
            let mut c10 = vld1q_f32(c.as_ptr().add((i0 + 1) * ldc + j0));
            let mut c11 = vld1q_f32(c.as_ptr().add((i0 + 1) * ldc + j0 + 4));
            let mut c20 = vld1q_f32(c.as_ptr().add((i0 + 2) * ldc + j0));
            let mut c21 = vld1q_f32(c.as_ptr().add((i0 + 2) * ldc + j0 + 4));
            let mut c30 = vld1q_f32(c.as_ptr().add((i0 + 3) * ldc + j0));
            let mut c31 = vld1q_f32(c.as_ptr().add((i0 + 3) * ldc + j0 + 4));
            let mut c40 = vld1q_f32(c.as_ptr().add((i0 + 4) * ldc + j0));
            let mut c41 = vld1q_f32(c.as_ptr().add((i0 + 4) * ldc + j0 + 4));
            let mut c50 = vld1q_f32(c.as_ptr().add((i0 + 5) * ldc + j0));
            let mut c51 = vld1q_f32(c.as_ptr().add((i0 + 5) * ldc + j0 + 4));
            let mut c60 = vld1q_f32(c.as_ptr().add((i0 + 6) * ldc + j0));
            let mut c61 = vld1q_f32(c.as_ptr().add((i0 + 6) * ldc + j0 + 4));
            let mut c70 = vld1q_f32(c.as_ptr().add((i0 + 7) * ldc + j0));
            let mut c71 = vld1q_f32(c.as_ptr().add((i0 + 7) * ldc + j0 + 4));

            for p in 0..k {
                let b0 = vld1q_f32(b.as_ptr().add(p * ldb + j0));
                let b1 = vld1q_f32(b.as_ptr().add(p * ldb + j0 + 4));

                let a0 = vdupq_n_f32(*a.as_ptr().add((i0 + 0) * lda + p));
                c00 = vfmaq_f32(c00, a0, b0);
                c01 = vfmaq_f32(c01, a0, b1);
                let a1 = vdupq_n_f32(*a.as_ptr().add((i0 + 1) * lda + p));
                c10 = vfmaq_f32(c10, a1, b0);
                c11 = vfmaq_f32(c11, a1, b1);
                let a2 = vdupq_n_f32(*a.as_ptr().add((i0 + 2) * lda + p));
                c20 = vfmaq_f32(c20, a2, b0);
                c21 = vfmaq_f32(c21, a2, b1);
                let a3 = vdupq_n_f32(*a.as_ptr().add((i0 + 3) * lda + p));
                c30 = vfmaq_f32(c30, a3, b0);
                c31 = vfmaq_f32(c31, a3, b1);
                let a4 = vdupq_n_f32(*a.as_ptr().add((i0 + 4) * lda + p));
                c40 = vfmaq_f32(c40, a4, b0);
                c41 = vfmaq_f32(c41, a4, b1);
                let a5 = vdupq_n_f32(*a.as_ptr().add((i0 + 5) * lda + p));
                c50 = vfmaq_f32(c50, a5, b0);
                c51 = vfmaq_f32(c51, a5, b1);
                let a6 = vdupq_n_f32(*a.as_ptr().add((i0 + 6) * lda + p));
                c60 = vfmaq_f32(c60, a6, b0);
                c61 = vfmaq_f32(c61, a6, b1);
                let a7 = vdupq_n_f32(*a.as_ptr().add((i0 + 7) * lda + p));
                c70 = vfmaq_f32(c70, a7, b0);
                c71 = vfmaq_f32(c71, a7, b1);
            }

            vst1q_f32(c.as_mut_ptr().add((i0 + 0) * ldc + j0), c00);
            vst1q_f32(c.as_mut_ptr().add((i0 + 0) * ldc + j0 + 4), c01);
            vst1q_f32(c.as_mut_ptr().add((i0 + 1) * ldc + j0), c10);
            vst1q_f32(c.as_mut_ptr().add((i0 + 1) * ldc + j0 + 4), c11);
            vst1q_f32(c.as_mut_ptr().add((i0 + 2) * ldc + j0), c20);
            vst1q_f32(c.as_mut_ptr().add((i0 + 2) * ldc + j0 + 4), c21);
            vst1q_f32(c.as_mut_ptr().add((i0 + 3) * ldc + j0), c30);
            vst1q_f32(c.as_mut_ptr().add((i0 + 3) * ldc + j0 + 4), c31);
            vst1q_f32(c.as_mut_ptr().add((i0 + 4) * ldc + j0), c40);
            vst1q_f32(c.as_mut_ptr().add((i0 + 4) * ldc + j0 + 4), c41);
            vst1q_f32(c.as_mut_ptr().add((i0 + 5) * ldc + j0), c50);
            vst1q_f32(c.as_mut_ptr().add((i0 + 5) * ldc + j0 + 4), c51);
            vst1q_f32(c.as_mut_ptr().add((i0 + 6) * ldc + j0), c60);
            vst1q_f32(c.as_mut_ptr().add((i0 + 6) * ldc + j0 + 4), c61);
            vst1q_f32(c.as_mut_ptr().add((i0 + 7) * ldc + j0), c70);
            vst1q_f32(c.as_mut_ptr().add((i0 + 7) * ldc + j0 + 4), c71);
        }
    }

    crate::gemm::avx2::tail_naive(m, n, k, a, lda, b, ldb, c, ldc, m_main, n_main);
}
