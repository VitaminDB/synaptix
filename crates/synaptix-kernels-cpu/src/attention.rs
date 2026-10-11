#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use rayon::prelude::*;
use std::sync::OnceLock;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::stream::Stream;
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::Storage;

use crate::common::{cpu, cpu_mut, elem_bytes, from_f32, threads, to_f32};
use crate::quant::{decode_e4m3, e8m0_decode, e8m0_scale_byte, encode_e4m3, MXFP8_BLOCK};

const KV_BLOCK: usize = 64;
const ROWS_PER_UNIT: usize = 64;

const EXP_LOG2E: f32 = std::f32::consts::LOG2_E;
const EXP_C1: f32 = 0.693_359_4_f32;
const EXP_C2: f32 = -2.121_944_4e-4_f32;
const EXP_P0: f32 = 1.987_569_2e-4_f32;
const EXP_P1: f32 = 1.398_199_9e-3_f32;
const EXP_P2: f32 = 8.333_452e-3_f32;
const EXP_P3: f32 = 4.166_579_6e-2_f32;
const EXP_P4: f32 = 1.666_666_5e-1_f32;
const EXP_P5: f32 = 0.5_f32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Isa {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

#[allow(unreachable_code)]
fn pick_isa() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        let f = crate::dispatch::cpu_features();
        if f.avx2 && f.fma {
            return Isa::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return Isa::Neon;
    }
    Isa::Scalar
}

#[inline]
fn qk(isa: Isa, q: &[f32], k: &[f32], n: usize, d: usize, s: &mut [f32]) {
    debug_assert!(q.len() >= d && k.len() >= n * d && s.len() >= n);
    match isa {
        Isa::Scalar => scalar::qk(q, k, n, d, s),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { x86::qk(q.as_ptr(), k.as_ptr(), n, d, s.as_mut_ptr()) },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => unsafe { arm::qk(q.as_ptr(), k.as_ptr(), n, d, s.as_mut_ptr()) },
    }
}

#[inline]
fn pv(isa: Isa, acc: &mut [f32], p: &[f32], v: &[f32], n: usize, d: usize) {
    debug_assert!(acc.len() >= d && v.len() >= n * d && p.len() >= n);
    match isa {
        Isa::Scalar => scalar::pv(acc, p, v, n, d),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { x86::pv(acc.as_mut_ptr(), p.as_ptr(), v.as_ptr(), n, d) },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => unsafe { arm::pv(acc.as_mut_ptr(), p.as_ptr(), v.as_ptr(), n, d) },
    }
}

#[inline]
fn exp_sum(isa: Isa, s: &mut [f32], m: f32) -> f32 {
    match isa {
        Isa::Scalar => scalar::exp_sum(s, m),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { x86::exp_sum(s.as_mut_ptr(), s.len(), m) },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => unsafe { arm::exp_sum(s.as_mut_ptr(), s.len(), m) },
    }
}

mod scalar {
    pub fn qk(q: &[f32], k: &[f32], n: usize, d: usize, s: &mut [f32]) {
        for j in 0..n {
            let kr = &k[j * d..(j + 1) * d];
            s[j] = q[..d].iter().zip(kr).map(|(a, b)| a * b).sum();
        }
    }

    pub fn pv(acc: &mut [f32], p: &[f32], v: &[f32], n: usize, d: usize) {
        for j in 0..n {
            let pj = p[j];
            let vr = &v[j * d..(j + 1) * d];
            for (a, &x) in acc[..d].iter_mut().zip(vr) {
                *a += pj * x;
            }
        }
    }

    pub fn exp_sum(s: &mut [f32], m: f32) -> f32 {
        let mut t = 0.0f32;
        for x in s.iter_mut() {
            *x = (*x - m).exp();
            t += *x;
        }
        t
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
        _mm_cvtss_f32(s)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_row(q: *const f32, k: *const f32, d: usize) -> f32 {
        let mut a = _mm256_setzero_ps();
        let mut b = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= d {
            a = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(i)), _mm256_loadu_ps(k.add(i)), a);
            b = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(i + 8)), _mm256_loadu_ps(k.add(i + 8)), b);
            i += 16;
        }
        if i + 8 <= d {
            a = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(i)), _mm256_loadu_ps(k.add(i)), a);
            i += 8;
        }
        let mut r = hsum(_mm256_add_ps(a, b));
        while i < d {
            r += *q.add(i) * *k.add(i);
            i += 1;
        }
        r
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn qk(q: *const f32, k: *const f32, n: usize, d: usize, s: *mut f32) {
        let mut j = 0;
        while j + 4 <= n {
            let k0 = k.add(j * d);
            let k1 = k0.add(d);
            let k2 = k1.add(d);
            let k3 = k2.add(d);
            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut a2 = _mm256_setzero_ps();
            let mut a3 = _mm256_setzero_ps();
            let mut b0 = _mm256_setzero_ps();
            let mut b1 = _mm256_setzero_ps();
            let mut b2 = _mm256_setzero_ps();
            let mut b3 = _mm256_setzero_ps();
            let mut i = 0;
            while i + 16 <= d {
                let q0 = _mm256_loadu_ps(q.add(i));
                let q1 = _mm256_loadu_ps(q.add(i + 8));
                a0 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(k0.add(i)), a0);
                b0 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(k0.add(i + 8)), b0);
                a1 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(k1.add(i)), a1);
                b1 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(k1.add(i + 8)), b1);
                a2 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(k2.add(i)), a2);
                b2 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(k2.add(i + 8)), b2);
                a3 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(k3.add(i)), a3);
                b3 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(k3.add(i + 8)), b3);
                i += 16;
            }
            if i + 8 <= d {
                let q0 = _mm256_loadu_ps(q.add(i));
                a0 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(k0.add(i)), a0);
                a1 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(k1.add(i)), a1);
                a2 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(k2.add(i)), a2);
                a3 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(k3.add(i)), a3);
                i += 8;
            }
            let mut r0 = hsum(_mm256_add_ps(a0, b0));
            let mut r1 = hsum(_mm256_add_ps(a1, b1));
            let mut r2 = hsum(_mm256_add_ps(a2, b2));
            let mut r3 = hsum(_mm256_add_ps(a3, b3));
            while i < d {
                let qi = *q.add(i);
                r0 += qi * *k0.add(i);
                r1 += qi * *k1.add(i);
                r2 += qi * *k2.add(i);
                r3 += qi * *k3.add(i);
                i += 1;
            }
            *s.add(j) = r0;
            *s.add(j + 1) = r1;
            *s.add(j + 2) = r2;
            *s.add(j + 3) = r3;
            j += 4;
        }
        while j < n {
            *s.add(j) = dot_row(q, k.add(j * d), d);
            j += 1;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn pv(acc: *mut f32, p: *const f32, v: *const f32, n: usize, d: usize) {
        let mut c = 0;
        while c + 32 <= d {
            let mut a0 = _mm256_loadu_ps(acc.add(c));
            let mut a1 = _mm256_loadu_ps(acc.add(c + 8));
            let mut a2 = _mm256_loadu_ps(acc.add(c + 16));
            let mut a3 = _mm256_loadu_ps(acc.add(c + 24));
            let mut b0 = _mm256_setzero_ps();
            let mut b1 = _mm256_setzero_ps();
            let mut b2 = _mm256_setzero_ps();
            let mut b3 = _mm256_setzero_ps();
            let mut j = 0;
            while j + 2 <= n {
                let p0 = _mm256_set1_ps(*p.add(j));
                let p1 = _mm256_set1_ps(*p.add(j + 1));
                let v0 = v.add(j * d + c);
                let v1 = v0.add(d);
                a0 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(v0), a0);
                a1 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(v0.add(8)), a1);
                a2 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(v0.add(16)), a2);
                a3 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(v0.add(24)), a3);
                b0 = _mm256_fmadd_ps(p1, _mm256_loadu_ps(v1), b0);
                b1 = _mm256_fmadd_ps(p1, _mm256_loadu_ps(v1.add(8)), b1);
                b2 = _mm256_fmadd_ps(p1, _mm256_loadu_ps(v1.add(16)), b2);
                b3 = _mm256_fmadd_ps(p1, _mm256_loadu_ps(v1.add(24)), b3);
                j += 2;
            }
            if j < n {
                let p0 = _mm256_set1_ps(*p.add(j));
                let v0 = v.add(j * d + c);
                a0 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(v0), a0);
                a1 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(v0.add(8)), a1);
                a2 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(v0.add(16)), a2);
                a3 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(v0.add(24)), a3);
            }
            _mm256_storeu_ps(acc.add(c), _mm256_add_ps(a0, b0));
            _mm256_storeu_ps(acc.add(c + 8), _mm256_add_ps(a1, b1));
            _mm256_storeu_ps(acc.add(c + 16), _mm256_add_ps(a2, b2));
            _mm256_storeu_ps(acc.add(c + 24), _mm256_add_ps(a3, b3));
            c += 32;
        }
        while c + 8 <= d {
            let mut a = _mm256_loadu_ps(acc.add(c));
            let mut b = _mm256_setzero_ps();
            let mut j = 0;
            while j + 2 <= n {
                a = _mm256_fmadd_ps(_mm256_set1_ps(*p.add(j)), _mm256_loadu_ps(v.add(j * d + c)), a);
                b = _mm256_fmadd_ps(_mm256_set1_ps(*p.add(j + 1)), _mm256_loadu_ps(v.add((j + 1) * d + c)), b);
                j += 2;
            }
            if j < n {
                a = _mm256_fmadd_ps(_mm256_set1_ps(*p.add(j)), _mm256_loadu_ps(v.add(j * d + c)), a);
            }
            _mm256_storeu_ps(acc.add(c), _mm256_add_ps(a, b));
            c += 8;
        }
        while c < d {
            let mut a = *acc.add(c);
            for j in 0..n {
                a += *p.add(j) * *v.add(j * d + c);
            }
            *acc.add(c) = a;
            c += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp8(x: __m256) -> __m256 {
        let x = _mm256_max_ps(x, _mm256_set1_ps(-87.0));
        let x = _mm256_min_ps(x, _mm256_set1_ps(88.0));
        let n = _mm256_cvtps_epi32(_mm256_mul_ps(x, _mm256_set1_ps(EXP_LOG2E)));
        let fx = _mm256_cvtepi32_ps(n);
        let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(EXP_C1), x);
        let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(EXP_C2), r);
        let mut y = _mm256_set1_ps(EXP_P0);
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(EXP_P1));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(EXP_P2));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(EXP_P3));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(EXP_P4));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(EXP_P5));
        let z = _mm256_mul_ps(r, r);
        y = _mm256_fmadd_ps(y, z, r);
        y = _mm256_add_ps(y, _mm256_set1_ps(1.0));
        let e = _mm256_slli_epi32::<23>(_mm256_add_epi32(n, _mm256_set1_epi32(127)));
        _mm256_mul_ps(y, _mm256_castsi256_ps(e))
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn exp_sum(s: *mut f32, n: usize, m: f32) -> f32 {
        let mv = _mm256_set1_ps(m);
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(s.add(i)), mv));
            _mm256_storeu_ps(s.add(i), e);
            sum = _mm256_add_ps(sum, e);
            i += 8;
        }
        let mut t = hsum(sum);
        while i < n {
            let e = (*s.add(i) - m).exp();
            *s.add(i) = e;
            t += e;
            i += 1;
        }
        t
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::*;
    use std::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn dot_row(q: *const f32, k: *const f32, d: usize) -> f32 {
        let mut a = vdupq_n_f32(0.0);
        let mut b = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 8 <= d {
            a = vfmaq_f32(a, vld1q_f32(q.add(i)), vld1q_f32(k.add(i)));
            b = vfmaq_f32(b, vld1q_f32(q.add(i + 4)), vld1q_f32(k.add(i + 4)));
            i += 8;
        }
        if i + 4 <= d {
            a = vfmaq_f32(a, vld1q_f32(q.add(i)), vld1q_f32(k.add(i)));
            i += 4;
        }
        let mut r = vaddvq_f32(vaddq_f32(a, b));
        while i < d {
            r += *q.add(i) * *k.add(i);
            i += 1;
        }
        r
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn qk(q: *const f32, k: *const f32, n: usize, d: usize, s: *mut f32) {
        let mut j = 0;
        while j + 4 <= n {
            let k0 = k.add(j * d);
            let k1 = k0.add(d);
            let k2 = k1.add(d);
            let k3 = k2.add(d);
            let mut a0 = vdupq_n_f32(0.0);
            let mut a1 = vdupq_n_f32(0.0);
            let mut a2 = vdupq_n_f32(0.0);
            let mut a3 = vdupq_n_f32(0.0);
            let mut b0 = vdupq_n_f32(0.0);
            let mut b1 = vdupq_n_f32(0.0);
            let mut b2 = vdupq_n_f32(0.0);
            let mut b3 = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 8 <= d {
                let q0 = vld1q_f32(q.add(i));
                let q1 = vld1q_f32(q.add(i + 4));
                a0 = vfmaq_f32(a0, q0, vld1q_f32(k0.add(i)));
                b0 = vfmaq_f32(b0, q1, vld1q_f32(k0.add(i + 4)));
                a1 = vfmaq_f32(a1, q0, vld1q_f32(k1.add(i)));
                b1 = vfmaq_f32(b1, q1, vld1q_f32(k1.add(i + 4)));
                a2 = vfmaq_f32(a2, q0, vld1q_f32(k2.add(i)));
                b2 = vfmaq_f32(b2, q1, vld1q_f32(k2.add(i + 4)));
                a3 = vfmaq_f32(a3, q0, vld1q_f32(k3.add(i)));
                b3 = vfmaq_f32(b3, q1, vld1q_f32(k3.add(i + 4)));
                i += 8;
            }
            if i + 4 <= d {
                let q0 = vld1q_f32(q.add(i));
                a0 = vfmaq_f32(a0, q0, vld1q_f32(k0.add(i)));
                a1 = vfmaq_f32(a1, q0, vld1q_f32(k1.add(i)));
                a2 = vfmaq_f32(a2, q0, vld1q_f32(k2.add(i)));
                a3 = vfmaq_f32(a3, q0, vld1q_f32(k3.add(i)));
                i += 4;
            }
            let mut r0 = vaddvq_f32(vaddq_f32(a0, b0));
            let mut r1 = vaddvq_f32(vaddq_f32(a1, b1));
            let mut r2 = vaddvq_f32(vaddq_f32(a2, b2));
            let mut r3 = vaddvq_f32(vaddq_f32(a3, b3));
            while i < d {
                let qi = *q.add(i);
                r0 += qi * *k0.add(i);
                r1 += qi * *k1.add(i);
                r2 += qi * *k2.add(i);
                r3 += qi * *k3.add(i);
                i += 1;
            }
            *s.add(j) = r0;
            *s.add(j + 1) = r1;
            *s.add(j + 2) = r2;
            *s.add(j + 3) = r3;
            j += 4;
        }
        while j < n {
            *s.add(j) = dot_row(q, k.add(j * d), d);
            j += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn pv(acc: *mut f32, p: *const f32, v: *const f32, n: usize, d: usize) {
        let mut c = 0;
        while c + 16 <= d {
            let mut a0 = vld1q_f32(acc.add(c));
            let mut a1 = vld1q_f32(acc.add(c + 4));
            let mut a2 = vld1q_f32(acc.add(c + 8));
            let mut a3 = vld1q_f32(acc.add(c + 12));
            let mut b0 = vdupq_n_f32(0.0);
            let mut b1 = vdupq_n_f32(0.0);
            let mut b2 = vdupq_n_f32(0.0);
            let mut b3 = vdupq_n_f32(0.0);
            let mut j = 0;
            while j + 2 <= n {
                let p0 = vdupq_n_f32(*p.add(j));
                let p1 = vdupq_n_f32(*p.add(j + 1));
                let v0 = v.add(j * d + c);
                let v1 = v0.add(d);
                a0 = vfmaq_f32(a0, p0, vld1q_f32(v0));
                a1 = vfmaq_f32(a1, p0, vld1q_f32(v0.add(4)));
                a2 = vfmaq_f32(a2, p0, vld1q_f32(v0.add(8)));
                a3 = vfmaq_f32(a3, p0, vld1q_f32(v0.add(12)));
                b0 = vfmaq_f32(b0, p1, vld1q_f32(v1));
                b1 = vfmaq_f32(b1, p1, vld1q_f32(v1.add(4)));
                b2 = vfmaq_f32(b2, p1, vld1q_f32(v1.add(8)));
                b3 = vfmaq_f32(b3, p1, vld1q_f32(v1.add(12)));
                j += 2;
            }
            if j < n {
                let p0 = vdupq_n_f32(*p.add(j));
                let v0 = v.add(j * d + c);
                a0 = vfmaq_f32(a0, p0, vld1q_f32(v0));
                a1 = vfmaq_f32(a1, p0, vld1q_f32(v0.add(4)));
                a2 = vfmaq_f32(a2, p0, vld1q_f32(v0.add(8)));
                a3 = vfmaq_f32(a3, p0, vld1q_f32(v0.add(12)));
            }
            vst1q_f32(acc.add(c), vaddq_f32(a0, b0));
            vst1q_f32(acc.add(c + 4), vaddq_f32(a1, b1));
            vst1q_f32(acc.add(c + 8), vaddq_f32(a2, b2));
            vst1q_f32(acc.add(c + 12), vaddq_f32(a3, b3));
            c += 16;
        }
        while c + 4 <= d {
            let mut a = vld1q_f32(acc.add(c));
            let mut b = vdupq_n_f32(0.0);
            let mut j = 0;
            while j + 2 <= n {
                a = vfmaq_f32(a, vdupq_n_f32(*p.add(j)), vld1q_f32(v.add(j * d + c)));
                b = vfmaq_f32(b, vdupq_n_f32(*p.add(j + 1)), vld1q_f32(v.add((j + 1) * d + c)));
                j += 2;
            }
            if j < n {
                a = vfmaq_f32(a, vdupq_n_f32(*p.add(j)), vld1q_f32(v.add(j * d + c)));
            }
            vst1q_f32(acc.add(c), vaddq_f32(a, b));
            c += 4;
        }
        while c < d {
            let mut a = *acc.add(c);
            for j in 0..n {
                a += *p.add(j) * *v.add(j * d + c);
            }
            *acc.add(c) = a;
            c += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn exp4(x: float32x4_t) -> float32x4_t {
        let x = vmaxq_f32(x, vdupq_n_f32(-87.0));
        let x = vminq_f32(x, vdupq_n_f32(88.0));
        let n = vcvtnq_s32_f32(vmulq_f32(x, vdupq_n_f32(EXP_LOG2E)));
        let fx = vcvtq_f32_s32(n);
        let r = vfmsq_f32(x, fx, vdupq_n_f32(EXP_C1));
        let r = vfmsq_f32(r, fx, vdupq_n_f32(EXP_C2));
        let mut y = vdupq_n_f32(EXP_P0);
        y = vfmaq_f32(vdupq_n_f32(EXP_P1), y, r);
        y = vfmaq_f32(vdupq_n_f32(EXP_P2), y, r);
        y = vfmaq_f32(vdupq_n_f32(EXP_P3), y, r);
        y = vfmaq_f32(vdupq_n_f32(EXP_P4), y, r);
        y = vfmaq_f32(vdupq_n_f32(EXP_P5), y, r);
        let z = vmulq_f32(r, r);
        y = vfmaq_f32(r, y, z);
        y = vaddq_f32(y, vdupq_n_f32(1.0));
        let e = vshlq_n_s32::<23>(vaddq_s32(n, vdupq_n_s32(127)));
        vmulq_f32(y, vreinterpretq_f32_s32(e))
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn exp_sum(s: *mut f32, n: usize, m: f32) -> f32 {
        let mv = vdupq_n_f32(m);
        let mut sum = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 4 <= n {
            let e = exp4(vsubq_f32(vld1q_f32(s.add(i)), mv));
            vst1q_f32(s.add(i), e);
            sum = vaddq_f32(sum, e);
            i += 4;
        }
        let mut t = vaddvq_f32(sum);
        while i < n {
            let e = (*s.add(i) - m).exp();
            *s.add(i) = e;
            t += e;
            i += 1;
        }
        t
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[inline]
    fn get(&self) -> *mut u8 {
        self.0
    }
}

#[derive(Clone, Copy)]
struct SendPtrF(*mut f32);
unsafe impl Send for SendPtrF {}
unsafe impl Sync for SendPtrF {}
impl SendPtrF {
    #[inline]
    fn get(&self) -> *mut f32 {
        self.0
    }
}

#[derive(Clone, Copy)]
struct View<'a> {
    bytes: &'a [u8],
    dtype: DType,
    off: usize,
    sb: usize,
    sh: usize,
    st: usize,
    d: usize,
}

impl View<'_> {
    #[inline]
    fn row(&self, b: usize, h: usize, t: usize) -> usize {
        self.off + b * self.sb + h * self.sh + t * self.st
    }
}

fn check_float(dt: DType, what: &'static str) -> Result<()> {
    if matches!(dt, DType::F32 | DType::F16 | DType::BF16) {
        Ok(())
    } else {
        Err(SynaptixError::Unsupported(what))
    }
}

fn stride_u(s: isize, what: &'static str) -> Result<usize> {
    if s < 0 {
        Err(SynaptixError::Unsupported(what))
    } else {
        Ok(s as usize)
    }
}

fn check_bounds(len_bytes: usize, elem: usize, off: usize, dims: [usize; 3], strides: [usize; 3], d: usize, what: &'static str) -> Result<()> {
    if dims.contains(&0) || d == 0 {
        return Ok(());
    }
    let last = off + (dims[0] - 1) * strides[0] + (dims[1] - 1) * strides[1] + (dims[2] - 1) * strides[2] + d;
    if last * elem > len_bytes {
        return Err(SynaptixError::Unsupported(what));
    }
    Ok(())
}

fn view_bhtd<'a>(bytes: &'a [u8], lo: &Layout, elem: usize, what: &'static str) -> Result<View<'a>> {
    if lo.dims().len() != 4 {
        return Err(SynaptixError::Unsupported(what));
    }
    let s = lo.strides().as_slice();
    if s[3] != 1 {
        return Err(SynaptixError::Unsupported(what));
    }
    let dm = lo.dims();
    let v = View {
        bytes,
        dtype: lo.dtype(),
        off: lo.offset(),
        sb: stride_u(s[0], what)?,
        sh: stride_u(s[1], what)?,
        st: stride_u(s[2], what)?,
        d: dm[3],
    };
    check_bounds(bytes.len(), elem, v.off, [dm[0], dm[1], dm[2]], [v.sb, v.sh, v.st], v.d, what)?;
    Ok(v)
}

fn view_bshd<'a>(bytes: &'a [u8], lo: &Layout, elem: usize, what: &'static str) -> Result<View<'a>> {
    if lo.dims().len() != 4 {
        return Err(SynaptixError::Unsupported(what));
    }
    let s = lo.strides().as_slice();
    if s[3] != 1 {
        return Err(SynaptixError::Unsupported(what));
    }
    let dm = lo.dims();
    let v = View {
        bytes,
        dtype: lo.dtype(),
        off: lo.offset(),
        sb: stride_u(s[0], what)?,
        st: stride_u(s[1], what)?,
        sh: stride_u(s[2], what)?,
        d: dm[3],
    };
    check_bounds(bytes.len(), elem, v.off, [dm[0], dm[2], dm[1]], [v.sb, v.sh, v.st], v.d, what)?;
    Ok(v)
}

#[derive(Clone, Copy)]
struct OutView {
    ptr: SendPtr,
    len: usize,
    dtype: DType,
    off: usize,
    sb: usize,
    sh: usize,
    st: usize,
    d: usize,
}

impl OutView {
    fn new(bytes: &mut [u8], lo: &Layout, bshd: bool, what: &'static str) -> Result<Self> {
        check_float(lo.dtype(), what)?;
        let elem = elem_bytes(lo.dtype());
        let v = if bshd { view_bshd(bytes, lo, elem, what)? } else { view_bhtd(bytes, lo, elem, what)? };
        let (dtype, off, sb, sh, st, d) = (v.dtype, v.off, v.sb, v.sh, v.st, v.d);
        Ok(Self { ptr: SendPtr(bytes.as_mut_ptr()), len: bytes.len(), dtype, off, sb, sh, st, d })
    }

    fn write_row(&self, b: usize, h: usize, t: usize, src: &[f32]) {
        let e = self.off + b * self.sb + h * self.sh + t * self.st;
        let nb = elem_bytes(self.dtype);
        debug_assert!((e + self.d) * nb <= self.len);
        let dst = unsafe { std::slice::from_raw_parts_mut(self.ptr.get().add(e * nb), self.d * nb) };
        from_f32(self.dtype, &src[..self.d], dst, 0).expect("attention out dtype");
    }
}

trait KvSource: Sync {
    fn len(&self, b: usize) -> usize;
    fn load(&self, b: usize, kvh: usize, t0: usize, n: usize, k: &mut [f32], v: &mut [f32]);
}

struct DenseKv<'a> {
    k: View<'a>,
    v: View<'a>,
    len: usize,
}

fn load_dense(view: &View, b: usize, kvh: usize, t0: usize, n: usize, dst: &mut [f32]) {
    let d = view.d;
    if view.st == d {
        to_f32(view.dtype, view.bytes, view.row(b, kvh, t0), &mut dst[..n * d]).expect("kv dtype");
    } else {
        for r in 0..n {
            to_f32(view.dtype, view.bytes, view.row(b, kvh, t0 + r), &mut dst[r * d..(r + 1) * d]).expect("kv dtype");
        }
    }
}

impl KvSource for DenseKv<'_> {
    fn len(&self, _b: usize) -> usize {
        self.len
    }
    fn load(&self, b: usize, kvh: usize, t0: usize, n: usize, k: &mut [f32], v: &mut [f32]) {
        load_dense(&self.k, b, kvh, t0, n, k);
        load_dense(&self.v, b, kvh, t0, n, v);
    }
}

static E4M3_LUT: OnceLock<[f32; 256]> = OnceLock::new();

fn e4m3_lut() -> &'static [f32; 256] {
    E4M3_LUT.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (i, x) in t.iter_mut().enumerate() {
            *x = decode_e4m3(i as u8);
        }
        t
    })
}

struct Mxfp8Kv<'a> {
    k: View<'a>,
    v: View<'a>,
    ks: View<'a>,
    vs: View<'a>,
    len: usize,
    lut: &'static [f32; 256],
}

impl Mxfp8Kv<'_> {
    fn load_row(&self, data: &View, sc: &View, b: usize, kvh: usize, t: usize, dst: &mut [f32]) {
        let r = data.row(b, kvh, t);
        let sr = sc.row(b, kvh, t);
        let nb = data.d / MXFP8_BLOCK;
        for blk in 0..nb {
            let sv = e8m0_decode(sc.bytes[sr + blk]);
            let src = &data.bytes[r + blk * MXFP8_BLOCK..r + (blk + 1) * MXFP8_BLOCK];
            let out = &mut dst[blk * MXFP8_BLOCK..(blk + 1) * MXFP8_BLOCK];
            for (o, &byte) in out.iter_mut().zip(src) {
                *o = self.lut[byte as usize] * sv;
            }
        }
    }
}

impl KvSource for Mxfp8Kv<'_> {
    fn len(&self, _b: usize) -> usize {
        self.len
    }
    fn load(&self, b: usize, kvh: usize, t0: usize, n: usize, k: &mut [f32], v: &mut [f32]) {
        let d = self.k.d;
        for r in 0..n {
            self.load_row(&self.k, &self.ks, b, kvh, t0 + r, &mut k[r * d..(r + 1) * d]);
            self.load_row(&self.v, &self.vs, b, kvh, t0 + r, &mut v[r * d..(r + 1) * d]);
        }
    }
}

struct GatheredKv<S: KvSource> {
    inner: S,
    positions: Vec<Vec<u32>>,
    d: usize,
}

impl<S: KvSource> KvSource for GatheredKv<S> {
    fn len(&self, b: usize) -> usize {
        self.positions[b].len()
    }
    fn load(&self, b: usize, kvh: usize, t0: usize, n: usize, k: &mut [f32], v: &mut [f32]) {
        let d = self.d;
        let pos = &self.positions[b];
        for r in 0..n {
            let p = pos[t0 + r] as usize;
            self.inner.load(0, kvh, p, 1, &mut k[r * d..(r + 1) * d], &mut v[r * d..(r + 1) * d]);
        }
    }
}

struct Attn<'a> {
    q: View<'a>,
    out: OutView,
    b: usize,
    nh: usize,
    nkv: usize,
    tq: usize,
    d: usize,
    scale: f32,
    causal: bool,
    window: i32,
    isa: Isa,
}

struct State {
    nq: usize,
    bq_cur: usize,
    d: usize,
    q: Vec<f32>,
    m: Vec<f32>,
    l: Vec<f32>,
    acc: Vec<f32>,
    lo: Vec<usize>,
    hi: Vec<usize>,
    kbuf: Vec<f32>,
    vbuf: Vec<f32>,
    s: Vec<f32>,
    tmp: Vec<f32>,
}

impl State {
    fn new(nq_max: usize, d: usize) -> Self {
        Self {
            nq: 0,
            bq_cur: 0,
            d,
            q: vec![0.0; nq_max * d],
            m: vec![f32::NEG_INFINITY; nq_max],
            l: vec![0.0; nq_max],
            acc: vec![0.0; nq_max * d],
            lo: vec![0; nq_max],
            hi: vec![0; nq_max],
            kbuf: vec![0.0; KV_BLOCK * d],
            vbuf: vec![0.0; KV_BLOCK * d],
            s: vec![0.0; KV_BLOCK],
            tmp: vec![0.0; d],
        }
    }

    fn compute<S: KvSource>(&mut self, p: &Attn, kv: &S, bi: usize, kvh: usize, qb: usize, bq: usize, t_begin: usize, t_end: usize) {
        let group = p.nh / p.nkv;
        let d = self.d;
        self.bq_cur = bq.min(p.tq - qb * bq);
        self.nq = group * self.bq_cur;
        let nq = self.nq;
        let len = kv.len(bi);
        let q_pos_base = len.saturating_sub(p.tq);
        for r in 0..nq {
            let hi_ = r / self.bq_cur;
            let qi = r % self.bq_cur;
            let h = kvh * group + hi_;
            let ti = qb * bq + qi;
            let row = &mut self.q[r * d..(r + 1) * d];
            to_f32(p.q.dtype, p.q.bytes, p.q.row(bi, h, ti), row).expect("q dtype");
            for x in row.iter_mut() {
                *x *= p.scale;
            }
            self.m[r] = f32::NEG_INFINITY;
            self.l[r] = 0.0;
            self.acc[r * d..(r + 1) * d].fill(0.0);
            let q_pos = q_pos_base + ti;
            let mut lo = 0usize;
            let mut hi = len;
            if p.causal {
                hi = hi.min(q_pos + 1);
            }
            if p.window > 0 {
                let w = p.window as usize;
                lo = lo.max(q_pos.saturating_sub(w));
                hi = hi.min(q_pos + w + 1);
            }
            lo = lo.max(t_begin);
            hi = hi.min(t_end);
            self.lo[r] = lo;
            self.hi[r] = hi.max(lo);
        }
        let unit_lo = self.lo[..nq].iter().copied().min().unwrap_or(0);
        let unit_hi = self.hi[..nq].iter().copied().max().unwrap_or(0);
        if unit_lo >= unit_hi {
            return;
        }
        let mut t0 = unit_lo;
        while t0 < unit_hi {
            let n = KV_BLOCK.min(unit_hi - t0);
            kv.load(bi, kvh, t0, n, &mut self.kbuf[..n * d], &mut self.vbuf[..n * d]);
            for r in 0..nq {
                let a = self.lo[r].max(t0);
                let e = self.hi[r].min(t0 + n);
                if a >= e {
                    continue;
                }
                let cnt = e - a;
                let qrow = &self.q[r * d..(r + 1) * d];
                let kb = &self.kbuf[(a - t0) * d..(e - t0) * d];
                let vb = &self.vbuf[(a - t0) * d..(e - t0) * d];
                let s = &mut self.s[..cnt];
                qk(p.isa, qrow, kb, cnt, d, s);
                let bmax = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let m_old = self.m[r];
                let m_new = m_old.max(bmax);
                let alpha = if m_old == f32::NEG_INFINITY { 0.0 } else { (m_old - m_new).exp() };
                let sum = exp_sum(p.isa, s, m_new);
                self.l[r] = self.l[r] * alpha + sum;
                let acc = &mut self.acc[r * d..(r + 1) * d];
                if alpha != 1.0 {
                    for x in acc.iter_mut() {
                        *x *= alpha;
                    }
                }
                pv(p.isa, acc, s, vb, cnt, d);
                self.m[r] = m_new;
            }
            t0 += n;
        }
    }

    fn row_target(&self, p: &Attn, kvh: usize, qb: usize, bq: usize, r: usize) -> (usize, usize) {
        let group = p.nh / p.nkv;
        let h = kvh * group + r / self.bq_cur;
        let ti = qb * bq + r % self.bq_cur;
        (h, ti)
    }

    fn write_out(&mut self, p: &Attn, bi: usize, kvh: usize, qb: usize, bq: usize) {
        let d = self.d;
        for r in 0..self.nq {
            let inv = if self.l[r] > 0.0 { 1.0 / self.l[r] } else { 0.0 };
            for (t, &a) in self.tmp.iter_mut().zip(&self.acc[r * d..(r + 1) * d]) {
                *t = a * inv;
            }
            let (h, ti) = self.row_target(p, kvh, qb, bq, r);
            p.out.write_row(bi, h, ti, &self.tmp);
        }
    }

    fn store_partial(&self, dst: &mut [f32]) {
        let d = self.d;
        for r in 0..self.nq {
            let base = r * (d + 2);
            dst[base] = self.m[r];
            dst[base + 1] = self.l[r];
            dst[base + 2..base + 2 + d].copy_from_slice(&self.acc[r * d..(r + 1) * d]);
        }
    }
}

fn run_attention<S: KvSource>(p: &Attn, kv: &S) {
    if p.b == 0 || p.nh == 0 || p.tq == 0 || p.d == 0 {
        return;
    }
    let group = p.nh / p.nkv;
    let bq = (ROWS_PER_UNIT / group).clamp(4, ROWS_PER_UNIT).min(p.tq);
    let nqb = p.tq.div_ceil(bq);
    let max_len = (0..p.b).map(|b| kv.len(b)).max().unwrap_or(0);
    let nkb = max_len.div_ceil(KV_BLOCK).max(1);
    let base_units = p.b * p.nkv * nqb;
    let t = threads();
    let splits = if base_units >= 2 * t { 1 } else { (2 * t).div_ceil(base_units).min(nkb).max(1) };
    let blocks_per_split = nkb.div_ceil(splits);
    let span = blocks_per_split * KV_BLOCK;
    let nq_max = group * bq;
    let d = p.d;
    let unit_coords = |unit: usize| -> (usize, usize, usize) {
        let qb = unit % nqb;
        let kvh = (unit / nqb) % p.nkv;
        let bi = unit / (nqb * p.nkv);
        (bi, kvh, qb)
    };
    if splits == 1 {
        (0..base_units).into_par_iter().for_each(|unit| {
            let (bi, kvh, qb) = unit_coords(unit);
            let mut st = State::new(nq_max, d);
            st.compute(p, kv, bi, kvh, qb, bq, 0, kv.len(bi));
            st.write_out(p, bi, kvh, qb, bq);
        });
        return;
    }
    let part_stride = nq_max * (d + 2);
    let mut partials = vec![0f32; base_units * splits * part_stride];
    let part_ptr = SendPtrF(partials.as_mut_ptr());
    (0..base_units * splits).into_par_iter().for_each(|u| {
        let unit = u / splits;
        let sp = u % splits;
        let (bi, kvh, qb) = unit_coords(unit);
        let len = kv.len(bi);
        let t_begin = (sp * span).min(len);
        let t_end = ((sp + 1) * span).min(len);
        let mut st = State::new(nq_max, d);
        st.compute(p, kv, bi, kvh, qb, bq, t_begin, t_end);
        let dst = unsafe { std::slice::from_raw_parts_mut(part_ptr.get().add(u * part_stride), part_stride) };
        st.store_partial(dst);
    });
    let partials = &partials;
    (0..base_units).into_par_iter().for_each(|unit| {
        let (bi, kvh, qb) = unit_coords(unit);
        let bq_cur = bq.min(p.tq - qb * bq);
        let nq = group * bq_cur;
        let mut tmp = vec![0f32; d];
        for r in 0..nq {
            let base = r * (d + 2);
            let mut mm = f32::NEG_INFINITY;
            for sp in 0..splits {
                mm = mm.max(partials[(unit * splits + sp) * part_stride + base]);
            }
            tmp.fill(0.0);
            let mut l = 0.0f32;
            if mm > f32::NEG_INFINITY {
                for sp in 0..splits {
                    let pb = (unit * splits + sp) * part_stride + base;
                    let ms = partials[pb];
                    if ms == f32::NEG_INFINITY {
                        continue;
                    }
                    let w = (ms - mm).exp();
                    l += partials[pb + 1] * w;
                    for (t, &a) in tmp.iter_mut().zip(&partials[pb + 2..pb + 2 + d]) {
                        *t += a * w;
                    }
                }
            }
            let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
            for t in tmp.iter_mut() {
                *t *= inv;
            }
            let h = kvh * group + r / bq_cur;
            let ti = qb * bq + r % bq_cur;
            p.out.write_row(bi, h, ti, &tmp);
        }
    });
}

fn read_u32(st: &Storage, lo: &Layout, what: &'static str) -> Result<usize> {
    if lo.dtype() != DType::U32 {
        return Err(SynaptixError::Unsupported(what));
    }
    let buf = cpu(st, what)?;
    let o = lo.offset() * 4;
    let b = buf.as_bytes().get(o..o + 4).ok_or(SynaptixError::Unsupported(what))?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
}

fn read_u32_slice(st: &Storage, lo: &Layout, what: &'static str) -> Result<Vec<u32>> {
    if lo.dtype() != DType::U32 || !lo.is_contiguous() {
        return Err(SynaptixError::Unsupported(what));
    }
    let buf = cpu(st, what)?;
    let n = lo.numel();
    let o = lo.offset() * 4;
    let b = buf.as_bytes().get(o..o + n * 4).ok_or(SynaptixError::Unsupported(what))?;
    Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn dense_attention(
    q: (&Storage, &Layout),
    k: (&Storage, &Layout),
    v: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    scale: f32,
    causal: bool,
    window: i32,
    active_len: Option<usize>,
    bshd: bool,
) -> Result<()> {
    let (q_st, q_lo) = q;
    let (k_st, k_lo) = k;
    let (v_st, v_lo) = v;
    let (out_st, out_lo) = out;
    check_float(q_lo.dtype(), "cpu flash: q dtype")?;
    check_float(k_lo.dtype(), "cpu flash: k dtype")?;
    check_float(v_lo.dtype(), "cpu flash: v dtype")?;
    if q_lo.dims().len() != 4 || k_lo.dims().len() != 4 || v_lo.dims() != k_lo.dims() || out_lo.dims() != q_lo.dims() {
        return Err(SynaptixError::Unsupported("cpu flash: expect rank-4 q/k/v/out"));
    }
    let qd = q_lo.dims();
    let kd = k_lo.dims();
    let (b, nh, tq, d, nkv, tk) = if bshd { (qd[0], qd[2], qd[1], qd[3], kd[2], kd[1]) } else { (qd[0], qd[1], qd[2], qd[3], kd[1], kd[2]) };
    if kd[0] != b || kd[3] != d {
        return Err(SynaptixError::Unsupported("cpu flash: k/v shape"));
    }
    if nkv == 0 || nh % nkv != 0 {
        return Err(SynaptixError::Unsupported("cpu flash: GQA nh % nkv != 0"));
    }
    let len = match active_len {
        Some(n) => {
            if n > tk {
                return Err(SynaptixError::Other(format!("cpu flash: t_cache {n} > max_seq {tk}")));
            }
            n
        }
        None => tk,
    };
    let q_buf = cpu(q_st, "cpu flash: q non-cpu")?;
    let k_buf = cpu(k_st, "cpu flash: k non-cpu")?;
    let v_buf = cpu(v_st, "cpu flash: v non-cpu")?;
    let (qv, kv_k, kv_v) = if bshd {
        (
            view_bshd(q_buf.as_bytes(), q_lo, elem_bytes(q_lo.dtype()), "cpu flash: q layout")?,
            view_bshd(k_buf.as_bytes(), k_lo, elem_bytes(k_lo.dtype()), "cpu flash: k layout")?,
            view_bshd(v_buf.as_bytes(), v_lo, elem_bytes(v_lo.dtype()), "cpu flash: v layout")?,
        )
    } else {
        (
            view_bhtd(q_buf.as_bytes(), q_lo, elem_bytes(q_lo.dtype()), "cpu flash: q layout")?,
            view_bhtd(k_buf.as_bytes(), k_lo, elem_bytes(k_lo.dtype()), "cpu flash: k layout")?,
            view_bhtd(v_buf.as_bytes(), v_lo, elem_bytes(v_lo.dtype()), "cpu flash: v layout")?,
        )
    };
    let out_buf = cpu_mut(out_st, "cpu flash: out non-cpu")?;
    let out = OutView::new(out_buf.as_bytes_mut(), out_lo, bshd, "cpu flash: out layout")?;
    let kv = DenseKv { k: kv_k, v: kv_v, len };
    let p = Attn { q: qv, out, b, nh, nkv, tq, d, scale, causal, window, isa: pick_isa() };
    run_attention(&p, &kv);
    Ok(())
}

pub fn flash_attention(
    q: (&Storage, &Layout),
    k: (&Storage, &Layout),
    v: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    scale: f32,
    causal: bool,
    _stream: &Stream,
) -> Result<()> {
    dense_attention(q, k, v, out, scale, causal, 0, None, false)
}

pub fn flash_attention_dev(
    q: (&Storage, &Layout),
    k: (&Storage, &Layout),
    v: (&Storage, &Layout),
    t_cache: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    scale: f32,
    causal: bool,
    _stream: &Stream,
) -> Result<()> {
    let len = read_u32(t_cache.0, t_cache.1, "cpu flash_dev: t_cache must be U32[1]")?;
    dense_attention(q, k, v, out, scale, causal, 0, Some(len), false)
}

pub fn flash_attention_window(
    q: (&Storage, &Layout),
    k: (&Storage, &Layout),
    v: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    scale: f32,
    window: i32,
    causal: bool,
    _stream: &Stream,
) -> Result<()> {
    dense_attention(q, k, v, out, scale, causal, window, None, false)
}

pub fn flash_attention_window_dev(
    q: (&Storage, &Layout),
    k: (&Storage, &Layout),
    v: (&Storage, &Layout),
    t_cache: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    scale: f32,
    window: i32,
    causal: bool,
    _stream: &Stream,
) -> Result<()> {
    let len = read_u32(t_cache.0, t_cache.1, "cpu flash_window_dev: t_cache must be U32[1]")?;
    dense_attention(q, k, v, out, scale, causal, window, Some(len), false)
}

pub fn flash_attention_bshd(
    q: (&Storage, &Layout),
    k: (&Storage, &Layout),
    v: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    scale: f32,
    causal: bool,
    _stream: &Stream,
) -> Result<()> {
    dense_attention(q, k, v, out, scale, causal, 0, None, true)
}

fn mxfp8_attention(
    q: (&Storage, &Layout),
    k: (&Storage, &Layout),
    v: (&Storage, &Layout),
    k_scale: (&Storage, &Layout),
    v_scale: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    scale: f32,
    causal: bool,
    active_len: Option<usize>,
) -> Result<()> {
    let (q_st, q_lo) = q;
    let (k_st, k_lo) = k;
    let (v_st, v_lo) = v;
    let (ks_st, ks_lo) = k_scale;
    let (vs_st, vs_lo) = v_scale;
    let (out_st, out_lo) = out;
    check_float(q_lo.dtype(), "cpu flash mxfp8: q must be float")?;
    if k_lo.dtype() != DType::MXFP8 || v_lo.dtype() != DType::MXFP8 {
        return Err(SynaptixError::Unsupported("cpu flash mxfp8: k/v must be MXFP8"));
    }
    if ks_lo.dtype() != DType::U8 || vs_lo.dtype() != DType::U8 {
        return Err(SynaptixError::Unsupported("cpu flash mxfp8: scales must be U8"));
    }
    if q_lo.dims().len() != 4 || k_lo.dims().len() != 4 || v_lo.dims() != k_lo.dims() || out_lo.dims() != q_lo.dims() {
        return Err(SynaptixError::Unsupported("cpu flash mxfp8: expect rank-4 q/k/v/out"));
    }
    let (b, nh, tq, d) = (q_lo.dims()[0], q_lo.dims()[1], q_lo.dims()[2], q_lo.dims()[3]);
    let (nkv, tk) = (k_lo.dims()[1], k_lo.dims()[2]);
    if k_lo.dims()[0] != b || k_lo.dims()[3] != d {
        return Err(SynaptixError::Unsupported("cpu flash mxfp8: k/v shape"));
    }
    if nkv == 0 || nh % nkv != 0 {
        return Err(SynaptixError::Unsupported("cpu flash mxfp8: GQA"));
    }
    if d % MXFP8_BLOCK != 0 {
        return Err(SynaptixError::Unsupported("cpu flash mxfp8: d % 32 != 0"));
    }
    let nb = d / MXFP8_BLOCK;
    if ks_lo.dims() != [b, nkv, tk, nb] || vs_lo.dims() != [b, nkv, tk, nb] {
        return Err(SynaptixError::Unsupported("cpu flash mxfp8: scale shape"));
    }
    let len = match active_len {
        Some(n) => {
            if n > tk {
                return Err(SynaptixError::Other(format!("cpu flash mxfp8: t_cache {n} > max_seq {tk}")));
            }
            n
        }
        None => tk,
    };
    let q_buf = cpu(q_st, "cpu flash mxfp8: q non-cpu")?;
    let k_buf = cpu(k_st, "cpu flash mxfp8: k non-cpu")?;
    let v_buf = cpu(v_st, "cpu flash mxfp8: v non-cpu")?;
    let ks_buf = cpu(ks_st, "cpu flash mxfp8: ks non-cpu")?;
    let vs_buf = cpu(vs_st, "cpu flash mxfp8: vs non-cpu")?;
    let qv = view_bhtd(q_buf.as_bytes(), q_lo, elem_bytes(q_lo.dtype()), "cpu flash mxfp8: q layout")?;
    let kv = Mxfp8Kv {
        k: view_bhtd(k_buf.as_bytes(), k_lo, 1, "cpu flash mxfp8: k layout")?,
        v: view_bhtd(v_buf.as_bytes(), v_lo, 1, "cpu flash mxfp8: v layout")?,
        ks: view_bhtd(ks_buf.as_bytes(), ks_lo, 1, "cpu flash mxfp8: k_scale layout")?,
        vs: view_bhtd(vs_buf.as_bytes(), vs_lo, 1, "cpu flash mxfp8: v_scale layout")?,
        len,
        lut: e4m3_lut(),
    };
    let out_buf = cpu_mut(out_st, "cpu flash mxfp8: out non-cpu")?;
    let out = OutView::new(out_buf.as_bytes_mut(), out_lo, false, "cpu flash mxfp8: out layout")?;
    let p = Attn { q: qv, out, b, nh, nkv, tq, d, scale, causal, window: 0, isa: pick_isa() };
    run_attention(&p, &kv);
    Ok(())
}

pub fn flash_attention_mxfp8kv(
    q: (&Storage, &Layout),
    k: (&Storage, &Layout),
    v: (&Storage, &Layout),
    k_scale: (&Storage, &Layout),
    v_scale: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    scale: f32,
    causal: bool,
    _stream: &Stream,
) -> Result<()> {
    mxfp8_attention(q, k, v, k_scale, v_scale, out, scale, causal, None)
}

pub fn flash_attention_mxfp8kv_dev(
    q: (&Storage, &Layout),
    k: (&Storage, &Layout),
    v: (&Storage, &Layout),
    k_scale: (&Storage, &Layout),
    v_scale: (&Storage, &Layout),
    t_cache: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    scale: f32,
    causal: bool,
    _stream: &Stream,
) -> Result<()> {
    let len = read_u32(t_cache.0, t_cache.1, "cpu flash mxfp8 dev: t_cache must be U32[1]")?;
    mxfp8_attention(q, k, v, k_scale, v_scale, out, scale, causal, Some(len))
}

fn kv_append_quant_impl(
    dst: (&mut Storage, &Layout),
    scale_dst: (&mut Storage, &Layout),
    src: (&Storage, &Layout),
    seq_pos: usize,
) -> Result<()> {
    let (dst_st, dst_lo) = dst;
    let (sc_st, sc_lo) = scale_dst;
    let (src_st, src_lo) = src;
    if dst_lo.dtype() != DType::MXFP8 {
        return Err(SynaptixError::Unsupported("cpu kv_quant mxfp8: dst must be MXFP8"));
    }
    if sc_lo.dtype() != DType::U8 {
        return Err(SynaptixError::Unsupported("cpu kv_quant mxfp8: scale must be U8"));
    }
    check_float(src_lo.dtype(), "cpu kv_quant mxfp8: src must be F32/F16/BF16")?;
    if dst_lo.dims().len() != 4 || src_lo.dims().len() != 4 || sc_lo.dims().len() != 4 {
        return Err(SynaptixError::Unsupported("cpu kv_quant mxfp8: ranks [4,4,4]"));
    }
    if !dst_lo.is_contiguous() || !src_lo.is_contiguous() || !sc_lo.is_contiguous() {
        return Err(SynaptixError::NonContiguous);
    }
    let (b, nkv, max_seq, hd) = (dst_lo.dims()[0], dst_lo.dims()[1], dst_lo.dims()[2], dst_lo.dims()[3]);
    let (sb, snkv, t_new, shd) = (src_lo.dims()[0], src_lo.dims()[1], src_lo.dims()[2], src_lo.dims()[3]);
    if hd % MXFP8_BLOCK != 0 {
        return Err(SynaptixError::Unsupported("cpu kv_quant mxfp8: hd % 32 != 0"));
    }
    let nb = hd / MXFP8_BLOCK;
    if sb != b || snkv != nkv || shd != hd || sc_lo.dims() != [b, nkv, max_seq, nb] {
        return Err(SynaptixError::shape_mismatch(src_lo.dims(), dst_lo.dims()));
    }
    if seq_pos + t_new > max_seq {
        return Err(SynaptixError::Other(format!(
            "cpu kv_quant mxfp8: seq_pos {seq_pos} + t_new {t_new} > max_seq {max_seq}"
        )));
    }
    let src_buf = cpu(src_st, "cpu kv_quant mxfp8: src non-cpu")?;
    let src_dt = src_lo.dtype();
    let src_bytes = src_buf.as_bytes();
    let so = src_lo.offset();
    if (so + b * nkv * t_new * hd) * elem_bytes(src_dt) > src_bytes.len() {
        return Err(SynaptixError::Unsupported("cpu kv_quant mxfp8: src out of bounds"));
    }
    let dst_buf = cpu_mut(dst_st, "cpu kv_quant mxfp8: dst non-cpu")?;
    let dst_bytes = dst_buf.as_bytes_mut();
    let dst_o0 = dst_lo.offset();
    if dst_o0 + b * nkv * max_seq * hd > dst_bytes.len() {
        return Err(SynaptixError::Unsupported("cpu kv_quant mxfp8: dst out of bounds"));
    }
    let sc_buf = cpu_mut(sc_st, "cpu kv_quant mxfp8: scale non-cpu")?;
    let sc_bytes = sc_buf.as_bytes_mut();
    let sc_o0 = sc_lo.offset();
    if sc_o0 + b * nkv * max_seq * nb > sc_bytes.len() {
        return Err(SynaptixError::Unsupported("cpu kv_quant mxfp8: scale out of bounds"));
    }
    let dst_ptr = SendPtr(dst_bytes.as_mut_ptr());
    let sc_ptr = SendPtr(sc_bytes.as_mut_ptr());
    let rows = b * nkv * t_new;
    threads();
    (0..rows).into_par_iter().for_each(|r| {
        let bh = r / t_new;
        let t = r % t_new;
        let mut row = vec![0f32; hd];
        to_f32(src_dt, src_bytes, so + r * hd, &mut row).expect("kv_quant src dtype");
        let dst_row = dst_o0 + (bh * max_seq + seq_pos + t) * hd;
        let sc_row = sc_o0 + (bh * max_seq + seq_pos + t) * nb;
        let drow = unsafe { std::slice::from_raw_parts_mut(dst_ptr.get().add(dst_row), hd) };
        let srow = unsafe { std::slice::from_raw_parts_mut(sc_ptr.get().add(sc_row), nb) };
        for blk in 0..nb {
            let base = blk * MXFP8_BLOCK;
            let mut amax = 0.0f32;
            for i in 0..MXFP8_BLOCK {
                amax = amax.max(row[base + i].abs());
            }
            let sbyte = e8m0_scale_byte(amax);
            let sv = e8m0_decode(sbyte);
            for i in 0..MXFP8_BLOCK {
                drow[base + i] = encode_e4m3(row[base + i] / sv);
            }
            srow[blk] = sbyte;
        }
    });
    Ok(())
}

pub fn kv_append_quant_mxfp8(
    dst: (&mut Storage, &Layout),
    scale_dst: (&mut Storage, &Layout),
    src: (&Storage, &Layout),
    seq_pos: usize,
    _stream: &Stream,
) -> Result<()> {
    kv_append_quant_impl(dst, scale_dst, src, seq_pos)
}

pub fn kv_append_quant_mxfp8_dev(
    dst: (&mut Storage, &Layout),
    scale_dst: (&mut Storage, &Layout),
    src: (&Storage, &Layout),
    seq_pos: (&Storage, &Layout),
    _stream: &Stream,
) -> Result<()> {
    let pos = read_u32(seq_pos.0, seq_pos.1, "cpu kv_quant mxfp8 dev: seq_pos must be U32[1]")?;
    kv_append_quant_impl(dst, scale_dst, src, pos)
}

fn block_positions(
    table: (&Storage, &Layout),
    tail_from: (&Storage, &Layout),
    tail_len: (&Storage, &Layout),
    b: usize,
    ratio: usize,
    row_offset: usize,
    cap: usize,
) -> Result<Vec<Vec<u32>>> {
    let (tb_st, tb_lo) = table;
    if tb_lo.dims().len() != 2 {
        return Err(SynaptixError::Unsupported("cpu flash_blocks: table must be [rows, NB] U32"));
    }
    let rows = tb_lo.dims()[0];
    let nb = tb_lo.dims()[1];
    if rows < row_offset + b {
        return Err(SynaptixError::Unsupported("cpu flash_blocks: table rows < row_offset + B"));
    }
    let tbl = read_u32_slice(tb_st, tb_lo, "cpu flash_blocks: table must be [rows, NB] U32")?;
    let tf = read_u32_slice(tail_from.0, tail_from.1, "cpu flash_blocks: tail_from must be U32")?;
    let tl = read_u32_slice(tail_len.0, tail_len.1, "cpu flash_blocks: tail_len must be U32")?;
    if tf.len() < row_offset + b || tl.len() < row_offset + b {
        return Err(SynaptixError::Unsupported("cpu flash_blocks: tail rows < row_offset + B"));
    }
    let ratio = ratio.max(1);
    let mut out = Vec::with_capacity(b);
    for bi in 0..b {
        let r = bi + row_offset;
        let from = tf[r] as usize;
        let len = tl[r] as usize;
        let mut pos = Vec::with_capacity(nb * ratio + len);
        for &blk in &tbl[r * nb..(r + 1) * nb] {
            if blk == u32::MAX {
                continue;
            }
            let start = blk as usize * ratio;
            for t in start..start + ratio {
                if t < cap {
                    pos.push(t as u32);
                }
            }
        }
        for t in from..from + len {
            if t < cap {
                pos.push(t as u32);
            }
        }
        out.push(pos);
    }
    Ok(out)
}

fn view_blocks_q<'a>(bytes: &'a [u8], lo: &Layout, what: &'static str) -> Result<View<'a>> {
    if lo.dims().len() != 3 {
        return Err(SynaptixError::Unsupported(what));
    }
    let s = lo.strides().as_slice();
    if s[2] != 1 {
        return Err(SynaptixError::Unsupported(what));
    }
    let dm = lo.dims();
    let v = View {
        bytes,
        dtype: lo.dtype(),
        off: lo.offset(),
        sb: stride_u(s[0], what)?,
        sh: stride_u(s[1], what)?,
        st: 0,
        d: dm[2],
    };
    check_bounds(bytes.len(), elem_bytes(lo.dtype()), v.off, [dm[0], dm[1], 1], [v.sb, v.sh, 0], v.d, what)?;
    Ok(v)
}

fn view_blocks_kv<'a>(bytes: &'a [u8], lo: &Layout, elem: usize, what: &'static str) -> Result<View<'a>> {
    if lo.dims().len() != 3 {
        return Err(SynaptixError::Unsupported(what));
    }
    let s = lo.strides().as_slice();
    if s[2] != 1 {
        return Err(SynaptixError::Unsupported(what));
    }
    let dm = lo.dims();
    let v = View {
        bytes,
        dtype: lo.dtype(),
        off: lo.offset(),
        sb: 0,
        sh: stride_u(s[0], what)?,
        st: stride_u(s[1], what)?,
        d: dm[2],
    };
    check_bounds(bytes.len(), elem, v.off, [1, dm[0], dm[1]], [0, v.sh, v.st], v.d, what)?;
    Ok(v)
}

fn blocks_out(out_st: &mut Storage, out_lo: &Layout, q_dims: &[usize]) -> Result<OutView> {
    if out_lo.dims() != q_dims {
        return Err(SynaptixError::Unsupported("cpu flash_blocks: out shape != q shape"));
    }
    let s = out_lo.strides().as_slice();
    if s[2] != 1 {
        return Err(SynaptixError::Unsupported("cpu flash_blocks: out layout"));
    }
    check_float(out_lo.dtype(), "cpu flash_blocks: out dtype")?;
    let buf = cpu_mut(out_st, "cpu flash_blocks: out non-cpu")?;
    let bytes = buf.as_bytes_mut();
    let what = "cpu flash_blocks: out layout";
    let sb = stride_u(s[0], what)?;
    let sh = stride_u(s[1], what)?;
    check_bounds(bytes.len(), elem_bytes(out_lo.dtype()), out_lo.offset(), [q_dims[0], q_dims[1], 1], [sb, sh, 0], q_dims[2], what)?;
    Ok(OutView {
        ptr: SendPtr(bytes.as_mut_ptr()),
        len: bytes.len(),
        dtype: out_lo.dtype(),
        off: out_lo.offset(),
        sb,
        sh,
        st: 0,
        d: q_dims[2],
    })
}

pub fn flash_attention_blocks(
    q: (&Storage, &Layout),
    k: (&Storage, &Layout),
    v: (&Storage, &Layout),
    table: (&Storage, &Layout),
    tail_from: (&Storage, &Layout),
    tail_len: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    ratio: usize,
    scale: f32,
    row_offset: usize,
    _stream: &Stream,
) -> Result<()> {
    let (q_st, q_lo) = q;
    let (k_st, k_lo) = k;
    let (v_st, v_lo) = v;
    let (out_st, out_lo) = out;
    check_float(q_lo.dtype(), "cpu flash_blocks: q dtype")?;
    check_float(k_lo.dtype(), "cpu flash_blocks: k dtype")?;
    check_float(v_lo.dtype(), "cpu flash_blocks: v dtype")?;
    if q_lo.dims().len() != 3 || k_lo.dims().len() != 3 || v_lo.dims() != k_lo.dims() {
        return Err(SynaptixError::Unsupported("cpu flash_blocks: expect q [B,NH,D], k/v [NKV,CAP,D]"));
    }
    let (b, nh, d) = (q_lo.dims()[0], q_lo.dims()[1], q_lo.dims()[2]);
    let (nkv, cap) = (k_lo.dims()[0], k_lo.dims()[1]);
    if k_lo.dims()[2] != d || nkv == 0 || nh % nkv != 0 {
        return Err(SynaptixError::Unsupported("cpu flash_blocks: shapes/GQA"));
    }
    let positions = block_positions(table, tail_from, tail_len, b, ratio, row_offset, cap)?;
    let q_buf = cpu(q_st, "cpu flash_blocks: q non-cpu")?;
    let k_buf = cpu(k_st, "cpu flash_blocks: k non-cpu")?;
    let v_buf = cpu(v_st, "cpu flash_blocks: v non-cpu")?;
    let qv = view_blocks_q(q_buf.as_bytes(), q_lo, "cpu flash_blocks: q layout")?;
    let inner = DenseKv {
        k: view_blocks_kv(k_buf.as_bytes(), k_lo, elem_bytes(k_lo.dtype()), "cpu flash_blocks: k layout")?,
        v: view_blocks_kv(v_buf.as_bytes(), v_lo, elem_bytes(v_lo.dtype()), "cpu flash_blocks: v layout")?,
        len: cap,
    };
    let kv = GatheredKv { inner, positions, d };
    let out = blocks_out(out_st, out_lo, q_lo.dims())?;
    let p = Attn { q: qv, out, b, nh, nkv, tq: 1, d, scale, causal: false, window: 0, isa: pick_isa() };
    run_attention(&p, &kv);
    Ok(())
}

pub fn flash_attention_blocks_mxfp8(
    q: (&Storage, &Layout),
    k: (&Storage, &Layout),
    v: (&Storage, &Layout),
    k_scale: (&Storage, &Layout),
    v_scale: (&Storage, &Layout),
    table: (&Storage, &Layout),
    tail_from: (&Storage, &Layout),
    tail_len: (&Storage, &Layout),
    out: (&mut Storage, &Layout),
    ratio: usize,
    scale: f32,
    row_offset: usize,
    _stream: &Stream,
) -> Result<()> {
    let (q_st, q_lo) = q;
    let (k_st, k_lo) = k;
    let (v_st, v_lo) = v;
    let (ks_st, ks_lo) = k_scale;
    let (vs_st, vs_lo) = v_scale;
    let (out_st, out_lo) = out;
    check_float(q_lo.dtype(), "cpu flash_blocks mxfp8: q dtype")?;
    if k_lo.dtype() != DType::MXFP8 || v_lo.dtype() != DType::MXFP8 {
        return Err(SynaptixError::Unsupported("cpu flash_blocks mxfp8: k/v must be MXFP8"));
    }
    if ks_lo.dtype() != DType::U8 || vs_lo.dtype() != DType::U8 {
        return Err(SynaptixError::Unsupported("cpu flash_blocks mxfp8: scales must be U8"));
    }
    if q_lo.dims().len() != 3 || k_lo.dims().len() != 3 || v_lo.dims() != k_lo.dims() {
        return Err(SynaptixError::Unsupported("cpu flash_blocks mxfp8: expect q [B,NH,D], k/v [NKV,CAP,D]"));
    }
    let (b, nh, d) = (q_lo.dims()[0], q_lo.dims()[1], q_lo.dims()[2]);
    let (nkv, cap) = (k_lo.dims()[0], k_lo.dims()[1]);
    if k_lo.dims()[2] != d || nkv == 0 || nh % nkv != 0 {
        return Err(SynaptixError::Unsupported("cpu flash_blocks mxfp8: shapes/GQA"));
    }
    if d % MXFP8_BLOCK != 0 {
        return Err(SynaptixError::Unsupported("cpu flash_blocks mxfp8: d % 32 != 0"));
    }
    let nb = d / MXFP8_BLOCK;
    if ks_lo.dims() != [nkv, cap, nb] || vs_lo.dims() != [nkv, cap, nb] {
        return Err(SynaptixError::Unsupported("cpu flash_blocks mxfp8: scale shape"));
    }
    let positions = block_positions(table, tail_from, tail_len, b, ratio, row_offset, cap)?;
    let q_buf = cpu(q_st, "cpu flash_blocks mxfp8: q non-cpu")?;
    let k_buf = cpu(k_st, "cpu flash_blocks mxfp8: k non-cpu")?;
    let v_buf = cpu(v_st, "cpu flash_blocks mxfp8: v non-cpu")?;
    let ks_buf = cpu(ks_st, "cpu flash_blocks mxfp8: ks non-cpu")?;
    let vs_buf = cpu(vs_st, "cpu flash_blocks mxfp8: vs non-cpu")?;
    let qv = view_blocks_q(q_buf.as_bytes(), q_lo, "cpu flash_blocks mxfp8: q layout")?;
    let inner = Mxfp8Kv {
        k: view_blocks_kv(k_buf.as_bytes(), k_lo, 1, "cpu flash_blocks mxfp8: k layout")?,
        v: view_blocks_kv(v_buf.as_bytes(), v_lo, 1, "cpu flash_blocks mxfp8: v layout")?,
        ks: view_blocks_kv(ks_buf.as_bytes(), ks_lo, 1, "cpu flash_blocks mxfp8: k_scale layout")?,
        vs: view_blocks_kv(vs_buf.as_bytes(), vs_lo, 1, "cpu flash_blocks mxfp8: v_scale layout")?,
        len: cap,
        lut: e4m3_lut(),
    };
    let kv = GatheredKv { inner, positions, d };
    let out = blocks_out(out_st, out_lo, q_lo.dims())?;
    let p = Attn { q: qv, out, b, nh, nkv, tq: 1, d, scale, causal: false, window: 0, isa: pick_isa() };
    run_attention(&p, &kv);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_core::device::Device;
    use synaptix_core::tensor::shape::Shape;
    use synaptix_core::tensor::storage::CpuBuf;

    fn stream() -> Stream {
        Stream::default_for(Device::Cpu).unwrap()
    }

    fn gen(seed: u64, n: usize, amp: f32) -> Vec<f32> {
        let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let u = (x >> 33) as u32;
                ((u as f32 / u32::MAX as f32) * 2.0 - 1.0) * amp
            })
            .collect()
    }

    fn pack(dt: DType, v: &[f32]) -> Vec<u8> {
        let mut b = vec![0u8; v.len() * elem_bytes(dt)];
        from_f32(dt, v, &mut b, 0).unwrap();
        b
    }

    fn unpack(dt: DType, b: &[u8], n: usize) -> Vec<f32> {
        let mut v = vec![0f32; n];
        to_f32(dt, b, 0, &mut v).unwrap();
        v
    }

    fn round(dt: DType, v: &[f32]) -> Vec<f32> {
        unpack(dt, &pack(dt, v), v.len())
    }

    fn stor(bytes: Vec<u8>) -> Storage {
        Storage::Cpu(CpuBuf::from_vec(bytes))
    }

    fn lay(dims: &[usize], dt: DType) -> Layout {
        Layout::contiguous(Shape::new(dims.to_vec()), dt)
    }

    fn u32_tensor(v: &[u32], dims: &[usize]) -> (Storage, Layout) {
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        (stor(b), lay(dims, DType::U32))
    }

    fn naive(
        q: &[f32], k: &[f32], v: &[f32],
        b: usize, nh: usize, nkv: usize, tq: usize, tkv: usize, d: usize,
        scale: f32, causal: bool, window: i32,
    ) -> Vec<f32> {
        let group = nh / nkv;
        let mut out = vec![0f32; b * nh * tq * d];
        let q_pos_base = tkv.saturating_sub(tq);
        let mut s = vec![0f32; tkv];
        for bi in 0..b {
            for h in 0..nh {
                let kvh = h / group;
                for ti in 0..tq {
                    let q_pos = q_pos_base + ti;
                    let qr = &q[((bi * nh + h) * tq + ti) * d..][..d];
                    let mut mx = f32::NEG_INFINITY;
                    for t in 0..tkv {
                        let masked = (causal && t > q_pos)
                            || (window > 0 && (q_pos as i64 - t as i64).abs() > window as i64);
                        if masked {
                            s[t] = f32::NEG_INFINITY;
                            continue;
                        }
                        let kr = &k[((bi * nkv + kvh) * tkv + t) * d..][..d];
                        let dot: f32 = qr.iter().zip(kr).map(|(a, b)| a * b).sum();
                        s[t] = dot * scale;
                        mx = mx.max(s[t]);
                    }
                    let mut sum = 0f32;
                    for t in 0..tkv {
                        s[t] = if s[t] == f32::NEG_INFINITY { 0.0 } else { (s[t] - mx).exp() };
                        sum += s[t];
                    }
                    let o = &mut out[((bi * nh + h) * tq + ti) * d..][..d];
                    if sum > 0.0 {
                        for t in 0..tkv {
                            if s[t] == 0.0 {
                                continue;
                            }
                            let p = s[t] / sum;
                            let vr = &v[((bi * nkv + kvh) * tkv + t) * d..][..d];
                            for (oo, &x) in o.iter_mut().zip(vr) {
                                *oo += p * x;
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: len");
        let mut worst = 0f32;
        for i in 0..got.len() {
            let err = (got[i] - want[i]).abs() / (1.0 + want[i].abs());
            assert!(err <= tol && got[i].is_finite(), "{what}: idx {i} got {} want {} err {err}", got[i], want[i]);
            worst = worst.max(err);
        }
    }

    fn tol(dt: DType) -> f32 {
        if dt == DType::F32 { 1e-4 } else { 2e-2 }
    }

    const DTYPES: [DType; 3] = [DType::F32, DType::F16, DType::BF16];

    struct Case {
        b: usize,
        nh: usize,
        nkv: usize,
        tq: usize,
        tkv: usize,
        d: usize,
        causal: bool,
    }

    fn run_dense(dt: DType, c: &Case, window: i32, max_seq: usize, dev: bool) {
        let Case { b, nh, nkv, tq, tkv, d, causal } = *c;
        let scale = 1.0 / (d as f32).sqrt();
        let q = round(dt, &gen(1, b * nh * tq * d, 1.0));
        let kbuf = round(dt, &gen(2, b * nkv * max_seq * d, 1.0));
        let vbuf = round(dt, &gen(3, b * nkv * max_seq * d, 1.0));
        let mut k = vec![0f32; b * nkv * tkv * d];
        let mut v = vec![0f32; b * nkv * tkv * d];
        for bh in 0..b * nkv {
            for t in 0..tkv {
                k[(bh * tkv + t) * d..][..d].copy_from_slice(&kbuf[(bh * max_seq + t) * d..][..d]);
                v[(bh * tkv + t) * d..][..d].copy_from_slice(&vbuf[(bh * max_seq + t) * d..][..d]);
            }
        }
        let want = naive(&q, &k, &v, b, nh, nkv, tq, tkv, d, scale, causal, window);
        let q_st = stor(pack(dt, &q));
        let q_lo = lay(&[b, nh, tq, d], dt);
        let k_st = stor(pack(dt, &kbuf));
        let v_st = stor(pack(dt, &vbuf));
        let full = lay(&[b, nkv, max_seq, d], dt);
        let kv_lo = if dev { full.clone() } else { full.narrow(2, 0, tkv).unwrap() };
        let mut out_st = stor(vec![0u8; b * nh * tq * d * elem_bytes(dt)]);
        let out_lo = lay(&[b, nh, tq, d], dt);
        let st = stream();
        let tc = u32_tensor(&[tkv as u32], &[1]);
        match (dev, window) {
            (false, 0) => flash_attention((&q_st, &q_lo), (&k_st, &kv_lo), (&v_st, &kv_lo), (&mut out_st, &out_lo), scale, causal, &st).unwrap(),
            (true, 0) => flash_attention_dev((&q_st, &q_lo), (&k_st, &kv_lo), (&v_st, &kv_lo), (&tc.0, &tc.1), (&mut out_st, &out_lo), scale, causal, &st).unwrap(),
            (false, w) => flash_attention_window((&q_st, &q_lo), (&k_st, &kv_lo), (&v_st, &kv_lo), (&mut out_st, &out_lo), scale, w, causal, &st).unwrap(),
            (true, w) => flash_attention_window_dev((&q_st, &q_lo), (&k_st, &kv_lo), (&v_st, &kv_lo), (&tc.0, &tc.1), (&mut out_st, &out_lo), scale, w, causal, &st).unwrap(),
        }
        let got = unpack(dt, out_st.as_cpu().unwrap().as_bytes(), want.len());
        assert_close(&got, &want, tol(dt), &format!("{dt:?} b{b} nh{nh} nkv{nkv} tq{tq} tkv{tkv} d{d} causal{causal} window{window} max_seq{max_seq} dev{dev}"));
    }

    #[test]
    fn dense_matches_naive() {
        let cases = [
            Case { b: 1, nh: 4, nkv: 4, tq: 1, tkv: 64, d: 64, causal: true },
            Case { b: 2, nh: 8, nkv: 2, tq: 1, tkv: 200, d: 128, causal: true },
            Case { b: 1, nh: 8, nkv: 2, tq: 32, tkv: 32, d: 96, causal: true },
            Case { b: 1, nh: 8, nkv: 2, tq: 32, tkv: 32, d: 96, causal: false },
            Case { b: 2, nh: 2, nkv: 1, tq: 70, tkv: 70, d: 128, causal: true },
            Case { b: 1, nh: 3, nkv: 1, tq: 7, tkv: 150, d: 64, causal: true },
            Case { b: 1, nh: 3, nkv: 1, tq: 7, tkv: 150, d: 64, causal: false },
            Case { b: 1, nh: 2, nkv: 2, tq: 5, tkv: 3, d: 40, causal: true },
            Case { b: 1, nh: 1, nkv: 1, tq: 9, tkv: 17, d: 20, causal: false },
        ];
        for dt in DTYPES {
            for c in &cases {
                run_dense(dt, c, 0, c.tkv, false);
            }
        }
    }

    #[test]
    fn strided_kv_and_dev() {
        let cases = [
            Case { b: 2, nh: 8, nkv: 2, tq: 1, tkv: 50, d: 128, causal: true },
            Case { b: 1, nh: 4, nkv: 2, tq: 16, tkv: 100, d: 64, causal: true },
            Case { b: 1, nh: 4, nkv: 2, tq: 16, tkv: 100, d: 64, causal: false },
        ];
        for dt in DTYPES {
            for c in &cases {
                run_dense(dt, c, 0, c.tkv + 46, false);
                run_dense(dt, c, 0, c.tkv + 46, true);
            }
        }
    }

    #[test]
    fn window_band() {
        let cases = [
            Case { b: 1, nh: 4, nkv: 2, tq: 16, tkv: 16, d: 64, causal: true },
            Case { b: 1, nh: 4, nkv: 2, tq: 16, tkv: 16, d: 64, causal: false },
            Case { b: 1, nh: 2, nkv: 1, tq: 4, tkv: 90, d: 128, causal: true },
            Case { b: 1, nh: 2, nkv: 1, tq: 90, tkv: 90, d: 32, causal: false },
            Case { b: 1, nh: 2, nkv: 2, tq: 1, tkv: 300, d: 64, causal: true },
        ];
        for dt in DTYPES {
            for c in &cases {
                for w in [3, 20, -1] {
                    run_dense(dt, c, w, c.tkv, false);
                    run_dense(dt, c, w, c.tkv + 10, true);
                }
            }
        }
    }

    #[test]
    fn decode_split_kv_path() {
        for dt in DTYPES {
            run_dense(dt, &Case { b: 1, nh: 1, nkv: 1, tq: 1, tkv: 4096, d: 128, causal: true }, 0, 4096, false);
            run_dense(dt, &Case { b: 1, nh: 4, nkv: 1, tq: 3, tkv: 1000, d: 64, causal: false }, 0, 1000, false);
        }
    }

    #[test]
    fn bshd_layout() {
        for dt in DTYPES {
            for &(b, s, h, hkv, d, causal) in &[(1usize, 20usize, 4usize, 2usize, 64usize, false), (2, 33, 2, 2, 128, true), (1, 1, 4, 1, 96, false)] {
                let scale = 1.0 / (d as f32).sqrt();
                let q = round(dt, &gen(11, b * s * h * d, 1.0));
                let k = round(dt, &gen(12, b * s * hkv * d, 1.0));
                let v = round(dt, &gen(13, b * s * hkv * d, 1.0));
                let to_bhsd = |x: &[f32], hh: usize| -> Vec<f32> {
                    let mut o = vec![0f32; x.len()];
                    for bi in 0..b {
                        for t in 0..s {
                            for hi in 0..hh {
                                o[((bi * hh + hi) * s + t) * d..][..d].copy_from_slice(&x[((bi * s + t) * hh + hi) * d..][..d]);
                            }
                        }
                    }
                    o
                };
                let want_bhsd = naive(&to_bhsd(&q, h), &to_bhsd(&k, hkv), &to_bhsd(&v, hkv), b, h, hkv, s, s, d, scale, causal, 0);
                let mut want = vec![0f32; want_bhsd.len()];
                for bi in 0..b {
                    for t in 0..s {
                        for hi in 0..h {
                            want[((bi * s + t) * h + hi) * d..][..d].copy_from_slice(&want_bhsd[((bi * h + hi) * s + t) * d..][..d]);
                        }
                    }
                }
                let q_st = stor(pack(dt, &q));
                let k_st = stor(pack(dt, &k));
                let v_st = stor(pack(dt, &v));
                let q_lo = lay(&[b, s, h, d], dt);
                let kv_lo = lay(&[b, s, hkv, d], dt);
                let mut out_st = stor(vec![0u8; want.len() * elem_bytes(dt)]);
                flash_attention_bshd((&q_st, &q_lo), (&k_st, &kv_lo), (&v_st, &kv_lo), (&mut out_st, &q_lo), scale, causal, &stream()).unwrap();
                let got = unpack(dt, out_st.as_cpu().unwrap().as_bytes(), want.len());
                assert_close(&got, &want, tol(dt), &format!("bshd {dt:?} b{b} s{s} h{h} hkv{hkv} d{d}"));
            }
        }
    }

    fn quantize_rows(x: &[f32], d: usize) -> (Vec<u8>, Vec<u8>) {
        let nb = d / MXFP8_BLOCK;
        let rows = x.len() / d;
        let mut data = vec![0u8; x.len()];
        let mut sc = vec![0u8; rows * nb];
        for r in 0..rows {
            for blk in 0..nb {
                let base = r * d + blk * MXFP8_BLOCK;
                let mut amax = 0f32;
                for i in 0..MXFP8_BLOCK {
                    amax = amax.max(x[base + i].abs());
                }
                let sbyte = e8m0_scale_byte(amax);
                let sv = e8m0_decode(sbyte);
                for i in 0..MXFP8_BLOCK {
                    data[base + i] = encode_e4m3(x[base + i] / sv);
                }
                sc[r * nb + blk] = sbyte;
            }
        }
        (data, sc)
    }

    fn ref_mxfp8(
        qb: &[u8], q_dtype: DType, q_lo: &Layout,
        kb: &[u8], k_lo: &Layout, vb: &[u8], v_lo: &Layout,
        ksb: &[u8], ks_lo: &Layout, vsb: &[u8], vs_lo: &Layout,
        tkv: usize, scale: f32, causal: bool,
    ) -> Vec<f32> {
        let (b, nh, tq, d) = (q_lo.dims()[0], q_lo.dims()[1], q_lo.dims()[2], q_lo.dims()[3]);
        let nkv = k_lo.dims()[1];
        let nb = d / MXFP8_BLOCK;
        let group = nh / nkv;
        let qs = q_lo.strides().as_slice().to_vec();
        let ks = k_lo.strides().as_slice().to_vec();
        let vs = v_lo.strides().as_slice().to_vec();
        let kss = ks_lo.strides().as_slice().to_vec();
        let vss = vs_lo.strides().as_slice().to_vec();
        let (qo, ko, vo, kso, vso) = (q_lo.offset(), k_lo.offset(), v_lo.offset(), ks_lo.offset(), vs_lo.offset());
        let read_qf = |idx: usize| -> f32 {
            match q_dtype {
                DType::F32 => f32::from_le_bytes(qb[idx * 4..idx * 4 + 4].try_into().unwrap()),
                DType::F16 => half::f16::from_bits(u16::from_le_bytes(qb[idx * 2..idx * 2 + 2].try_into().unwrap())).to_f32(),
                DType::BF16 => half::bf16::from_bits(u16::from_le_bytes(qb[idx * 2..idx * 2 + 2].try_into().unwrap())).to_f32(),
                _ => 0.0,
            }
        };
        let mut out_f32 = vec![0.0f32; b * nh * tq * d];
        for bi in 0..b {
            for h in 0..nh {
                let h_kv = h / group;
                for ti in 0..tq {
                    let q_pos = if tkv >= tq { tkv - tq + ti } else { ti };
                    let mut m = f32::NEG_INFINITY;
                    let mut l = 0.0f32;
                    let mut acc = vec![0.0f32; d];
                    for t in 0..tkv {
                        if causal && t > q_pos {
                            break;
                        }
                        let mut dot = 0.0f32;
                        for blk in 0..nb {
                            let ks_idx = (kso as isize + bi as isize * kss[0] + h_kv as isize * kss[1] + t as isize * kss[2] + blk as isize * kss[3]) as usize;
                            let ksc = e8m0_decode(ksb[ks_idx]);
                            let mut bsum = 0.0f32;
                            for i in 0..MXFP8_BLOCK {
                                let dd = blk * MXFP8_BLOCK + i;
                                let qi = (qo as isize + bi as isize * qs[0] + h as isize * qs[1] + ti as isize * qs[2] + dd as isize * qs[3]) as usize;
                                let ki = (ko as isize + bi as isize * ks[0] + h_kv as isize * ks[1] + t as isize * ks[2] + dd as isize * ks[3]) as usize;
                                bsum += read_qf(qi) * decode_e4m3(kb[ki]);
                            }
                            dot += bsum * ksc;
                        }
                        let s = dot * scale;
                        let m_new = m.max(s);
                        let alpha = if m == f32::NEG_INFINITY { 0.0 } else { (m - m_new).exp() };
                        let p = (s - m_new).exp();
                        l = l * alpha + p;
                        for dd in 0..d {
                            let blk = dd / MXFP8_BLOCK;
                            let vs_idx = (vso as isize + bi as isize * vss[0] + h_kv as isize * vss[1] + t as isize * vss[2] + blk as isize * vss[3]) as usize;
                            let vsc = e8m0_decode(vsb[vs_idx]);
                            let vi = (vo as isize + bi as isize * vs[0] + h_kv as isize * vs[1] + t as isize * vs[2] + dd as isize * vs[3]) as usize;
                            acc[dd] = acc[dd] * alpha + p * decode_e4m3(vb[vi]) * vsc;
                        }
                        m = m_new;
                    }
                    let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
                    let obase = ((bi * nh + h) * tq + ti) * d;
                    for dd in 0..d {
                        out_f32[obase + dd] = acc[dd] * inv;
                    }
                }
            }
        }
        out_f32
    }

    #[test]
    fn mxfp8_matches_reference() {
        let cases = [
            Case { b: 1, nh: 4, nkv: 4, tq: 1, tkv: 64, d: 64, causal: true },
            Case { b: 2, nh: 8, nkv: 2, tq: 1, tkv: 130, d: 128, causal: true },
            Case { b: 1, nh: 8, nkv: 2, tq: 24, tkv: 24, d: 96, causal: true },
            Case { b: 1, nh: 2, nkv: 1, tq: 24, tkv: 24, d: 64, causal: false },
            Case { b: 1, nh: 2, nkv: 1, tq: 5, tkv: 70, d: 128, causal: true },
        ];
        for dt in DTYPES {
            for c in &cases {
                for (max_seq, dev) in [(c.tkv, false), (c.tkv + 13, false), (c.tkv + 13, true)] {
                    let Case { b, nh, nkv, tq, tkv, d, causal } = *c;
                    let scale = 1.0 / (d as f32).sqrt();
                    let q = round(dt, &gen(21, b * nh * tq * d, 1.0));
                    let kf = gen(22, b * nkv * max_seq * d, 3.0);
                    let vf = gen(23, b * nkv * max_seq * d, 3.0);
                    let (kq, ksq) = quantize_rows(&kf, d);
                    let (vq, vsq) = quantize_rows(&vf, d);
                    let q_lo = lay(&[b, nh, tq, d], dt);
                    let kv_full = lay(&[b, nkv, max_seq, d], DType::MXFP8);
                    let sc_full = lay(&[b, nkv, max_seq, d / MXFP8_BLOCK], DType::U8);
                    let (kv_lo, sc_lo) = if dev {
                        (kv_full.clone(), sc_full.clone())
                    } else {
                        (kv_full.narrow(2, 0, tkv).unwrap(), sc_full.narrow(2, 0, tkv).unwrap())
                    };
                    let qb = pack(dt, &q);
                    let want = ref_mxfp8(&qb, dt, &q_lo, &kq, &kv_lo, &vq, &kv_lo, &ksq, &sc_lo, &vsq, &sc_lo, tkv, scale, causal);
                    let q_st = stor(qb);
                    let k_st = stor(kq);
                    let v_st = stor(vq);
                    let ks_st = stor(ksq);
                    let vs_st = stor(vsq);
                    let mut out_st = stor(vec![0u8; want.len() * elem_bytes(dt)]);
                    let tc = u32_tensor(&[tkv as u32], &[1]);
                    if dev {
                        flash_attention_mxfp8kv_dev((&q_st, &q_lo), (&k_st, &kv_lo), (&v_st, &kv_lo), (&ks_st, &sc_lo), (&vs_st, &sc_lo), (&tc.0, &tc.1), (&mut out_st, &q_lo), scale, causal, &stream()).unwrap();
                    } else {
                        flash_attention_mxfp8kv((&q_st, &q_lo), (&k_st, &kv_lo), (&v_st, &kv_lo), (&ks_st, &sc_lo), (&vs_st, &sc_lo), (&mut out_st, &q_lo), scale, causal, &stream()).unwrap();
                    }
                    let got = unpack(dt, out_st.as_cpu().unwrap().as_bytes(), want.len());
                    let t = if dt == DType::F32 { 1e-3 } else { 2e-2 };
                    assert_close(&got, &want, t, &format!("mxfp8 {dt:?} b{b} nh{nh} nkv{nkv} tq{tq} tkv{tkv} d{d} causal{causal} max_seq{max_seq} dev{dev}"));
                }
            }
        }
    }

    fn ref_kv_append(src: &[f32], b: usize, nkv: usize, t_new: usize, hd: usize, max_seq: usize, seq_pos: usize, dst: &mut [u8], sc: &mut [u8]) {
        let nb = hd / MXFP8_BLOCK;
        for bh in 0..(b * nkv) {
            for t in 0..t_new {
                let src_row = (bh * t_new + t) * hd;
                let dst_row = (bh * max_seq + seq_pos + t) * hd;
                let sc_row = (bh * max_seq + seq_pos + t) * nb;
                for blk in 0..nb {
                    let base = blk * MXFP8_BLOCK;
                    let mut amax = 0.0f32;
                    for i in 0..MXFP8_BLOCK {
                        amax = amax.max(src[src_row + base + i].abs());
                    }
                    let sbyte = e8m0_scale_byte(amax);
                    let sv = e8m0_decode(sbyte);
                    for i in 0..MXFP8_BLOCK {
                        dst[dst_row + base + i] = encode_e4m3(src[src_row + base + i] / sv);
                    }
                    sc[sc_row + blk] = sbyte;
                }
            }
        }
    }

    #[test]
    fn kv_append_quant_bit_exact() {
        let (b, nkv, max_seq, hd, t_new) = (2usize, 3usize, 40usize, 96usize, 7usize);
        let nb = hd / MXFP8_BLOCK;
        for dt in DTYPES {
            for (seq_pos, dev) in [(0usize, false), (5, false), (33, true)] {
                let raw = gen(31 + seq_pos as u64, b * nkv * t_new * hd, 4.0);
                let src_bytes = pack(dt, &raw);
                let src_f = unpack(dt, &src_bytes, raw.len());
                let mut want_dst = vec![0u8; b * nkv * max_seq * hd];
                let mut want_sc = vec![0u8; b * nkv * max_seq * nb];
                if dt == DType::BF16 {
                    let bf: Vec<half::bf16> = src_f.iter().map(|&x| half::bf16::from_f32(x)).collect();
                    let as_f32: Vec<f32> = bf.iter().map(|x| x.to_f32()).collect();
                    ref_kv_append(&as_f32, b, nkv, t_new, hd, max_seq, seq_pos, &mut want_dst, &mut want_sc);
                } else {
                    ref_kv_append(&src_f, b, nkv, t_new, hd, max_seq, seq_pos, &mut want_dst, &mut want_sc);
                }
                let mut dst_st = stor(vec![0u8; b * nkv * max_seq * hd]);
                let mut sc_st = stor(vec![0u8; b * nkv * max_seq * nb]);
                let dst_lo = lay(&[b, nkv, max_seq, hd], DType::MXFP8);
                let sc_lo = lay(&[b, nkv, max_seq, nb], DType::U8);
                let src_st = stor(src_bytes);
                let src_lo = lay(&[b, nkv, t_new, hd], dt);
                if dev {
                    let sp = u32_tensor(&[seq_pos as u32], &[1]);
                    kv_append_quant_mxfp8_dev((&mut dst_st, &dst_lo), (&mut sc_st, &sc_lo), (&src_st, &src_lo), (&sp.0, &sp.1), &stream()).unwrap();
                } else {
                    kv_append_quant_mxfp8((&mut dst_st, &dst_lo), (&mut sc_st, &sc_lo), (&src_st, &src_lo), seq_pos, &stream()).unwrap();
                }
                assert_eq!(dst_st.as_cpu().unwrap().as_bytes(), &want_dst[..], "{dt:?} seq_pos {seq_pos} data");
                assert_eq!(sc_st.as_cpu().unwrap().as_bytes(), &want_sc[..], "{dt:?} seq_pos {seq_pos} scale");
            }
        }
    }

    #[test]
    fn blocks_sparse_attention() {
        let (b, nh, nkv, d, cap, ratio, nb, row_offset) = (3usize, 4usize, 2usize, 64usize, 64usize, 4usize, 3usize, 2usize);
        let rows = row_offset + b;
        let table: Vec<u32> = vec![
            0, 0, 0,
            0, 0, 0,
            1, 5, u32::MAX,
            u32::MAX, u32::MAX, u32::MAX,
            3, 0, 9,
        ];
        let tail_from: Vec<u32> = vec![0, 0, 24, 40, 60];
        let tail_len: Vec<u32> = vec![0, 0, 3, 0, 4];
        let scale = 0.2;
        let group = nh / nkv;
        for dt in DTYPES {
            for quant in [false, true] {
                let q = round(dt, &gen(41, b * nh * d, 1.0));
                let kf = gen(42, nkv * cap * d, 2.0);
                let vf = gen(43, nkv * cap * d, 2.0);
                let (k, v, kq, ksq, vq, vsq) = if quant {
                    let (kq, ksq) = quantize_rows(&kf, d);
                    let (vq, vsq) = quantize_rows(&vf, d);
                    let lut = e4m3_lut();
                    let deq = |data: &[u8], sc: &[u8]| -> Vec<f32> {
                        let mut o = vec![0f32; data.len()];
                        for r in 0..nkv * cap {
                            for blk in 0..d / MXFP8_BLOCK {
                                let sv = e8m0_decode(sc[r * (d / MXFP8_BLOCK) + blk]);
                                for i in 0..MXFP8_BLOCK {
                                    o[r * d + blk * MXFP8_BLOCK + i] = lut[data[r * d + blk * MXFP8_BLOCK + i] as usize] * sv;
                                }
                            }
                        }
                        o
                    };
                    (deq(&kq, &ksq), deq(&vq, &vsq), kq, ksq, vq, vsq)
                } else {
                    (round(dt, &kf), round(dt, &vf), Vec::new(), Vec::new(), Vec::new(), Vec::new())
                };
                let mut want = vec![0f32; b * nh * d];
                for bi in 0..b {
                    let r = bi + row_offset;
                    let mut pos: Vec<usize> = Vec::new();
                    for &blk in &table[r * nb..(r + 1) * nb] {
                        if blk != u32::MAX {
                            pos.extend((blk as usize * ratio)..(blk as usize * ratio + ratio));
                        }
                    }
                    pos.extend(tail_from[r] as usize..(tail_from[r] + tail_len[r]) as usize);
                    for h in 0..nh {
                        let kvh = h / group;
                        let qr = &q[(bi * nh + h) * d..][..d];
                        let s: Vec<f32> = pos.iter().map(|&t| qr.iter().zip(&k[(kvh * cap + t) * d..][..d]).map(|(a, b)| a * b).sum::<f32>() * scale).collect();
                        let mx = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let e: Vec<f32> = s.iter().map(|&x| (x - mx).exp()).collect();
                        let sum: f32 = e.iter().sum();
                        let o = &mut want[(bi * nh + h) * d..][..d];
                        if sum > 0.0 {
                            for (j, &t) in pos.iter().enumerate() {
                                let p = e[j] / sum;
                                for (oo, &x) in o.iter_mut().zip(&v[(kvh * cap + t) * d..][..d]) {
                                    *oo += p * x;
                                }
                            }
                        }
                    }
                }
                let q_st = stor(pack(dt, &q));
                let q_lo = lay(&[b, nh, d], dt);
                let tb = u32_tensor(&table, &[rows, nb]);
                let tf = u32_tensor(&tail_from, &[rows]);
                let tl = u32_tensor(&tail_len, &[rows]);
                let mut out_st = stor(vec![0u8; b * nh * d * elem_bytes(dt)]);
                if quant {
                    let kv_lo = lay(&[nkv, cap, d], DType::MXFP8);
                    let sc_lo = lay(&[nkv, cap, d / MXFP8_BLOCK], DType::U8);
                    let (k_st, v_st, ks_st, vs_st) = (stor(kq), stor(vq), stor(ksq), stor(vsq));
                    flash_attention_blocks_mxfp8((&q_st, &q_lo), (&k_st, &kv_lo), (&v_st, &kv_lo), (&ks_st, &sc_lo), (&vs_st, &sc_lo), (&tb.0, &tb.1), (&tf.0, &tf.1), (&tl.0, &tl.1), (&mut out_st, &q_lo), ratio, scale, row_offset, &stream()).unwrap();
                } else {
                    let kv_lo = lay(&[nkv, cap, d], dt);
                    let (k_st, v_st) = (stor(pack(dt, &k)), stor(pack(dt, &v)));
                    flash_attention_blocks((&q_st, &q_lo), (&k_st, &kv_lo), (&v_st, &kv_lo), (&tb.0, &tb.1), (&tf.0, &tf.1), (&tl.0, &tl.1), (&mut out_st, &q_lo), ratio, scale, row_offset, &stream()).unwrap();
                }
                let got = unpack(dt, out_st.as_cpu().unwrap().as_bytes(), want.len());
                let t = if dt == DType::F32 { 1e-3 } else { 2e-2 };
                assert_close(&got, &want, t, &format!("blocks {dt:?} quant{quant}"));
                let zero_row = &got[nh * d..2 * nh * d];
                assert!(zero_row.iter().all(|&x| x == 0.0), "empty selection must give zeros");
            }
        }
    }

    #[test]
    fn simd_kernels_match_scalar() {
        let isa = pick_isa();
        for &(n, d) in &[(1usize, 1usize), (3, 7), (4, 8), (5, 16), (9, 20), (64, 64), (63, 96), (13, 128), (130, 40)] {
            let q = gen(61, d, 1.0);
            let k = gen(62, n * d, 1.0);
            let v = gen(63, n * d, 1.0);
            let mut s_ref = vec![0f32; n];
            let mut s_got = vec![0f32; n];
            qk(Isa::Scalar, &q, &k, n, d, &mut s_ref);
            qk(isa, &q, &k, n, d, &mut s_got);
            assert_close(&s_got, &s_ref, 1e-5, &format!("qk n{n} d{d}"));
            let m = s_ref.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut e_ref = s_ref.clone();
            let mut e_got = s_ref.clone();
            let sum_ref = exp_sum(Isa::Scalar, &mut e_ref, m);
            let sum_got = exp_sum(isa, &mut e_got, m);
            assert_close(&e_got, &e_ref, 1e-6, &format!("exp n{n}"));
            assert_close(&[sum_got], &[sum_ref], 1e-5, &format!("exp sum n{n}"));
            let mut a_ref = gen(64, d, 1.0);
            let mut a_got = a_ref.clone();
            pv(Isa::Scalar, &mut a_ref, &e_ref, &v, n, d);
            pv(isa, &mut a_got, &e_ref, &v, n, d);
            assert_close(&a_got, &a_ref, 1e-5, &format!("pv n{n} d{d}"));
        }
        let mut big = vec![0f32; 1000];
        for (i, x) in big.iter_mut().enumerate() {
            *x = -(i as f32) * 0.1;
        }
        let mut big_ref = big.clone();
        let sum_ref = exp_sum(Isa::Scalar, &mut big_ref, 0.0);
        let sum_got = exp_sum(isa, &mut big, 0.0);
        assert_close(&big, &big_ref, 1e-6, "exp wide range");
        assert_close(&[sum_got], &[sum_ref], 1e-5, "exp wide range sum");
    }

    #[test]
    fn unsupported_inputs() {
        let dt = DType::F32;
        let q_st = stor(vec![0u8; 4 * 16]);
        let q_lo = lay(&[1, 1, 1, 16], dt);
        let k_lo = lay(&[1, 1, 4, 16], dt).transpose(2, 3).unwrap();
        let k_st = stor(vec![0u8; 4 * 64]);
        let mut out_st = stor(vec![0u8; 4 * 16]);
        let r = flash_attention((&q_st, &q_lo), (&k_st, &k_lo), (&k_st, &k_lo), (&mut out_st, &q_lo), 1.0, false, &stream());
        assert!(matches!(r, Err(SynaptixError::Unsupported(_))));
        let bad_dt = lay(&[1, 1, 4, 16], DType::U8);
        let r = flash_attention((&q_st, &q_lo), (&k_st, &bad_dt), (&k_st, &bad_dt), (&mut out_st, &q_lo), 1.0, false, &stream());
        assert!(matches!(r, Err(SynaptixError::Unsupported(_))));
    }

    #[test]
    fn timing_decode_and_prefill() {
        let dt = DType::F16;
        let st = stream();
        let run = |b: usize, nh: usize, nkv: usize, tq: usize, tkv: usize, d: usize, causal: bool, label: &str| {
            let scale = 1.0 / (d as f32).sqrt();
            let q_st = stor(pack(dt, &gen(51, b * nh * tq * d, 1.0)));
            let k_st = stor(pack(dt, &gen(52, b * nkv * tkv * d, 1.0)));
            let v_st = stor(pack(dt, &gen(53, b * nkv * tkv * d, 1.0)));
            let q_lo = lay(&[b, nh, tq, d], dt);
            let kv_lo = lay(&[b, nkv, tkv, d], dt);
            let mut out_st = stor(vec![0u8; b * nh * tq * d * 2]);
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                flash_attention((&q_st, &q_lo), (&k_st, &kv_lo), (&v_st, &kv_lo), (&mut out_st, &q_lo), scale, causal, &st).unwrap();
                best = best.min(t0.elapsed().as_secs_f64());
            }
            let got = unpack(dt, out_st.as_cpu().unwrap().as_bytes(), 64);
            assert!(got.iter().all(|x| x.is_finite()));
            println!("{label}: {:.3} ms (threads {}, isa {:?})", best * 1e3, threads(), pick_isa());
        };
        run(1, 32, 8, 1, 4096, 128, true, "decode Tq=1 Tkv=4096 nh=32 nkv=8 hd=128 F16");
        run(1, 32, 8, 2048, 2048, 128, true, "prefill Tq=Tkv=2048 causal nh=32 nkv=8 hd=128 F16");
    }
}
