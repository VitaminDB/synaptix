use synaptix_core::quant::{ggml_dequant, GgmlType};

pub fn dequant_row(t: GgmlType, src: &[u8], k: usize, dst: &mut [f32]) {
    let be = t.block_elems();
    let bb = t.block_bytes();
    if k % be != 0 {
        ggml_dequant::dequantize(t, src, k, dst).expect("ggml dequant");
        return;
    }
    let fast = match t {
        GgmlType::Q8_0 => Some(deq_q8_0 as fn(&[u8], &mut [f32])),
        GgmlType::Q4_0 => Some(deq_q4_0 as fn(&[u8], &mut [f32])),
        GgmlType::Q4K => Some(deq_q4_k as fn(&[u8], &mut [f32])),
        GgmlType::Q6K => Some(deq_q6_k as fn(&[u8], &mut [f32])),
        _ => None,
    };
    match fast {
        Some(f) => {
            for (ib, blk) in src[..k / be * bb].chunks_exact(bb).enumerate() {
                f(blk, &mut dst[ib * be..(ib + 1) * be]);
            }
        }
        None => ggml_dequant::dequantize(t, src, k, dst).expect("ggml dequant"),
    }
}

#[inline]
fn f16_at(b: &[u8], o: usize) -> f32 {
    half::f16::from_le_bytes([b[o], b[o + 1]]).to_f32()
}

#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    ggml_dequant::scale_min_k4(j, q)
}

fn deq_q8_0(b: &[u8], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx2 {
            unsafe { x86::q8_0(b, y) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { arm::q8_0(b, y) };
        return;
    }
    #[allow(unreachable_code)]
    ggml_dequant::dequantize(GgmlType::Q8_0, b, 32, y).unwrap();
}

fn deq_q4_0(b: &[u8], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx2 {
            unsafe { x86::q4_0(b, y) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { arm::q4_0(b, y) };
        return;
    }
    #[allow(unreachable_code)]
    ggml_dequant::dequantize(GgmlType::Q4_0, b, 32, y).unwrap();
}

fn deq_q4_k(b: &[u8], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx2 {
            unsafe { x86::q4_k(b, y) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { arm::q4_k(b, y) };
        return;
    }
    #[allow(unreachable_code)]
    ggml_dequant::dequantize(GgmlType::Q4K, b, 256, y).unwrap();
}

fn deq_q6_k(b: &[u8], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx2 {
            unsafe { x86::q6_k(b, y) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { arm::q6_k(b, y) };
        return;
    }
    #[allow(unreachable_code)]
    ggml_dequant::dequantize(GgmlType::Q6K, b, 256, y).unwrap();
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn i8x8_to_ps(p: *const u8) -> __m256 {
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(p as *const __m128i)))
    }

    #[inline]
    unsafe fn u8x8_to_ps(v: __m128i) -> __m256 {
        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn q8_0(b: &[u8], y: &mut [f32]) {
        let d = _mm256_set1_ps(super::f16_at(b, 0));
        let p = b.as_ptr().add(2);
        for j in 0..4 {
            _mm256_storeu_ps(y.as_mut_ptr().add(j * 8), _mm256_mul_ps(i8x8_to_ps(p.add(j * 8)), d));
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn q4_0(b: &[u8], y: &mut [f32]) {
        let d = _mm256_set1_ps(super::f16_at(b, 0));
        let qs = _mm_loadu_si128(b.as_ptr().add(2) as *const __m128i);
        let lo = _mm_and_si128(qs, _mm_set1_epi8(0x0f));
        let hi = _mm_and_si128(_mm_srli_epi16(qs, 4), _mm_set1_epi8(0x0f));
        let eight = _mm256_set1_ps(8.0);
        let yp = y.as_mut_ptr();
        _mm256_storeu_ps(yp, _mm256_mul_ps(_mm256_sub_ps(u8x8_to_ps(lo), eight), d));
        _mm256_storeu_ps(yp.add(8), _mm256_mul_ps(_mm256_sub_ps(u8x8_to_ps(_mm_srli_si128(lo, 8)), eight), d));
        _mm256_storeu_ps(yp.add(16), _mm256_mul_ps(_mm256_sub_ps(u8x8_to_ps(hi), eight), d));
        _mm256_storeu_ps(yp.add(24), _mm256_mul_ps(_mm256_sub_ps(u8x8_to_ps(_mm_srli_si128(hi, 8)), eight), d));
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn q4_k(b: &[u8], y: &mut [f32]) {
        let d = super::f16_at(b, 0);
        let dmin = super::f16_at(b, 2);
        let scales = &b[4..16];
        let qs = b.as_ptr().add(16);
        let mask = _mm_set1_epi8(0x0f);
        let yp = y.as_mut_ptr();
        let mut is = 0usize;
        for j in (0..256).step_by(64) {
            let (sc, m) = super::scale_min_k4(is, scales);
            let d1 = _mm256_set1_ps(d * sc as f32);
            let m1 = _mm256_set1_ps(dmin * m as f32);
            let (sc, m) = super::scale_min_k4(is + 1, scales);
            let d2 = _mm256_set1_ps(d * sc as f32);
            let m2 = _mm256_set1_ps(dmin * m as f32);
            let q0 = _mm_loadu_si128(qs.add(j / 2) as *const __m128i);
            let q1 = _mm_loadu_si128(qs.add(j / 2 + 16) as *const __m128i);
            let lo0 = _mm_and_si128(q0, mask);
            let lo1 = _mm_and_si128(q1, mask);
            let hi0 = _mm_and_si128(_mm_srli_epi16(q0, 4), mask);
            let hi1 = _mm_and_si128(_mm_srli_epi16(q1, 4), mask);
            let o = yp.add(j);
            _mm256_storeu_ps(o, _mm256_sub_ps(_mm256_mul_ps(d1, u8x8_to_ps(lo0)), m1));
            _mm256_storeu_ps(o.add(8), _mm256_sub_ps(_mm256_mul_ps(d1, u8x8_to_ps(_mm_srli_si128(lo0, 8))), m1));
            _mm256_storeu_ps(o.add(16), _mm256_sub_ps(_mm256_mul_ps(d1, u8x8_to_ps(lo1)), m1));
            _mm256_storeu_ps(o.add(24), _mm256_sub_ps(_mm256_mul_ps(d1, u8x8_to_ps(_mm_srli_si128(lo1, 8))), m1));
            _mm256_storeu_ps(o.add(32), _mm256_sub_ps(_mm256_mul_ps(d2, u8x8_to_ps(hi0)), m2));
            _mm256_storeu_ps(o.add(40), _mm256_sub_ps(_mm256_mul_ps(d2, u8x8_to_ps(_mm_srli_si128(hi0, 8))), m2));
            _mm256_storeu_ps(o.add(48), _mm256_sub_ps(_mm256_mul_ps(d2, u8x8_to_ps(hi1)), m2));
            _mm256_storeu_ps(o.add(56), _mm256_sub_ps(_mm256_mul_ps(d2, u8x8_to_ps(_mm_srli_si128(hi1, 8))), m2));
            is += 2;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn q6_k(b: &[u8], y: &mut [f32]) {
        let d = super::f16_at(b, 208);
        let mask = _mm_set1_epi8(0x0f);
        let m3 = _mm_set1_epi8(3);
        let off = _mm256_set1_ps(32.0);
        for n in 0..2 {
            let ql = b.as_ptr().add(n * 64);
            let qh = b.as_ptr().add(128 + n * 32);
            let sc = &b[192 + n * 8..192 + n * 8 + 8];
            let yp = y.as_mut_ptr().add(n * 128);
            for half in 0..2 {
                let l0 = half * 16;
                let is = l0 / 16;
                let qlv = _mm_loadu_si128(ql.add(l0) as *const __m128i);
                let qlv32 = _mm_loadu_si128(ql.add(l0 + 32) as *const __m128i);
                let qhv = _mm_loadu_si128(qh.add(l0) as *const __m128i);
                let q1 = _mm_or_si128(_mm_and_si128(qlv, mask), _mm_slli_epi16(_mm_and_si128(qhv, m3), 4));
                let q2 = _mm_or_si128(_mm_and_si128(qlv32, mask), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 2), m3), 4));
                let q3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(qlv, 4), mask), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 4), m3), 4));
                let q4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(qlv32, 4), mask), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qhv, 6), m3), 4));
                let s1 = _mm256_set1_ps(d * sc[is] as i8 as f32);
                let s2 = _mm256_set1_ps(d * sc[is + 2] as i8 as f32);
                let s3 = _mm256_set1_ps(d * sc[is + 4] as i8 as f32);
                let s4 = _mm256_set1_ps(d * sc[is + 6] as i8 as f32);
                let st = |o: *mut f32, q: __m128i, s: __m256| {
                    _mm256_storeu_ps(o, _mm256_mul_ps(s, _mm256_sub_ps(u8x8_to_ps(q), off)));
                    _mm256_storeu_ps(o.add(8), _mm256_mul_ps(s, _mm256_sub_ps(u8x8_to_ps(_mm_srli_si128(q, 8)), off)));
                };
                st(yp.add(l0), q1, s1);
                st(yp.add(l0 + 32), q2, s2);
                st(yp.add(l0 + 64), q3, s3);
                st(yp.add(l0 + 96), q4, s4);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::*;

    #[inline]
    unsafe fn u8x8_to_f32(v: uint8x8_t) -> (float32x4_t, float32x4_t) {
        let w = vmovl_u8(v);
        (vcvtq_f32_u32(vmovl_u16(vget_low_u16(w))), vcvtq_f32_u32(vmovl_u16(vget_high_u16(w))))
    }

    #[inline]
    unsafe fn i8x8_to_f32(v: int8x8_t) -> (float32x4_t, float32x4_t) {
        let w = vmovl_s8(v);
        (vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w))))
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn q8_0(b: &[u8], y: &mut [f32]) {
        let d = vdupq_n_f32(super::f16_at(b, 0));
        let p = b.as_ptr().add(2) as *const i8;
        for j in 0..4 {
            let (a, c) = i8x8_to_f32(vld1_s8(p.add(j * 8)));
            vst1q_f32(y.as_mut_ptr().add(j * 8), vmulq_f32(a, d));
            vst1q_f32(y.as_mut_ptr().add(j * 8 + 4), vmulq_f32(c, d));
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn q4_0(b: &[u8], y: &mut [f32]) {
        let d = vdupq_n_f32(super::f16_at(b, 0));
        let qs = vld1q_u8(b.as_ptr().add(2));
        let lo = vandq_u8(qs, vdupq_n_u8(0x0f));
        let hi = vshrq_n_u8(qs, 4);
        let eight = vdupq_n_f32(8.0);
        let yp = y.as_mut_ptr();
        let mut st = |o: usize, v: uint8x8_t| {
            let (a, c) = u8x8_to_f32(v);
            vst1q_f32(yp.add(o), vmulq_f32(vsubq_f32(a, eight), d));
            vst1q_f32(yp.add(o + 4), vmulq_f32(vsubq_f32(c, eight), d));
        };
        st(0, vget_low_u8(lo));
        st(8, vget_high_u8(lo));
        st(16, vget_low_u8(hi));
        st(24, vget_high_u8(hi));
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn q4_k(b: &[u8], y: &mut [f32]) {
        let d = super::f16_at(b, 0);
        let dmin = super::f16_at(b, 2);
        let scales = &b[4..16];
        let qs = b.as_ptr().add(16);
        let mask = vdupq_n_u8(0x0f);
        let yp = y.as_mut_ptr();
        let mut is = 0usize;
        for j in (0..256).step_by(64) {
            let (sc, m) = super::scale_min_k4(is, scales);
            let d1 = vdupq_n_f32(d * sc as f32);
            let m1 = vdupq_n_f32(dmin * m as f32);
            let (sc, m) = super::scale_min_k4(is + 1, scales);
            let d2 = vdupq_n_f32(d * sc as f32);
            let m2 = vdupq_n_f32(dmin * m as f32);
            let q0 = vld1q_u8(qs.add(j / 2));
            let q1 = vld1q_u8(qs.add(j / 2 + 16));
            let mut st = |o: usize, v: uint8x8_t, dd: float32x4_t, mm: float32x4_t| {
                let (a, c) = u8x8_to_f32(v);
                vst1q_f32(yp.add(o), vsubq_f32(vmulq_f32(dd, a), mm));
                vst1q_f32(yp.add(o + 4), vsubq_f32(vmulq_f32(dd, c), mm));
            };
            let lo0 = vandq_u8(q0, mask);
            let lo1 = vandq_u8(q1, mask);
            let hi0 = vshrq_n_u8(q0, 4);
            let hi1 = vshrq_n_u8(q1, 4);
            st(j, vget_low_u8(lo0), d1, m1);
            st(j + 8, vget_high_u8(lo0), d1, m1);
            st(j + 16, vget_low_u8(lo1), d1, m1);
            st(j + 24, vget_high_u8(lo1), d1, m1);
            st(j + 32, vget_low_u8(hi0), d2, m2);
            st(j + 40, vget_high_u8(hi0), d2, m2);
            st(j + 48, vget_low_u8(hi1), d2, m2);
            st(j + 56, vget_high_u8(hi1), d2, m2);
            is += 2;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn q6_k(b: &[u8], y: &mut [f32]) {
        let d = super::f16_at(b, 208);
        let mask = vdupq_n_u8(0x0f);
        let m3 = vdupq_n_u8(3);
        let off = vdupq_n_f32(32.0);
        for n in 0..2 {
            let ql = b.as_ptr().add(n * 64);
            let qh = b.as_ptr().add(128 + n * 32);
            let sc = &b[192 + n * 8..192 + n * 8 + 8];
            let yp = y.as_mut_ptr().add(n * 128);
            for half in 0..2 {
                let l0 = half * 16;
                let is = l0 / 16;
                let qlv = vld1q_u8(ql.add(l0));
                let qlv32 = vld1q_u8(ql.add(l0 + 32));
                let qhv = vld1q_u8(qh.add(l0));
                let q1 = vorrq_u8(vandq_u8(qlv, mask), vshlq_n_u8(vandq_u8(qhv, m3), 4));
                let q2 = vorrq_u8(vandq_u8(qlv32, mask), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 2), m3), 4));
                let q3 = vorrq_u8(vshrq_n_u8(qlv, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 4), m3), 4));
                let q4 = vorrq_u8(vshrq_n_u8(qlv32, 4), vshlq_n_u8(vshrq_n_u8(qhv, 6), 4));
                let s1 = vdupq_n_f32(d * sc[is] as i8 as f32);
                let s2 = vdupq_n_f32(d * sc[is + 2] as i8 as f32);
                let s3 = vdupq_n_f32(d * sc[is + 4] as i8 as f32);
                let s4 = vdupq_n_f32(d * sc[is + 6] as i8 as f32);
                let mut st = |o: usize, q: uint8x16_t, s: float32x4_t| {
                    let (a, c) = u8x8_to_f32(vget_low_u8(q));
                    vst1q_f32(yp.add(o), vmulq_f32(s, vsubq_f32(a, off)));
                    vst1q_f32(yp.add(o + 4), vmulq_f32(s, vsubq_f32(c, off)));
                    let (a, c) = u8x8_to_f32(vget_high_u8(q));
                    vst1q_f32(yp.add(o + 8), vmulq_f32(s, vsubq_f32(a, off)));
                    vst1q_f32(yp.add(o + 12), vmulq_f32(s, vsubq_f32(c, off)));
                };
                st(l0, q1, s1);
                st(l0 + 32, q2, s2);
                st(l0 + 64, q3, s3);
                st(l0 + 96, q4, s4);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd_bytes(n: usize, seed: &mut u64) -> Vec<u8> {
        (0..n)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                (*seed & 0xff) as u8
            })
            .collect()
    }

    fn check(t: GgmlType, scale_offsets: &[usize]) {
        let mut seed = 7u64;
        let k = t.block_elems() * 5;
        let mut src = rnd_bytes(t.bytes_for(k), &mut seed);
        let bb = t.block_bytes();
        for ib in 0..5 {
            for &o in scale_offsets {
                let v = half::f16::from_f32(0.01 * (ib as f32 + 1.0)).to_le_bytes();
                src[ib * bb + o] = v[0];
                src[ib * bb + o + 1] = v[1];
            }
        }
        let mut want = vec![0f32; k];
        ggml_dequant::dequantize(t, &src, k, &mut want).unwrap();
        let mut got = vec![0f32; k];
        dequant_row(t, &src, k, &mut got);
        for i in 0..k {
            assert!((got[i] - want[i]).abs() <= 1e-6 * (1.0 + want[i].abs()), "{t:?} [{i}] {} vs {}", got[i], want[i]);
        }
    }

    #[test]
    fn simd_matches_reference() {
        check(GgmlType::Q8_0, &[0]);
        check(GgmlType::Q4_0, &[0]);
        check(GgmlType::Q4K, &[0, 2]);
        check(GgmlType::Q6K, &[208]);
    }
}
