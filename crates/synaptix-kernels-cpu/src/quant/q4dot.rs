//! Q4_0 (ggml: блок 32 = f16 `d` + 16 байт полубайтов; младшие — значения 0..15, старшие — 16..31,
//! вес `d·(q − 8)`) × активации int8 блоками по 32 (как [`super::q8dot`]): целочисленное
//! скалярное произведение (NEON dotprod), шкалы — в f32. Вес читается вдвое меньше, чем у Q8Row.

use super::q8dot::QK;

pub const BLOCK_BYTES: usize = 18;

#[inline]
fn d_of(b: &[u8]) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
}

fn dot_scalar(xq: &[i8], xs: &[f32], w: &[u8]) -> f32 {
    let nb = xq.len() / QK;
    let mut acc = 0f32;
    for b in 0..nb {
        let blk = &w[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        let qs = &blk[2..];
        let x = &xq[b * QK..(b + 1) * QK];
        let mut s = 0i32;
        for j in 0..16 {
            s += ((qs[j] & 0x0F) as i32 - 8) * x[j] as i32;
            s += ((qs[j] >> 4) as i32 - 8) * x[j + 16] as i32;
        }
        acc += s as f32 * d_of(blk) * xs[b];
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn sdot(acc: std::arch::aarch64::int32x4_t, a: std::arch::aarch64::int8x16_t, b: std::arch::aarch64::int8x16_t) -> std::arch::aarch64::int32x4_t {
    let mut acc = acc;
    std::arch::asm!("sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b, options(pure, nomem, nostack));
    acc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_dotprod(xq: &[i8], xs: &[f32], w: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    let nb = xq.len() / QK;
    let xp = xq.as_ptr();
    let wp = w.as_ptr();
    let m4 = vdupq_n_u8(0x0F);
    let s8 = vdupq_n_s8(8);
    let z = vdupq_n_s32(0);
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut b = 0;
    while b + 1 < nb {
        let p0 = wp.add(b * BLOCK_BYTES);
        let p1 = wp.add((b + 1) * BLOCK_BYTES);
        let v0 = vld1q_u8(p0.add(2));
        let v1 = vld1q_u8(p1.add(2));
        let l0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4)), s8);
        let h0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), s8);
        let l1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, m4)), s8);
        let h1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)), s8);
        let o = b * QK;
        let s0 = sdot(sdot(z, l0, vld1q_s8(xp.add(o))), h0, vld1q_s8(xp.add(o + 16)));
        let s1 = sdot(sdot(z, l1, vld1q_s8(xp.add(o + QK))), h1, vld1q_s8(xp.add(o + QK + 16)));
        let d0 = d_of(std::slice::from_raw_parts(p0, 2)) * *xs.get_unchecked(b);
        let d1 = d_of(std::slice::from_raw_parts(p1, 2)) * *xs.get_unchecked(b + 1);
        acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(s0), d0);
        acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(s1), d1);
        b += 2;
    }
    if b < nb {
        let p0 = wp.add(b * BLOCK_BYTES);
        let v0 = vld1q_u8(p0.add(2));
        let l0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4)), s8);
        let h0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), s8);
        let o = b * QK;
        let s0 = sdot(sdot(z, l0, vld1q_s8(xp.add(o))), h0, vld1q_s8(xp.add(o + 16)));
        acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(s0), d_of(std::slice::from_raw_parts(p0, 2)) * *xs.get_unchecked(b));
    }
    vaddvq_f32(vaddq_f32(acc0, acc1))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(xq: &[i8], xs: &[f32], w: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let nb = xq.len() / QK;
    let ones = _mm256_set1_epi16(1);
    let m4 = _mm_set1_epi8(0x0F);
    let s8 = _mm256_set1_epi8(8);
    let mut acc = _mm256_setzero_ps();
    for b in 0..nb {
        let p = w.as_ptr().add(b * BLOCK_BYTES);
        let v = _mm_loadu_si128(p.add(2) as *const __m128i);
        let lo = _mm_and_si128(v, m4);
        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), m4);
        let q = _mm256_sub_epi8(_mm256_set_m128i(hi, lo), s8);
        let x = _mm256_loadu_si256(xq.as_ptr().add(b * QK) as *const __m256i);
        let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(q, x));
        let f = _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones));
        acc = _mm256_fmadd_ps(f, _mm256_set1_ps(d_of(std::slice::from_raw_parts(p, 2)) * xs[b]), acc);
    }
    let s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(_mm_hadd_ps(s, s))
}

/// Строка весов Q4_0 (`k/32` блоков) · активации int8 (`xq`, шкалы `xs` по 32).
pub fn dot(xq: &[i8], xs: &[f32], w: &[u8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { dot_dotprod(xq, xs, w) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(xq, xs, w) };
        }
    }
    dot_scalar(xq, xs, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar_and_dequant() {
        let k = 256;
        let nb = k / QK;
        let mut w = vec![0u8; nb * BLOCK_BYTES];
        for b in 0..nb {
            let d = half::f16::from_f32(0.01 * (b as f32 + 1.0)).to_bits().to_le_bytes();
            w[b * BLOCK_BYTES] = d[0];
            w[b * BLOCK_BYTES + 1] = d[1];
            for j in 0..16 {
                w[b * BLOCK_BYTES + 2 + j] = ((j * 7 + b) % 256) as u8;
            }
        }
        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.31).sin()).collect();
        let mut xq = vec![0i8; k];
        let mut xs = vec![0f32; nb];
        super::super::q8dot::quantize_act(&x, &mut xq, &mut xs);
        let a = dot(&xq, &xs, &w);
        let b = dot_scalar(&xq, &xs, &w);
        assert!((a - b).abs() <= 1e-3 * b.abs().max(1.0), "{a} vs {b}");
        // против деквантованной строки (ошибка — только квант активаций)
        let mut row = vec![0f32; k];
        crate::quant::ggml_simd::dequant_row(synaptix_core::quant::ggml::GgmlType::Q4_0, &w, k, &mut row);
        let exact: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((a - exact).abs() <= 0.02 * exact.abs().max(1.0), "{a} vs {exact}");
    }
}
