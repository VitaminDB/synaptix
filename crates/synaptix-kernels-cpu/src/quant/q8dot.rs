pub const QK: usize = 32;

pub fn quantize_act(x: &[f32], q: &mut [i8], s: &mut [f32]) {
    for (b, (xc, qc)) in x.chunks_exact(QK).zip(q.chunks_exact_mut(QK)).enumerate() {
        let amax = xc.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        for (o, &v) in qc.iter_mut().zip(xc) {
            *o = (v * inv).round() as i8;
        }
        s[b] = d;
    }
}

#[inline]
fn dot_scalar(xq: &[i8], xs: &[f32], w: &[u8]) -> f32 {
    let mut acc = 0f32;
    for (b, (xc, wc)) in xq.chunks_exact(QK).zip(w.chunks_exact(QK)).enumerate() {
        let mut s = 0i32;
        for (a, &c) in xc.iter().zip(wc) {
            s += *a as i32 * (c as i8) as i32;
        }
        acc += s as f32 * xs[b];
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
    let wp = w.as_ptr() as *const i8;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let z = vdupq_n_s32(0);
    let mut b = 0;
    while b + 1 < nb {
        let o = b * QK;
        let s0 = sdot(sdot(z, vld1q_s8(xp.add(o)), vld1q_s8(wp.add(o))), vld1q_s8(xp.add(o + 16)), vld1q_s8(wp.add(o + 16)));
        let o1 = o + QK;
        let s1 = sdot(sdot(z, vld1q_s8(xp.add(o1)), vld1q_s8(wp.add(o1))), vld1q_s8(xp.add(o1 + 16)), vld1q_s8(wp.add(o1 + 16)));
        acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(s0), *xs.get_unchecked(b));
        acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(s1), *xs.get_unchecked(b + 1));
        b += 2;
    }
    if b < nb {
        let o = b * QK;
        let s0 = sdot(sdot(z, vld1q_s8(xp.add(o)), vld1q_s8(wp.add(o))), vld1q_s8(xp.add(o + 16)), vld1q_s8(wp.add(o + 16)));
        acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(s0), *xs.get_unchecked(b));
    }
    vaddvq_f32(vaddq_f32(acc0, acc1))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(xq: &[i8], xs: &[f32], w: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let nb = xq.len() / QK;
    let mut acc = _mm256_setzero_ps();
    let ones = _mm256_set1_epi16(1);
    for b in 0..nb {
        let o = b * QK;
        let x = _mm256_loadu_si256(xq.as_ptr().add(o) as *const __m256i);
        let wv = _mm256_loadu_si256(w.as_ptr().add(o) as *const __m256i);
        let ax = _mm256_sign_epi8(x, x);
        let sw = _mm256_sign_epi8(wv, x);
        let p16 = _mm256_maddubs_epi16(ax, sw);
        let p32 = _mm256_madd_epi16(p16, ones);
        let f = _mm256_cvtepi32_ps(p32);
        acc = _mm256_fmadd_ps(f, _mm256_set1_ps(xs[b]), acc);
    }
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

fn dot4_scalar(xq: &[i8], xs: &[f32], k: usize, w: &[u8]) -> [f32; 4] {
    let nb = k / QK;
    let mut out = [0f32; 4];
    for (r, o) in out.iter_mut().enumerate() {
        *o = dot_scalar(&xq[r * k..(r + 1) * k], &xs[r * nb..(r + 1) * nb], w);
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot4_dotprod(xq: &[i8], xs: &[f32], k: usize, w: &[u8]) -> [f32; 4] {
    use std::arch::aarch64::*;
    let nb = k / QK;
    let xp = xq.as_ptr();
    let wp = w.as_ptr() as *const i8;
    let sp = xs.as_ptr();
    let z = vdupq_n_s32(0);
    let (mut a0, mut a1, mut a2, mut a3) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    for b in 0..nb {
        let o = b * QK;
        let w0 = vld1q_s8(wp.add(o));
        let w1 = vld1q_s8(wp.add(o + 16));
        let x0 = xp.add(o);
        let x1 = xp.add(k + o);
        let x2 = xp.add(2 * k + o);
        let x3 = xp.add(3 * k + o);
        let s0 = sdot(sdot(z, vld1q_s8(x0), w0), vld1q_s8(x0.add(16)), w1);
        let s1 = sdot(sdot(z, vld1q_s8(x1), w0), vld1q_s8(x1.add(16)), w1);
        let s2 = sdot(sdot(z, vld1q_s8(x2), w0), vld1q_s8(x2.add(16)), w1);
        let s3 = sdot(sdot(z, vld1q_s8(x3), w0), vld1q_s8(x3.add(16)), w1);
        a0 = vfmaq_n_f32(a0, vcvtq_f32_s32(s0), *sp.add(b));
        a1 = vfmaq_n_f32(a1, vcvtq_f32_s32(s1), *sp.add(nb + b));
        a2 = vfmaq_n_f32(a2, vcvtq_f32_s32(s2), *sp.add(2 * nb + b));
        a3 = vfmaq_n_f32(a3, vcvtq_f32_s32(s3), *sp.add(3 * nb + b));
    }
    [vaddvq_f32(a0), vaddvq_f32(a1), vaddvq_f32(a2), vaddvq_f32(a3)]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot4_avx2(xq: &[i8], xs: &[f32], k: usize, w: &[u8]) -> [f32; 4] {
    use std::arch::x86_64::*;
    let nb = k / QK;
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_ps(); 4];
    for b in 0..nb {
        let o = b * QK;
        let wv = _mm256_loadu_si256(w.as_ptr().add(o) as *const __m256i);
        for (r, a) in acc.iter_mut().enumerate() {
            let x = _mm256_loadu_si256(xq.as_ptr().add(r * k + o) as *const __m256i);
            let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(wv, x));
            let f = _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones));
            *a = _mm256_fmadd_ps(f, _mm256_set1_ps(*xs.get_unchecked(r * nb + b)), *a);
        }
    }
    let mut out = [0f32; 4];
    for (o, a) in out.iter_mut().zip(acc) {
        let s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        let s = _mm_hadd_ps(s, s);
        *o = _mm_cvtss_f32(_mm_hadd_ps(s, s));
    }
    out
}

pub fn dot4(xq: &[i8], xs: &[f32], k: usize, w: &[u8]) -> [f32; 4] {
    debug_assert!(xq.len() >= 4 * k && xs.len() >= 4 * (k / QK) && w.len() >= k);
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { dot4_dotprod(xq, xs, k, w) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return unsafe { dot4_avx2(xq, xs, k, w) };
        }
    }
    dot4_scalar(xq, xs, k, w)
}

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
    fn dot4_matches_dot() {
        let k = 512;
        let x: Vec<f32> = (0..4 * k).map(|i| ((i as f32) * 0.173).sin() * (1.0 + (i / k) as f32)).collect();
        let w: Vec<u8> = (0..k).map(|i| (((i * 53) % 255) as i32 - 127) as i8 as u8).collect();
        let mut xq = vec![0i8; 4 * k];
        let mut xs = vec![0f32; 4 * k / QK];
        for r in 0..4 {
            quantize_act(&x[r * k..(r + 1) * k], &mut xq[r * k..(r + 1) * k], &mut xs[r * k / QK..(r + 1) * k / QK]);
        }
        let got = dot4(&xq, &xs, k, &w);
        for r in 0..4 {
            let want = dot(&xq[r * k..(r + 1) * k], &xs[r * k / QK..(r + 1) * k / QK], &w);
            assert!((got[r] - want).abs() < 1e-3 * want.abs().max(1.0), "{r}: {} {want}", got[r]);
        }
    }

    #[test]
    fn matches_scalar_and_float() {
        let k = 256;
        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.37).sin()).collect();
        let w: Vec<u8> = (0..k).map(|i| (((i * 37) % 255) as i32 - 127) as i8 as u8).collect();
        let mut xq = vec![0i8; k];
        let mut xs = vec![0f32; k / QK];
        quantize_act(&x, &mut xq, &mut xs);
        let a = dot(&xq, &xs, &w);
        let b = dot_scalar(&xq, &xs, &w);
        let f: f32 = x.iter().zip(&w).map(|(a, &b)| a * (b as i8) as f32).sum();
        assert!((a - b).abs() < 1e-2 * b.abs().max(1.0), "{a} {b} {f}");
        let mag: f32 = x.iter().zip(&w).map(|(a, &b)| (a * (b as i8) as f32).abs()).sum();
        assert!((a - f).abs() < 1e-2 * mag, "{a} {f} {mag}");
    }
}
