//! SIMD dot-product для linear-пути: `Σ_k x[k]·w[k]` (оба операнда непрерывны).
//! AVX2+FMA с 4 независимыми аккумуляторами (ILP) + хвост скаляром. F16 через F16C
//! (`vcvtph2ps`), BF16 через widening-shift (bf16→f32 = `<<16`). Горизонтальная
//! сумма в конце меняет порядок суммирования относительно скалярного цикла — для
//! inference допустимо (bit-exact не требуется на linear-пути, нет ref-теста).

#[inline]
pub fn scalar_dot_f32(x: &[f32], w: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..x.len() {
        acc += x[i] * w[i];
    }
    acc
}

#[inline]
pub fn scalar_dot_f16(x: &[half::f16], w: &[half::f16]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..x.len() {
        acc += x[i].to_f32() * w[i].to_f32();
    }
    acc
}

#[inline]
pub fn scalar_dot_bf16(x: &[half::bf16], w: &[half::bf16]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..x.len() {
        acc += x[i].to_f32() * w[i].to_f32();
    }
    acc
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn hsum256(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(s)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32(x: &[f32], w: &[f32]) -> f32 {
        let k = x.len();
        let xp = x.as_ptr();
        let wp = w.as_ptr();
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 32 <= k {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i)), _mm256_loadu_ps(wp.add(i)), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i + 8)), _mm256_loadu_ps(wp.add(i + 8)), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i + 16)), _mm256_loadu_ps(wp.add(i + 16)), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i + 24)), _mm256_loadu_ps(wp.add(i + 24)), a3);
            i += 32;
        }
        while i + 8 <= k {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i)), _mm256_loadu_ps(wp.add(i)), a0);
            i += 8;
        }
        let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let mut s = hsum256(acc);
        while i < k {
            s += *xp.add(i) * *wp.add(i);
            i += 1;
        }
        s
    }

    #[inline]
    unsafe fn load8_bf16_to_f32(p: *const u16) -> __m256 {
        // bf16→f32: расширить 16→32 бит и сдвинуть в старшую половину (мантисса/экспонента).
        let raw = _mm_loadu_si128(p as *const __m128i);
        let widened = _mm256_cvtepu16_epi32(raw);
        _mm256_castsi256_ps(_mm256_slli_epi32(widened, 16))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_bf16(x: &[half::bf16], w: &[half::bf16]) -> f32 {
        let k = x.len();
        let xp = x.as_ptr() as *const u16;
        let wp = w.as_ptr() as *const u16;
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= k {
            a0 = _mm256_fmadd_ps(load8_bf16_to_f32(xp.add(i)), load8_bf16_to_f32(wp.add(i)), a0);
            a1 = _mm256_fmadd_ps(load8_bf16_to_f32(xp.add(i + 8)), load8_bf16_to_f32(wp.add(i + 8)), a1);
            i += 16;
        }
        while i + 8 <= k {
            a0 = _mm256_fmadd_ps(load8_bf16_to_f32(xp.add(i)), load8_bf16_to_f32(wp.add(i)), a0);
            i += 8;
        }
        let mut s = hsum256(_mm256_add_ps(a0, a1));
        while i < k {
            s += x[i].to_f32() * w[i].to_f32();
            i += 1;
        }
        s
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f16(x: &[half::f16], w: &[half::f16]) -> f32 {
        let k = x.len();
        let xp = x.as_ptr() as *const __m128i;
        let wp = w.as_ptr() as *const __m128i;
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= k {
            let xv0 = _mm256_cvtph_ps(_mm_loadu_si128(xp.add(i / 8)));
            let wv0 = _mm256_cvtph_ps(_mm_loadu_si128(wp.add(i / 8)));
            a0 = _mm256_fmadd_ps(xv0, wv0, a0);
            let xv1 = _mm256_cvtph_ps(_mm_loadu_si128(xp.add(i / 8 + 1)));
            let wv1 = _mm256_cvtph_ps(_mm_loadu_si128(wp.add(i / 8 + 1)));
            a1 = _mm256_fmadd_ps(xv1, wv1, a1);
            i += 16;
        }
        while i + 8 <= k {
            let xv = _mm256_cvtph_ps(_mm_loadu_si128(xp.add(i / 8)));
            let wv = _mm256_cvtph_ps(_mm_loadu_si128(wp.add(i / 8)));
            a0 = _mm256_fmadd_ps(xv, wv, a0);
            i += 8;
        }
        let mut s = hsum256(_mm256_add_ps(a0, a1));
        while i < k {
            s += x[i].to_f32() * w[i].to_f32();
            i += 1;
        }
        s
    }
}

#[inline]
pub fn dot_f32(x: &[f32], w: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { x86::dot_f32(x, w) };
        }
    }
    scalar_dot_f32(x, w)
}

#[inline]
pub fn dot_bf16(x: &[half::bf16], w: &[half::bf16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { x86::dot_bf16(x, w) };
        }
    }
    scalar_dot_bf16(x, w)
}

#[inline]
pub fn dot_f16(x: &[half::f16], w: &[half::f16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
        {
            return unsafe { x86::dot_f16(x, w) };
        }
    }
    scalar_dot_f16(x, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_dot_matches_scalar() {
        let k = 2053; // нечётный → проверяет хвост
        let xf: Vec<f32> = (0..k).map(|i| (i as f32 * 0.013).sin()).collect();
        let wf: Vec<f32> = (0..k).map(|i| (i as f32 * 0.027).cos()).collect();
        let s = scalar_dot_f32(&xf, &wf);
        let v = dot_f32(&xf, &wf);
        assert!((s - v).abs() < 1e-2, "f32 simd {v} vs scalar {s}");

        let xb: Vec<half::bf16> = xf.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let wb: Vec<half::bf16> = wf.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let sb = scalar_dot_bf16(&xb, &wb);
        let vb = dot_bf16(&xb, &wb);
        assert!((sb - vb).abs() < 0.5, "bf16 simd {vb} vs scalar {sb}");

        let xh: Vec<half::f16> = xf.iter().map(|&v| half::f16::from_f32(v)).collect();
        let wh: Vec<half::f16> = wf.iter().map(|&v| half::f16::from_f32(v)).collect();
        let sh = scalar_dot_f16(&xh, &wh);
        let vh = dot_f16(&xh, &wh);
        assert!((sh - vh).abs() < 0.1, "f16 simd {vh} vs scalar {sh}");
    }
}
