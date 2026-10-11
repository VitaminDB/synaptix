use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::layout::Layout;
use synaptix_core::tensor::storage::{CpuBuf, Storage};

pub fn cpu<'a>(st: &'a Storage, what: &'static str) -> Result<&'a CpuBuf> {
    st.as_cpu().ok_or(SynaptixError::Unsupported(what))
}

pub fn cpu_mut<'a>(st: &'a mut Storage, what: &'static str) -> Result<&'a mut CpuBuf> {
    st.as_cpu_mut().ok_or(SynaptixError::Unsupported(what))
}

pub fn threads() -> usize {
    crate::threads::pool();
    rayon::current_num_threads().max(1)
}

pub fn par_chunks(n: usize, min_chunk: usize, f: impl Fn(usize, usize) + Sync) {
    let t = threads();
    if n == 0 {
        return;
    }
    let chunk = n.div_ceil(t).max(min_chunk).max(1);
    if n <= chunk {
        f(0, n);
        return;
    }
    use rayon::prelude::*;
    (0..n.div_ceil(chunk)).into_par_iter().for_each(|i| {
        let a = i * chunk;
        f(a, (a + chunk).min(n));
    });
}

pub fn par_chunks_cost(n: usize, ns_per_item: usize, f: impl Fn(usize, usize) + Sync) {
    let min_chunk = (60_000 / ns_per_item.max(1)).max(1);
    par_chunks(n, min_chunk, f)
}

#[inline]
pub fn is_half(dt: DType) -> bool {
    matches!(dt, DType::F16 | DType::BF16)
}

#[inline]
pub fn elem_bytes(dt: DType) -> usize {
    dt.size_in_bits() / 8
}

pub fn to_f32(dt: DType, bytes: &[u8], off: usize, dst: &mut [f32]) -> Result<()> {
    let n = dst.len();
    match dt {
        DType::F32 => dst.copy_from_slice(bytemuck::cast_slice(&bytes[off * 4..(off + n) * 4])),
        DType::F16 => f16_to_f32(bytemuck::cast_slice(&bytes[off * 2..(off + n) * 2]), dst),
        DType::BF16 => bf16_to_f32(bytemuck::cast_slice(&bytes[off * 2..(off + n) * 2]), dst),
        DType::F64 => {
            let s: &[f64] = bytemuck::cast_slice(&bytes[off * 8..(off + n) * 8]);
            for (d, &v) in dst.iter_mut().zip(s) {
                *d = v as f32;
            }
        }
        _ => return Err(SynaptixError::Unsupported("to_f32: dtype")),
    }
    Ok(())
}

pub fn from_f32(dt: DType, src: &[f32], bytes: &mut [u8], off: usize) -> Result<()> {
    let n = src.len();
    match dt {
        DType::F32 => bytes[off * 4..(off + n) * 4].copy_from_slice(bytemuck::cast_slice(src)),
        DType::F16 => f32_to_f16(src, bytemuck::cast_slice_mut(&mut bytes[off * 2..(off + n) * 2])),
        DType::BF16 => f32_to_bf16(src, bytemuck::cast_slice_mut(&mut bytes[off * 2..(off + n) * 2])),
        DType::F64 => {
            let d: &mut [f64] = bytemuck::cast_slice_mut(&mut bytes[off * 8..(off + n) * 8]);
            for (d, &v) in d.iter_mut().zip(src) {
                *d = v as f64;
            }
        }
        _ => return Err(SynaptixError::Unsupported("from_f32: dtype")),
    }
    Ok(())
}

#[inline]
pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[inline]
pub fn f32_to_bf16_bits(v: f32) -> u16 {
    let u = v.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        return ((u >> 16) | 0x40) as u16;
    }
    let round = 0x7fff + ((u >> 16) & 1);
    ((u.wrapping_add(round)) >> 16) as u16
}

pub fn f16_to_f32(src: &[u16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().f16c {
            unsafe { x86::f16_to_f32(src, dst) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { arm::f16_to_f32(src, dst) };
        return;
    }
    #[allow(unreachable_code)]
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = half::f16::from_bits(s).to_f32();
    }
}

pub fn f32_to_f16(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().f16c {
            unsafe { x86::f32_to_f16(src, dst) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { arm::f32_to_f16(src, dst) };
        return;
    }
    #[allow(unreachable_code)]
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = half::f16::from_f32(s).to_bits();
    }
}

pub fn bf16_to_f32(src: &[u16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx2 {
            unsafe { x86::bf16_to_f32(src, dst) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { arm::bf16_to_f32(src, dst) };
        return;
    }
    #[allow(unreachable_code)]
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = bf16_bits_to_f32(s);
    }
}

pub fn f32_to_bf16(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx2 {
            unsafe { x86::f32_to_bf16(src, dst) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { arm::f32_to_bf16(src, dst) };
        return;
    }
    #[allow(unreachable_code)]
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = f32_to_bf16_bits(s);
    }
}

pub fn row_to_f32(dt: DType, bytes: &[u8], off: usize, dst: &mut [f32]) {
    to_f32(dt, bytes, off, dst).expect("row_to_f32: dtype")
}

pub fn row_from_f32(dt: DType, src: &[f32], bytes: &mut [u8], off: usize) {
    from_f32(dt, src, bytes, off).expect("row_from_f32: dtype")
}

pub struct RowBuf {
    buf: Vec<f32>,
}

impl RowBuf {
    pub fn new(n: usize) -> Self {
        Self { buf: vec![0.0; n] }
    }
    pub fn load(&mut self, dt: DType, bytes: &[u8], off: usize) -> &[f32] {
        row_to_f32(dt, bytes, off, &mut self.buf);
        &self.buf
    }
}

pub fn check_contig(lo: &Layout) -> Result<()> {
    if lo.is_contiguous() {
        Ok(())
    } else {
        Err(SynaptixError::NonContiguous)
    }
}

pub fn same_dtype(a: DType, b: DType) -> Result<()> {
    if a == b {
        Ok(())
    } else {
        Err(SynaptixError::dtype_mismatch(a, b))
    }
}

#[inline]
pub fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn gelu_tanh_f32(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,f16c")]
    pub unsafe fn f16_to_f32(src: &[u16], dst: &mut [f32]) {
        let n = src.len();
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        while i + 8 <= n {
            let h = _mm_loadu_si128(sp.add(i) as *const __m128i);
            _mm256_storeu_ps(dp.add(i), _mm256_cvtph_ps(h));
            i += 8;
        }
        while i < n {
            *dp.add(i) = half::f16::from_bits(*sp.add(i)).to_f32();
            i += 1;
        }
    }

    #[target_feature(enable = "avx2,f16c")]
    pub unsafe fn f32_to_f16(src: &[f32], dst: &mut [u16]) {
        let n = src.len();
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(sp.add(i));
            _mm_storeu_si128(dp.add(i) as *mut __m128i, _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
            i += 8;
        }
        while i < n {
            *dp.add(i) = half::f16::from_f32(*sp.add(i)).to_bits();
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn bf16_to_f32(src: &[u16], dst: &mut [f32]) {
        let n = src.len();
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        while i + 8 <= n {
            let h = _mm_loadu_si128(sp.add(i) as *const __m128i);
            let w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
            _mm256_storeu_ps(dp.add(i), _mm256_castsi256_ps(w));
            i += 8;
        }
        while i < n {
            *dp.add(i) = super::bf16_bits_to_f32(*sp.add(i));
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn f32_to_bf16(src: &[f32], dst: &mut [u16]) {
        let n = src.len();
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        let bias = _mm256_set1_epi32(0x7fff);
        let one = _mm256_set1_epi32(1);
        let nan_mask = _mm256_set1_epi32(0x7fff_ffff);
        let inf = _mm256_set1_epi32(0x7f80_0000);
        while i + 8 <= n {
            let u = _mm256_castps_si256(_mm256_loadu_ps(sp.add(i)));
            let lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), one);
            let rounded = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(u, bias), lsb), 16);
            let is_nan = _mm256_cmpgt_epi32(_mm256_and_si256(u, nan_mask), inf);
            let nan_v = _mm256_or_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(0x40));
            let r = _mm256_blendv_epi8(rounded, nan_v, is_nan);
            let packed = _mm256_packus_epi32(r, r);
            let packed = _mm256_permute4x64_epi64(packed, 0b11011000);
            _mm_storeu_si128(dp.add(i) as *mut __m128i, _mm256_castsi256_si128(packed));
            i += 8;
        }
        while i < n {
            *dp.add(i) = super::f32_to_bf16_bits(*sp.add(i));
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub unsafe fn f16_to_f32(src: &[u16], dst: &mut [f32]) {
        let n = src.len();
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        while i + 4 <= n {
            let h: float16x4_t = vreinterpret_f16_u16(vld1_u16(sp.add(i)));
            vst1q_f32(dp.add(i), vcvt_f32_f16(h));
            i += 4;
        }
        while i < n {
            *dp.add(i) = half::f16::from_bits(*sp.add(i)).to_f32();
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn f32_to_f16(src: &[f32], dst: &mut [u16]) {
        let n = src.len();
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        while i + 4 <= n {
            let h = vcvt_f16_f32(vld1q_f32(sp.add(i)));
            vst1_u16(dp.add(i), vreinterpret_u16_f16(h));
            i += 4;
        }
        while i < n {
            *dp.add(i) = half::f16::from_f32(*sp.add(i)).to_bits();
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn bf16_to_f32(src: &[u16], dst: &mut [f32]) {
        let n = src.len();
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        while i + 4 <= n {
            let h = vmovl_u16(vld1_u16(sp.add(i)));
            vst1q_f32(dp.add(i), vreinterpretq_f32_u32(vshlq_n_u32(h, 16)));
            i += 4;
        }
        while i < n {
            *dp.add(i) = super::bf16_bits_to_f32(*sp.add(i));
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn f32_to_bf16(src: &[f32], dst: &mut [u16]) {
        let n = src.len();
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        let bias = vdupq_n_u32(0x7fff);
        let one = vdupq_n_u32(1);
        let inf = vdupq_n_u32(0x7f80_0000);
        let abs = vdupq_n_u32(0x7fff_ffff);
        while i + 4 <= n {
            let u = vreinterpretq_u32_f32(vld1q_f32(sp.add(i)));
            let lsb = vandq_u32(vshrq_n_u32(u, 16), one);
            let r = vshrq_n_u32(vaddq_u32(vaddq_u32(u, bias), lsb), 16);
            let is_nan = vcgtq_u32(vandq_u32(u, abs), inf);
            let nan_v = vorrq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(0x40));
            let r = vbslq_u32(is_nan, nan_v, r);
            vst1_u16(dp.add(i), vmovn_u32(r));
            i += 4;
        }
        while i < n {
            *dp.add(i) = super::f32_to_bf16_bits(*sp.add(i));
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversions_match_scalar() {
        let xs: Vec<f32> = (0..1037).map(|i| ((i as f32) * 0.37 - 190.0) * 1.7).chain([f32::NAN, 0.0, -0.0, 65504.0, 1e-8]).collect();
        let mut h = vec![0u16; xs.len()];
        f32_to_f16(&xs, &mut h);
        for (i, &x) in xs.iter().enumerate() {
            let want = half::f16::from_f32(x).to_bits();
            assert!(h[i] == want || (x.is_nan() && half::f16::from_bits(h[i]).is_nan()), "f16 {i}: {x}");
        }
        let mut back = vec![0f32; xs.len()];
        f16_to_f32(&h, &mut back);
        for (i, &b) in back.iter().enumerate() {
            let want = half::f16::from_bits(h[i]).to_f32();
            assert!(b.to_bits() == want.to_bits() || (b.is_nan() && want.is_nan()), "f16->f32 {i}");
        }
        let mut b16 = vec![0u16; xs.len()];
        f32_to_bf16(&xs, &mut b16);
        for (i, &x) in xs.iter().enumerate() {
            let want = half::bf16::from_f32(x).to_bits();
            assert!(b16[i] == want || (x.is_nan() && half::bf16::from_bits(b16[i]).is_nan()), "bf16 {i}: {x} {:x} {:x}", b16[i], want);
        }
        bf16_to_f32(&b16, &mut back);
        for (i, &b) in back.iter().enumerate() {
            let want = half::bf16::from_bits(b16[i]).to_f32();
            assert!(b.to_bits() == want.to_bits() || (b.is_nan() && want.is_nan()), "bf16->f32 {i}");
        }
    }
}
