use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::quant::sq;

pub const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

pub fn nvfp4_sf_inner_dim(k: usize) -> usize {
    k.div_ceil(64) * 4
}

pub fn nvfp4_scale_bytes(n: usize, k: usize) -> usize {
    nvfp4_sf_inner_dim(k) * n.div_ceil(128) * 128
}

#[inline]
pub fn nvfp4_scale_index(outer: usize, block_col: usize, sf_inner_dim: usize) -> usize {
    let tile_row = outer >> 7;
    let tile_col = block_col >> 2;
    let local_outer = outer & 127;
    let local_inner = block_col & 3;
    (tile_col * 4 + tile_row * sf_inner_dim) * 128 + (local_outer & 31) * 16 + (local_outer >> 5) * 4 + local_inner
}

pub struct WeightView<'a> {
    pub dtype: DType,
    pub n: usize,
    pub k: usize,
    pub packed: &'a [u8],
    pub scales: Option<&'a [u8]>,
    pub row_bytes: usize,
}

impl<'a> WeightView<'a> {
    pub fn new(dtype: DType, n: usize, k: usize, packed: &'a [u8], scales: Option<&'a [u8]>) -> Result<Self> {
        let row_bytes = match dtype {
            DType::NVFP4 => {
                if k % 64 != 0 || scales.is_none() {
                    return Err(SynaptixError::Unsupported("WeightView: NVFP4 требует k % 64 == 0 и шкалы"));
                }
                k / 2
            }
            DType::MXFP8 => {
                if k % 32 != 0 || scales.is_none() {
                    return Err(SynaptixError::Unsupported("WeightView: MXFP8 требует k % 32 == 0 и шкалы"));
                }
                k
            }
            DType::Sq { .. } | DType::Ggml(_) => synaptix_core::quant::block_row_bytes(dtype, k)
                .ok_or(SynaptixError::Unsupported("WeightView: K не кратен блоку формата"))?,
            _ => return Err(SynaptixError::Unsupported("WeightView: не квантованный dtype")),
        };
        if packed.len() < n * row_bytes {
            return Err(SynaptixError::Unsupported("WeightView: блоб короче n·row_bytes"));
        }
        Ok(Self { dtype, n, k, packed, scales, row_bytes })
    }

    #[inline]
    pub fn row_bytes_of(&self, i: usize) -> &'a [u8] {
        &self.packed[i * self.row_bytes..(i + 1) * self.row_bytes]
    }

    pub fn row(&self, i: usize, dst: &mut [f32]) {
        debug_assert!(dst.len() >= self.k);
        let src = self.row_bytes_of(i);
        match self.dtype {
            DType::Sq { bits } => sq_dequant_row(bits, src, self.k, dst),
            DType::Ggml(t) => super::ggml_simd::dequant_row(t, src, self.k, dst),
            DType::NVFP4 => nvfp4_dequant_row(src, self.scales.unwrap(), i, self.k, dst),
            DType::MXFP8 => {
                let nb = self.k / 32;
                mxfp8_dequant_row(src, &self.scales.unwrap()[i * nb..(i + 1) * nb], dst)
            }
            _ => unreachable!(),
        }
    }
}

pub fn mxfp8_dequant_row(packed: &[u8], scales: &[u8], dst: &mut [f32]) {
    for (b, blk) in packed.chunks_exact(32).enumerate() {
        let s = crate::quant::e8m0_decode(scales[b]);
        let o = &mut dst[b * 32..b * 32 + 32];
        for (d, &q) in o.iter_mut().zip(blk) {
            *d = crate::quant::decode_e4m3(q) * s;
        }
    }
}

pub fn nvfp4_dequant_row(packed: &[u8], scales: &[u8], outer: usize, k: usize, dst: &mut [f32]) {
    let sf = nvfp4_sf_inner_dim(k);
    for blk in 0..k / 16 {
        let s = crate::quant::decode_e4m3(scales[nvfp4_scale_index(outer, blk, sf)]);
        let bytes = &packed[blk * 8..blk * 8 + 8];
        let o = &mut dst[blk * 16..blk * 16 + 16];
        for (j, &b) in bytes.iter().enumerate() {
            o[2 * j] = E2M1[(b & 0x0f) as usize] * s;
            o[2 * j + 1] = E2M1[(b >> 4) as usize] * s;
        }
    }
}

pub fn sq_dequant_row(bits: u8, src: &[u8], k: usize, dst: &mut [f32]) {
    let sbb = sq::super_block_bytes(bits);
    let mut scratch = [0f32; sq::SUPER_BLOCK];
    for (ib, blk) in src.chunks_exact(sbb).enumerate() {
        let off = ib * sq::SUPER_BLOCK;
        if off >= k {
            break;
        }
        let take = (k - off).min(sq::SUPER_BLOCK);
        if take == sq::SUPER_BLOCK {
            sq_dequant_super_block(bits, blk, &mut dst[off..off + sq::SUPER_BLOCK]);
        } else {
            sq_dequant_super_block(bits, blk, &mut scratch);
            dst[off..off + take].copy_from_slice(&scratch[..take]);
        }
    }
}

pub fn sq_dequant_super_block(bits: u8, blk: &[u8], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::cpu_features().avx2 {
            unsafe { x86::sq_super_block(bits as usize, blk, out) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { arm::sq_super_block(bits as usize, blk, out) };
        return;
    }
    #[allow(unreachable_code)]
    sq::dequant_super_block(bits, blk, out);
}

#[inline]
fn sq_header(blk: &[u8]) -> (f32, f32) {
    let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
    (d, dmin)
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;
    use synaptix_core::quant::sq::{HEADER_BYTES, SUBS, SUB_BLOCK};

    #[target_feature(enable = "avx2")]
    pub unsafe fn sq_super_block(b: usize, blk: &[u8], out: &mut [f32]) {
        let (d, dmin) = super::sq_header(blk);
        let one = _mm256_set1_epi32(1);
        let shifts = [
            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
            _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15),
            _mm256_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23),
            _mm256_setr_epi32(24, 25, 26, 27, 28, 29, 30, 31),
        ];
        let op = out.as_mut_ptr();
        for s in 0..SUBS {
            let dl = _mm256_set1_ps(d * blk[4 + s] as f32);
            let ml = _mm256_set1_ps(dmin * blk[12 + s] as f32);
            let base = HEADER_BYTES + s * b * 4;
            let mut planes = [_mm256_setzero_si256(); 8];
            for (j, p) in planes.iter_mut().enumerate().take(b) {
                let o = base + j * 4;
                *p = _mm256_set1_epi32(i32::from_le_bytes([blk[o], blk[o + 1], blk[o + 2], blk[o + 3]]));
            }
            for (g, sh) in shifts.iter().enumerate() {
                let mut q = _mm256_setzero_si256();
                for (j, p) in planes.iter().enumerate().take(b) {
                    let bit = _mm256_and_si256(_mm256_srlv_epi32(*p, *sh), one);
                    q = _mm256_or_si256(q, _mm256_sll_epi32(bit, _mm_cvtsi32_si128(j as i32)));
                }
                let f = _mm256_cvtepi32_ps(q);
                let r = _mm256_sub_ps(_mm256_mul_ps(dl, f), ml);
                _mm256_storeu_ps(op.add(s * SUB_BLOCK + g * 8), r);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::*;
    use synaptix_core::quant::sq::{HEADER_BYTES, SUBS, SUB_BLOCK};

    #[target_feature(enable = "neon")]
    pub unsafe fn sq_super_block(b: usize, blk: &[u8], out: &mut [f32]) {
        let (d, dmin) = super::sq_header(blk);
        let one = vdupq_n_u32(1);
        let op = out.as_mut_ptr();
        for s in 0..SUBS {
            let dl = vdupq_n_f32(d * blk[4 + s] as f32);
            let ml = vdupq_n_f32(dmin * blk[12 + s] as f32);
            let base = HEADER_BYTES + s * b * 4;
            let mut planes = [vdupq_n_u32(0); 8];
            for (j, p) in planes.iter_mut().enumerate().take(b) {
                let o = base + j * 4;
                *p = vdupq_n_u32(u32::from_le_bytes([blk[o], blk[o + 1], blk[o + 2], blk[o + 3]]));
            }
            for g in 0..8 {
                let l0 = (g * 4) as i32;
                let sh = vnegq_s32(vaddq_s32(vdupq_n_s32(l0), vld1q_s32([0i32, 1, 2, 3].as_ptr())));
                let mut q = vdupq_n_u32(0);
                for (j, p) in planes.iter().enumerate().take(b) {
                    let bit = vandq_u32(vshlq_u32(*p, sh), one);
                    q = vorrq_u32(q, vshlq_u32(bit, vdupq_n_s32(j as i32)));
                }
                let f = vcvtq_f32_u32(q);
                let r = vsubq_f32(vmulq_f32(dl, f), ml);
                vst1q_f32(op.add(s * SUB_BLOCK + g * 4), r);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq_simd_matches_reference() {
        for bits in 1..=8u8 {
            let k = 512;
            let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.731).sin() * 3.0 - 0.7).collect();
            let blob = sq::quantize_matrix(bits, &x, 1, k).unwrap();
            let mut want = vec![0f32; k];
            sq::dequant_row(bits, &blob, k, &mut want).unwrap();
            let mut got = vec![0f32; k];
            sq_dequant_row(bits, &blob, k, &mut got);
            assert_eq!(want, got, "bits {bits}");
        }
    }

    #[test]
    fn sq_row_tail_block() {
        let bits = 4;
        let k = 320;
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.1).cos()).collect();
        let blob = sq::quantize_matrix(bits, &x, 1, k).unwrap();
        let mut want = vec![0f32; k];
        sq::dequant_row(bits, &blob, k, &mut want).unwrap();
        let mut got = vec![7f32; k];
        sq_dequant_row(bits, &blob, k, &mut got);
        assert_eq!(want, got);
    }
}
