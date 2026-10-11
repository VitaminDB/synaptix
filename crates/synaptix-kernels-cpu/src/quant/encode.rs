use rayon::prelude::*;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::quant::sq;

use super::dequant::{nvfp4_scale_bytes, nvfp4_scale_index, nvfp4_sf_inner_dim};

pub fn quantize_sq_rows(bits: u8, w: &[f32], n: usize, k: usize) -> Result<Vec<u8>> {
    sq::check_bits(bits)?;
    if k % sq::SUB_BLOCK != 0 || w.len() != n * k {
        return Err(SynaptixError::Unsupported("quantize_sq_rows: K % 32 != 0 или размер"));
    }
    let sbb = sq::super_block_bytes(bits);
    let rb = sq::row_bytes(bits, k);
    let mut out = vec![0u8; n * rb];
    crate::threads::pool();
    out.par_chunks_mut(rb).enumerate().for_each(|(r, row_out)| {
        let row = &w[r * k..(r + 1) * k];
        for (ib, chunk) in row.chunks(sq::SUPER_BLOCK).enumerate() {
            sq::quant_super_block(bits, chunk, &mut row_out[ib * sbb..(ib + 1) * sbb]);
        }
    });
    Ok(out)
}

pub fn quantize_mxfp8_rows(w: &[f32], n: usize, k: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    if k % 32 != 0 || w.len() != n * k {
        return Err(SynaptixError::Unsupported("quantize_mxfp8_rows: K % 32 != 0 или размер"));
    }
    let nb = k / 32;
    let mut packed = vec![0u8; n * k];
    let mut scales = vec![0u8; n * nb];
    crate::threads::pool();
    packed.par_chunks_mut(k).zip(scales.par_chunks_mut(nb)).enumerate().for_each(|(r, (p, s))| {
        let row = &w[r * k..(r + 1) * k];
        for b in 0..nb {
            let blk = &row[b * 32..b * 32 + 32];
            let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let sb = super::e8m0_scale_byte(amax);
            let sv = super::e8m0_decode(sb);
            s[b] = sb;
            for (i, &v) in blk.iter().enumerate() {
                p[b * 32 + i] = super::encode_e4m3(v / sv);
            }
        }
    });
    Ok((packed, scales))
}

#[inline]
pub fn encode_e2m1_rtne(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 8u8 } else { 0 };
    let a = x.abs();
    let idx = if a >= 5.0 {
        7
    } else if a >= 3.5 {
        6
    } else if a >= 2.5 {
        5
    } else if a >= 1.75 {
        4
    } else if a >= 1.25 {
        3
    } else if a >= 0.75 {
        2
    } else if a >= 0.25 {
        1
    } else {
        0
    };
    sign | idx
}

pub fn quantize_nvfp4_rows(w: &[f32], n: usize, k: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    if k % 64 != 0 || w.len() != n * k {
        return Err(SynaptixError::Unsupported("quantize_nvfp4_rows: K % 64 != 0 или размер"));
    }
    let sf = nvfp4_sf_inner_dim(k);
    let mut packed = vec![0u8; n * k / 2];
    let mut scales = vec![0u8; nvfp4_scale_bytes(n, k)];
    let nblk = k / 16;
    crate::threads::pool();
    let mut row_scales: Vec<Vec<u8>> = Vec::with_capacity(n);
    packed
        .par_chunks_mut(k / 2)
        .enumerate()
        .map(|(r, p)| {
            let row = &w[r * k..(r + 1) * k];
            let mut sc = vec![0u8; nblk];
            for b in 0..nblk {
                let blk = &row[b * 16..b * 16 + 16];
                let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
                let raw = if amax > 0.0 { amax / 6.0 } else { 1e-9 };
                let sb = super::encode_e4m3(raw);
                let mut sq = super::decode_e4m3(sb);
                if sq == 0.0 {
                    sq = 1e-9;
                }
                sc[b] = sb;
                for j in 0..8 {
                    let lo = encode_e2m1_rtne(blk[2 * j] / sq);
                    let hi = encode_e2m1_rtne(blk[2 * j + 1] / sq);
                    p[b * 8 + j] = lo | (hi << 4);
                }
            }
            sc
        })
        .collect_into_vec(&mut row_scales);
    for (r, sc) in row_scales.iter().enumerate() {
        for (b, &s) in sc.iter().enumerate() {
            scales[nvfp4_scale_index(r, b, sf)] = s;
        }
    }
    Ok((packed, scales))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::dequant::WeightView;
    use synaptix_core::dtype::DType;

    fn data(n: usize, k: usize) -> Vec<f32> {
        (0..n * k).map(|i| ((i as f32) * 0.37).sin() * ((i % 7) as f32 - 3.0)).collect()
    }

    #[test]
    fn sq_rows_match_reference_encoder() {
        let (n, k) = (5, 288);
        let w = data(n, k);
        for bits in [2u8, 4, 8] {
            let got = quantize_sq_rows(bits, &w, n, k).unwrap();
            let want = sq::quantize_matrix(bits, &w, n, k).unwrap();
            assert_eq!(got, want, "bits {bits}");
        }
    }

    #[test]
    fn mxfp8_and_nvfp4_round_trip() {
        let (n, k) = (130, 192);
        let w = data(n, k);
        let (p, s) = quantize_mxfp8_rows(&w, n, k).unwrap();
        let v = WeightView::new(DType::MXFP8, n, k, &p, Some(&s)).unwrap();
        let mut row = vec![0f32; k];
        for r in 0..n {
            v.row(r, &mut row);
            for i in 0..k {
                let x = w[r * k + i];
                assert!((row[i] - x).abs() <= 0.07 * x.abs() + 1e-3, "mxfp8 {r} {i}: {} vs {x}", row[i]);
            }
        }
        let (p, s) = quantize_nvfp4_rows(&w, n, k).unwrap();
        assert_eq!(s.len(), nvfp4_scale_bytes(n, k));
        let v = WeightView::new(DType::NVFP4, n, k, &p, Some(&s)).unwrap();
        let mut err = 0f64;
        let mut norm = 0f64;
        for r in 0..n {
            v.row(r, &mut row);
            for i in 0..k {
                let x = w[r * k + i] as f64;
                err += (row[i] as f64 - x).powi(2);
                norm += x * x;
            }
        }
        assert!(err / norm < 0.02, "nvfp4 rel err {}", err / norm);
    }
}
