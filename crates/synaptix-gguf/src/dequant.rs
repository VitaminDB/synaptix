use half::{bf16, f16};

use crate::error::{GgufError, Result};
use crate::ggml::{GgmlType, QK_K};

const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[inline]
fn rd_f16(b: &[u8], off: usize) -> f32 {
    f16::from_le_bytes([b[off], b[off + 1]]).to_f32()
}

#[inline]
fn rd_bf16(b: &[u8], off: usize) -> f32 {
    bf16::from_le_bytes([b[off], b[off + 1]]).to_f32()
}

pub fn dequantize(ty: GgmlType, src: &[u8], n: usize, dst: &mut [f32]) -> Result<()> {
    let need = ty.bytes_for(n);
    if src.len() < need {
        return Err(GgufError::Truncated {
            at: 0,
            need,
            have: src.len(),
        });
    }
    if dst.len() < n {
        return Err(GgufError::Truncated {
            at: 0,
            need: n,
            have: dst.len(),
        });
    }
    use GgmlType::*;
    match ty {
        F32 => {
            for i in 0..n {
                dst[i] = f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        F16 => {
            for i in 0..n {
                dst[i] = rd_f16(src, i * 2);
            }
        }
        BF16 => {
            for i in 0..n {
                dst[i] = rd_bf16(src, i * 2);
            }
        }
        F64 => {
            for i in 0..n {
                dst[i] = f64::from_le_bytes(src[i * 8..i * 8 + 8].try_into().unwrap()) as f32;
            }
        }
        I8 => {
            for i in 0..n {
                dst[i] = src[i] as i8 as f32;
            }
        }
        I16 => {
            for i in 0..n {
                dst[i] = i16::from_le_bytes(src[i * 2..i * 2 + 2].try_into().unwrap()) as f32;
            }
        }
        I32 => {
            for i in 0..n {
                dst[i] = i32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap()) as f32;
            }
        }
        I64 => {
            for i in 0..n {
                dst[i] = i64::from_le_bytes(src[i * 8..i * 8 + 8].try_into().unwrap()) as f32;
            }
        }
        Q4_0 => blocks(ty, src, n, dst, deq_q4_0),
        Q4_1 => blocks(ty, src, n, dst, deq_q4_1),
        Q5_0 => blocks(ty, src, n, dst, deq_q5_0),
        Q5_1 => blocks(ty, src, n, dst, deq_q5_1),
        Q8_0 => blocks(ty, src, n, dst, deq_q8_0),
        Q2K => blocks(ty, src, n, dst, deq_q2_k),
        Q3K => blocks(ty, src, n, dst, deq_q3_k),
        Q4K => blocks(ty, src, n, dst, deq_q4_k),
        Q5K => blocks(ty, src, n, dst, deq_q5_k),
        Q6K => blocks(ty, src, n, dst, deq_q6_k),
        Iq4Nl => blocks(ty, src, n, dst, deq_iq4_nl),
        Iq4Xs => blocks(ty, src, n, dst, deq_iq4_xs),
        Mxfp4 => blocks(ty, src, n, dst, deq_mxfp4),
        other => return Err(GgufError::UnsupportedQuant(other.name())),
    }
    Ok(())
}

#[inline]
fn blocks(
    ty: GgmlType,
    src: &[u8],
    n: usize,
    dst: &mut [f32],
    f: impl Fn(&[u8], &mut [f32]),
) {
    let be = ty.block_elems();
    let bb = ty.block_bytes();
    let nb = n.div_ceil(be);
    let mut scratch = vec![0f32; be];
    for ib in 0..nb {
        let out_off = ib * be;
        let rest = n - out_off;
        let blk = &src[ib * bb..ib * bb + bb];
        if rest >= be {
            f(blk, &mut dst[out_off..out_off + be]);
        } else {
            f(blk, &mut scratch);
            dst[out_off..out_off + rest].copy_from_slice(&scratch[..rest]);
        }
    }
}

fn deq_q4_0(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..18];
    for j in 0..16 {
        y[j] = ((qs[j] & 0x0F) as i32 - 8) as f32 * d;
        y[j + 16] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
    }
}

fn deq_q4_1(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let m = rd_f16(b, 2);
    let qs = &b[4..20];
    for j in 0..16 {
        y[j] = (qs[j] & 0x0F) as f32 * d + m;
        y[j + 16] = (qs[j] >> 4) as f32 * d + m;
    }
}

fn deq_q5_0(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qh = u32::from_le_bytes(b[2..6].try_into().unwrap());
    let qs = &b[6..22];
    for j in 0..16 {
        let xh0 = (((qh >> j) << 4) & 0x10) as u8;
        let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
        y[j] = (((qs[j] & 0x0F) | xh0) as i32 - 16) as f32 * d;
        y[j + 16] = (((qs[j] >> 4) | xh1) as i32 - 16) as f32 * d;
    }
}

fn deq_q5_1(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let m = rd_f16(b, 2);
    let qh = u32::from_le_bytes(b[4..8].try_into().unwrap());
    let qs = &b[8..24];
    for j in 0..16 {
        let xh0 = (((qh >> j) << 4) & 0x10) as u8;
        let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
        y[j] = ((qs[j] & 0x0F) | xh0) as f32 * d + m;
        y[j + 16] = ((qs[j] >> 4) | xh1) as f32 * d + m;
    }
}

fn deq_q8_0(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    for j in 0..32 {
        y[j] = b[2 + j] as i8 as f32 * d;
    }
}

fn deq_mxfp4(b: &[u8], y: &mut [f32]) {

    const KV: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    let d = f32::from_bits((b[0] as u32) << 23);
    let qs = &b[1..17];
    for j in 0..16 {
        y[j] = KV[(qs[j] & 0x0F) as usize] * d;
        y[j + 16] = KV[(qs[j] >> 4) as usize] * d;
    }
}

fn deq_q2_k(b: &[u8], y: &mut [f32]) {

    let scales = &b[0..16];
    let qs = &b[16..80];
    let d = rd_f16(b, 80);
    let dmin = rd_f16(b, 82);
    let mut out = 0usize;
    let mut is = 0usize;
    for n in (0..QK_K).step_by(128) {
        let q = &qs[n / 4..];
        let mut shift = 0u32;
        for _ in 0..4 {
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0xF) as f32;
            let ml = dmin * (sc >> 4) as f32;
            for l in 0..16 {
                y[out] = dl * ((q[l] >> shift) & 3) as f32 - ml;
                out += 1;
            }
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0xF) as f32;
            let ml = dmin * (sc >> 4) as f32;
            for l in 0..16 {
                y[out] = dl * ((q[l + 16] >> shift) & 3) as f32 - ml;
                out += 1;
            }
            shift += 2;
        }
    }
}

fn deq_q3_k(b: &[u8], y: &mut [f32]) {

    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let hmask = &b[0..32];
    let qs = &b[32..96];
    let d_all = rd_f16(b, 108);

    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes(b[96..100].try_into().unwrap());
    aux[1] = u32::from_le_bytes(b[100..104].try_into().unwrap());
    aux[2] = u32::from_le_bytes(b[104..108].try_into().unwrap());
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut scales = [0i8; 16];
    for (i, w) in aux.iter().enumerate() {
        let bytes = w.to_le_bytes();
        for k in 0..4 {
            scales[i * 4 + k] = bytes[k] as i8;
        }
    }

    let mut out = 0usize;
    let mut is = 0usize;
    let mut m = 1u8;
    for n in (0..QK_K).step_by(128) {
        let q = &qs[n / 4..];
        let mut shift = 0u32;
        for _ in 0..4 {
            let dl = d_all * (scales[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let hi = if hmask[l] & m != 0 { 0 } else { 4 };
                y[out] = dl * (((q[l] >> shift) & 3) as i32 - hi) as f32;
                out += 1;
            }
            let dl = d_all * (scales[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let hi = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                y[out] = dl * (((q[l + 16] >> shift) & 3) as i32 - hi) as f32;
                out += 1;
            }
            shift += 2;
            m <<= 1;
        }
    }
}

#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn deq_q4_k(b: &[u8], y: &mut [f32]) {

    let d = rd_f16(b, 0);
    let dmin = rd_f16(b, 2);
    let scales = &b[4..16];
    let qs = &b[16..144];
    let mut out = 0usize;
    let mut is = 0usize;
    for j in (0..QK_K).step_by(64) {
        let q = &qs[j / 2..];
        let (sc, m) = scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = dmin * m as f32;
        let (sc, m) = scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = dmin * m as f32;
        for l in 0..32 {
            y[out] = d1 * (q[l] & 0xF) as f32 - m1;
            out += 1;
        }
        for l in 0..32 {
            y[out] = d2 * (q[l] >> 4) as f32 - m2;
            out += 1;
        }
        is += 2;
    }
}

fn deq_q5_k(b: &[u8], y: &mut [f32]) {

    let d = rd_f16(b, 0);
    let dmin = rd_f16(b, 2);
    let scales = &b[4..16];
    let qh = &b[16..48];
    let qs = &b[48..176];
    let mut out = 0usize;
    let mut is = 0usize;
    let mut u1 = 1u8;
    let mut u2 = 2u8;
    for j in (0..QK_K).step_by(64) {
        let ql = &qs[j / 2..];
        let (sc, m) = scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = dmin * m as f32;
        let (sc, m) = scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = dmin * m as f32;
        for l in 0..32 {
            let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
            y[out] = d1 * ((ql[l] & 0xF) as i32 + hi) as f32 - m1;
            out += 1;
        }
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
            y[out] = d2 * ((ql[l] >> 4) as i32 + hi) as f32 - m2;
            out += 1;
        }
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

fn deq_q6_k(b: &[u8], y: &mut [f32]) {

    let d = rd_f16(b, 208);
    for n in 0..2 {
        let ql = &b[n * 64..];
        let qh = &b[128 + n * 32..];
        let sc = &b[192 + n * 8..];
        let y = &mut y[n * 128..];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            y[l] = d * sc[is] as i8 as f32 * q1 as f32;
            y[l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
            y[l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
            y[l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
        }
    }
}

fn deq_iq4_nl(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..18];
    for j in 0..16 {
        y[j] = d * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
        y[j + 16] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
    }
}

fn deq_iq4_xs(b: &[u8], y: &mut [f32]) {

    let d = rd_f16(b, 0);
    let h = u16::from_le_bytes(b[2..4].try_into().unwrap());
    let scales_l = &b[4..8];
    let qs = &b[8..136];
    for ib in 0..(QK_K / 32) {
        let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32 | (((h >> (2 * ib)) & 3) as i32) << 4;
        let dl = d * (ls - 32) as f32;
        let q = &qs[ib * 16..ib * 16 + 16];
        let y = &mut y[ib * 32..ib * 32 + 32];
        for j in 0..16 {
            y[j] = dl * KVALUES_IQ4NL[(q[j] & 0xF) as usize] as f32;
            y[j + 16] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_round_trip_exact() {
        let d = f16::from_f32(0.01234);
        let mut blk = vec![0u8; 34];
        blk[0..2].copy_from_slice(&d.to_le_bytes());
        for i in 0..32 {
            blk[2 + i] = (i as i32 - 16) as i8 as u8;
        }
        let mut out = vec![0f32; 32];
        dequantize(GgmlType::Q8_0, &blk, 32, &mut out).unwrap();
        for i in 0..32 {
            let want = (i as i32 - 16) as f32 * d.to_f32();
            assert_eq!(out[i], want, "элемент {i}");
        }
    }

    #[test]
    fn q4_0_nibble_order() {
        let d = f16::from_f32(0.5);
        let mut blk = vec![0u8; 18];
        blk[0..2].copy_from_slice(&d.to_le_bytes());
        blk[2] = 0x0F;
        let mut out = vec![0f32; 32];
        dequantize(GgmlType::Q4_0, &blk, 32, &mut out).unwrap();
        assert_eq!(out[0], 7.0 * 0.5);
        assert_eq!(out[16], -8.0 * 0.5);
    }

    #[test]
    fn q4_k_scale_min_unpack() {
        let mut q = [0u8; 12];
        q[0] = 0b0011_1111;
        q[4] = 0b0010_1010;
        assert_eq!(scale_min_k4(0, &q), (63, 42));
        q[8] = 0x5A;
        q[0] |= 0b1100_0000;
        q[4] |= 0b1100_0000;

        assert_eq!(scale_min_k4(4, &q), (0x3A, 0x35));
    }

    #[test]
    fn partial_tail_block() {
        let d = f16::from_f32(1.0);
        let mut blk = vec![0u8; 34];
        blk[0..2].copy_from_slice(&d.to_le_bytes());
        for i in 0..32 {
            blk[2 + i] = i as u8;
        }
        let mut out = vec![0f32; 20];
        dequantize(GgmlType::Q8_0, &blk, 20, &mut out).unwrap();
        assert_eq!(out[19], 19.0);
    }

    #[test]
    fn iq4_xs_uses_shared_lut() {
        let d = f16::from_f32(1.0);
        let mut blk = vec![0u8; 136];
        blk[0..2].copy_from_slice(&d.to_le_bytes());

        blk[2] = 0;
        blk[3] = 0;
        blk[4] = 33;
        let mut out = vec![0f32; 256];
        dequantize(GgmlType::Iq4Xs, &blk, 256, &mut out).unwrap();

        assert_eq!(out[0], -31.0 * -127.0);
    }

    #[test]
    fn unsupported_iq_reports_type_name() {
        let src = vec![0u8; 1024];
        let mut out = vec![0f32; 256];
        let err = dequantize(GgmlType::Iq2Xs, &src, 256, &mut out).unwrap_err();
        assert!(matches!(err, GgufError::UnsupportedQuant("IQ2_XS")));
    }
}
