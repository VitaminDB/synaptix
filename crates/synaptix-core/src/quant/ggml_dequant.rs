//! CPU-эталон деквантования всех типов ggml (порт `dequantize_row_*` из
//! `ggml-quants.c`, llama.cpp, MIT). Выход — f32.
//!
//! В движке это **эталон для тестов и вспомогательный путь**: боевой деквант
//! идёт на карте (`blockq_dequant.cu` в synaptix-kernels-cuda), и он обязан
//! давать те же биты. Поэтому порядок операций здесь — контракт: каждое
//! произведение и сумма выписаны так, как их считает C-эталон, а GPU-модуль
//! собирается без слияния в FMA.

use half::{bf16, f16};

use super::ggml::{GgmlType, QK_K};
use super::ggml_tables::*;
use crate::error::{Result, SynaptixError};

const IQ1S_DELTA: f32 = 0.125;

#[inline]
fn rd_f16(b: &[u8], off: usize) -> f32 {
    f16::from_le_bytes([b[off], b[off + 1]]).to_f32()
}

#[inline]
fn rd_bf16(b: &[u8], off: usize) -> f32 {
    bf16::from_le_bytes([b[off], b[off + 1]]).to_f32()
}

#[inline]
fn rd_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

#[inline]
fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// `ggml_e8m0_to_fp32`.
#[inline]
pub fn e8m0_to_f32(x: u8) -> f32 {
    if x == 0 {
        f32::from_bits(0x0040_0000)
    } else if x == 0xFF {
        f32::NAN
    } else {
        f32::from_bits((x as u32) << 23)
    }
}

/// `ggml_e8m0_to_fp32_half` — половина шкалы, под удвоенные `kvalues_fp4`.
#[inline]
pub fn e8m0_to_f32_half(x: u8) -> f32 {
    if x < 2 {
        f32::from_bits(0x0020_0000 << x)
    } else {
        f32::from_bits(((x as u32) - 1) << 23)
    }
}

/// `ggml_ue4m3_to_fp32` — беззнаковый E4M3 (bias 7), ×0,5 под `kvalues_fp4`.
#[inline]
pub fn ue4m3_to_f32_half(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xF) as i32;
    let man = (x & 0x7) as f32;
    let raw = if exp == 0 { man * 2f32.powi(-9) } else { (1.0 + man / 8.0) * 2f32.powi(exp - 7) };
    raw * 0.5
}

/// Деквант `n` элементов типа `ty` из `src` в `dst` (f32). `src` должен
/// покрывать `ty.bytes_for(n)` байт; хвостовой блок читается целиком.
pub fn dequantize(ty: GgmlType, src: &[u8], n: usize, dst: &mut [f32]) -> Result<()> {
    let need = ty.bytes_for(n);
    if src.len() < need || dst.len() < n {
        return Err(SynaptixError::Unsupported("ggml dequantize: буфер короче раскладки"));
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
        Q8_1 => blocks(ty, src, n, dst, deq_q8_1),
        Q2K => blocks(ty, src, n, dst, deq_q2_k),
        Q3K => blocks(ty, src, n, dst, deq_q3_k),
        Q4K => blocks(ty, src, n, dst, deq_q4_k),
        Q5K => blocks(ty, src, n, dst, deq_q5_k),
        Q6K => blocks(ty, src, n, dst, deq_q6_k),
        Q8K => blocks(ty, src, n, dst, deq_q8_k),
        Iq2Xxs => blocks(ty, src, n, dst, deq_iq2_xxs),
        Iq2Xs => blocks(ty, src, n, dst, deq_iq2_xs),
        Iq2S => blocks(ty, src, n, dst, deq_iq2_s),
        Iq3Xxs => blocks(ty, src, n, dst, deq_iq3_xxs),
        Iq3S => blocks(ty, src, n, dst, deq_iq3_s),
        Iq1S => blocks(ty, src, n, dst, deq_iq1_s),
        Iq1M => blocks(ty, src, n, dst, deq_iq1_m),
        Iq4Nl => blocks(ty, src, n, dst, deq_iq4_nl),
        Iq4Xs => blocks(ty, src, n, dst, deq_iq4_xs),
        Tq1_0 => blocks(ty, src, n, dst, deq_tq1_0),
        Tq2_0 => blocks(ty, src, n, dst, deq_tq2_0),
        Mxfp4 => blocks(ty, src, n, dst, deq_mxfp4),
        Nvfp4 => blocks(ty, src, n, dst, deq_nvfp4),
        Q1_0 => blocks(ty, src, n, dst, deq_q1_0),
        Q2_0 => blocks(ty, src, n, dst, deq_q2_0),
    }
    Ok(())
}

#[inline]
fn blocks(ty: GgmlType, src: &[u8], n: usize, dst: &mut [f32], f: impl Fn(&[u8], &mut [f32])) {
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
    let qh = rd_u32(b, 2);
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
    let qh = rd_u32(b, 4);
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

fn deq_q8_1(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    for j in 0..32 {
        y[j] = b[4 + j] as i8 as f32 * d;
    }
}

fn deq_q8_k(b: &[u8], y: &mut [f32]) {
    let d = f32::from_le_bytes(b[0..4].try_into().unwrap());
    for j in 0..QK_K {
        y[j] = d * b[4 + j] as i8 as f32;
    }
}

fn deq_q1_0(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let neg_d = -d;
    for j in 0..128 {
        let bit = (b[2 + j / 8] >> (j % 8)) & 1;
        y[j] = if bit != 0 { d } else { neg_d };
    }
}

fn deq_q2_0(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    for j in 0..64 {
        let q = (b[2 + j / 4] >> ((j % 4) * 2)) & 3;
        y[j] = (q as i32 - 1) as f32 * d;
    }
}

fn deq_mxfp4(b: &[u8], y: &mut [f32]) {
    let d = e8m0_to_f32_half(b[0]);
    let qs = &b[1..17];
    for j in 0..16 {
        y[j] = KVALUES_FP4[(qs[j] & 0x0F) as usize] as f32 * d;
        y[j + 16] = KVALUES_FP4[(qs[j] >> 4) as usize] as f32 * d;
    }
}

fn deq_nvfp4(b: &[u8], y: &mut [f32]) {
    for s in 0..4 {
        let d = ue4m3_to_f32_half(b[s]);
        let qs = &b[4 + s * 8..4 + s * 8 + 8];
        let yb = &mut y[s * 16..s * 16 + 16];
        for j in 0..8 {
            yb[j] = KVALUES_FP4[(qs[j] & 0x0F) as usize] as f32 * d;
            yb[j + 8] = KVALUES_FP4[(qs[j] >> 4) as usize] as f32 * d;
        }
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
    aux[0] = rd_u32(b, 96);
    aux[1] = rd_u32(b, 100);
    aux[2] = rd_u32(b, 104);
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
pub fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
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
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
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
    let h = rd_u16(b, 2);
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

#[inline]
fn sign(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 {
        -1.0
    } else {
        1.0
    }
}

fn deq_iq2_xxs(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..];
    let mut out = 0usize;
    for ib32 in 0..(QK_K / 32) {
        let aux0 = rd_u32(qs, 8 * ib32);
        let aux1 = rd_u32(qs, 8 * ib32 + 4);
        let aux8 = aux0.to_le_bytes();
        let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
        for l in 0..4 {
            let grid = IQ2XXS_GRID[aux8[l] as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                y[out] = db * grid[j] as f32 * sign(signs, j);
                out += 1;
            }
        }
    }
}

fn deq_iq2_xs(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..66];
    let scales = &b[66..74];
    let mut out = 0usize;
    for ib32 in 0..(QK_K / 32) {
        let db = [
            d * (0.5 + (scales[ib32] & 0xF) as f32) * 0.25,
            d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let q = rd_u16(qs, 2 * (4 * ib32 + l));
            let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
            for j in 0..8 {
                y[out] = db[l / 2] * grid[j] as f32 * sign(signs, j);
                out += 1;
            }
        }
    }
}

fn deq_iq2_s(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..66];
    let qh = &b[66..74];
    let scales = &b[74..82];
    let signs = &qs[QK_K / 8..];
    let mut out = 0usize;
    for ib32 in 0..(QK_K / 32) {
        let db = [
            d * (0.5 + (scales[ib32] & 0xF) as f32) * 0.25,
            d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let dl = db[l / 2];
            let idx = qs[4 * ib32 + l] as usize | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
            let grid = IQ2S_GRID[idx].to_le_bytes();
            let sg = signs[4 * ib32 + l];
            for j in 0..8 {
                y[out] = dl * grid[j] as f32 * sign(sg, j);
                out += 1;
            }
        }
    }
}

fn deq_iq3_xxs(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..];
    let scales_and_signs = &qs[QK_K / 4..];
    let mut out = 0usize;
    for ib32 in 0..(QK_K / 32) {
        let aux32 = rd_u32(scales_and_signs, 4 * ib32);
        let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
            let grid1 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize].to_le_bytes();
            let grid2 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize].to_le_bytes();
            for j in 0..4 {
                y[out + j] = db * grid1[j] as f32 * sign(signs, j);
                y[out + j + 4] = db * grid2[j] as f32 * sign(signs, j + 4);
            }
            out += 8;
        }
    }
}

fn deq_iq3_s(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..66];
    let qh = &b[66..74];
    let signs = &b[74..106];
    let scales = &b[106..110];
    let mut out = 0usize;
    let mut qi = 0usize;
    let mut si = 0usize;
    let mut hi = 0usize;
    let mut ib32 = 0usize;
    while ib32 < QK_K / 32 {
        let db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xF) as i32) as f32;
        let db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4) as i32) as f32;
        for l in 0..4 {
            let g1 = qs[qi + 2 * l] as usize | (((qh[hi] as usize) << (8 - 2 * l)) & 256);
            let g2 = qs[qi + 2 * l + 1] as usize | (((qh[hi] as usize) << (7 - 2 * l)) & 256);
            let grid1 = IQ3S_GRID[g1].to_le_bytes();
            let grid2 = IQ3S_GRID[g2].to_le_bytes();
            let sg = signs[si + l];
            for j in 0..4 {
                y[out + j] = db1 * grid1[j] as f32 * sign(sg, j);
                y[out + j + 4] = db1 * grid2[j] as f32 * sign(sg, j + 4);
            }
            out += 8;
        }
        qi += 8;
        si += 4;
        for l in 0..4 {
            let g1 = qs[qi + 2 * l] as usize | (((qh[hi + 1] as usize) << (8 - 2 * l)) & 256);
            let g2 = qs[qi + 2 * l + 1] as usize | (((qh[hi + 1] as usize) << (7 - 2 * l)) & 256);
            let grid1 = IQ3S_GRID[g1].to_le_bytes();
            let grid2 = IQ3S_GRID[g2].to_le_bytes();
            let sg = signs[si + l];
            for j in 0..4 {
                y[out + j] = db2 * grid1[j] as f32 * sign(sg, j);
                y[out + j + 4] = db2 * grid2[j] as f32 * sign(sg, j + 4);
            }
            out += 8;
        }
        hi += 2;
        qi += 8;
        si += 4;
        ib32 += 2;
    }
}

fn deq_iq1_s(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..34];
    let qh = &b[34..50];
    let mut out = 0usize;
    for ib in 0..(QK_K / 32) {
        let h = rd_u16(qh, 2 * ib);
        let dl = d * (2 * ((h >> 12) & 7) as i32 + 1) as f32;
        let delta = if h & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
        for l in 0..4 {
            let idx = qs[4 * ib + l] as usize | ((((h >> (3 * l)) & 7) as usize) << 8);
            let grid = IQ1S_GRID[idx].to_le_bytes();
            for j in 0..8 {
                y[out] = dl * (grid[j] as i8 as f32 + delta);
                out += 1;
            }
        }
    }
}

fn deq_iq1_m(b: &[u8], y: &mut [f32]) {
    let qs = &b[0..32];
    let qh = &b[32..48];
    let sc: [u16; 4] = [rd_u16(b, 48), rd_u16(b, 50), rd_u16(b, 52), rd_u16(b, 54)];
    let scale_bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    let d = f16::from_bits(scale_bits).to_f32();
    let mut out = 0usize;
    for ib in 0..(QK_K / 32) {
        let dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) as i32 + 1) as f32;
        let dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) as i32 + 1) as f32;
        let q = &qs[4 * ib..4 * ib + 4];
        let h = &qh[2 * ib..2 * ib + 2];
        let idx = [
            q[0] as usize | (((h[0] as usize) << 8) & 0x700),
            q[1] as usize | (((h[0] as usize) << 4) & 0x700),
            q[2] as usize | (((h[1] as usize) << 8) & 0x700),
            q[3] as usize | (((h[1] as usize) << 4) & 0x700),
        ];
        let delta = [
            if h[0] & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
            if h[0] & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
            if h[1] & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
            if h[1] & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
        ];
        for l in 0..4 {
            let dl = if l < 2 { dl1 } else { dl2 };
            let grid = IQ1S_GRID[idx[l]].to_le_bytes();
            for j in 0..8 {
                y[out] = dl * (grid[j] as i8 as f32 + delta[l]);
                out += 1;
            }
        }
    }
}

fn deq_tq1_0(b: &[u8], y: &mut [f32]) {
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    let qs = &b[0..48];
    let qh = &b[48..52];
    let d = rd_f16(b, 52);
    let mut out = 0usize;
    // 32 байта × 5 троек = 160, затем 16 байт × 5 = 80, затем qh 4 байта × 4 = 16.
    for j in (0..32).step_by(32) {
        for n in 0..5 {
            for m in 0..32 {
                let q = qs[j + m].wrapping_mul(POW3[n]);
                let xi = ((q as u16 * 3) >> 8) as i16;
                y[out] = (xi - 1) as f32 * d;
                out += 1;
            }
        }
    }
    for j in (32..48).step_by(16) {
        for n in 0..5 {
            for m in 0..16 {
                let q = qs[j + m].wrapping_mul(POW3[n]);
                let xi = ((q as u16 * 3) >> 8) as i16;
                y[out] = (xi - 1) as f32 * d;
                out += 1;
            }
        }
    }
    for n in 0..4 {
        for j in 0..4 {
            let q = qh[j].wrapping_mul(POW3[n]);
            let xi = ((q as u16 * 3) >> 8) as i16;
            y[out] = (xi - 1) as f32 * d;
            out += 1;
        }
    }
}

fn deq_tq2_0(b: &[u8], y: &mut [f32]) {
    let qs = &b[0..64];
    let d = rd_f16(b, 64);
    let mut out = 0usize;
    for j in (0..64).step_by(32) {
        for l in 0..4 {
            for m in 0..32 {
                let q = ((qs[j + m] >> (l * 2)) & 3) as i8;
                y[out] = (q - 1) as f32 * d;
                out += 1;
            }
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
            assert_eq!(out[i], (i as i32 - 16) as f32 * d.to_f32(), "элемент {i}");
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
        blk[4] = 33;
        let mut out = vec![0f32; 256];
        dequantize(GgmlType::Iq4Xs, &blk, 256, &mut out).unwrap();
        assert_eq!(out[0], -31.0 * -127.0);
    }

    #[test]
    fn every_type_decodes_zero_block() {
        for t in GgmlType::ALL {
            let n = t.block_elems() * 2 + if t.block_elems() > 1 { 3 } else { 0 };
            let src = vec![0u8; t.bytes_for(n)];
            let mut out = vec![7f32; n];
            dequantize(t, &src, n, &mut out).unwrap_or_else(|e| panic!("{}: {e}", t.name()));
            assert!(out.iter().all(|v| v.is_finite()), "{}", t.name());
        }
    }

    #[test]
    fn tq1_0_counts_all_256() {
        // d=1, все байты 0 → все тройки = 0 → значение −1.
        let mut blk = vec![0u8; 54];
        blk[52..54].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        let mut out = vec![9f32; 256];
        dequantize(GgmlType::Tq1_0, &blk, 256, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == -1.0));
    }

    #[test]
    fn q1_0_sign_bits() {
        let mut blk = vec![0u8; 18];
        blk[0..2].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        blk[2] = 0b0000_0101;
        let mut out = vec![0f32; 128];
        dequantize(GgmlType::Q1_0, &blk, 128, &mut out).unwrap();
        assert_eq!(&out[..4], &[2.0, -2.0, 2.0, -2.0]);
    }

    #[test]
    fn ggml_nvfp4_sub_block_scales() {
        let mut blk = vec![0u8; 36];
        // ue4m3 0x38 = 1.0 → ×0.5; kvalues удвоены → итог = E2M1.
        blk[0] = 0x38;
        blk[4] = 0x21; // low nibble 1 → 0.5, high nibble 2 → 1.0
        let mut out = vec![0f32; 64];
        dequantize(GgmlType::Nvfp4, &blk, 64, &mut out).unwrap();
        assert_eq!(out[0], 0.5);
        assert_eq!(out[8], 1.0);
        assert_eq!(out[16], 0.0);
    }
}
