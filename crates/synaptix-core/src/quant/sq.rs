//! SQ — собственный низкобитный формат весов synaptix (`DType::Sq { bits }`).
//!
//! Цель: любое число бит `b ∈ 1..=8`, единая распаковка на GPU без ветвлений по
//! формату, накладные расходы на шкалы как у K-квантов и никаких решёток.
//!
//! **Раскладка** — вдоль K, супер-блок 256 значений = 8 под-блоков по 32:
//!
//! | смещение | поле            | что это                                    |
//! |----------|-----------------|--------------------------------------------|
//! | 0        | `d: f16`        | шкала шкал                                 |
//! | 2        | `dmin: f16`     | шкала минимумов                            |
//! | 4        | `sub_scale[8]`  | `u8` на под-блок                           |
//! | 12       | `sub_min[8]`    | `u8` на под-блок                           |
//! | 20       | плоскости       | 8 под-блоков × `b` слов `u32` (LE)         |
//!
//! Под-блок `s` хранит 32 значения `q ∈ [0, 2^b − 1]` **битовыми плоскостями**:
//! слово `j` (0..b), бит `i` (0..32) = бит `j` значения `q[s·32 + i]`. Одна и та
//! же распаковка для любого `b`, и ни одно значение не пересекает границу
//! байта — в отличие от контиг-упаковки 3/5/6-битных.
//!
//! **Деквант** (контракт формата; CPU-эталон здесь, GPU — `blockq_dequant.cu`):
//! `dl = f32(d)·sub_scale[s]`, `ml = f32(dmin)·sub_min[s]`, `w = dl·q − ml`.
//! Асимметрично, минимумы неотрицательны (как у Q4_K: блок с положительным
//! минимумом получает `min = 0`). Все произведения — в f32 без слияния в FMA
//! (на GPU модуль собирается с `-fmad=false`), поэтому CPU и GPU дают один бит.
//!
//! **Размер**: `20 + 32·b` байт на 256 значений → `b + 0,625` бит на вес
//! (SQ4 = 4,625, SQ2 = 2,625). Требование `K % 32 == 0`; хвостовой супер-блок
//! неполный (лишние под-блоки — нули), но занимает полный размер.
//!
//! **Энкодер** здесь — эталонный RTN: min/max по под-блоку, шкалы округляются
//! в `u8`, значения — к ближайшему по уже округлённым шкалам. GPU-энкодер
//! (этап 4) обязан давать те же байты при тех же входах; перебор поправок
//! шкалы по MSE добавляется как отдельный режим.

use half::f16;

use crate::error::{Result, SynaptixError};

pub const SUPER_BLOCK: usize = 256;
pub const SUB_BLOCK: usize = 32;
pub const SUBS: usize = SUPER_BLOCK / SUB_BLOCK;
pub const HEADER_BYTES: usize = 20;

/// Байт в супер-блоке при `bits` битах на вес.
pub const fn super_block_bytes(bits: u8) -> usize {
    HEADER_BYTES + SUBS * bits as usize * 4
}

/// Байт на строку из `k` значений (`k % 32 == 0`; хвостовой супер-блок целиком).
pub const fn row_bytes(bits: u8, k: usize) -> usize {
    k.div_ceil(SUPER_BLOCK) * super_block_bytes(bits)
}

pub fn check_bits(bits: u8) -> Result<()> {
    if (1..=8).contains(&bits) {
        Ok(())
    } else {
        Err(SynaptixError::Unsupported("SQ: bits должен быть в 1..=8"))
    }
}

/// Деквант одного супер-блока: `out.len() == 256`.
pub fn dequant_super_block(bits: u8, blk: &[u8], out: &mut [f32]) {
    let b = bits as usize;
    let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
    let dmin = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
    for s in 0..SUBS {
        let dl = d * blk[4 + s] as f32;
        let ml = dmin * blk[12 + s] as f32;
        let base = HEADER_BYTES + s * b * 4;
        let mut planes = [0u32; 8];
        for (j, p) in planes.iter_mut().enumerate().take(b) {
            let o = base + j * 4;
            *p = u32::from_le_bytes([blk[o], blk[o + 1], blk[o + 2], blk[o + 3]]);
        }
        for i in 0..SUB_BLOCK {
            let mut q = 0u32;
            for (j, p) in planes.iter().enumerate().take(b) {
                q |= ((p >> i) & 1) << j;
            }
            out[s * SUB_BLOCK + i] = dl * q as f32 - ml;
        }
    }
}

/// Деквант строки из `k` значений (`src.len() >= row_bytes(bits, k)`).
pub fn dequant_row(bits: u8, src: &[u8], k: usize, dst: &mut [f32]) -> Result<()> {
    check_bits(bits)?;
    if k % SUB_BLOCK != 0 {
        return Err(SynaptixError::Unsupported("SQ: K должно быть кратно 32"));
    }
    let sbb = super_block_bytes(bits);
    let need = row_bytes(bits, k);
    if src.len() < need || dst.len() < k {
        return Err(SynaptixError::Unsupported("SQ dequant_row: буфер короче раскладки"));
    }
    let mut scratch = [0f32; SUPER_BLOCK];
    for (ib, blk) in src[..need].chunks_exact(sbb).enumerate() {
        let off = ib * SUPER_BLOCK;
        let take = (k - off).min(SUPER_BLOCK);
        dequant_super_block(bits, blk, &mut scratch);
        dst[off..off + take].copy_from_slice(&scratch[..take]);
    }
    Ok(())
}

/// Эталонный энкодер одного супер-блока: `x.len() <= 256` (хвост дополняется
/// нулями), `out.len() == super_block_bytes(bits)`.
pub fn quant_super_block(bits: u8, x: &[f32], out: &mut [u8]) {
    let b = bits as usize;
    let qmax = ((1u32 << b) - 1) as f32;
    let mut vals = [0f32; SUPER_BLOCK];
    vals[..x.len()].copy_from_slice(x);

    // Под-блочные шкалы и минимумы (минимум ≥ 0, как у Q4_K).
    let mut scale = [0f32; SUBS];
    let mut min = [0f32; SUBS];
    for s in 0..SUBS {
        let v = &vals[s * SUB_BLOCK..(s + 1) * SUB_BLOCK];
        let mut mn = v[0];
        let mut mx = v[0];
        for &t in v {
            mn = mn.min(t);
            mx = mx.max(t);
        }
        let mn = mn.min(0.0);
        scale[s] = (mx - mn) / qmax;
        min[s] = -mn;
    }
    let smax = scale.iter().cloned().fold(0f32, f32::max);
    let mmax = min.iter().cloned().fold(0f32, f32::max);
    let d16 = f16::from_f32(smax / 255.0);
    let dmin16 = f16::from_f32(mmax / 255.0);
    let d = d16.to_f32();
    let dmin = dmin16.to_f32();
    out.fill(0);
    out[0..2].copy_from_slice(&d16.to_le_bytes());
    out[2..4].copy_from_slice(&dmin16.to_le_bytes());
    for s in 0..SUBS {
        let sc = if d > 0.0 { (scale[s] / d).round().clamp(0.0, 255.0) as u8 } else { 0 };
        let mi = if dmin > 0.0 { (min[s] / dmin).round().clamp(0.0, 255.0) as u8 } else { 0 };
        out[4 + s] = sc;
        out[12 + s] = mi;
        let dl = d * sc as f32;
        let ml = dmin * mi as f32;
        let mut planes = [0u32; 8];
        for i in 0..SUB_BLOCK {
            let t = vals[s * SUB_BLOCK + i];
            let q = if dl > 0.0 { ((t + ml) / dl).round().clamp(0.0, qmax) as u32 } else { 0 };
            for (j, p) in planes.iter_mut().enumerate().take(b) {
                *p |= ((q >> j) & 1) << i;
            }
        }
        let base = HEADER_BYTES + s * b * 4;
        for (j, p) in planes.iter().enumerate().take(b) {
            out[base + j * 4..base + j * 4 + 4].copy_from_slice(&p.to_le_bytes());
        }
    }
}

/// Эталонный энкодер матрицы `[n, k]` (row-major f32) → блоб SQ.
pub fn quantize_matrix(bits: u8, x: &[f32], n: usize, k: usize) -> Result<Vec<u8>> {
    check_bits(bits)?;
    if k % SUB_BLOCK != 0 {
        return Err(SynaptixError::Unsupported("SQ: K должно быть кратно 32"));
    }
    if x.len() != n * k {
        return Err(SynaptixError::Unsupported("SQ quantize_matrix: x.len() != n*k"));
    }
    let sbb = super_block_bytes(bits);
    let rb = row_bytes(bits, k);
    let mut out = vec![0u8; n * rb];
    for r in 0..n {
        let row = &x[r * k..(r + 1) * k];
        for (ib, chunk) in row.chunks(SUPER_BLOCK).enumerate() {
            let o = r * rb + ib * sbb;
            quant_super_block(bits, chunk, &mut out[o..o + sbb]);
        }
    }
    Ok(out)
}

/// Деквант матрицы `[n, k]` из блоба SQ.
pub fn dequantize_matrix(bits: u8, blob: &[u8], n: usize, k: usize) -> Result<Vec<f32>> {
    check_bits(bits)?;
    let rb = row_bytes(bits, k);
    if blob.len() < n * rb {
        return Err(SynaptixError::Unsupported("SQ dequantize_matrix: блоб короче раскладки"));
    }
    let mut out = vec![0f32; n * k];
    for r in 0..n {
        dequant_row(bits, &blob[r * rb..(r + 1) * rb], k, &mut out[r * k..(r + 1) * k])?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(seed: u32, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn sizes() {
        assert_eq!(super_block_bytes(4), 148);
        assert_eq!(super_block_bytes(1), 52);
        assert_eq!(super_block_bytes(8), 276);
        assert_eq!(row_bytes(4, 256), 148);
        assert_eq!(row_bytes(4, 288), 296);
        assert!(check_bits(0).is_err() && check_bits(9).is_err());
    }

    #[test]
    fn bit_planes_pack_every_bit() {
        // Значения 0..=qmax по кругу: после квантования с известной шкалой
        // распаковка обязана вернуть их же.
        for bits in 1..=8u8 {
            let qmax = (1u32 << bits) - 1;
            let x: Vec<f32> = (0..256).map(|i| (i as u32 % (qmax + 1)) as f32).collect();
            let mut blk = vec![0u8; super_block_bytes(bits)];
            quant_super_block(bits, &x, &mut blk);
            let mut y = [0f32; 256];
            dequant_super_block(bits, &blk, &mut y);
            for i in 0..256 {
                assert!((y[i] - x[i]).abs() <= 0.5 + 0.02 * x[i].abs(), "bits={bits} i={i}: {} vs {}", y[i], x[i]);
            }
            // Старшие плоскости при bits<8 отсутствуют физически.
            assert_eq!(blk.len(), 20 + 32 * bits as usize);
        }
    }

    #[test]
    fn round_trip_error_shrinks_with_bits() {
        let x = det(7, 4 * 512);
        let mut prev = f32::INFINITY;
        for bits in [2u8, 3, 4, 6, 8] {
            let blob = quantize_matrix(bits, &x, 4, 512).unwrap();
            assert_eq!(blob.len(), 4 * row_bytes(bits, 512));
            let y = dequantize_matrix(bits, &blob, 4, 512).unwrap();
            let err: f32 = x.iter().zip(&y).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / x.len() as f32;
            assert!(err < prev, "bits={bits}: mse {err} не меньше {prev}");
            prev = err;
        }
        // 8 бит на диапазоне ±1: ошибка на уровне шага 2/255.
        assert!(prev < 1e-4, "mse@8bit = {prev}");
    }

    #[test]
    fn tail_super_block_is_partial() {
        let k = 288; // 256 + 32
        let x = det(3, 2 * k);
        let blob = quantize_matrix(4, &x, 2, k).unwrap();
        let y = dequantize_matrix(4, &blob, 2, k).unwrap();
        assert_eq!(y.len(), 2 * k);
        let err = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < 0.1, "max err {err}");
        assert!(quantize_matrix(4, &x[..2 * 40], 2, 40).is_err());
    }

    #[test]
    fn positive_block_gets_zero_min() {
        let x: Vec<f32> = (0..256).map(|i| 1.0 + (i % 7) as f32 * 0.1).collect();
        let mut blk = vec![0u8; super_block_bytes(4)];
        quant_super_block(4, &x, &mut blk);
        assert!(blk[12..20].iter().all(|&m| m == 0), "минимумы положительного блока = 0");
        let mut y = [0f32; 256];
        dequant_super_block(4, &blk, &mut y);
        assert!(y.iter().zip(&x).all(|(a, b)| (a - b).abs() < 0.08));
    }
}
