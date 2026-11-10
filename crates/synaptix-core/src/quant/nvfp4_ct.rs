//! NVFP4 в раскладке compressed-tensors (чекпойнты llm-compressor/ModelOpt):
//! `weight_packed` `[n, k/2]` — два кода e2m1 на байт, младший полубайт —
//! чётный столбец; `weight_scale` `[n, k/16]` — e4m3 на блок 16;
//! `weight_global_scale` — число на тензор. Вес = e2m1 · scale / global.
//!
//! Это не `DType::NVFP4` движка (там шкалы тайлами и без общего множителя) —
//! только чтение чужого формата в f32.

const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// E4M3 → f32 (NaN-код в весах не встречается, даёт ноль).
pub fn e4m3_to_f32(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0x0f) as i32;
    let m = (b & 7) as f32;
    if e == 0 {
        s * m / 8.0 * 2f32.powi(-6)
    } else if e == 15 && (b & 7) == 7 {
        0.0
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// Строки `rows` матрицы `[n, k]` в `out` (`rows.len() · k` значений).
pub fn dequant_rows(packed: &[u8], scale: &[u8], global: f32, k: usize, rows: impl Iterator<Item = usize>, out: &mut [f32]) {
    let inv = 1.0 / global;
    for (i, r) in rows.enumerate() {
        let p = &packed[r * (k / 2)..(r + 1) * (k / 2)];
        let s = &scale[r * (k / 16)..(r + 1) * (k / 16)];
        let dst = &mut out[i * k..(i + 1) * k];
        for b in 0..k / 16 {
            let f = e4m3_to_f32(s[b]) * inv;
            for j in 0..8 {
                let byte = p[8 * b + j];
                dst[16 * b + 2 * j] = E2M1[(byte & 15) as usize] * f;
                dst[16 * b + 2 * j + 1] = E2M1[(byte >> 4) as usize] * f;
            }
        }
    }
}

/// Вся матрица `[n, k]`, строки параллельно.
pub fn dequant(packed: &[u8], scale: &[u8], global: f32, n: usize, k: usize) -> Vec<f32> {
    use rayon::prelude::*;
    assert_eq!(packed.len(), n * k / 2, "NVFP4: packed");
    assert_eq!(scale.len(), n * k / 16, "NVFP4: scale");
    let mut out = vec![0f32; n * k];
    out.par_chunks_mut(k).enumerate().for_each(|(r, row)| dequant_rows(packed, scale, global, k, std::iter::once(r), row));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nibble_order_and_scales() {
        // Одна строка, k = 16: коды 1 (0,5) и 9 (−0,5) в одном байте, шкала 2,0, общий 4.
        let mut packed = [0u8; 8];
        packed[0] = 0x91; // столбец 0 = код 1, столбец 1 = код 9
        packed[7] = 0x70; // столбец 15 = код 7 (6,0)
        let scale = [0x40u8]; // e=8, m=0 → 2,0
        let w = dequant(&packed, &scale, 4.0, 1, 16);
        assert_eq!(w[0], 0.25);
        assert_eq!(w[1], -0.25);
        assert_eq!(w[15], 3.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0);
    }
}
