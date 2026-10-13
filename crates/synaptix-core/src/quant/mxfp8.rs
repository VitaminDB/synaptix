//! MXFP8 на хосте: строки natural-раскладки (`packed [n, k]` e4m3 +
//! `scales [n, k/32]` E8M0) — для выборки строк таблиц эмбеддингов из
//! квантованного бандла без выгрузки всей таблицы.

/// e4m3 (fn) → f32: знак, 4 бита порядка (смещение 7), 3 бита мантиссы;
/// `0x7f`/`0xff` — NaN.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0x0f) as i32;
    let m = (b & 0x07) as f32;
    if e == 0x0f && (b & 0x07) == 0x07 {
        return f32::NAN;
    }
    if e == 0 {
        s * m / 8.0 * 2f32.powi(-6)
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// Строка `row` в f32: `packed[row·k..]` e4m3 · 2^(E8M0 − 127) по блокам 32.
pub fn row_f32(packed: &[u8], scales: &[u8], k: usize, row: usize, dst: &mut [f32]) {
    let p = &packed[row * k..(row + 1) * k];
    let s = &scales[row * (k / 32)..(row + 1) * (k / 32)];
    for (j, v) in dst.iter_mut().enumerate().take(k) {
        let sc = f32::from_bits((s[j / 32] as u32) << 23);
        *v = e4m3_to_f32(p[j]) * sc;
    }
}
