//! MXFP8 (Blackwell-нативный block-scale FP8) E8M0 codec — бит-идентично CUDA
//! `mxfp8_quant.cu` (per-32-block: один E8M0 scale-байт + E4M3 mantissa на 32
//! элемента вдоль оси). Используется для KV-кеша `--kv-dtype mxfp8`.

/// Размер MXFP8-блока: 32 элемента делят один E8M0 scale-байт.
pub const MXFP8_BLOCK: usize = 32;

/// E8M0 scale-байт из amax блока: `exp(amax)/256` → старший байт экспоненты.
/// Деление на 256 (а не 448) даёт `amax/sv ≈ 256 < 448` (запас в E4M3) —
/// бит-идентично `mxfp8_quant.cu:30-32` и остальному MXFP8-стеку.
pub fn e8m0_scale_byte(amax: f32) -> u8 {
    let exp_bits = amax.to_bits() & 0x7F80_0000;
    let scale_f = f32::from_bits(exp_bits) / 256.0;
    (scale_f.to_bits() >> 23) as u8
}

/// E8M0 decode: байт `b` → `2^(b-127)` (mantissa=0), пол `1e-12`.
/// Бит-идентично `mxfp8_quant.cu:33/88` (`uint_as_float(b<<23)`).
pub fn e8m0_decode(sbyte: u8) -> f32 {
    f32::from_bits((sbyte as u32) << 23).max(1e-12)
}
