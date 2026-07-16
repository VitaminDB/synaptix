pub const FP8_E4M3_MAX: f32 = 448.0;

pub fn encode_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let mut v = x;
    if v > FP8_E4M3_MAX {
        v = FP8_E4M3_MAX;
    } else if v < -FP8_E4M3_MAX {
        v = -FP8_E4M3_MAX;
    }
    let sign_bit: u8 = if v.is_sign_negative() { 1 } else { 0 };
    let abs = v.abs();
    if abs == 0.0 {
        return sign_bit << 7;
    }

    let log2 = abs.log2();
    let exp_raw = log2.floor() as i32;
    let exp_biased = exp_raw + 7;

    if exp_biased < 1 {
        let scale = (1u32 << 9) as f32;
        let m = (abs * scale).round() as i32;
        let m_clamped = m.clamp(0, 7) as u8;
        if m_clamped == 0 {
            return sign_bit << 7;
        }
        return (sign_bit << 7) | m_clamped;
    }
    if exp_biased > 15 {
        return (sign_bit << 7) | (15 << 3) | 0b110;
    }

    let pow2_exp = (exp_raw as f32).exp2();
    let m_f = ((abs / pow2_exp) - 1.0) * 8.0;
    let mut m = m_f.round() as i32;
    let mut exp_biased = exp_biased;
    if m == 8 {
        m = 0;
        exp_biased += 1;
    }
    if exp_biased > 15 {
        return (sign_bit << 7) | (15 << 3) | 0b110;
    }
    if exp_biased == 15 && m == 7 {
        m = 6;
    }
    let m_bits = (m & 0x07) as u8;
    let exp_bits = (exp_biased as u8) & 0x0F;
    (sign_bit << 7) | (exp_bits << 3) | m_bits
}

pub fn decode_e4m3(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0_f32 } else { 1.0 };
    let exp_bits = ((byte >> 3) & 0x0F) as i32;
    let mantissa = (byte & 0x07) as i32;

    if exp_bits == 15 && mantissa == 7 {
        return f32::NAN;
    }

    if exp_bits == 0 {
        if mantissa == 0 {
            return 0.0 * sign;
        }
        let m = mantissa as f32;
        return sign * m * 2f32.powi(-9);
    }

    let exp_raw = exp_bits - 7;
    let frac = 1.0_f32 + (mantissa as f32) / 8.0;
    sign * frac * (exp_raw as f32).exp2()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_zero() {
        assert_eq!(encode_e4m3(0.0), 0x00);
        assert_eq!(decode_e4m3(0x00), 0.0);
        assert_eq!(decode_e4m3(0x80), -0.0);
    }

    #[test]
    fn encode_decode_one() {
        let b = encode_e4m3(1.0);
        assert_eq!(decode_e4m3(b), 1.0);
        let bn = encode_e4m3(-1.0);
        assert_eq!(decode_e4m3(bn), -1.0);
    }

    #[test]
    fn encode_decode_max() {
        let b = encode_e4m3(448.0);
        assert_eq!(decode_e4m3(b), 448.0);
        let b2 = encode_e4m3(1e10);
        assert_eq!(decode_e4m3(b2), 448.0);
        let b3 = encode_e4m3(-1e10);
        assert_eq!(decode_e4m3(b3), -448.0);
    }

    #[test]
    fn roundtrip_no_nan_pattern() {
        for byte in 0u8..=255u8 {
            let v = decode_e4m3(byte);
            if v.is_nan() {
                assert!(byte == 0x7F || byte == 0xFF, "Unexpected NaN at byte 0x{byte:X}");
                continue;
            }
            let back = encode_e4m3(v);
            let v2 = decode_e4m3(back);
            assert_eq!(v, v2, "Round-trip failed for byte 0x{byte:X} (v={v})");
        }
    }

    #[test]
    fn encode_does_not_produce_nan_for_finite() {
        for x in [0.0_f32, 1.0, -1.0, 2.5, -3.7, 100.0, 447.9, 448.0, 449.0, 1e10, -1e10, 0.001, 1e-5] {
            let b = encode_e4m3(x);
            assert!(b != 0x7F && b != 0xFF, "x={x} produced NaN byte 0x{b:X}");
        }
    }

    #[test]
    fn subnormal_handling() {
        let v = 0.001_f32;
        let b = encode_e4m3(v);
        let back = decode_e4m3(b);
        assert!((back - v).abs() < 0.005, "subnormal too far: v={v}, back={back}");
    }
}
