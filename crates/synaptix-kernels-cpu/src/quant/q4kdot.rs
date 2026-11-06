//! Q4_K (ggml: суперблок 256 = f16 `d`, f16 `dmin`, 12 байт 6-битных шкал/минимумов восьми подблоков
//! по 32, 128 байт полубайтов) × активации int8 блоками по 32 (как [`super::q8dot`]): вес
//! `d·sc·q − dmin·m`, поэтому `Σ w·x = d·sc·xs·Σ q·xq − dmin·m·Σx`; `Σx` блока — `xsum`
//! (из квантованных активаций). Целочисленная часть — NEON dotprod.

use super::q8dot::QK;
use synaptix_core::quant::ggml::q4k_scale_min;

pub const QK_K: usize = 256;
pub const BLOCK_BYTES: usize = 144;

#[inline]
fn f16_at(b: &[u8], o: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f32()
}

/// Суммы блоков активаций: `xsum[b] = xs[b]·Σ xq` (по каждому блоку 32).
pub fn block_sums(xq: &[i8], xs: &[f32], out: &mut [f32]) {
    for (b, o) in out.iter_mut().enumerate() {
        let s: i32 = xq[b * QK..(b + 1) * QK].iter().map(|&v| v as i32).sum();
        *o = s as f32 * xs[b];
    }
}

fn dot_scalar(xq: &[i8], xs: &[f32], xsum: &[f32], w: &[u8]) -> f32 {
    let nsb = xq.len() / QK_K;
    let mut acc = 0f32;
    for sb in 0..nsb {
        let blk = &w[sb * BLOCK_BYTES..(sb + 1) * BLOCK_BYTES];
        let (d, dmin) = (f16_at(blk, 0), f16_at(blk, 2));
        let sc = &blk[4..16];
        let qs = &blk[16..];
        for c in 0..4 {
            for half_ in 0..2 {
                let j = 2 * c + half_;
                let b = sb * 8 + j;
                let (s, m) = q4k_scale_min(j, sc);
                let x = &xq[b * QK..(b + 1) * QK];
                let mut isum = 0i32;
                for t in 0..32 {
                    let q = if half_ == 0 { qs[c * 32 + t] & 0x0F } else { qs[c * 32 + t] >> 4 };
                    isum += q as i32 * x[t] as i32;
                }
                acc += d * s as f32 * xs[b] * isum as f32 - dmin * m as f32 * xsum[b];
            }
        }
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn sdot(acc: std::arch::aarch64::int32x4_t, a: std::arch::aarch64::int8x16_t, b: std::arch::aarch64::int8x16_t) -> std::arch::aarch64::int32x4_t {
    let mut acc = acc;
    std::arch::asm!("sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b, options(pure, nomem, nostack));
    acc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_dotprod(xq: &[i8], xs: &[f32], xsum: &[f32], w: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    let nsb = xq.len() / QK_K;
    let xp = xq.as_ptr();
    let m4 = vdupq_n_u8(0x0F);
    let z = vdupq_n_s32(0);
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut mins = 0f32;
    for sb in 0..nsb {
        let blk = w.as_ptr().add(sb * BLOCK_BYTES);
        let bs = std::slice::from_raw_parts(blk, BLOCK_BYTES);
        let (d, dmin) = (f16_at(bs, 0), f16_at(bs, 2));
        let sc = &bs[4..16];
        let qs = blk.add(16);
        for c in 0..4 {
            let v0 = vld1q_u8(qs.add(c * 32));
            let v1 = vld1q_u8(qs.add(c * 32 + 16));
            let l0 = vreinterpretq_s8_u8(vandq_u8(v0, m4));
            let l1 = vreinterpretq_s8_u8(vandq_u8(v1, m4));
            let h0 = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));
            let h1 = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));
            let b0 = sb * 8 + 2 * c;
            let x0 = xp.add(b0 * QK);
            let x1 = xp.add((b0 + 1) * QK);
            let s0 = sdot(sdot(z, l0, vld1q_s8(x0)), l1, vld1q_s8(x0.add(16)));
            let s1 = sdot(sdot(z, h0, vld1q_s8(x1)), h1, vld1q_s8(x1.add(16)));
            let (sc0, m0) = q4k_scale_min(2 * c, sc);
            let (sc1, m1) = q4k_scale_min(2 * c + 1, sc);
            acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(s0), d * sc0 as f32 * *xs.get_unchecked(b0));
            acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(s1), d * sc1 as f32 * *xs.get_unchecked(b0 + 1));
            mins += dmin * (m0 as f32 * *xsum.get_unchecked(b0) + m1 as f32 * *xsum.get_unchecked(b0 + 1));
        }
    }
    vaddvq_f32(vaddq_f32(acc0, acc1)) - mins
}

/// Строка Q4_K (`k/256` суперблоков) · активации (`xq`, `xs`, `xsum` по блокам 32).
pub fn dot(xq: &[i8], xs: &[f32], xsum: &[f32], w: &[u8]) -> f32 {
    debug_assert!(xq.len() % QK_K == 0 && w.len() >= xq.len() / QK_K * BLOCK_BYTES);
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { dot_dotprod(xq, xs, xsum, w) };
        }
    }
    dot_scalar(xq, xs, xsum, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::q8dot::quantize_act;

    #[test]
    fn dot_matches_dequant() {
        let k = 512;
        let w: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.71).sin() * 0.04) + 0.01).collect();
        let q = synaptix_core::quant::ggml::quantize_q4_k(&w, 1, k, None);
        let mut wd = vec![0f32; k];
        synaptix_core::quant::ggml_dequant::dequantize(synaptix_core::quant::ggml::GgmlType::Q4K, &q, k, &mut wd).unwrap();
        let x: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.37).cos()) * 0.8).collect();
        let nb = k / QK;
        let (mut xq, mut xs, mut xsum) = (vec![0i8; k], vec![0f32; nb], vec![0f32; nb]);
        quantize_act(&x, &mut xq, &mut xs);
        block_sums(&xq, &xs, &mut xsum);
        // эталон: деквантованные веса × деквантованные активации
        let xd: Vec<f32> = (0..k).map(|i| xq[i] as f32 * xs[i / QK]).collect();
        let r: f32 = wd.iter().zip(&xd).map(|(a, b)| a * b).sum();
        let a = dot(&xq, &xs, &xsum, &q);
        let s = dot_scalar(&xq, &xs, &xsum, &q);
        assert!((a - r).abs() < 1e-3 * r.abs().max(1.0), "{a} vs {r}");
        assert!((s - r).abs() < 1e-3 * r.abs().max(1.0), "{s} vs {r}");
    }
}
