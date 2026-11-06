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

/// Шкалы и минимумы восьми подблоков суперблока, уже умноженные на `d` и `dmin`.
#[inline(always)]
pub(crate) fn unpack_scales(d: f32, dmin: f32, q: &[u8], dsc: &mut [f32; 8], dmn: &mut [f32; 8]) {
    for j in 0..4 {
        dsc[j] = d * (q[j] & 63) as f32;
        dmn[j] = dmin * (q[j + 4] & 63) as f32;
    }
    for j in 4..8 {
        dsc[j] = d * ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)) as f32;
        dmn[j] = dmin * ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as f32;
    }
}

/// Свести 8 сумм подблоков (`int32x4` каждая) и добавить к аккумулятору:
/// `acc += Σ_j sum(s[j]) · dsc[j] · xs[j]` векторно (vpaddq вместо 8 горизонтальных сумм).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn fold8(acc: std::arch::aarch64::float32x4_t, s: &[std::arch::aarch64::int32x4_t; 8], dsc: &[f32; 8], xs: *const f32) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let q0 = vpaddq_s32(vpaddq_s32(s[0], s[1]), vpaddq_s32(s[2], s[3]));
    let q1 = vpaddq_s32(vpaddq_s32(s[4], s[5]), vpaddq_s32(s[6], s[7]));
    let f0 = vmulq_f32(vld1q_f32(dsc.as_ptr()), vld1q_f32(xs));
    let f1 = vmulq_f32(vld1q_f32(dsc.as_ptr().add(4)), vld1q_f32(xs.add(4)));
    vfmaq_f32(vfmaq_f32(acc, vcvtq_f32_s32(q0), f0), vcvtq_f32_s32(q1), f1)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_dotprod(xq: &[i8], xs: &[f32], xsum: &[f32], w: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    let nsb = xq.len() / QK_K;
    let xp = xq.as_ptr();
    let m4 = vdupq_n_u8(0x0F);
    let z = vdupq_n_s32(0);
    let mut acc = vdupq_n_f32(0.0);
    let mut macc = vdupq_n_f32(0.0);
    let (mut dsc, mut dmn) = ([0f32; 8], [0f32; 8]);
    let mut s = [z; 8];
    for sb in 0..nsb {
        let blk = w.as_ptr().add(sb * BLOCK_BYTES);
        let bs = std::slice::from_raw_parts(blk, BLOCK_BYTES);
        unpack_scales(f16_at(bs, 0), f16_at(bs, 2), &bs[4..16], &mut dsc, &mut dmn);
        let qs = blk.add(16);
        let xb = xp.add(sb * QK_K);
        for c in 0..4 {
            let v0 = vld1q_u8(qs.add(c * 32));
            let v1 = vld1q_u8(qs.add(c * 32 + 16));
            let x0 = xb.add(c * 64);
            s[2 * c] = sdot(sdot(z, vreinterpretq_s8_u8(vandq_u8(v0, m4)), vld1q_s8(x0)), vreinterpretq_s8_u8(vandq_u8(v1, m4)), vld1q_s8(x0.add(16)));
            s[2 * c + 1] = sdot(sdot(z, vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), vld1q_s8(x0.add(32))), vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)), vld1q_s8(x0.add(48)));
        }
        let o = sb * 8;
        acc = fold8(acc, &s, &dsc, xs.as_ptr().add(o));
        macc = vfmaq_f32(macc, vld1q_f32(dmn.as_ptr()), vld1q_f32(xsum.as_ptr().add(o)));
        macc = vfmaq_f32(macc, vld1q_f32(dmn.as_ptr().add(4)), vld1q_f32(xsum.as_ptr().add(o + 4)));
    }
    vaddvq_f32(acc) - vaddvq_f32(macc)
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
