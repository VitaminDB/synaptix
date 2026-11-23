#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
//! Префилл q8row на aarch64 с i8mm (SMMLA): плитка 8 строк активаций × 4 строки весов.
//!
//! Активации — int8 блоками по [`QK`] (как на декоде, та же точность), уложены парами строк
//! по 8 байт k (операнд SMMLA — матрица 2×8). Веса каждой задачи (полоса строк) один раз
//! перепаковываются так же парами и дальше читаются из L2 для всех строк активаций —
//! веса из памяти читаются один раз на матрицу, а не на каждые 4 токена, как в `dot4`.

use rayon::prelude::*;

use super::dequant::WeightView;
use super::q8dot::{quantize_act, QK};

/// Строк активаций в плитке и строк весов в полосе задачи.
const MT: usize = 8;

pub fn available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("i8mm")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Активации: пары строк `[пара][k/8][16 байт]`, масштабы `[пара][блок][s_i0, s_i0, s_i1, s_i1]`.
fn pack_act(row: &(dyn Fn(usize, &mut [f32]) + Sync), m: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    let mp = m.div_ceil(MT) * MT;
    let nb = k / QK;
    let mut xq = vec![0i8; mp * k];
    let mut xs = vec![0f32; mp / 2 * nb * 4];
    xq.par_chunks_mut(2 * k).zip(xs.par_chunks_mut(nb * 4)).enumerate().for_each_init(
        || (vec![0i8; 2 * k], vec![0f32; 2 * nb], vec![0f32; k]),
        |(rq, rs, rf), (p, (q, s))| {
            for r in 0..2 {
                let i = 2 * p + r;
                if i < m {
                    row(i, rf);
                    quantize_act(rf, &mut rq[r * k..(r + 1) * k], &mut rs[r * nb..(r + 1) * nb]);
                } else {
                    rq[r * k..(r + 1) * k].fill(0);
                    rs[r * nb..(r + 1) * nb].fill(0.0);
                }
            }
            for st in 0..k / 8 {
                q[st * 16..st * 16 + 8].copy_from_slice(&rq[st * 8..st * 8 + 8]);
                q[st * 16 + 8..st * 16 + 16].copy_from_slice(&rq[k + st * 8..k + st * 8 + 8]);
            }
            for b in 0..nb {
                s[b * 4..b * 4 + 4].copy_from_slice(&[rs[b], rs[b], rs[nb + b], rs[nb + b]]);
            }
        },
    );
    (xq, xs)
}

/// Полоса весов `[j0, j1)` парами строк (хвост до кратного 4 — нули) и шкалы строк.
/// Q8Row копируется как есть; прочие форматы деквантуются в F32 и квантуются в int8 по строке.
fn pack_w(view: &WeightView, j0: usize, j1: usize, buf: &mut Vec<i8>, ws: &mut Vec<f32>) {
    let k = view.k;
    let rows = (j1 - j0).div_ceil(4) * 4;
    buf.clear();
    buf.resize(rows * k, 0);
    ws.clear();
    ws.resize(rows, 0.0);
    let q8 = view.dtype == synaptix_core::dtype::DType::Q8Row;
    let mut rf = if q8 { Vec::new() } else { vec![0f32; k] };
    let mut rq = if q8 { Vec::new() } else { vec![0i8; k] };
    for pr in 0..rows / 2 {
        let dst = &mut buf[pr * 2 * k..(pr + 1) * 2 * k];
        for r in 0..2 {
            let j = j0 + 2 * pr + r;
            if j >= j1 {
                continue;
            }
            let src: &[i8] = if q8 {
                ws[2 * pr + r] = view.q8_scale(j);
                let b = view.row_bytes_of(j);
                // SAFETY: u8 → i8 той же ширины.
                unsafe { std::slice::from_raw_parts(b.as_ptr() as *const i8, k) }
            } else {
                view.row(j, &mut rf);
                let amax = rf.iter().fold(0f32, |a, v| a.max(v.abs()));
                let d = amax / 127.0;
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                for (q, &v) in rq.iter_mut().zip(&rf) {
                    *q = (v * inv).round() as i8;
                }
                ws[2 * pr + r] = d;
                &rq
            };
            for st in 0..k / 8 {
                dst[st * 16 + r * 8..st * 16 + r * 8 + 8].copy_from_slice(&src[st * 8..st * 8 + 8]);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn smmla(acc: std::arch::aarch64::int32x4_t, a: std::arch::aarch64::int8x16_t, b: std::arch::aarch64::int8x16_t) -> std::arch::aarch64::int32x4_t {
    let mut acc = acc;
    std::arch::asm!("smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b, options(pure, nomem, nostack));
    acc
}

/// Строка плиток 8 × полоса: `xp` — 4 пары активаций (шаг `2k`), `wp` — полоса весов
/// парами строк (`rows` кратно 4), `ws` — шкалы строк весов. Результат с шкалами —
/// в `ob[r * rows + j]` (r — строка плитки 0..8, j — столбец полосы).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
unsafe fn tile_row(xp: *const i8, xs: *const f32, wp: *const i8, ws: *const f32, k: usize, rows: usize, ob: *mut f32) {
    use std::arch::aarch64::*;
    let nb = k / QK;
    let k2 = 2 * k;
    let z = vdupq_n_s32(0);
    for jt in 0..rows / 4 {
        let wt = wp.add(jt * 4 * k);
        let mut f = [[vdupq_n_f32(0.0); 2]; 4];
        for b in 0..nb {
            let (mut c00, mut c01, mut c10, mut c11, mut c20, mut c21, mut c30, mut c31) = (z, z, z, z, z, z, z, z);
            for st in 0..QK / 8 {
                let o = (b * QK / 8 + st) * 16;
                let w0 = vld1q_s8(wt.add(o));
                let w1 = vld1q_s8(wt.add(k2 + o));
                let x0 = vld1q_s8(xp.add(o));
                let x1 = vld1q_s8(xp.add(k2 + o));
                let x2 = vld1q_s8(xp.add(2 * k2 + o));
                let x3 = vld1q_s8(xp.add(3 * k2 + o));
                c00 = smmla(c00, x0, w0);
                c01 = smmla(c01, x0, w1);
                c10 = smmla(c10, x1, w0);
                c11 = smmla(c11, x1, w1);
                c20 = smmla(c20, x2, w0);
                c21 = smmla(c21, x2, w1);
                c30 = smmla(c30, x3, w0);
                c31 = smmla(c31, x3, w1);
            }
            let s0 = vld1q_f32(xs.add(b * 4));
            let s1 = vld1q_f32(xs.add((nb + b) * 4));
            let s2 = vld1q_f32(xs.add((2 * nb + b) * 4));
            let s3 = vld1q_f32(xs.add((3 * nb + b) * 4));
            f[0][0] = vfmaq_f32(f[0][0], vcvtq_f32_s32(c00), s0);
            f[0][1] = vfmaq_f32(f[0][1], vcvtq_f32_s32(c01), s0);
            f[1][0] = vfmaq_f32(f[1][0], vcvtq_f32_s32(c10), s1);
            f[1][1] = vfmaq_f32(f[1][1], vcvtq_f32_s32(c11), s1);
            f[2][0] = vfmaq_f32(f[2][0], vcvtq_f32_s32(c20), s2);
            f[2][1] = vfmaq_f32(f[2][1], vcvtq_f32_s32(c21), s2);
            f[3][0] = vfmaq_f32(f[3][0], vcvtq_f32_s32(c30), s3);
            f[3][1] = vfmaq_f32(f[3][1], vcvtq_f32_s32(c31), s3);
        }
        // Результат SMMLA — 2×2 по строкам: [r0c0, r0c1, r1c0, r1c1].
        for q in 0..2 {
            let jb = jt * 4 + q * 2;
            let w2 = vld1_f32(ws.add(jb));
            let wv = vcombine_f32(w2, w2);
            for p in 0..4 {
                let v = vmulq_f32(f[p][q], wv);
                vst1_f32(ob.add((2 * p) * rows + jb), vget_low_f32(v));
                vst1_f32(ob.add((2 * p + 1) * rows + jb), vget_high_f32(v));
            }
        }
    }
}

/// `out[m, n] = x[m, k] · wᵀ` для q8row; `k % 32 == 0`. Только при [`available`].
pub fn gemm(view: &WeightView, xf: &[f32], m: usize, out: &mut [f32]) {
    let k = view.k;
    let n = view.n;
    let op = crate::gemm::blocked::SendPtr::new(out.as_mut_ptr());
    gemm_with(view, m, &|i, r| r.copy_from_slice(&xf[i * k..(i + 1) * k]), &|i, j0, v| {
        // SAFETY: каждый отрезок строки (i, [j0, j0 + len)) пишет одна задача.
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), op.ptr().add(i * n + j0), v.len()) }
    });
}

/// Как [`gemm`], но строки активаций читаются, а выход пишется замыканиями (любой тип в
/// памяти): `put_row(i, j0, v)` — значения `out[i, j0..j0 + v.len()]`.
pub fn gemm_with(view: &WeightView, m: usize, row: &(dyn Fn(usize, &mut [f32]) + Sync), put_row: &(dyn Fn(usize, usize, &[f32]) + Sync)) {
    #[cfg(target_arch = "aarch64")]
    {
        let (n, k) = (view.n, view.k);
        let nb = k / QK;
        crate::threads::pool();
        let (xq, xs) = synaptix_core::prof::span("q8mm.pack_act", || pack_act(row, m, k));
        let _t = std::time::Instant::now();
        let mp = m.div_ceil(MT) * MT;
        // Полоса весов задачи в L2: ~128 КБ.
        let band = ((128 << 10) / k).clamp(4, 64) / 4 * 4;
        let nbands = n.div_ceil(band);
        // Задач — с запасом на поток (big.LITTLE: мелкие задачи добирают быстрые ядра);
        // полос мало (n = 1280 при k = 1280 — 14 полос) — режем ещё и по строкам активаций.
        let tiles = mp / MT;
        let want = 8 * crate::common::threads();
        let msplit = want.div_ceil(nbands).clamp(1, tiles.max(1));
        let mchunk = tiles.div_ceil(msplit);
        let msplit = tiles.div_ceil(mchunk.max(1)).max(1);
        (0..nbands * msplit).into_par_iter().for_each_init(
            || (Vec::new(), Vec::new(), Vec::new()),
            |(wbuf, ws, ob), task| {
                let (bi, mi) = (task / msplit, task % msplit);
                let (j0, j1) = (bi * band, ((bi + 1) * band).min(n));
                pack_w(view, j0, j1, wbuf, ws);
                let rows = (j1 - j0).div_ceil(4) * 4;
                ob.clear();
                ob.resize(MT * rows, 0f32);
                for it in mi * mchunk..((mi + 1) * mchunk).min(tiles) {
                    // SAFETY: границы — по построению пачек (mp, rows кратны плиткам).
                    unsafe {
                        let xp = xq.as_ptr().add(it * MT * k);
                        let xsp = xs.as_ptr().add(it * (MT / 2) * nb * 4);
                        tile_row(xp, xsp, wbuf.as_ptr(), ws.as_ptr(), k, rows, ob.as_mut_ptr());
                    }
                    for r in 0..MT {
                        let i = it * MT + r;
                        if i < m {
                            put_row(i, j0, &ob[r * rows..r * rows + (j1 - j0)]);
                        }
                    }
                }
            },
        );
        if synaptix_core::prof::enabled() {
            synaptix_core::prof::add("q8mm.compute", _t.elapsed().as_secs_f64());
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (view, m, row, put_row);
        unreachable!("q8mm: только aarch64");
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use synaptix_core::dtype::DType;

    #[test]
    fn matches_dot_path() {
        if !available() {
            return;
        }
        let (m, n, k) = (13usize, 10usize, 96usize);
        let packed: Vec<u8> = (0..n * k).map(|i| ((i * 7919 % 251) as i32 - 125) as i8 as u8).collect();
        let scales: Vec<u8> = (0..n).flat_map(|j| (0.01f32 + j as f32 * 1e-3).to_le_bytes()).collect();
        let view = WeightView::new(DType::Q8Row, n, k, &packed, Some(&scales)).unwrap();
        let x: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.013).collect();
        let mut out = vec![0f32; m * n];
        gemm(&view, &x, m, &mut out);
        let nb = k / QK;
        for i in 0..m {
            let mut q = vec![0i8; k];
            let mut s = vec![0f32; nb];
            quantize_act(&x[i * k..(i + 1) * k], &mut q, &mut s);
            for j in 0..n {
                let r = super::super::q8dot::dot(&q, &s, view.row_bytes_of(j)) * view.q8_scale(j);
                assert!((out[i * n + j] - r).abs() <= 1e-4 * r.abs().max(1.0), "({i},{j}): {} vs {r}", out[i * n + j]);
            }
        }
    }
}
