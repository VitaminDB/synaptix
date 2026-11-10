//! Кодировщики низкобитных форматов ggml: IQ2_XXS (2,06 бита на вес) и
//! IQ1_S (1,56). Раскладка блоков — штатная ggml (см. `ggml_dequant`), так
//! что результат читает любой декодер этих форматов.
//!
//! Это эталон на CPU с полным перебором сетки: медленный, зато без
//! эвристик поиска соседей. Вес ошибки — как у IQ3_S-128: важность столбца
//! (E[x²] входа), смягчённая модулем веса.

use super::ggml::{GgmlType, QK_K};
use super::ggml_tables::{IQ1S_GRID, IQ2XXS_GRID, KSIGNS_IQ2XS};

fn iq2xxs_grid() -> &'static [[f32; 8]; 256] {
    static G: std::sync::OnceLock<[[f32; 8]; 256]> = std::sync::OnceLock::new();
    G.get_or_init(|| {
        let mut g = [[0f32; 8]; 256];
        for (i, v) in IQ2XXS_GRID.iter().enumerate() {
            let b = v.to_le_bytes();
            for j in 0..8 {
                g[i][j] = b[j] as f32;
            }
        }
        g
    })
}

fn iq1s_grid() -> &'static Vec<[f32; 8]> {
    static G: std::sync::OnceLock<Vec<[f32; 8]>> = std::sync::OnceLock::new();
    G.get_or_init(|| {
        IQ1S_GRID
            .iter()
            .map(|v| {
                let b = v.to_le_bytes();
                let mut g = [0f32; 8];
                for j in 0..8 {
                    g[j] = b[j] as i8 as f32;
                }
                g
            })
            .collect()
    })
}

/// Ближайшая точка сетки к `v/s − shift` с весами `w`: индекс и ошибка в
/// исходном масштабе (`Σ w·(v − s·(g + shift))²`).
#[inline]
fn nearest(grid: &[[f32; 8]], v: &[f32], w: &[f32], s: f32, shift: f32) -> (usize, f32) {
    let inv = 1.0 / s;
    let mut t = [0f32; 8];
    for j in 0..8 {
        t[j] = v[j] * inv - shift;
    }
    let mut best = 0usize;
    let mut be = f32::INFINITY;
    for (i, p) in grid.iter().enumerate() {
        let mut e = 0f32;
        for j in 0..8 {
            let d = t[j] - p[j];
            e += w[j] * d * d;
        }
        if e < be {
            be = e;
            best = i;
        }
    }
    (best, be * s * s)
}

fn weights(x: &[f32], imp: Option<&[f32]>) -> Vec<f32> {
    let sigma2 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    (0..x.len()).map(|i| imp.map(|h| h[i]).unwrap_or(1.0) * (sigma2 + x[i] * x[i]).sqrt()).collect()
}

/// Под-блок 32 при масштабе `s`: индексы 4 групп и ошибка.
fn sub_codes(grid: &[[f32; 8]], v: &[f32], w: &[f32], s: f32, shift: f32, idx: &mut [u16; 4]) -> f32 {
    let mut err = 0f32;
    for g in 0..4 {
        let (b, e) = nearest(grid, &v[8 * g..8 * g + 8], &w[8 * g..8 * g + 8], s, shift);
        idx[g] = b as u16;
        err += e;
    }
    err
}

/// Блок IQ2_XXS: 256 значений → 66 байт.
pub fn quantize_iq2_xxs_block(x: &[f32], imp: Option<&[f32]>, out: &mut [u8]) {
    debug_assert_eq!(x.len(), QK_K);
    out.fill(0);
    let grid = iq2xxs_grid();
    let w = weights(x, imp);
    // Знаки: 7 бит на группу, восьмой — чётность. При нечётном числе минусов
    // переворачивается знак самого дешёвого элемента, и его «модуль» уходит
    // в минус — сетка положительна, ошибка там неизбежна, но мала.
    let mut v = vec![0f32; QK_K];
    let mut signs = [0u8; 32];
    for g in 0..32 {
        let mut mask = 0u8;
        for j in 0..8 {
            if x[8 * g + j] < 0.0 {
                mask |= 1 << j;
            }
        }
        if mask.count_ones() % 2 == 1 {
            let j = (0..8)
                .min_by(|a, b| {
                    let (ia, ib) = (8 * g + a, 8 * g + b);
                    (w[ia] * x[ia] * x[ia]).total_cmp(&(w[ib] * x[ib] * x[ib]))
                })
                .unwrap();
            mask ^= 1 << j;
        }
        for j in 0..8 {
            let i = 8 * g + j;
            v[i] = if mask & (1 << j) != 0 { -x[i] } else { x[i] };
        }
        debug_assert_eq!(KSIGNS_IQ2XS[(mask & 127) as usize], mask);
        signs[g] = mask & 127;
    }
    // 1) лучший вещественный масштаб под-блока.
    let mut best_s = [0f32; 8];
    let mut idx = [0u16; 4];
    for sb in 0..8 {
        let r = 32 * sb..32 * sb + 32;
        let amax = v[r.clone()].iter().fold(0f32, |m, a| m.max(a.abs()));
        if amax == 0.0 {
            continue;
        }
        let mut be = f32::INFINITY;
        for c in 0..19 {
            let s = amax / 43.0 * (0.45 + 0.05 * c as f32);
            let e = sub_codes(grid, &v[r.clone()], &w[r.clone()], s, 0.0, &mut idx);
            if e < be {
                be = e;
                best_s[sb] = s;
            }
        }
    }
    let smax = best_s.iter().copied().fold(0f32, f32::max);
    if smax == 0.0 {
        return;
    }
    // 2) общий d и 4-битные шкалы: s = d·(2·sc + 1)/8.
    let mut best: Option<(f32, f32, [u8; 8], [[u16; 4]; 8])> = None;
    for c in 0..9 {
        let d = half::f16::from_f32(8.0 * smax / 31.0 * (0.9 + 0.025 * c as f32)).to_f32();
        if d <= 0.0 {
            continue;
        }
        let mut sc = [0u8; 8];
        let mut codes = [[0u16; 4]; 8];
        let mut tot = 0f32;
        for sb in 0..8 {
            let r = 32 * sb..32 * sb + 32;
            let q = ((8.0 * best_s[sb] / d - 1.0) / 2.0).round().clamp(0.0, 15.0) as u8;
            let mut be = f32::INFINITY;
            for qq in [q.saturating_sub(1), q, (q + 1).min(15)] {
                let s = d * (2 * qq as i32 + 1) as f32 / 8.0;
                let e = sub_codes(grid, &v[r.clone()], &w[r.clone()], s, 0.0, &mut idx);
                if e < be {
                    be = e;
                    sc[sb] = qq;
                    codes[sb] = idx;
                }
            }
            tot += be;
        }
        if best.as_ref().is_none_or(|b| tot < b.0) {
            best = Some((tot, d, sc, codes));
        }
    }
    let (_, d, sc, codes) = best.expect("хотя бы один d");
    out[0..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    for sb in 0..8 {
        let aux0 = u32::from_le_bytes([codes[sb][0] as u8, codes[sb][1] as u8, codes[sb][2] as u8, codes[sb][3] as u8]);
        let mut aux1 = (sc[sb] as u32) << 28;
        for l in 0..4 {
            aux1 |= (signs[4 * sb + l] as u32) << (7 * l);
        }
        out[2 + 8 * sb..6 + 8 * sb].copy_from_slice(&aux0.to_le_bytes());
        out[6 + 8 * sb..10 + 8 * sb].copy_from_slice(&aux1.to_le_bytes());
    }
}

/// Блок IQ1_S: 256 значений → 50 байт.
pub fn quantize_iq1_s_block(x: &[f32], imp: Option<&[f32]>, out: &mut [u8]) {
    debug_assert_eq!(x.len(), QK_K);
    out.fill(0);
    let grid = iq1s_grid();
    let w = weights(x, imp);
    const DELTA: f32 = 0.125;
    // 1) лучший вещественный масштаб и знак сдвига под-блока.
    let mut best_s = [0f32; 8];
    let mut neg = [false; 8];
    let mut idx = [0u16; 4];
    for sb in 0..8 {
        let r = 32 * sb..32 * sb + 32;
        let amax = x[r.clone()].iter().fold(0f32, |m, a| m.max(a.abs()));
        if amax == 0.0 {
            continue;
        }
        let mut be = f32::INFINITY;
        for minus in [false, true] {
            let shift = if minus { -DELTA } else { DELTA };
            for c in 0..12 {
                let s = amax * (0.2 + 0.06 * c as f32);
                let e = sub_codes(grid, &x[r.clone()], &w[r.clone()], s, shift, &mut idx);
                if e < be {
                    be = e;
                    best_s[sb] = s;
                    neg[sb] = minus;
                }
            }
        }
    }
    let smax = best_s.iter().copied().fold(0f32, f32::max);
    if smax == 0.0 {
        return;
    }
    // 2) общий d и 3-битные шкалы: s = d·(2·sc + 1).
    let mut best: Option<(f32, f32, [u8; 8], [[u16; 4]; 8])> = None;
    for c in 0..7 {
        let d = half::f16::from_f32(smax / 15.0 * (0.9 + 0.035 * c as f32)).to_f32();
        if d <= 0.0 {
            continue;
        }
        let mut sc = [0u8; 8];
        let mut codes = [[0u16; 4]; 8];
        let mut tot = 0f32;
        for sb in 0..8 {
            let r = 32 * sb..32 * sb + 32;
            let shift = if neg[sb] { -DELTA } else { DELTA };
            let q = ((best_s[sb] / d - 1.0) / 2.0).round().clamp(0.0, 7.0) as u8;
            let mut be = f32::INFINITY;
            for qq in [q.saturating_sub(1), q, (q + 1).min(7)] {
                let s = d * (2 * qq as i32 + 1) as f32;
                let e = sub_codes(grid, &x[r.clone()], &w[r.clone()], s, shift, &mut idx);
                if e < be {
                    be = e;
                    sc[sb] = qq;
                    codes[sb] = idx;
                }
            }
            tot += be;
        }
        if best.as_ref().is_none_or(|b| tot < b.0) {
            best = Some((tot, d, sc, codes));
        }
    }
    let (_, d, sc, codes) = best.expect("хотя бы один d");
    out[0..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    for sb in 0..8 {
        let mut h = (sc[sb] as u16) << 12;
        if neg[sb] {
            h |= 0x8000;
        }
        for l in 0..4 {
            out[2 + 4 * sb + l] = codes[sb][l] as u8;
            h |= (codes[sb][l] >> 8) << (3 * l);
        }
        out[34 + 2 * sb..36 + 2 * sb].copy_from_slice(&h.to_le_bytes());
    }
}

/// Матрица `[n, k]` (`k % 256 == 0`) в IQ2_XXS или IQ1_S; `imp` — важность
/// столбцов длины `k`.
pub fn quantize_lowbit(ty: GgmlType, x: &[f32], n: usize, k: usize, imp: Option<&[f32]>) -> Vec<u8> {
    use rayon::prelude::*;
    assert!(k % QK_K == 0, "{}: K={k} не кратно {QK_K}", ty.name());
    let f: fn(&[f32], Option<&[f32]>, &mut [u8]) = match ty {
        GgmlType::Iq2Xxs => quantize_iq2_xxs_block,
        GgmlType::Iq1S => quantize_iq1_s_block,
        _ => panic!("quantize_lowbit: {}", ty.name()),
    };
    let bb = ty.bytes_for(QK_K);
    let rb = k / QK_K * bb;
    let mut out = vec![0u8; n * rb];
    out.par_chunks_mut(rb).enumerate().for_each(|(r, row)| {
        for b in 0..k / QK_K {
            let h = imp.map(|h| &h[QK_K * b..QK_K * (b + 1)]);
            f(&x[r * k + QK_K * b..r * k + QK_K * (b + 1)], h, &mut row[b * bb..(b + 1) * bb]);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss(n: usize) -> Vec<f32> {
        let mut s = 11u64;
        (0..n)
            .map(|_| {
                let mut acc = 0f32;
                for _ in 0..6 {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    acc += (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                }
                acc * 0.02
            })
            .collect()
    }

    fn rel_err(ty: GgmlType, x: &[f32], n: usize, k: usize) -> f32 {
        let q = quantize_lowbit(ty, x, n, k, None);
        assert_eq!(q.len(), ty.bytes_for(k) * n);
        let mut y = vec![0f32; n * k];
        crate::quant::ggml_dequant::dequantize(ty, &q, n * k, &mut y).unwrap();
        let num: f32 = x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum();
        (num / x.iter().map(|a| a * a).sum::<f32>()).sqrt()
    }

    /// Ошибка округления на нормальном шуме: 2 бита — около трети, 1,56 —
    /// около половины; порог ловит поломку раскладки (тогда ошибка ≥ 1).
    #[test]
    fn lowbit_roundtrip_error() {
        let (n, k) = (8usize, 768usize);
        let x = gauss(n * k);
        let e2 = rel_err(GgmlType::Iq2Xxs, &x, n, k);
        let e1 = rel_err(GgmlType::Iq1S, &x, n, k);
        eprintln!("IQ2_XXS {e2:.4}, IQ1_S {e1:.4}");
        assert!(e2 < 0.40, "IQ2_XXS {e2}");
        assert!(e1 < 0.62 && e1 > e2, "IQ1_S {e1}");
    }
}
