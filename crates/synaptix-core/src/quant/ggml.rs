//! Типы тензоров ggml (GGUF) — идентификаторы, геометрия блоков, имена.
//!
//! Это **формат исполнения моделей llama.cpp**: движок читает блоки байт в
//! байт как в файле (один непрерывный блоб на тензор, шкалы внутри блока),
//! деквантует их на карте и умножает. Энкодеров у ggml-форматов в synaptix
//! нет — файлы берутся готовыми (решение Р1 плана
//! `quant_lowbit_gguf_plan_2026.md`); собственный квант движка — [`super::sq`].
//!
//! Номера — `enum ggml_type` из `ggml.h`; удалённые типы (4, 5, 31–33, 36–38)
//! не представлены. Размеры блоков — `GGML_QUANT_SIZES` из gguf-py и
//! `static_assert`'ы `ggml-common.h`.

use serde::{Deserialize, Serialize};

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    BF16 = 30,
    Tq1_0 = 34,
    Tq2_0 = 35,
    Mxfp4 = 39,
    /// NVFP4 в раскладке ggml: блок 64 = 4 под-блока по 16 с UE4M3-шкалами
    /// внутри блока. Это не `DType::NVFP4` движка (там шкалы отдельно и
    /// тайл-мажорно) — только формат файла.
    Nvfp4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
}

impl GgmlType {
    pub const ALL: [GgmlType; 35] = {
        use GgmlType::*;
        [
            F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K, Iq2Xxs,
            Iq2Xs, Iq3Xxs, Iq1S, Iq4Nl, Iq3S, Iq2S, Iq4Xs, I8, I16, I32, I64, F64, Iq1M, BF16,
            Tq1_0, Tq2_0, Mxfp4, Nvfp4, Q1_0, Q2_0,
        ]
    };

    pub fn from_u32(v: u32) -> Option<Self> {
        use GgmlType::*;
        Some(match v {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            15 => Q8K,
            16 => Iq2Xxs,
            17 => Iq2Xs,
            18 => Iq3Xxs,
            19 => Iq1S,
            20 => Iq4Nl,
            21 => Iq3S,
            22 => Iq2S,
            23 => Iq4Xs,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => Iq1M,
            30 => BF16,
            34 => Tq1_0,
            35 => Tq2_0,
            39 => Mxfp4,
            40 => Nvfp4,
            41 => Q1_0,
            42 => Q2_0,
            _ => return None,
        })
    }

    /// Имя как в ggml (`GGML_TYPE_*` без префикса).
    pub const fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "F32",
            F16 => "F16",
            Q4_0 => "Q4_0",
            Q4_1 => "Q4_1",
            Q5_0 => "Q5_0",
            Q5_1 => "Q5_1",
            Q8_0 => "Q8_0",
            Q8_1 => "Q8_1",
            Q2K => "Q2_K",
            Q3K => "Q3_K",
            Q4K => "Q4_K",
            Q5K => "Q5_K",
            Q6K => "Q6_K",
            Q8K => "Q8_K",
            Iq2Xxs => "IQ2_XXS",
            Iq2Xs => "IQ2_XS",
            Iq3Xxs => "IQ3_XXS",
            Iq1S => "IQ1_S",
            Iq4Nl => "IQ4_NL",
            Iq3S => "IQ3_S",
            Iq2S => "IQ2_S",
            Iq4Xs => "IQ4_XS",
            I8 => "I8",
            I16 => "I16",
            I32 => "I32",
            I64 => "I64",
            F64 => "F64",
            Iq1M => "IQ1_M",
            BF16 => "BF16",
            Tq1_0 => "TQ1_0",
            Tq2_0 => "TQ2_0",
            Mxfp4 => "MXFP4",
            Nvfp4 => "NVFP4",
            Q1_0 => "Q1_0",
            Q2_0 => "Q2_0",
        }
    }

    /// Машинный ключ формата (манифест бандла, CLI): имя в нижнем регистре.
    pub fn key(self) -> String {
        self.name().to_ascii_lowercase()
    }

    pub fn from_key(s: &str) -> Option<Self> {
        let up = s.trim().to_ascii_uppercase();
        Self::ALL.iter().copied().find(|t| t.name() == up)
    }

    /// Элементов в блоке.
    pub const fn block_elems(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | F16 | BF16 | F64 | I8 | I16 | I32 | I64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 | Iq4Nl | Mxfp4 => 32,
            Nvfp4 => 64,
            Q2_0 => 64,
            Q1_0 => 128,
            _ => QK_K,
        }
    }

    /// Байт в блоке.
    pub const fn block_bytes(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | I32 => 4,
            F16 | BF16 | I16 => 2,
            F64 | I64 => 8,
            I8 => 1,
            Q4_0 => 2 + 16,
            Q4_1 => 4 + 16,
            Q5_0 => 2 + 4 + 16,
            Q5_1 => 4 + 4 + 16,
            Q8_0 => 2 + 32,
            Q8_1 => 4 + 32,
            Q2K => QK_K / 16 + QK_K / 4 + 4,
            Q3K => QK_K / 8 + QK_K / 4 + 12 + 2,
            Q4K => 4 + K_SCALE_SIZE + QK_K / 2,
            Q5K => 4 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2,
            Q6K => QK_K / 2 + QK_K / 4 + QK_K / 16 + 2,
            Q8K => 4 + QK_K + QK_K / 16 * 2,
            Iq2Xxs => 2 + QK_K / 8 * 2,
            Iq2Xs => 2 + QK_K / 8 * 2 + QK_K / 32,
            Iq2S => 2 + QK_K / 4 + QK_K / 16,
            Iq3Xxs => 2 + 3 * QK_K / 8,
            Iq3S => 2 + 13 * QK_K / 32 + QK_K / 64,
            Iq1S => 2 + QK_K / 8 + QK_K / 16,
            Iq1M => QK_K / 8 + QK_K / 16 + QK_K / 32,
            Iq4Nl => 2 + 16,
            Iq4Xs => 2 + 2 + QK_K / 64 + QK_K / 2,
            Tq1_0 => 2 + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5,
            Tq2_0 => 2 + QK_K / 4,
            Mxfp4 => 1 + 16,
            Nvfp4 => 4 + 32,
            Q1_0 => 2 + 16,
            Q2_0 => 2 + 16,
        }
    }

    /// Байт на `elems` элементов одной строки (хвостовой блок — целиком).
    pub const fn bytes_for(self, elems: usize) -> usize {
        elems.div_ceil(self.block_elems()) * self.block_bytes()
    }

    /// Средних бит на вес (для сводок и `DType::size_in_bits`).
    pub const fn bits_per_weight_x8(self) -> usize {
        self.block_bytes() * 8 * 8 / self.block_elems()
    }

    pub const fn is_quantized(self) -> bool {
        !matches!(
            self,
            GgmlType::F32
                | GgmlType::F16
                | GgmlType::BF16
                | GgmlType::F64
                | GgmlType::I8
                | GgmlType::I16
                | GgmlType::I32
                | GgmlType::I64
        )
    }

    /// Формат весов, который умеют ядра движка (деквант на карте). Q8_1 и
    /// Q8_K — форматы активаций для dp4a, как весов они не встречаются.
    pub const fn is_weight_format(self) -> bool {
        self.is_quantized() && !matches!(self, GgmlType::Q8_1 | GgmlType::Q8K)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_ggml() {
        use GgmlType::*;
        let want = [
            (Q4_0, 32, 18),
            (Q4_1, 32, 20),
            (Q5_0, 32, 22),
            (Q5_1, 32, 24),
            (Q8_0, 32, 34),
            (Q8_1, 32, 36),
            (Q2K, 256, 84),
            (Q3K, 256, 110),
            (Q4K, 256, 144),
            (Q5K, 256, 176),
            (Q6K, 256, 210),
            (Q8K, 256, 292),
            (Iq2Xxs, 256, 66),
            (Iq2Xs, 256, 74),
            (Iq3Xxs, 256, 98),
            (Iq1S, 256, 50),
            (Iq4Nl, 32, 18),
            (Iq3S, 256, 110),
            (Iq2S, 256, 82),
            (Iq4Xs, 256, 136),
            (Iq1M, 256, 56),
            (Tq1_0, 256, 54),
            (Tq2_0, 256, 66),
            (Mxfp4, 32, 17),
            (Nvfp4, 64, 36),
            (Q1_0, 128, 18),
            (Q2_0, 64, 18),
        ];
        for (t, be, bb) in want {
            assert_eq!(t.block_elems(), be, "{}", t.name());
            assert_eq!(t.block_bytes(), bb, "{}", t.name());
        }
    }

    #[test]
    fn ids_and_keys_round_trip() {
        for t in GgmlType::ALL {
            assert_eq!(GgmlType::from_u32(t as u32), Some(t));
            assert_eq!(GgmlType::from_key(&t.key()), Some(t), "{}", t.name());
        }
        assert_eq!(GgmlType::from_u32(4), None);
        assert_eq!(GgmlType::from_key("q4_k"), Some(GgmlType::Q4K));
        assert_eq!(GgmlType::from_key("iq2_xxs"), Some(GgmlType::Iq2Xxs));
    }

    #[test]
    fn bytes_for_rounds_up() {
        assert_eq!(GgmlType::Q4_0.bytes_for(33), 36);
        assert_eq!(GgmlType::Q4K.bytes_for(256), 144);
        assert_eq!(GgmlType::Q4K.bytes_for(257), 288);
    }
}

/// Кодировщик Q4_0 (как `quantize_row_q4_0_ref` ggml): блок 32 — `d = max/−8` по значению с
/// наибольшим модулем (со знаком), `q = min(15, x/d + 8.5)`; f16 `d` + 16 байт (младший полубайт —
/// значение j, старший — j + 16). `x` — матрица `[n, k]`, `k % 32 == 0`.
pub fn quantize_q4_0(x: &[f32], n: usize, k: usize) -> Vec<u8> {
    let nb = k / 32;
    let mut out = vec![0u8; n * nb * 18];
    for r in 0..n {
        for b in 0..nb {
            let v = &x[r * k + b * 32..r * k + b * 32 + 32];
            let (mut amax, mut max) = (0f32, 0f32);
            for &e in v {
                if e.abs() > amax {
                    amax = e.abs();
                    max = e;
                }
            }
            let d = max / -8.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let o = (r * nb + b) * 18;
            out[o..o + 2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            for j in 0..16 {
                let q0 = ((v[j] * id + 8.5) as i32).clamp(0, 15) as u8;
                let q1 = ((v[j + 16] * id + 8.5) as i32).clamp(0, 15) as u8;
                out[o + 2 + j] = q0 | (q1 << 4);
            }
        }
    }
    out
}

#[inline]
fn nearest_int(v: f32) -> i32 {
    v.round() as i32
}

/// `make_qkx2_quants` ggml: шкала и минимум блока по взвешенному МНК с перебором `nstep` сдвигов.
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(x: &[f32], w: &[f32], nmax: i32, l: &mut [u8], laux: &mut [u8], rmin: f32, rdelta: f32, nstep: i32) -> (f32, f32) {
    let n = x.len();
    let (mut min, mut max) = (x[0], x[0]);
    let (mut sum_w, mut sum_x) = (w[0], w[0] * x[0]);
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += w[i];
        sum_x += w[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l[..n].iter_mut().for_each(|v| *v = 0);
        return (0.0, -min);
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best = 0f32;
    for i in 0..n {
        let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = q as u8;
        let d = scale * q as f32 + min - x[i];
        best += w[i] * d * d;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sl, mut sl2, mut sxl) = (0f32, 0f32, 0f32);
        for i in 0..n {
            let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = q as u8;
            let q = q as f32;
            sl += w[i] * q;
            sl2 += w[i] * q * q;
            sxl += w[i] * q * x[i];
        }
        let dd = sum_w * sl2 - sl * sl;
        if dd > 0.0 {
            let mut this_scale = (sum_w * sxl - sum_x * sl) / dd;
            let mut this_min = (sl2 * sum_x - sl * sxl) / dd;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sxl / sl2;
            }
            let mut mad = 0f32;
            for i in 0..n {
                let d = this_scale * laux[i] as f32 + this_min - x[i];
                mad += w[i] * d * d;
            }
            if mad < best {
                l[..n].copy_from_slice(&laux[..n]);
                best = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// Шкала и минимум подблока `j` суперблока Q4_K/Q5_K (6 бит, `get_scale_min_k4` ggml).
#[inline]
pub fn q4k_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Кодировщик Q4_K (как `quantize_row_q4_K_ref` ggml): суперблок 256 = f16 `d`, f16 `dmin`,
/// 12 байт 6-битных шкал/минимумов восьми подблоков по 32 и 128 байт полубайтов (в каждой
/// четверти по 64: младшие — первые 32 значения, старшие — вторые); вес `d·sc·q − dmin·m`.
/// `x` — матрица `[n, k]`, `k % 256 == 0`. `imp` — важность столбцов (E[x²] входа из калибровки,
/// как imatrix ggml): ошибка взвешивается `imp·√(σ² + w²)`, без неё — `rms + |w|`.
pub fn quantize_q4_k(x: &[f32], n: usize, k: usize, imp: Option<&[f32]>) -> Vec<u8> {
    quantize_k(x, n, k, imp, false)
}

/// Кодировщик Q5_K (`quantize_row_q5_K_ref` ggml): как Q4_K, но 5 бит — плюс 32 байта `qh`
/// (пятый бит: для четверти c значения 0..31 — бит 2c, 32..63 — бит 2c + 1); 176 байт на 256.
pub fn quantize_q5_k(x: &[f32], n: usize, k: usize, imp: Option<&[f32]>) -> Vec<u8> {
    quantize_k(x, n, k, imp, true)
}

fn quantize_k(x: &[f32], n: usize, k: usize, imp: Option<&[f32]>, five: bool) -> Vec<u8> {
    const QK_K: usize = 256;
    // Калибровка (imatrix) — только для 4 бит. LFM3-1.7B, средняя KL к эталону по 628 позициям:
    // Q4_K 0,150 → 0,086 (калибровка MLP), а Q5_K 0,037 → 0,120 (выбросы на отдельных позициях).
    let imp = if five { None } else { imp.filter(|h| h.len() == k) };
    let nb = k / QK_K;
    let bb = if five { 176 } else { 144 };
    let nmax = if five { 31 } else { 15 };
    let mut out = vec![0u8; n * nb * bb];
    use rayon::prelude::*;
    out.par_chunks_mut(nb * bb).enumerate().for_each(|(r, orow)| {
        let mut l = [0u8; QK_K];
        let mut laux = [0u8; 32];
        let mut w = [0f32; 32];
        for b in 0..nb {
            let xb = &x[r * k + b * QK_K..r * k + (b + 1) * QK_K];
            let y = &mut orow[b * bb..(b + 1) * bb];
            let (mut scales, mut mins) = ([0f32; 8], [0f32; 8]);
            let (mut max_scale, mut max_min) = (0f32, 0f32);
            let sigma2 = 2.0 * xb.iter().map(|v| v * v).sum::<f32>() / QK_K as f32;
            for j in 0..8 {
                let xs = &xb[32 * j..32 * j + 32];
                let (s, m) = match imp {
                    Some(h) => {
                        let hb = &h[b * QK_K + 32 * j..b * QK_K + 32 * j + 32];
                        for ((wi, &v), &q) in w.iter_mut().zip(xs).zip(hb) {
                            *wi = q * (sigma2 + v * v).sqrt();
                        }
                        make_qkx2_quants(xs, &w, nmax, &mut l[32 * j..32 * j + 32], &mut laux, -0.9, 0.05, 36)
                    }
                    None => {
                        let av = (xs.iter().map(|v| v * v).sum::<f32>() / 32.0).sqrt();
                        for (wi, &v) in w.iter_mut().zip(xs) {
                            *wi = av + v.abs();
                        }
                        let (rmin, nstep) = if five { (-0.5, 15) } else { (-1.0, 20) };
                        make_qkx2_quants(xs, &w, nmax, &mut l[32 * j..32 * j + 32], &mut laux, rmin, 0.1, nstep)
                    }
                };
                scales[j] = s;
                mins[j] = m;
                max_scale = max_scale.max(s);
                max_min = max_min.max(m);
            }
            let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
            let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
            let sc = &mut y[4..16];
            sc.iter_mut().for_each(|v| *v = 0);
            for j in 0..8 {
                let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
                let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
                if j < 4 {
                    sc[j] = ls;
                    sc[j + 4] = lm;
                } else {
                    sc[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                    sc[j - 4] |= (ls >> 4) << 6;
                    sc[j] |= (lm >> 4) << 6;
                }
            }
            let d = half::f16::from_f32(max_scale / 63.0);
            let dmin = half::f16::from_f32(max_min / 63.0);
            y[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            y[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            let scb: [u8; 12] = y[4..16].try_into().unwrap();
            for j in 0..8 {
                let (s, m) = q4k_scale_min(j, &scb);
                let dd = d.to_f32() * s as f32;
                if dd == 0.0 {
                    continue;
                }
                let dm = dmin.to_f32() * m as f32;
                for ii in 0..32 {
                    l[32 * j + ii] = nearest_int((xb[32 * j + ii] + dm) / dd).clamp(0, nmax) as u8;
                }
            }
            if five {
                let (qh, q) = y[16..176].split_at_mut(32);
                qh.iter_mut().for_each(|v| *v = 0);
                for c in 0..4 {
                    for t in 0..32 {
                        let (mut l1, mut l2) = (l[c * 64 + t], l[c * 64 + t + 32]);
                        if l1 > 15 {
                            l1 -= 16;
                            qh[t] |= 1 << (2 * c);
                        }
                        if l2 > 15 {
                            l2 -= 16;
                            qh[t] |= 1 << (2 * c + 1);
                        }
                        q[c * 32 + t] = l1 | (l2 << 4);
                    }
                }
            } else {
                let q = &mut y[16..144];
                for c in 0..4 {
                    for t in 0..32 {
                        q[c * 32 + t] = l[c * 64 + t] | (l[c * 64 + t + 32] << 4);
                    }
                }
            }
        }
    });
    out
}

#[cfg(test)]
mod q4k_tests {
    use super::*;

    #[test]
    fn q4k_roundtrip_beats_q4_0() {
        let (n, k) = (4usize, 512usize);
        let x: Vec<f32> = (0..n * k).map(|i| ((i as f32 * 0.37).sin() * 0.05) + if i % 97 == 0 { 0.3 } else { 0.0 }).collect();
        let deq = |t: GgmlType, b: &[u8]| {
            let mut y = vec![0f32; n * k];
            crate::quant::ggml_dequant::dequantize(t, b, n * k, &mut y).unwrap();
            y
        };
        let err = |y: &[f32]| (x.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / x.iter().map(|a| a * a).sum::<f32>()).sqrt();
        let e4k = err(&deq(GgmlType::Q4K, &quantize_q4_k(&x, n, k, None)));
        let e40 = err(&deq(GgmlType::Q4_0, &quantize_q4_0(&x, n, k)));
        let e5k = err(&deq(GgmlType::Q5K, &quantize_q5_k(&x, n, k, None)));
        assert!(e4k < 0.1 && e4k < e40, "Q4_K {e4k} против Q4_0 {e40}");
        assert!(e5k < e4k * 0.7, "Q5_K {e5k} против Q4_K {e4k}");
    }
}
