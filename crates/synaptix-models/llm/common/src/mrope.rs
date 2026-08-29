//! M-RoPE — мультимодальные позиции семейства Qwen-VL / Qwen3.5.
//!
//! Позиция каждого токена трёхмерна: (время, строка, столбец). У текста все
//! три равны и растут на 1 — это обычный 1D RoPE. Токены картинки получают
//! `t = start`, `h = start + строка`, `w = start + столбец` по merged-сетке
//! башни, а текст после блока продолжается с `start + max(h, w)` — блок в
//! 1000 токенов «занимает» лишь ~32 позиции. Видео после HF-процессора —
//! это последовательность групп кадров, каждая как отдельная картинка
//! (таймкод идёт текстом), так что ось времени ведёт себя как у картинок.
//!
//! Частоты rotary-части поделены между осями `mrope_section`
//! (у Qwen3.5 — `[11, 11, 10]` на 32 частоты) — интерливингом
//! (`T H W T H W …`, `mrope_interleaved`) или подряд блоками (Qwen2-VL).
//! Таблицы cos/sin собираются на host на весь промпт и уходят в
//! `RopePositions::Tables`; декод после промпта идёт по 1D-позициям со
//! сдвигом `max_pos + 1 − L` (`RopePositions::Shifted`).
//!
//! Порт `get_rope_index` и `apply_interleaved_mrope` из HF
//! `modeling_qwen3_vl.py`.

/// Merged-сетка одного блока: кадров, строк, столбцов. У картинки `t = 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid3 {
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

impl Grid3 {
    pub fn image(h: usize, w: usize) -> Self {
        Self { t: 1, h, w }
    }

    pub fn tokens(&self) -> usize {
        self.t * self.h * self.w
    }
}

/// Прогоны токенов-заполнителей одной модальности: id заполнителя и
/// merged-сетка каждого блока в порядке появления в промпте.
pub struct MediaRuns<'a> {
    pub pad: u32,
    pub grids: &'a [Grid3],
}

/// 3D-позиции токенов промпта и максимальная позиция (для сдвига декода).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Positions {
    pub pos: Vec<[u32; 3]>,
    pub max_pos: u32,
}

impl Positions {
    /// Сдвиг 1D-позиций декода относительно индекса токена:
    /// первый сгенерированный токен стоит на `max_pos + 1`, а его индекс — `L`.
    pub fn decode_delta(&self) -> i64 {
        self.max_pos as i64 + 1 - self.pos.len() as i64
    }
}

/// Позиции по `get_rope_index`: текст — последовательно, блок картинки —
/// сетка от текущей позиции, после блока — `+max(h, w)`.
pub fn positions_3d(ids: &[u32], runs: &[MediaRuns]) -> Result<Positions, String> {
    let mut cursors = vec![0usize; runs.len()];
    let mut pos: Vec<[u32; 3]> = Vec::with_capacity(ids.len());
    let mut cur: u32 = 0;
    let mut i = 0usize;
    while i < ids.len() {
        if let Some(ri) = runs.iter().position(|r| r.pad == ids[i]) {
            let pad = ids[i];
            let start = i;
            while i < ids.len() && ids[i] == pad {
                i += 1;
            }
            let run = i - start;
            let g = *runs[ri].grids.get(cursors[ri]).ok_or_else(|| {
                format!("mrope: блоков заполнителя {pad} в промпте больше, чем сеток ({})", runs[ri].grids.len())
            })?;
            cursors[ri] += 1;
            if g.t == 0 || g.h == 0 || g.w == 0 || g.tokens() != run {
                return Err(format!(
                    "mrope: блок заполнителя {pad} из {run} токенов не совпадает с сеткой {}×{}×{}",
                    g.t, g.h, g.w
                ));
            }
            let plane = g.h * g.w;
            for j in 0..run {
                let ti = (j / plane) as u32;
                let (r, c) = (((j % plane) / g.w) as u32, (j % g.w) as u32);
                pos.push([cur + ti, cur + r, cur + c]);
            }
            cur += g.t.max(g.h).max(g.w) as u32;
        } else {
            pos.push([cur, cur, cur]);
            cur += 1;
            i += 1;
        }
    }
    for (ri, r) in runs.iter().enumerate() {
        if cursors[ri] != r.grids.len() {
            return Err(format!(
                "mrope: сеток заполнителя {} — {}, а блоков в промпте — {}",
                r.pad,
                r.grids.len(),
                cursors[ri]
            ));
        }
    }
    let max_pos = cur.saturating_sub(1);
    Ok(Positions { pos, max_pos })
}

/// Ось позиции для частоты `j` из `half` частот.
pub fn axis_of(j: usize, section: &[usize], interleaved: bool) -> usize {
    let s = |k: usize| section.get(k).copied().unwrap_or(0);
    if interleaved {
        // HF apply_interleaved_mrope: T по умолчанию; H — индексы 1,4,7,… < 3·s1;
        // W — 2,5,8,… < 3·s2.
        match j % 3 {
            1 if j < 3 * s(1) => 1,
            2 if j < 3 * s(2) => 2,
            _ => 0,
        }
    } else if j < s(0) {
        0
    } else if j < s(0) + s(1) {
        1
    } else {
        2
    }
}

/// Таблицы `cos`/`sin` формы `[L, half]` (row-major) по 3D-позициям:
/// частота `j` берёт позицию своей оси. Та же f32-арифметика, что в
/// `RopeCache`, поэтому для текста строки совпадают с обычными.
pub fn rope_tables(
    pos: &[[u32; 3]],
    inv_freq: &[f32],
    section: &[usize],
    interleaved: bool,
) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let axes: Vec<usize> = (0..half).map(|j| axis_of(j, section, interleaved)).collect();
    let mut cos = vec![0f32; pos.len() * half];
    let mut sin = vec![0f32; pos.len() * half];
    for (i, p) in pos.iter().enumerate() {
        for j in 0..half {
            let angle = (p[axes[j]] as f32) * inv_freq[j];
            cos[i * half + j] = angle.cos();
            sin[i * half + j] = angle.sin();
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_only_positions_are_sequential() {
        let ids = [5u32, 6, 7, 8];
        let p = positions_3d(&ids, &[]).unwrap();
        assert_eq!(p.pos, vec![[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]);
        assert_eq!(p.max_pos, 3);
        assert_eq!(p.decode_delta(), 0);
    }

    /// Текст (2) + картинка 2×3 + текст (2): как в HF get_rope_index.
    #[test]
    fn image_block_gets_grid_positions_and_compresses_tail() {
        const PAD: u32 = 99;
        let ids = [1u32, 2, PAD, PAD, PAD, PAD, PAD, PAD, 3, 4];
        let grids = [Grid3::image(2, 3)];
        let p = positions_3d(&ids, &[MediaRuns { pad: PAD, grids: &grids }]).unwrap();
        let expect = vec![
            [0, 0, 0], [1, 1, 1],
            [2, 2, 2], [2, 2, 3], [2, 2, 4],
            [2, 3, 2], [2, 3, 3], [2, 3, 4],
            [5, 5, 5], [6, 6, 6],
        ];
        assert_eq!(p.pos, expect);
        assert_eq!(p.max_pos, 6);
        // 10 токенов, max_pos 6 → первый сгенерированный стоит на 7 = 10 + (−3).
        assert_eq!(p.decode_delta(), -3);
    }

    #[test]
    fn video_groups_are_independent_blocks() {
        const VID: u32 = 7;
        // <t><start> pad pad <end> <t><start> pad pad <end>
        let ids = [10u32, 11, VID, VID, 12, 10, 11, VID, VID, 12];
        let grids = [Grid3::image(1, 2), Grid3::image(1, 2)];
        let p = positions_3d(&ids, &[MediaRuns { pad: VID, grids: &grids }]).unwrap();
        assert_eq!(p.pos[2], [2, 2, 2]);
        assert_eq!(p.pos[3], [2, 2, 3]);
        assert_eq!(p.pos[4], [4, 4, 4]); // после блока 1×2 → +2
        assert_eq!(p.pos[7], [7, 7, 7]);
        assert_eq!(p.pos[8], [7, 7, 8]);
        assert_eq!(p.pos[9], [9, 9, 9]);
    }

    /// Видео одним блоком: t·h·w токенов, ось времени растёт по группам
    /// кадров, а текст после блока продолжается с max(t, h, w).
    #[test]
    fn video_block_uses_temporal_axis() {
        const VID: u32 = 7;
        let ids = [1u32, VID, VID, VID, VID, VID, VID, VID, VID, 2];
        let grids = [Grid3 { t: 2, h: 2, w: 2 }];
        let p = positions_3d(&ids, &[MediaRuns { pad: VID, grids: &grids }]).unwrap();
        assert_eq!(&p.pos[1..9], &[
            [1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2],
            [2, 1, 1], [2, 1, 2], [2, 2, 1], [2, 2, 2],
        ]);
        assert_eq!(p.pos[9], [3, 3, 3]);
    }

    #[test]
    fn grid_mismatch_is_an_error() {
        const PAD: u32 = 99;
        let ids = [PAD, PAD, PAD];
        let grids = [Grid3::image(2, 2)];
        assert!(positions_3d(&ids, &[MediaRuns { pad: PAD, grids: &grids }]).is_err());
        let ids = [PAD, PAD, PAD, PAD, 1, PAD, PAD, PAD, PAD];
        assert!(positions_3d(&ids, &[MediaRuns { pad: PAD, grids: &grids }]).is_err());
    }

    #[test]
    fn interleaved_axes_match_hf_layout() {
        let section = [11usize, 11, 10];
        let axes: Vec<usize> = (0..32).map(|j| axis_of(j, &section, true)).collect();
        assert_eq!(axes.iter().filter(|a| **a == 0).count(), 11);
        assert_eq!(axes.iter().filter(|a| **a == 1).count(), 11);
        assert_eq!(axes.iter().filter(|a| **a == 2).count(), 10);
        assert_eq!(&axes[..6], &[0, 1, 2, 0, 1, 2]);
        assert_eq!(axes[31], 1); // 31 < 33 → H
        assert_eq!(axis_of(29, &section, true), 2); // 29 < 30 → W
        let chunked: Vec<usize> = (0..32).map(|j| axis_of(j, &section, false)).collect();
        assert_eq!(chunked[10], 0);
        assert_eq!(chunked[11], 1);
        assert_eq!(chunked[22], 2);
    }

    /// Для текста таблицы совпадают с 1D RoPE (`angle = pos · inv_freq`).
    #[test]
    fn tables_reduce_to_plain_rope_for_text() {
        let inv: Vec<f32> = (0..32).map(|i| 10_000_000f32.powf(-(2.0 * i as f32) / 64.0)).collect();
        let pos: Vec<[u32; 3]> = (0..5).map(|p| [p, p, p]).collect();
        let (cos, sin) = rope_tables(&pos, &inv, &[11, 11, 10], true);
        for t in 0..5usize {
            for j in 0..32 {
                let a = (t as f32) * inv[j];
                assert_eq!(cos[t * 32 + j], a.cos());
                assert_eq!(sin[t * 32 + j], a.sin());
            }
        }
        // Токен картинки: строка/столбец видны только на своих осях.
        let (cos, _) = rope_tables(&[[2, 3, 4]], &inv, &[11, 11, 10], true);
        assert_eq!(cos[0], (2.0 * inv[0]).cos());
        assert_eq!(cos[1], (3.0 * inv[1]).cos());
        assert_eq!(cos[2], (4.0 * inv[2]).cos());
    }
}
