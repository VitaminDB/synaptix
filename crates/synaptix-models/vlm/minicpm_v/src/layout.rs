//! Заполнитель картинки в промпте и «холстовые» позиции M-RoPE MiniCPM-V 4.7.
//!
//! Заполнитель (порт `replace_image_token` + `_prepend_local_ids`):
//! `<image_id>N</image_id><image>⟨pad×T⟩</image>` и, если есть срезы, строки
//! `<slice>⟨pad×S⟩</slice>…` по `cols` штук, соединённые `\n`.
//!
//! Позиции (порт `get_rope_index` для картинок, `mrope_mode = "canvas"`):
//! у текста все три координаты равны и растут на 1; картинка — «холст»
//! `rows·slice_h × cols·slice_w` токенов (или сетка миниатюры, если срезов
//! нет) с началом `origin`: `<image>` стоит на гало перед углом холста,
//! токены миниатюры растягиваются `linspace` по холсту, срезы лежат на своих
//! местах, маркеры `<slice>`/`</slice>` — на углах среза, `\n` между
//! строками срезов — у правого края строки, `</image>` — за дальним углом.
//! Текст после картинки продолжается с `origin + max(h, w) + 1`.

use crate::preprocess::{CropGrid, ImageLayout};

/// Строки спецтокенов заполнителя (из `tokenizer_config.json`).
#[derive(Debug, Clone)]
pub struct PlaceholderTokens {
    pub image_pad: String,
    pub image_start: String,
    pub image_end: String,
    pub slice_start: String,
    pub slice_end: String,
    pub image_id_start: String,
    pub image_id_end: String,
}

impl Default for PlaceholderTokens {
    fn default() -> Self {
        Self {
            image_pad: "<|image_pad|>".into(),
            image_start: "<image>".into(),
            image_end: "</image>".into(),
            slice_start: "<slice>".into(),
            slice_end: "</slice>".into(),
            image_id_start: "<image_id>".into(),
            image_id_end: "</image_id>".into(),
        }
    }
}

/// Заполнитель картинки; `image_index` — локальный номер картинки в
/// сообщении (`use_image_id`), `None` — без префикса `<image_id>`.
pub fn placeholder(
    layout: &ImageLayout,
    merge_factor: usize,
    image_index: Option<usize>,
    t: &PlaceholderTokens,
) -> String {
    let mut s = String::new();
    if let Some(i) = image_index {
        s.push_str(&t.image_id_start);
        s.push_str(&i.to_string());
        s.push_str(&t.image_id_end);
    }
    s.push_str(&t.image_start);
    s.push_str(&t.image_pad.repeat(layout.thumb.tokens(merge_factor)));
    s.push_str(&t.image_end);
    if let Some(sl) = layout.slice.filter(|_| layout.rows > 0 && layout.cols > 0) {
        let one = format!("{}{}{}", t.slice_start, t.image_pad.repeat(sl.tokens(merge_factor)), t.slice_end);
        let row = one.repeat(layout.cols);
        let rows: Vec<&str> = std::iter::repeat_n(row.as_str(), layout.rows).collect();
        s.push_str(&rows.join("\n"));
    }
    s
}

/// Сколько токенов-заполнителей (`<|image_pad|>`) несёт картинка.
pub fn placeholder_token_count(layout: &ImageLayout, merge_factor: usize) -> usize {
    layout.tokens(merge_factor)
}

/// Id токенов холста из `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanvasIds {
    pub image_token: u32,
    pub image_start: u32,
    pub image_end: u32,
    pub slice_start: u32,
    pub slice_end: u32,
    pub newline: u32,
}

impl CanvasIds {
    pub fn from_config(c: &crate::config::MiniCpmConfig) -> Option<Self> {
        Some(Self {
            image_token: c.image_token_id?,
            image_start: c.image_start_id?,
            image_end: c.image_end_id?,
            slice_start: c.slice_start_id?,
            slice_end: c.slice_end_id?,
            newline: c.newline_id?,
        })
    }

    fn is_marker(&self, id: u32) -> bool {
        id == self.slice_start
            || id == self.slice_end
            || id == self.image_start
            || id == self.image_end
            || id == self.newline
    }
}

/// 3D-позиции `[T, H, W]` каждого токена промпта и максимум по всем.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanvasPositions {
    pub pos: Vec<[u32; 3]>,
    pub max_pos: u32,
}

impl CanvasPositions {
    /// Сдвиг 1D-позиций декода: первый сгенерированный токен стоит на
    /// `max_pos + 1`, а его индекс — `L` (`rope_deltas` HF).
    pub fn decode_delta(&self) -> i64 {
        self.max_pos as i64 + 1 - self.pos.len() as i64
    }
}

/// Один кадр группы: миниатюра `[thumb_start, thumb_end)`, конец кадра с
/// закрывающими маркерами, срезы `[start, end)`.
struct Frame {
    thumb: (usize, usize),
    end: usize,
    slices: Vec<(usize, usize)>,
}

/// `_group_visual_frames` для картинок: группы кадров, разделённых только
/// маркерами.
fn group_frames(ids: &[u32], c: &CanvasIds) -> Vec<(usize, usize, Vec<Frame>)> {
    let n = ids.len();
    // (group_start_idx, frames)
    let mut groups: Vec<(usize, Vec<Frame>)> = Vec::new();
    let mut has_open_frame = false;
    let mut previous_end = 0usize;
    let mut i = 0usize;
    while i < n {
        if ids[i] != c.image_token {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && ids[i] == c.image_token {
            i += 1;
        }
        let crop = (start, i);
        let before = crop.0.checked_sub(1).map(|k| ids[k]);
        if has_open_frame && before == Some(c.slice_start) {
            groups.last_mut().expect("группа").1.last_mut().expect("кадр").slices.push(crop);
        } else {
            let frame = Frame { thumb: crop, end: crop.1, slices: Vec::new() };
            let gap_end = crop.0.saturating_sub(1).max(previous_end);
            let gap = &ids[previous_end..gap_end];
            let adjacent = !groups.is_empty() && gap.iter().all(|t| c.is_marker(*t));
            if adjacent {
                groups.last_mut().expect("группа").1.push(frame);
            } else {
                groups.push((crop.0.saturating_sub(1), vec![frame]));
            }
            has_open_frame = true;
        }
        previous_end = crop.1;
    }
    groups
        .into_iter()
        .map(|(gs, mut frames)| {
            for f in frames.iter_mut() {
                let mut end = f.slices.last().map(|s| s.1).unwrap_or(f.thumb.1);
                while end < n && c.is_marker(ids[end]) {
                    end += 1;
                }
                f.end = end;
            }
            let ge = frames.last().map(|f| f.end).unwrap_or(gs);
            (gs, ge, frames)
        })
        .collect()
}

/// `torch.linspace(0, end, steps)` в f32 с округлением `round` (к чётному),
/// как `linspace(...).round().long()`.
fn linspace_round(end: usize, steps: usize) -> Vec<u32> {
    if steps == 0 {
        return Vec::new();
    }
    if steps == 1 {
        return vec![0];
    }
    let e = end as f32;
    let step = e / (steps - 1) as f32;
    let half = steps / 2;
    (0..steps)
        .map(|i| {
            let v = if i < half { step * i as f32 } else { e - step * (steps - 1 - i) as f32 };
            v.round_ties_even().max(0.0) as u32
        })
        .collect()
}

/// `get_vision_position_ids`: позиции `[T, H, W]` токенов кропа на холсте.
fn vision_positions(
    start: u32,
    grid: CropGrid,
    canvas_h: usize,
    canvas_w: usize,
    h_off: usize,
    w_off: usize,
    mf: usize,
) -> Vec<[u32; 3]> {
    let (lh, lw) = (grid.h / mf, grid.w / mf);
    let ch = if canvas_h == 0 { lh } else { canvas_h };
    let cw = if canvas_w == 0 { lw } else { canvas_w };
    let hs = linspace_round(ch.saturating_sub(1), lh);
    let ws = linspace_round(cw.saturating_sub(1), lw);
    let mut out = Vec::with_capacity(lh * lw);
    for hc in &hs {
        for wc in &ws {
            out.push([start, start + hc + h_off as u32, start + wc + w_off as u32]);
        }
    }
    out
}

/// Холстовые позиции M-RoPE всего промпта. `layouts` — по одной на картинку
/// в порядке появления в `ids`; кропы каждой — [`ImageLayout::crops`].
pub fn canvas_positions(
    ids: &[u32],
    layouts: &[ImageLayout],
    merge_factor: usize,
    c: &CanvasIds,
) -> Result<CanvasPositions, String> {
    let n = ids.len();
    let mf = merge_factor.max(1);
    let mut pos: Vec<[u32; 3]> = (0..n as u32).map(|i| [i, i, i]).collect();
    let mut grids = layouts.iter().flat_map(|l| l.crops()).peekable();
    let mut next_grid = |what: &str| -> Result<CropGrid, String> {
        grids.next().ok_or_else(|| format!("холст: в промпте больше кропов, чем раскладок ({what})"))
    };
    let fill_text = |pos: &mut Vec<[u32; 3]>, from: usize, to: usize, base: u32| {
        for (k, p) in pos[from..to].iter_mut().enumerate() {
            let v = base + k as u32;
            *p = [v, v, v];
        }
    };

    let mut current_pos: u32 = 0;
    let mut current_idx = 0usize;
    for (gs, ge, frames) in group_frames(ids, c) {
        if gs > current_idx {
            fill_text(&mut pos, current_idx, gs, current_pos);
            current_pos += (gs - current_idx) as u32;
        }
        let mut frame_cursor = gs;
        for f in frames {
            let thumb = next_grid("миниатюра")?;
            let (ts, te) = f.thumb;
            let fs = ts.saturating_sub(1);
            if fs > frame_cursor {
                let gap = fs - frame_cursor;
                fill_text(&mut pos, frame_cursor, fs, current_pos);
                current_pos += gap as u32 + 1;
            }
            let fe = f.end.min(ge);
            let origin = current_pos;
            let halo = origin.saturating_sub(1);

            let mut canvas_h = thumb.h / mf;
            let mut canvas_w = thumb.w / mf;
            let mut num_cols = 0usize;
            let (mut lsh, mut lsw) = (0usize, 0usize);
            let mut first_slice: Option<CropGrid> = None;
            if !f.slices.is_empty() {
                num_cols = f.slices.len();
                for k in 0..f.slices.len() - 1 {
                    if f.slices[k + 1].0 - f.slices[k].1 > 2 {
                        num_cols = k + 1;
                        break;
                    }
                }
                let mut num_rows = if num_cols > 0 { f.slices.len() / num_cols } else { 1 };
                if num_rows * num_cols != f.slices.len() {
                    num_rows = 1;
                    num_cols = f.slices.len();
                }
                let s0 = next_grid("первый срез")?;
                first_slice = Some(s0);
                lsh = s0.h / mf;
                lsw = s0.w / mf;
                canvas_h = num_rows * lsh;
                canvas_w = num_cols * lsw;
            }

            for p in pos[fs..fe].iter_mut() {
                *p = [origin, origin, origin];
            }
            if fs < n {
                pos[fs][1] = halo;
                pos[fs][2] = halo;
            }
            if te < fe {
                pos[te][1] = origin + canvas_h as u32;
                pos[te][2] = origin + canvas_w as u32;
            }
            let tp = vision_positions(origin, thumb, canvas_h, canvas_w, 0, 0, mf);
            if tp.len() != te - ts {
                return Err(format!(
                    "холст: миниатюра {}×{} патчей даёт {} токенов, а в промпте {}",
                    thumb.h,
                    thumb.w,
                    tp.len(),
                    te - ts
                ));
            }
            pos[ts..te].copy_from_slice(&tp);

            for (k, (ss, se)) in f.slices.iter().copied().enumerate() {
                let h_off = (k / num_cols) * lsh;
                let w_off = (k % num_cols) * lsw;
                let g = if k == 0 { first_slice.expect("первый срез") } else { next_grid("срез")? };
                let (sh, sw) = (g.h / mf, g.w / mf);
                let ssm = ss.saturating_sub(1);
                if ss >= 1 && ssm >= fs {
                    pos[ssm][1] = origin + h_off as u32;
                    pos[ssm][2] = origin + w_off as u32;
                }
                if se < fe {
                    pos[se][1] = origin + (h_off + sh) as u32 - 1;
                    pos[se][2] = origin + (w_off + sw) as u32 - 1;
                }
                let sp = vision_positions(origin, g, 0, 0, h_off, w_off, mf);
                if sp.len() != se - ss {
                    return Err(format!(
                        "холст: срез {}×{} патчей даёт {} токенов, а в промпте {}",
                        g.h,
                        g.w,
                        sp.len(),
                        se - ss
                    ));
                }
                pos[ss..se].copy_from_slice(&sp);
            }

            for k in 0..f.slices.len().saturating_sub(1) {
                let (gs_, ge_) = (f.slices[k].1, f.slices[k + 1].0);
                if ge_ - gs_ <= 2 {
                    continue;
                }
                let boundary_h = (k / num_cols + 1) * lsh - 1;
                let right_w = num_cols * lsw;
                for nl in gs_ + 1..ge_ - 1 {
                    pos[nl][1] = origin + boundary_h as u32;
                    pos[nl][2] = origin + right_w as u32;
                }
            }

            current_pos = origin + canvas_h.max(canvas_w) as u32 + 1;
            frame_cursor = fe;
        }
        if frame_cursor < ge {
            fill_text(&mut pos, frame_cursor, ge, current_pos);
            current_pos += (ge - frame_cursor) as u32;
        }
        current_idx = ge;
    }
    if current_idx < n {
        fill_text(&mut pos, current_idx, n, current_pos);
    }
    if grids.peek().is_some() {
        return Err("холст: раскладок картинок больше, чем кропов в промпте".into());
    }
    let max_pos = pos.iter().flat_map(|p| p.iter().copied()).max().unwrap_or(0);
    Ok(CanvasPositions { pos, max_pos })
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDS: CanvasIds = CanvasIds {
        image_token: 100,
        image_start: 101,
        image_end: 102,
        slice_start: 103,
        slice_end: 104,
        newline: 105,
    };

    fn thumb_only(h: usize, w: usize) -> ImageLayout {
        ImageLayout { thumb: CropGrid { h, w }, slice: None, rows: 0, cols: 0 }
    }

    #[test]
    fn placeholder_strings() {
        let t = PlaceholderTokens::default();
        let l = thumb_only(8, 8);
        assert_eq!(
            placeholder(&l, 4, Some(0), &t),
            format!("<image_id>0</image_id><image>{}</image>", "<|image_pad|>".repeat(4))
        );
        assert_eq!(placeholder(&l, 4, None, &t), format!("<image>{}</image>", "<|image_pad|>".repeat(4)));
        let s = ImageLayout { thumb: CropGrid { h: 8, w: 12 }, slice: Some(CropGrid { h: 8, w: 8 }), rows: 2, cols: 3 };
        let p = placeholder(&s, 4, Some(1), &t);
        let row = format!("<slice>{}</slice>", "<|image_pad|>".repeat(4)).repeat(3);
        assert_eq!(
            p,
            format!("<image_id>1</image_id><image>{}</image>{row}\n{row}", "<|image_pad|>".repeat(6))
        );
        assert_eq!(placeholder_token_count(&s, 4), 6 + 6 * 4);
        assert_eq!(p.matches("<|image_pad|>").count(), 30);
    }

    #[test]
    fn linspace_round_matches_torch() {
        assert_eq!(linspace_round(0, 1), vec![0]);
        assert_eq!(linspace_round(3, 4), vec![0, 1, 2, 3]);
        // linspace(0, 3, 3) = [0, 1.5, 3] → round к чётному: 1.5 → 2
        assert_eq!(linspace_round(3, 3), vec![0, 2, 3]);
        // linspace(0, 7, 4) = [0, 2.333, 4.667, 7] → [0, 2, 5, 7]
        assert_eq!(linspace_round(7, 4), vec![0, 2, 5, 7]);
        // linspace(0, 5, 3) = [0, 2.5, 5] → 2.5 → 2
        assert_eq!(linspace_round(5, 3), vec![0, 2, 5]);
    }

    /// Текст, картинка 8×8 патчей (2×2 токена) без срезов, текст.
    #[test]
    fn thumbnail_only_canvas() {
        let mut ids = vec![1u32, 2, 3];
        ids.push(IDS.image_start);
        ids.extend([IDS.image_token; 4]);
        ids.push(IDS.image_end);
        ids.extend([4u32, 5]);
        let p = canvas_positions(&ids, &[thumb_only(8, 8)], 4, &IDS).unwrap();
        assert_eq!(&p.pos[0..3], &[[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
        // <image>: origin 3, halo 2
        assert_eq!(p.pos[3], [3, 2, 2]);
        assert_eq!(&p.pos[4..8], &[[3, 3, 3], [3, 3, 4], [3, 4, 3], [3, 4, 4]]);
        // </image>: (origin, origin+h, origin+w)
        assert_eq!(p.pos[8], [3, 5, 5]);
        // текст дальше с origin + max(2,2) + 1 = 6
        assert_eq!(&p.pos[9..11], &[[6, 6, 6], [7, 7, 7]]);
        assert_eq!(p.max_pos, 7);
        assert_eq!(p.decode_delta(), 8 - 11);
    }

    /// Картинка в самом начале промпта: гало упирается в 0.
    #[test]
    fn image_at_start() {
        let mut ids = vec![IDS.image_start];
        ids.extend([IDS.image_token; 4]);
        ids.push(IDS.image_end);
        ids.push(9);
        let p = canvas_positions(&ids, &[thumb_only(8, 8)], 4, &IDS).unwrap();
        assert_eq!(p.pos[0], [0, 0, 0]);
        assert_eq!(p.pos[1], [0, 0, 0]);
        assert_eq!(p.pos[5], [0, 2, 2]);
        assert_eq!(p.pos[6], [3, 3, 3]);
    }

    /// Миниатюра 8×8 (2×2) + срезы 1×2 по 8×8 (2×2 каждый): холст 2×4.
    #[test]
    fn slices_canvas() {
        let layout = ImageLayout { thumb: CropGrid { h: 8, w: 8 }, slice: Some(CropGrid { h: 8, w: 8 }), rows: 1, cols: 2 };
        let mut ids = vec![7u32];
        ids.push(IDS.image_start); // 1
        ids.extend([IDS.image_token; 4]); // 2..6
        ids.push(IDS.image_end); // 6
        ids.push(IDS.slice_start); // 7
        ids.extend([IDS.image_token; 4]); // 8..12
        ids.push(IDS.slice_end); // 12
        ids.push(IDS.slice_start); // 13
        ids.extend([IDS.image_token; 4]); // 14..18
        ids.push(IDS.slice_end); // 18
        ids.push(8); // 19
        let p = canvas_positions(&ids, &[layout], 4, &IDS).unwrap();
        let o = 1u32;
        assert_eq!(p.pos[1], [o, 0, 0]);
        // миниатюра 2×2 растянута на холст 2×4: h → linspace(0,1,2) = [0,1]; w → linspace(0,3,2) = [0,3]
        assert_eq!(&p.pos[2..6], &[[o, o, o], [o, o, o + 3], [o, o + 1, o], [o, o + 1, o + 3]]);
        assert_eq!(p.pos[6], [o, o + 2, o + 4]);
        // <slice> k=0 на (0,0), </slice> на (1,1)
        assert_eq!(p.pos[7], [o, o, o]);
        assert_eq!(&p.pos[8..12], &[[o, o, o], [o, o, o + 1], [o, o + 1, o], [o, o + 1, o + 1]]);
        assert_eq!(p.pos[12], [o, o + 1, o + 1]);
        // срез k=1: смещение w 2
        assert_eq!(p.pos[13], [o, o, o + 2]);
        assert_eq!(&p.pos[14..18], &[[o, o, o + 2], [o, o, o + 3], [o, o + 1, o + 2], [o, o + 1, o + 3]]);
        assert_eq!(p.pos[18], [o, o + 1, o + 3]);
        // текст: origin + max(2, 4) + 1 = 6
        assert_eq!(p.pos[19], [6, 6, 6]);
        assert_eq!(p.max_pos, 6);
    }

    /// Две строки срезов с переводом строки между ними.
    #[test]
    fn slice_rows_newline() {
        let layout = ImageLayout { thumb: CropGrid { h: 8, w: 8 }, slice: Some(CropGrid { h: 8, w: 8 }), rows: 2, cols: 1 };
        let mut ids = vec![IDS.image_start]; // 0
        ids.extend([IDS.image_token; 4]); // 1..5
        ids.push(IDS.image_end); // 5
        ids.push(IDS.slice_start); // 6
        ids.extend([IDS.image_token; 4]); // 7..11
        ids.push(IDS.slice_end); // 11
        ids.push(IDS.newline); // 12
        ids.push(IDS.slice_start); // 13
        ids.extend([IDS.image_token; 4]); // 14..18
        ids.push(IDS.slice_end); // 18
        ids.push(3); // 19
        let p = canvas_positions(&ids, &[layout], 4, &IDS).unwrap();
        // HF: `slices[k+1][0] - slices[k][1] > 2` → 14 − 11 = 3 > 2 → cols = 1.
        // холст 4×2; второй срез со смещением h 2.
        assert_eq!(&p.pos[14..18], &[[0, 2, 0], [0, 2, 1], [0, 3, 0], [0, 3, 1]]);
        // \n — у правого края первой строки: h = 1·2 − 1 = 1, w = 1·2 = 2
        assert_eq!(p.pos[12], [0, 1, 2]);
        assert_eq!(p.pos[5], [0, 4, 2]);
        assert_eq!(p.pos[19], [5, 5, 5]);
    }

    /// Две картинки подряд (только маркеры между ними) — одна группа; вторая
    /// начинается на шаг позже.
    #[test]
    fn two_adjacent_images() {
        let mut ids = vec![IDS.image_start];
        ids.extend([IDS.image_token; 4]);
        ids.push(IDS.image_end);
        ids.push(IDS.image_start);
        ids.extend([IDS.image_token; 4]);
        ids.push(IDS.image_end);
        let p = canvas_positions(&ids, &[thumb_only(8, 8), thumb_only(8, 8)], 4, &IDS).unwrap();
        // первая: origin 0, текст-курсор после неё 3; второй кадр начинается с
        // индекса 6 = frame_cursor → без разрыва, origin 3.
        assert_eq!(p.pos[6], [3, 2, 2]);
        assert_eq!(p.pos[7], [3, 3, 3]);
        assert_eq!(p.pos[11], [3, 5, 5]);
        assert_eq!(p.max_pos, 5);
    }

    #[test]
    fn mismatch_is_error() {
        let mut ids = vec![IDS.image_start];
        ids.extend([IDS.image_token; 3]);
        ids.push(IDS.image_end);
        assert!(canvas_positions(&ids, &[thumb_only(8, 8)], 4, &IDS).is_err());
        assert!(canvas_positions(&[1, 2], &[thumb_only(8, 8)], 4, &IDS).is_err());
    }
}
