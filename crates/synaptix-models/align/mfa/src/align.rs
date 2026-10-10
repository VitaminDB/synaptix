//! Принудительное выравнивание: граф фразы (трифоны с вариантами контекста на
//! границах слов, необязательные паузы) и Витерби по состояниям HMM — как
//! `gmm-align-compiled` в первом проходе MFA.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::model::MfaModel;
use crate::{MfaError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignOptions {
    pub acoustic_scale: f32,
    pub transition_scale: f32,
    pub self_loop_scale: f32,
    /// Луч отсечения (как `--beam` у `gmm-align`); `None` — точный Витерби.
    pub beam: Option<f32>,
    /// Луч повторной попытки, если с `beam` путь не нашёлся.
    pub retry_beam: Option<f32>,
}

impl Default for AlignOptions {
    /// Параметры MFA: акустика 0.1, self-loop 0.1, beam 10, retry 40.
    fn default() -> Self {
        Self { acoustic_scale: 0.1, transition_scale: 1.0, self_loop_scale: 0.1, beam: Some(10.0), retry_beam: Some(40.0) }
    }
}

/// Вариант произношения слова: фонемы (символы модели), вероятность
/// варианта и вероятность паузы после слова.
#[derive(Debug, Clone, Default)]
pub struct AlignWord {
    pub phones: Vec<String>,
    pub silence_after: Option<f32>,
    /// Множители вероятности входа в слово после паузы / без неё (словарь MFA).
    pub silence_before_correction: Option<f32>,
    pub non_silence_before_correction: Option<f32>,
}

impl AlignWord {
    pub fn from_pron(p: &crate::lexicon::Pron) -> Self {
        Self {
            phones: p.phones.clone(),
            silence_after: p.silence_after,
            silence_before_correction: p.silence_before_correction,
            non_silence_before_correction: p.non_silence_before_correction,
        }
    }
}

/// Слово фразы — один или несколько вариантов произношения.
#[derive(Debug, Clone)]
pub struct AlignWordAlts {
    pub alts: Vec<(AlignWord, f32)>,
}

impl From<AlignWord> for AlignWordAlts {
    fn from(w: AlignWord) -> Self {
        Self { alts: vec![(w, 1.0)] }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhoneSpan {
    pub phone: String,
    /// Индекс слова (`None` — пауза).
    pub word: Option<usize>,
    /// Какой вариант произношения слова выбран.
    pub alt: usize,
    /// Кадры `[start, end)`.
    pub start: usize,
    pub end: usize,
    /// Среднее (по кадрам) лог-правдоподобие выбранных состояний, без масштаба.
    pub loglike: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alignment {
    pub phones: Vec<PhoneSpan>,
    pub words: Vec<WordSpan>,
    /// Секунд на кадр.
    pub frame_shift: f32,
    /// Итоговая стоимость пути (лог, с масштабами).
    pub score: f32,
}

struct Unit {
    phone: i32,
    word: Option<usize>,
    alt: usize,
    /// Эмитирующие состояния топологии: (индекс состояния, pdf).
    states: Vec<(usize, usize)>,
}

const NEG: f32 = f32::NEG_INFINITY;

/// Выровнять фразу. `feats` — признаки `[T × dim]` из [`MfaModel::features`].
pub fn align(model: &MfaModel, feats: &[f32], words: &[AlignWord], opts: &AlignOptions) -> Result<Alignment> {
    let alts: Vec<AlignWordAlts> = words.iter().cloned().map(Into::into).collect();
    align_alts(model, feats, &alts, opts)
}

/// Выровнять фразу с вариантами произношения слов (выбирается лучший путь;
/// при заданном луче — как `gmm-align`, с повтором на `retry_beam`).
pub fn align_alts(model: &MfaModel, feats: &[f32], words: &[AlignWordAlts], opts: &AlignOptions) -> Result<Alignment> {
    match align_beam(model, feats, words, opts, opts.beam) {
        Err(MfaError::Align(_)) if opts.retry_beam.is_some() && opts.beam.is_some() => {
            align_beam(model, feats, words, opts, opts.retry_beam)
        }
        r => r,
    }
}

fn align_beam(model: &MfaModel, feats: &[f32], words: &[AlignWordAlts], opts: &AlignOptions, beam: Option<f32>) -> Result<Alignment> {
    let dim = model.lda_rows;
    let frames = feats.len() / dim;
    if frames == 0 || words.is_empty() || words.iter().any(|w| w.alts.is_empty()) {
        return Err(MfaError::Align("пустая запись или фраза".into()));
    }
    let sil = model.phone_id(&model.silence_phone).ok_or_else(|| MfaError::Align("нет фонемы паузы".into()))?;
    // ids[i][a] — фонемы варианта a слова i.
    let ids: Vec<Vec<Vec<i32>>> = words
        .iter()
        .map(|w| {
            w.alts
                .iter()
                .map(|(a, _)| {
                    a.phones
                        .iter()
                        .map(|p| model.phone_id(p).ok_or_else(|| MfaError::Align(format!("неизвестная фонема `{p}`"))))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<_>>()?;
    if ids.iter().flatten().any(|w| w.is_empty()) {
        return Err(MfaError::Align("слово без фонем".into()));
    }

    // ── Граф ──────────────────────────────────────────────────────────────
    let mut units: Vec<Unit> = Vec::new();
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    let mut start: Vec<(usize, f32)> = Vec::new();
    let mut finals: Vec<(usize, f32)> = Vec::new();
    let make = |units: &mut Vec<Unit>, phone: i32, l: i32, r: i32, word: Option<usize>, alt: usize| -> Result<usize> {
        let topo = model.transitions.topology(phone).ok_or_else(|| MfaError::Align(format!("нет топологии фонемы {phone}")))?;
        let ctx = [l, phone, r];
        let states = topo
            .states
            .iter()
            .enumerate()
            .filter_map(|(si, st)| st.pdf_class.map(|pc| (si, pc)))
            .map(|(si, pc)| {
                model.tree.pdf(&ctx, pc).map(|pdf| (si, pdf)).ok_or_else(|| MfaError::Align(format!("дерево: нет pdf для {ctx:?}/{pc}")))
            })
            .collect::<Result<Vec<_>>>()?;
        units.push(Unit { phone, word, alt, states });
        Ok(units.len() - 1)
    };
    let cost = |p: f32| -(p.clamp(1e-6, 1.0 - 1e-6)).ln();
    // Поправки — множители, могут быть > 1.
    let corr = |c: Option<f32>| -(c.unwrap_or(1.0).max(1e-6)).ln();
    let final_sil = corr(Some(model.final_silence_correction));
    let final_nonsil = corr(Some(model.final_non_silence_correction));
    let p_sil_init = model.initial_silence_probability;
    let n = ids.len();
    let uniq = |v: Vec<i32>| {
        let mut v = v;
        v.sort_unstable();
        v.dedup();
        v
    };
    // Выходы предыдущего слова: (unit, правый контекст, последняя фонема, стоимость).
    let mut prev_exits: Vec<(usize, i32, i32, f32)> = Vec::new();
    let mut prev_sil: Option<usize> = None;
    let s0 = make(&mut units, sil, 0, 0, None, 0)?;
    start.push((s0, cost(p_sil_init)));
    for i in 0..n {
        let lefts: Vec<i32> = if i == 0 { vec![0] } else { uniq(ids[i - 1].iter().map(|a| *a.last().unwrap()).collect()) };
        let rights: Vec<i32> = if i + 1 == n { vec![0] } else { uniq(ids[i + 1].iter().map(|a| a[0]).collect()) };
        let sil_unit = make(&mut units, sil, 0, 0, None, 0)?;
        let mut exits = Vec::new();
        for (ai, w) in ids[i].iter().enumerate() {
            let (aw, prob) = &words[i].alts[ai];
            let alt_cost = cost(*prob);
            let after_sil = alt_cost + corr(aw.silence_before_correction);
            let after_word = alt_cost + corr(aw.non_silence_before_correction);
            let p_sil = aw.silence_after.unwrap_or(model.silence_probability);
            let k = w.len();
            let mut entries: Vec<(usize, i32)> = Vec::new(); // (unit, левый контекст)
            let mut lasts: Vec<(usize, i32)> = Vec::new(); // (unit, правый контекст)
            let left_all: Vec<i32> = std::iter::once(sil).chain(lefts.iter().copied()).collect();
            let right_all: Vec<i32> = std::iter::once(sil).chain(rights.iter().copied()).collect();
            if k == 1 {
                for &l in &left_all {
                    for &r in &right_all {
                        let u = make(&mut units, w[0], l, r, Some(i), ai)?;
                        entries.push((u, l));
                        lasts.push((u, r));
                    }
                }
            } else {
                let mut prev: Vec<usize> = Vec::new();
                for &l in &left_all {
                    let u = make(&mut units, w[0], l, w[1], Some(i), ai)?;
                    entries.push((u, l));
                    prev.push(u);
                }
                for j in 1..k - 1 {
                    let u = make(&mut units, w[j], w[j - 1], w[j + 1], Some(i), ai)?;
                    for &p in &prev {
                        edges.push((p, u, 0.0));
                    }
                    prev = vec![u];
                }
                for &r in &right_all {
                    let u = make(&mut units, w[k - 1], w[k - 2], r, Some(i), ai)?;
                    for &p in &prev {
                        edges.push((p, u, 0.0));
                    }
                    lasts.push((u, r));
                }
            }
            // Вход в вариант.
            for &(u, l) in &entries {
                if i == 0 {
                    if l == sil {
                        edges.push((s0, u, after_sil));
                    } else {
                        start.push((u, cost(1.0 - p_sil_init) + after_word));
                    }
                } else if l == sil {
                    if let Some(ps) = prev_sil {
                        edges.push((ps, u, after_sil));
                    }
                } else {
                    // Прямой стык: правый контекст выхода = наша первая фонема,
                    // наш левый контекст = последняя фонема выхода.
                    for &(pu, pr, plast, c) in &prev_exits {
                        if pr == w[0] && plast == l {
                            edges.push((pu, u, c + after_word));
                        }
                    }
                }
            }
            // Выходы варианта.
            for &(u, r) in &lasts {
                if r == sil {
                    edges.push((u, sil_unit, cost(p_sil)));
                } else if i + 1 == n {
                    finals.push((u, cost(1.0 - p_sil) + final_nonsil));
                } else {
                    exits.push((u, r, w[k - 1], cost(1.0 - p_sil)));
                }
            }
        }
        if i + 1 == n {
            finals.push((sil_unit, final_sil));
        }
        prev_exits = exits;
        prev_sil = Some(sil_unit);
    }

    // ── Веса переходов внутри единиц ─────────────────────────────────────
    // trans[u][k] = [(dst эмит. индекс или None=финал, вес)] для эмит. состояния k.
    let trans: Vec<Vec<Vec<(Option<usize>, f32)>>> = units
        .iter()
        .map(|u| {
            let topo = model.transitions.topology(u.phone).unwrap();
            u.states
                .iter()
                .map(|&(si, pdf)| {
                    let st = &topo.states[si];
                    let lp: Vec<f32> = (0..st.transitions.len())
                        .map(|j| model.transitions.log_prob(u.phone, si, pdf, j).unwrap_or_else(|| st.transitions[j].1.max(1e-10).ln()))
                        .collect();
                    let p_self: f32 = st.transitions.iter().zip(&lp).filter(|((d, _), _)| *d == si).map(|(_, &l)| l.exp()).sum();
                    st.transitions
                        .iter()
                        .zip(&lp)
                        .map(|(&(d, _), &l)| {
                            // GetScaledTransitionLogProb (Kaldi): self-loop —
                            // self_loop_scale·ln p; остальные — self_loop_scale·ln(1−p_self)
                            // + transition_scale·ln(p/(1−p_self)).
                            let w = if d == si {
                                opts.self_loop_scale * l
                            } else {
                                let rest = (1.0 - p_self).max(1e-10).ln();
                                opts.transition_scale * (l - rest) + opts.self_loop_scale * rest
                            };
                            (u.states.iter().position(|&(s2, _)| s2 == d), w)
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    // Вход в единицу — эмитирующее состояние топологии 0.
    let entry: Vec<usize> = units.iter().map(|u| u.states.iter().position(|&(s, _)| s == 0).unwrap_or(0)).collect();

    // ── Акустика: только используемые pdf ─────────────────────────────────
    let mut pdfs: Vec<usize> = units.iter().flat_map(|u| u.states.iter().map(|&(_, p)| p)).collect();
    pdfs.sort_unstable();
    pdfs.dedup();
    let pidx = |p: usize| pdfs.binary_search(&p).unwrap();
    let ll: Vec<f32> = (0..frames)
        .into_par_iter()
        .flat_map_iter(|t| {
            let x = &feats[t * dim..(t + 1) * dim];
            pdfs.iter().map(move |&p| model.gmms.loglike(p, x)).collect::<Vec<_>>()
        })
        .collect();
    let np = pdfs.len();

    // ── Витерби ──────────────────────────────────────────────────────────
    let offs: Vec<usize> = units.iter().scan(0, |a, u| {
        let o = *a;
        *a += u.states.len();
        Some(o)
    }).collect();
    let total: usize = units.iter().map(|u| u.states.len()).sum();
    let mut delta = vec![NEG; total];
    // Обратные указатели: глобальный индекс предыдущего состояния (u32::MAX — старт).
    let mut bp = vec![u32::MAX; frames * total];
    let mut incoming: Vec<Vec<(usize, f32)>> = vec![Vec::new(); units.len()];
    for &(a, b, c) in &edges {
        incoming[b].push((a, c));
    }
    // Стоимости графа — −ln p: вычитаются из лог-оценки пути.
    for &(u, c) in &start {
        let g = offs[u] + entry[u];
        delta[g] = delta[g].max(-c);
    }
    let add_ac = |delta: &mut [f32], t: usize| {
        for (ui, u) in units.iter().enumerate() {
            for (k, &(_, pdf)) in u.states.iter().enumerate() {
                let g = offs[ui] + k;
                if delta[g] > NEG {
                    delta[g] += opts.acoustic_scale * ll[t * np + pidx(pdf)];
                }
            }
        }
    };
    add_ac(&mut delta, 0);
    // Лучший выход из единицы (в финал) после кадра t: (оценка, глоб. состояние).
    let exits_of = |delta: &[f32]| -> Vec<(f32, u32)> {
        units
            .iter()
            .enumerate()
            .map(|(ui, u)| {
                let mut best = (NEG, u32::MAX);
                for k in 0..u.states.len() {
                    let g = offs[ui] + k;
                    if delta[g] == NEG {
                        continue;
                    }
                    for &(d, w) in &trans[ui][k] {
                        if d.is_none() && delta[g] + w > best.0 {
                            best = (delta[g] + w, g as u32);
                        }
                    }
                }
                best
            })
            .collect()
    };
    let mut exit_hist: Vec<Vec<(f32, u32)>> = Vec::with_capacity(frames);
    for t in 1..frames {
        let ex = exits_of(&delta);
        let mut next = vec![NEG; total];
        let row = &mut bp[t * total..(t + 1) * total];
        for (ui, u) in units.iter().enumerate() {
            for k in 0..u.states.len() {
                let g = offs[ui] + k;
                if delta[g] == NEG {
                    continue;
                }
                for &(d, w) in &trans[ui][k] {
                    if let Some(dk) = d {
                        let gd = offs[ui] + dk;
                        let c = delta[g] + w;
                        if c > next[gd] {
                            next[gd] = c;
                            row[gd] = g as u32;
                        }
                    }
                }
            }
            // Вход из предыдущих единиц.
            let ge = offs[ui] + entry[ui];
            for &(from, c) in &incoming[ui] {
                let (e, src) = ex[from];
                if e > NEG && e - c > next[ge] {
                    next[ge] = e - c;
                    row[ge] = src;
                }
            }
        }
        exit_hist.push(ex);
        delta = next;
        add_ac(&mut delta, t);
        if let Some(b) = beam {
            let best = delta.iter().cloned().fold(NEG, f32::max);
            for d in delta.iter_mut() {
                if *d < best - b {
                    *d = NEG;
                }
            }
        }
    }
    let ex = exits_of(&delta);
    let mut best = (NEG, u32::MAX);
    for &(u, c) in &finals {
        if ex[u].0 - c > best.0 {
            best = (ex[u].0 - c, ex[u].1);
        }
    }
    if best.0 == NEG {
        return Err(MfaError::Align("нет пути: запись короче фразы".into()));
    }
    // Обратный проход.
    let unit_of = |g: usize| -> (usize, usize) {
        let ui = offs.partition_point(|&o| o <= g) - 1;
        (ui, g - offs[ui])
    };
    let mut path = vec![0usize; frames];
    let mut g = best.1 as usize;
    for t in (0..frames).rev() {
        path[t] = g;
        if t > 0 {
            g = bp[t * total + g] as usize;
        }
    }
    let _ = exit_hist;

    // ── Отрезки фонем ────────────────────────────────────────────────────
    let sym: std::collections::HashMap<i32, &str> = model.phones.iter().map(|(k, &v)| (v, k.as_str())).collect();
    let mut phones: Vec<PhoneSpan> = Vec::new();
    let mut t = 0;
    while t < frames {
        let (ui, _) = unit_of(path[t]);
        let mut e = t + 1;
        let mut acc = 0f32;
        // Новая единица начинается, когда путь вошёл в состояние входа из другой единицы.
        while e < frames {
            let (uj, kj) = unit_of(path[e]);
            if uj != ui || (kj == entry[uj] && unit_of(path[e - 1]).0 != uj) {
                break;
            }
            e += 1;
        }
        for tt in t..e {
            let (u2, k2) = unit_of(path[tt]);
            acc += ll[tt * np + pidx(units[u2].states[k2].1)];
        }
        phones.push(PhoneSpan {
            phone: sym.get(&units[ui].phone).copied().unwrap_or("?").to_string(),
            word: units[ui].word,
            alt: units[ui].alt,
            start: t,
            end: e,
            loglike: acc / (e - t) as f32,
        });
        t = e;
    }
    let words_out: Vec<WordSpan> = (0..n)
        .map(|wi| {
            let s: Vec<&PhoneSpan> = phones.iter().filter(|p| p.word == Some(wi)).collect();
            WordSpan { start: s.first().map_or(0, |p| p.start), end: s.last().map_or(0, |p| p.end) }
        })
        .collect();
    Ok(Alignment { phones, words: words_out, frame_shift: model.features.frame_shift / 1000.0, score: best.0 })
}
