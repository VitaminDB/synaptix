//! Склеенные события → доли, интервалы и ноты → ABC (`exports_sheetsage2.py`).
//!
//! Релиз передаёт мелодию в нотацию через MIDI-файл (pretty_midi, 960 тиков на
//! четверть при 120 уд/мин): времена нот при этом округляются до тика 1/1920 с
//! половинами к чётному. Это округление здесь повторено явно — без него ноты на
//! границе шага сетки попадали бы в соседний шаг.

use crate::notation::{self, Interval, MidiNote, Score};
use crate::stitch::{median, py_round};
use crate::vocab::Event;
use crate::SheetError;

/// Секунд на тик MIDI релиза: `60 / (120 · 960)`.
const MIDI_TICK: f64 = 60.0 / (120.0 * 960.0);

/// Время после записи в MIDI и чтения обратно.
pub fn midi_quantize(time: f64) -> f64 {
    if time <= 0.0 {
        return 0.0;
    }
    let tick = py_round(time / MIDI_TICK);
    tick * MIDI_TICK
}

/// Нота с источником: `track` 0 — вокал, 1 — инструмент.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawNote {
    pub start: f64,
    pub end: f64,
    pub pitch: u32,
    pub track: u32,
}

/// Ноты всех событий, по возрастанию (как `notes.sort()` релиза).
pub fn collect_notes(events: &[Event], duration: f64) -> Vec<RawNote> {
    let mut notes = Vec::new();
    for event in events {
        let start = event.time.unwrap_or(0.0);
        if let Some(melody) = &event.values.melody {
            for n in melody {
                let end = duration.min(n.end_time.unwrap_or(start));
                if end > start {
                    notes.push(RawNote { start, end, pitch: n.pitch, track: n.track });
                }
            }
        }
    }
    notes.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then(a.end.total_cmp(&b.end))
            .then(a.pitch.cmp(&b.pitch))
            .then(a.track.cmp(&b.track))
    });
    notes
}

/// Строки интервалов поля: от события до следующего такого же (последнее — до конца).
pub fn interval_rows(events: &[Event], field: crate::vocab::Field, duration: f64) -> Vec<Interval> {
    use crate::vocab::Field;
    let mut rows: Vec<(f64, f64, String)> = events
        .iter()
        .filter_map(|e| {
            let value = match field {
                Field::Chord => e.values.chord.clone(),
                Field::Key => e.values.key.clone(),
                Field::Structure => e.values.structure.clone(),
                _ => None,
            }?;
            Some((e.time.unwrap_or(0.0), 0.0, value))
        })
        .collect();
    for i in 0..rows.len() {
        rows[i].1 = if i + 1 < rows.len() { rows[i + 1].0 } else { duration };
    }
    rows.retain(|r| r.1 > r.0);
    rows
}

/// Доли `[время, номер доли, числитель, знаменатель]`: позиция восьмой
/// переводится в долю размера; позиция вне сетки размера — ошибка ABC.
pub fn rhythm_rows(events: &[Event]) -> Result<Vec<(f64, i64, i64, i64)>, SheetError> {
    let mut rows = Vec::new();
    let mut meter: Option<(u32, u32)> = None;
    for event in events {
        let rhythm = event.values.rhythm.unwrap_or_default();
        if rhythm.meter.is_some() {
            meter = rhythm.meter;
        }
        let (Some(eighth), Some((num, den))) = (rhythm.eighth_position, meter) else {
            continue;
        };
        let scaled = eighth as i64 * den as i64;
        if scaled % 8 != 0 {
            return Err(SheetError::Notation(format!("Eighth position {eighth} is off the {num}/{den} beat grid")));
        }
        let position = scaled / 8;
        if !(0 <= position && position < num as i64) {
            return Err(SheetError::Notation(format!("Eighth position {eighth} is outside meter ({num}, {den})")));
        }
        rows.push((event.time.unwrap_or(0.0), position + 1, num as i64, den as i64));
    }
    Ok(rows)
}

/// Монофоничный вид для нотации: нота срезается по началу следующей в том же
/// голосе, пустые выпадают.
pub fn notation_notes(notes: &[RawNote]) -> (Vec<RawNote>, Vec<String>) {
    let mut result = Vec::new();
    let mut diagnostics = Vec::new();
    for track in 0..2u32 {
        let mut ordered: Vec<RawNote> = notes.iter().copied().filter(|n| n.track == track).collect();
        ordered.sort_by(|a, b| {
            a.start.total_cmp(&b.start).then(a.pitch.cmp(&b.pitch)).then(a.end.total_cmp(&b.end))
        });
        for i in 0..ordered.len() {
            if i + 1 < ordered.len() && ordered[i].end > ordered[i + 1].start + 1e-6 {
                ordered[i].end = ordered[i + 1].start;
                diagnostics.push(format!(
                    "notation only: clipped track {track} note at {:.3} to next onset",
                    ordered[i].start
                ));
            }
            if ordered[i].end > ordered[i].start + 1e-6 {
                result.push(ordered[i]);
            }
        }
    }
    result.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then(a.end.total_cmp(&b.end))
            .then(a.pitch.cmp(&b.pitch))
            .then(a.track.cmp(&b.track))
    });
    (result, diagnostics)
}

/// Какие голоса оставить в партитуре. Выключенный голос становится паузами на
/// той же сетке — двухголосный формат сохраняется.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VoiceSelect {
    #[default]
    Both,
    Vocal,
    Instrumental,
}

/// Результат сборки ABC.
#[derive(Debug, Clone)]
pub struct AbcExport {
    pub abc: Option<String>,
    pub abc_error: Option<String>,
    pub measures: usize,
    pub diagnostics: Vec<String>,
    pub melody_notes: usize,
    pub vocal_notes: usize,
    pub instrumental_notes: usize,
    /// Партитура — для отладки и повторной сериализации.
    pub score: Option<Score>,
}

/// Склеенные события → ABC. Ошибка сборки не роняет транскрипцию, а
/// возвращается в `abc_error` (как у релиза).
pub fn export_abc(events: &[Event], duration: f64, melody_only: bool, voices: VoiceSelect) -> AbcExport {
    let notes = collect_notes(events, duration);
    let mut out = AbcExport {
        abc: None,
        abc_error: None,
        measures: 0,
        diagnostics: Vec::new(),
        melody_notes: notes.len(),
        vocal_notes: notes.iter().filter(|n| n.track == 0).count(),
        instrumental_notes: notes.iter().filter(|n| n.track == 1).count(),
        score: None,
    };
    match build(events, &notes, duration, melody_only, voices, &mut out.diagnostics) {
        Ok((text, score)) => {
            out.measures = score.measures.len();
            out.diagnostics.extend(score.diagnostics.iter().cloned());
            out.abc = Some(text);
            out.score = Some(score);
        }
        Err(e) => {
            out.abc_error = Some(match e {
                SheetError::Notation(msg) => msg,
                other => other.to_string(),
            });
        }
    }
    out
}

fn build(
    events: &[Event],
    notes: &[RawNote],
    duration: f64,
    melody_only: bool,
    voices: VoiceSelect,
    diagnostics: &mut Vec<String>,
) -> Result<(String, Score), SheetError> {
    use crate::vocab::Field;
    let beats = rhythm_rows(events)?;
    if beats.len() < 2 {
        return Err(SheetError::Notation("At least two decoded beats are required for ABC".into()));
    }
    // Последняя доля нотации — граница: продлеваем последним темпом до конца
    // записи или последней ноты, такт дополняется паузой.
    let mut abc_beats = beats.clone();
    let tail: Vec<f64> = beats[beats.len().saturating_sub(9)..].iter().map(|b| b.0).collect();
    let diffs: Vec<f64> = tail.windows(2).map(|w| w[1] - w[0]).collect();
    let period = median(&diffs);
    if !(period > 0.0) {
        return Err(SheetError::Notation("Decoded beats must increase in time".into()));
    }
    let end = notes.iter().map(|n| n.end).fold(duration.max(0.0), f64::max);
    while abc_beats.last().unwrap().0 < end - 1e-6 {
        let prev = *abc_beats.last().unwrap();
        abc_beats.push((prev.0 + period, prev.1 % prev.2 + 1, prev.2, prev.3));
    }
    let (first_beat, last_beat) = (abc_beats[0].0, abc_beats.last().unwrap().0);
    let clip = |rows: Vec<Interval>| -> Vec<Interval> {
        rows.into_iter()
            .filter(|(a, b, _)| *b > first_beat && *a < last_beat)
            .map(|(a, b, v)| (first_beat.max(a), last_beat.min(b), v))
            .collect()
    };
    let chords = clip(interval_rows(events, Field::Chord, duration));
    let keys = clip(interval_rows(events, Field::Key, duration));
    let structures = clip(interval_rows(events, Field::Structure, duration));
    if interval_rows(events, Field::Key, duration).is_empty() {
        return Err(SheetError::Notation("No key was decoded; cannot construct a keyed ABC score".into()));
    }
    let (clean, adjustments) = notation_notes(notes);
    diagnostics.extend(adjustments);
    // Через MIDI: тиковое округление. Нулевые после округления ноты pretty_midi
    // не читает как ноты — выпадают и здесь.
    let mut tracks: [Vec<MidiNote>; 2] = Default::default();
    for n in &clean {
        let (start, end) = (midi_quantize(n.start), midi_quantize(n.end));
        if end > start {
            tracks[n.track as usize].push(MidiNote { start, end, pitch: n.pitch });
        }
    }
    let mut score = notation::build_score(
        [&tracks[0], &tracks[1]],
        &abc_beats,
        &chords,
        &keys,
        &structures,
        melody_only,
    )?;
    match voices {
        VoiceSelect::Both => {}
        VoiceSelect::Vocal => score.voice_arrs[1].iter_mut().for_each(|v| *v = 0),
        VoiceSelect::Instrumental => score.voice_arrs[0].iter_mut().for_each(|v| *v = 0),
    }
    let text = notation::score_to_abc(&score)?;
    Ok((text, score))
}
