//! Сборка двухголосного ABC (`Vocal` и `Ins`) из долей, интервалов и нот —
//! порт `notation_sheetsage2.py` релиза, включая его самопроверку.
//!
//! Сетка: каждая доля делится на четыре шага; ноты, тональности и аккорды
//! квантуются на шаги по серединам между ними. Такты выводятся из сильных
//! долей: затакт и неполный последний такт дополняются паузой до объявленного
//! размера. Группы по 1–4 такта, новая группа — на смене размера, тональности
//! или секции (`% verse`). Длительности — только те, что понимают строгие
//! парсеры (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48 единиц `L:`), остальное
//! разбивается лигами.
//!
//! Этот ABC — родной диалект YuE2: его партитуру без аккордов модель берёт как
//! план кавера (`cot = melody`).

use std::collections::HashMap;

use crate::stitch::py_round;
use crate::SheetError;

pub const SUBBEAT_DIVISION: usize = 4;
pub const VOICE_IDS: [&str; 2] = ["Vocal", "Ins"];

fn err(msg: impl Into<String>) -> SheetError {
    SheetError::Notation(msg.into())
}

type R<T> = Result<T, SheetError>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeatEvent {
    pub time: f64,
    pub beat_id: i64,
    pub declared_numerator: i64,
    pub denominator: i64,
    pub line_no: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measure {
    pub index: usize,
    pub start_beat: usize,
    pub end_beat: usize,
    pub numerator: i64,
    pub denominator: i64,
    pub pickup: bool,
    pub partial: bool,
    pub inferred: bool,
    pub notated_numerator: Option<i64>,
    pub notated_denominator: Option<i64>,
    pub pad_before: bool,
}

impl Measure {
    pub fn start_t(&self) -> usize {
        self.start_beat * SUBBEAT_DIVISION
    }
    pub fn end_t(&self) -> usize {
        self.end_beat * SUBBEAT_DIVISION
    }
    pub fn abc_numerator(&self) -> i64 {
        self.notated_numerator.filter(|&n| n != 0).unwrap_or(self.numerator)
    }
    pub fn abc_denominator(&self) -> i64 {
        self.notated_denominator.filter(|&d| d != 0).unwrap_or(self.denominator)
    }
}

/// Нота MIDI-вида: секунды начала и конца, высота.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MidiNote {
    pub start: f64,
    pub end: f64,
    pub pitch: u32,
}

/// Интервал `[start, end)` со значением (аккорд, тональность, секция).
pub type Interval = (f64, f64, String);

#[derive(Debug, Clone)]
pub struct Score {
    pub beats: Vec<BeatEvent>,
    pub measures: Vec<Measure>,
    pub subbeat_times: Vec<f64>,
    pub subbeat_quarters: Vec<f64>,
    pub subbeat_denominators: Vec<i64>,
    pub key_arr: Vec<String>,
    pub chord_arr: Vec<String>,
    pub structure_events: Vec<(usize, String)>,
    /// `[Vocal, Ins]`: 0 — пауза, `2p+2` — тянется высота `p`, `2p+3` — атака.
    pub voice_arrs: [Vec<i32>; 2],
    pub diagnostics: Vec<String>,
}

// ── Словари нотации ───────────────────────────────────────────────────────

fn quality_to_abc(q: &str) -> Option<&'static str> {
    Some(match q {
        "maj" => "",
        "min" => "m",
        "dim" => "dim",
        "aug" => "aug",
        "7" => "7",
        "maj7" => "maj7",
        "min7" => "m7",
        "dim7" => "dim7",
        "hdim7" => "m7b5",
        "sus4" => "sus4",
        "sus2" => "sus2",
        "maj6" => "6",
        "min6" => "m6",
        "sus4(b7)" => "7sus4",
        "minmaj7" => "m(maj7)",
        _ => return None,
    })
}

const LETTERS: &str = "CDEFGAB";
const SHARP_NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const FLAT_NAMES: [&str; 12] = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];

fn natural_pc(letter: char) -> i64 {
    match letter {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => 0,
    }
}

fn key_signature_accidentals(key: &str) -> Option<i64> {
    Some(match key {
        "C" => 0,
        "G" => 1,
        "D" => 2,
        "A" => 3,
        "E" => 4,
        "B" => 5,
        "F#" => 6,
        "C#" => 7,
        "F" => -1,
        "Bb" => -2,
        "Eb" => -3,
        "Ab" => -4,
        "Db" => -5,
        "Gb" => -6,
        "Cb" => -7,
        "Am" => 0,
        "Em" => 1,
        "Bm" => 2,
        "F#m" => 3,
        "C#m" => 4,
        "G#m" => 5,
        "D#m" => 6,
        "A#m" => 7,
        "Dm" => -1,
        "Gm" => -2,
        "Cm" => -3,
        "Fm" => -4,
        "Bbm" => -5,
        "Ebm" => -6,
        "Abm" => -7,
        _ => return None,
    })
}

/// Написание высоты относительно тональности (двойные знаки — в далёких тональностях).
fn key_relative_names(count: i64) -> Option<[&'static str; 12]> {
    Some(match count {
        7 => ["B#", "C#", "C##", "D#", "D##", "E#", "F#", "F##", "G#", "G##", "A#", "B"],
        6 => ["B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "G##", "A#", "B"],
        5 => ["B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"],
        4 => ["B#", "C#", "D", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"],
        3 => ["B#", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"],
        2 => ["C", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"],
        1 => ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
        0 => ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"],
        -1 => ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"],
        -2 => ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
        -3 => ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
        -4 => ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"],
        -5 => ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"],
        -6 => ["C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"],
        -7 => ["C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "Bbb", "Bb", "Cb"],
        _ => return None,
    })
}

/// Разбор записи высоты `^[A-G](#{0,2}|b{0,2})$` → (класс высоты, буква, знаки).
fn pitch_class(root: &str) -> R<(i64, char, String)> {
    let mut chars = root.chars();
    let letter = chars.next().filter(|c| LETTERS.contains(*c));
    let accidental: String = chars.collect();
    let valid = matches!(accidental.as_str(), "" | "#" | "##" | "b" | "bb");
    let Some(letter) = letter.filter(|_| valid) else {
        return Err(err(format!("Invalid pitch spelling {root:?}")));
    };
    let offset = accidental.matches('#').count() as i64 - accidental.matches('b').count() as i64;
    Ok(((natural_pc(letter) + offset).rem_euclid(12), letter, accidental))
}

pub fn portable_pitch_name(root: &str, preserve_double: bool) -> R<String> {
    let (pc, _, accidental) = pitch_class(root)?;
    if preserve_double || accidental.len() <= 1 {
        return Ok(root.to_string());
    }
    let names = if accidental.starts_with('#') { SHARP_NAMES } else { FLAT_NAMES };
    Ok(names[pc as usize].to_string())
}

fn bass_degree_to_pitch(root: &str, degree_text: &str) -> R<String> {
    if pitch_class(degree_text).is_ok() {
        return portable_pitch_name(degree_text, true);
    }
    // ^(#{0,2}|b{0,2})([1-9]|1[0-3])$
    let digits_at = degree_text.find(|c: char| c.is_ascii_digit()).unwrap_or(degree_text.len());
    let (degree_accidental, digits) = degree_text.split_at(digits_at);
    let degree: i64 = digits.parse().unwrap_or(0);
    let accidental_ok = matches!(degree_accidental, "" | "#" | "##" | "b" | "bb");
    let digits_ok = !digits.is_empty() && !digits.starts_with('0') && (1..=13).contains(&degree);
    if !accidental_ok || !digits_ok {
        return Err(err(format!("Invalid chord bass degree {degree_text:?}")));
    }
    let (root_pc, root_letter, root_accidental) = pitch_class(root)?;
    const SCALE: [i64; 7] = [0, 2, 4, 5, 7, 9, 11];
    let mut interval = SCALE[((degree - 1) % 7) as usize] + 12 * ((degree - 1) / 7);
    interval += degree_accidental.matches('#').count() as i64 - degree_accidental.matches('b').count() as i64;
    let target_pc = (root_pc + interval).rem_euclid(12);
    let letter_index = (LETTERS.find(root_letter).unwrap() as i64 + degree - 1).rem_euclid(7) as usize;
    let target_letter = LETTERS.as_bytes()[letter_index] as char;
    let difference = (target_pc - natural_pc(target_letter) + 6).rem_euclid(12) - 6;
    if (-2..=2).contains(&difference) {
        let acc = match difference {
            -2 => "bb",
            -1 => "b",
            0 => "",
            1 => "#",
            _ => "##",
        };
        return Ok(format!("{target_letter}{acc}"));
    }
    let names = if format!("{root_accidental}{degree_accidental}").contains('#') { SHARP_NAMES } else { FLAT_NAMES };
    Ok(names[target_pc as usize].to_string())
}

/// Метка аккорда (`C:maj7/3`) → символ ABC (`Cmaj7/E`); `None` — «без аккорда».
pub fn chord_symbol_to_abc(chord: &str) -> R<Option<String>> {
    let chord = chord.trim();
    if matches!(chord, "N" | "X" | "?") {
        return Ok(None);
    }
    let Some((root, descriptor)) = chord.split_once(':') else {
        return Err(err(format!("Chord {chord:?} is missing the ':' quality separator")));
    };
    let (quality, bass) = match descriptor.split_once('/') {
        Some((q, b)) => (q, Some(b)),
        None => (descriptor, None),
    };
    let Some(suffix) = quality_to_abc(quality) else {
        return Err(err(format!("Unsupported chord quality {quality:?} in {chord:?}")));
    };
    let mut text = portable_pitch_name(root, true)? + suffix;
    if let Some(b) = bass.filter(|b| !b.is_empty()) {
        text.push('/');
        text.push_str(&bass_degree_to_pitch(root, b)?);
    }
    Ok(Some(text))
}

/// Тональность (`G:minor`, `Gm`, `G`) → поле `K:` (`Gm`).
pub fn key_symbol_to_abc(key: &str) -> R<String> {
    let key = key.trim();
    let (root, minor) = if let Some((r, mode)) = key.split_once(':') {
        match mode {
            "major" => (r, false),
            "minor" => (r, true),
            _ => return Err(err(format!("Unsupported key mode {mode:?} in {key:?}"))),
        }
    } else if let Some(r) = key.strip_suffix('m') {
        (r, true)
    } else {
        (key, false)
    };
    let (pc, _, accidental) = pitch_class(root)?;
    let suffix = if minor { "m" } else { "" };
    let candidate = portable_pitch_name(root, false)? + suffix;
    if key_signature_accidentals(&candidate).is_some() {
        return Ok(candidate);
    }
    let flats = accidental.contains('b');
    let names = if flats { FLAT_NAMES } else { SHARP_NAMES };
    let mut candidate = format!("{}{suffix}", names[pc as usize]);
    if key_signature_accidentals(&candidate).is_none() {
        let fallback = if flats { SHARP_NAMES } else { FLAT_NAMES };
        candidate = format!("{}{suffix}", fallback[pc as usize]);
    }
    if key_signature_accidentals(&candidate).is_none() {
        return Err(err(format!("Cannot encode portable ABC key for {key:?}")));
    }
    Ok(candidate)
}

pub fn get_key_accidentals(key: &str) -> R<[i64; 7]> {
    let count = key_signature_accidentals(key).ok_or_else(|| err(format!("Unsupported ABC key signature {key:?}")))?;
    let mut acc = [0i64; 7];
    let order = if count > 0 { "FCGDAEB" } else { "BEADGCF" };
    for letter in order.chars().take(count.unsigned_abs() as usize) {
        acc[LETTERS.find(letter).unwrap()] = if count > 0 { 1 } else { -1 };
    }
    Ok(acc)
}

/// Высота MIDI → нота ABC в тональности; знак пишется, только если меняет
/// состояние такта (знаки действуют на букву во всех октавах до тактовой черты).
pub fn note_to_abc(note: i64, key_acc: &[i64; 7], measure_acc: &mut HashMap<usize, i64>) -> R<String> {
    let count: i64 = key_acc.iter().sum();
    let names = key_relative_names(count)
        .ok_or_else(|| err(format!("Unsupported key signature accidental count {count}")))?;
    let name = names[note.rem_euclid(12) as usize];
    let letter = name.chars().next().unwrap();
    let accidental = &name[1..];
    let acc_number = match accidental {
        "" => 0,
        "#" => 1,
        "##" => 2,
        "b" => -1,
        _ => -2,
    };
    let mut octave = (note - 60).div_euclid(12);
    if note.rem_euclid(12) == 11 && acc_number == -1 {
        octave += 1;
    } else if note.rem_euclid(12) == 0 && acc_number == 1 {
        octave -= 1;
    }
    let scale_index = LETTERS.find(letter).unwrap();
    let current = *measure_acc.get(&scale_index).unwrap_or(&key_acc[scale_index]);
    let mut text = String::new();
    if current != acc_number {
        measure_acc.insert(scale_index, acc_number);
        text.push_str(match acc_number {
            -2 => "__",
            -1 => "_",
            0 => "=",
            1 => "^",
            _ => "^^",
        });
    }
    if octave > 0 {
        text.push(letter.to_ascii_lowercase());
        for _ in 1..octave {
            text.push('\'');
        }
    } else {
        text.push(letter);
        for _ in 0..(-octave) {
            text.push(',');
        }
    }
    Ok(text)
}

// ── Доли, такты, сетка ─────────────────────────────────────────────────────

fn parse_beats(rows: &[(f64, i64, i64, i64)]) -> R<Vec<BeatEvent>> {
    let mut beats: Vec<BeatEvent> = Vec::with_capacity(rows.len());
    for (i, &(time, beat_id, num, den)) in rows.iter().enumerate() {
        let line_no = i + 1;
        if beat_id < 1 {
            return Err(err(format!("beats:{line_no}: beat ID must be positive")));
        }
        if num < 1 {
            return Err(err(format!("beats:{line_no}: meter numerator must be positive")));
        }
        if den < 1 || den & (den - 1) != 0 {
            return Err(err(format!("beats:{line_no}: meter denominator must be a positive power of two")));
        }
        if let Some(prev) = beats.last() {
            if time <= prev.time {
                return Err(err(format!("beats:{line_no}: beat times must be strictly increasing")));
            }
        }
        beats.push(BeatEvent { time, beat_id, declared_numerator: num, denominator: den, line_no });
    }
    if beats.len() < 2 {
        return Err(err("beats: at least two beat events are required"));
    }
    Ok(beats)
}

fn parse_intervals(rows: &[Interval], what: &str) -> R<Vec<Interval>> {
    let mut out: Vec<Interval> = Vec::with_capacity(rows.len());
    let mut previous_end: Option<f64> = None;
    for (i, (start, end, value)) in rows.iter().enumerate() {
        if end <= start {
            return Err(err(format!("{what}:{}: {what} end must be after start", i + 1)));
        }
        if let Some(p) = previous_end {
            if *start < p - 1e-6 {
                return Err(err(format!("{what}:{}: overlapping {what} intervals", i + 1)));
            }
        }
        out.push((*start, *end, value.trim().to_string()));
        previous_end = Some(*end);
    }
    Ok(out)
}

fn mode_first_tiebreak(values: &[i64]) -> i64 {
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for &v in values {
        *counts.entry(v).or_default() += 1;
    }
    let max = counts.values().copied().max().unwrap_or(0);
    values.iter().copied().find(|v| counts[v] == max).unwrap_or(0)
}

/// Такты из сильных долей (доля с номером 1).
pub fn infer_measures(beats: &[BeatEvent]) -> R<(Vec<Measure>, Vec<String>)> {
    let downbeats: Vec<usize> = beats.iter().enumerate().filter(|(_, b)| b.beat_id == 1).map(|(i, _)| i).collect();
    if downbeats.is_empty() {
        return Err(err("No downbeat (beat ID 1) exists in the beat lab"));
    }
    let mut spans: Vec<(usize, usize, bool, bool)> = Vec::new();
    if downbeats[0] > 0 {
        spans.push((0, downbeats[0], true, false));
    }
    for w in downbeats.windows(2) {
        spans.push((w[0], w[1], false, false));
    }
    let last_down = *downbeats.last().unwrap();
    if last_down < beats.len() - 1 {
        // Последняя строка долей — граница конца; неполный последний такт.
        spans.push((last_down, beats.len() - 1, false, true));
    }
    if spans.is_empty() {
        return Err(err("No positive-length measure exists between downbeats"));
    }
    let mut diagnostics = Vec::new();
    let mut measures = Vec::with_capacity(spans.len());
    for (index, &(start, end, pickup, partial)) in spans.iter().enumerate() {
        let events = &beats[start..end];
        let count = events.len() as i64;
        if count < 1 {
            return Err(err(format!("Measure {index}: empty downbeat span")));
        }
        let ids: Vec<i64> = events.iter().map(|e| e.beat_id).collect();
        let consecutive = ids.iter().enumerate().all(|(k, &id)| id == ids[0] + k as i64);
        if !consecutive {
            return Err(err(format!(
                "Measure {index} (beat rows {}-{}): non-consecutive beat IDs {ids:?}",
                events[0].line_no,
                events[events.len() - 1].line_no
            )));
        }
        if !pickup && ids[0] != 1 {
            return Err(err(format!("Measure {index}: full measure does not start at beat ID 1")));
        }
        let denominators: Vec<i64> = events.iter().map(|e| e.denominator).collect();
        let denominator = mode_first_tiebreak(&denominators);
        let declared: Vec<i64> = events.iter().map(|e| e.declared_numerator).collect();
        let declared_numerator = mode_first_tiebreak(&declared);
        let numerator_conflict = declared.iter().any(|&v| v != count);
        let denominator_conflict = denominators.iter().any(|&v| v != denominator);
        let all_same_declared = declared.iter().all(|&v| v == declared[0]);
        let pad_final_partial =
            partial && all_same_declared && !denominator_conflict && declared_numerator >= count;
        let inferred = pickup || partial || numerator_conflict || denominator_conflict;
        if pad_final_partial && declared_numerator > count {
            diagnostics.push(format!(
                "measure {index}: padded final {count}/{denominator} span to declared {declared_numerator}/{denominator} with trailing rest"
            ));
        } else if numerator_conflict {
            diagnostics.push(format!(
                "measure {index}: inferred {count}/{denominator} from downbeat span; declared numerators were {}",
                py_list(&declared)
            ));
        }
        if denominator_conflict {
            diagnostics.push(format!(
                "measure {index}: placed denominator {denominator} at the measure boundary; row declarations were {}",
                py_list(&denominators)
            ));
        }
        measures.push(Measure {
            index,
            start_beat: start,
            end_beat: end,
            numerator: count,
            denominator,
            pickup,
            partial,
            inferred,
            notated_numerator: Some(if pad_final_partial { declared_numerator } else { count }),
            notated_denominator: None,
            pad_before: false,
        });
    }
    if measures.len() >= 2 {
        let first = measures[0];
        let following = measures[1];
        let first_duration = first.numerator as f64 / first.denominator as f64;
        let following_duration = following.abc_numerator() as f64 / following.abc_denominator() as f64;
        if first_duration < following_duration {
            measures[0] = Measure {
                inferred: true,
                notated_numerator: Some(following.abc_numerator()),
                notated_denominator: Some(following.abc_denominator()),
                pad_before: true,
                ..first
            };
            diagnostics.push(format!(
                "measure 0: padded leading {}/{} span to {}/{} with preceding rest",
                first.numerator,
                first.denominator,
                following.abc_numerator(),
                following.abc_denominator()
            ));
        }
    }
    Ok((measures, diagnostics))
}

fn py_list(values: &[i64]) -> String {
    let parts: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

fn build_grid(beats: &[BeatEvent], measures: &[Measure]) -> R<(Vec<f64>, Vec<f64>, Vec<i64>)> {
    let mut interval_den = vec![0i64; beats.len() - 1];
    for m in measures {
        for d in &mut interval_den[m.start_beat..m.end_beat] {
            *d = m.denominator;
        }
    }
    if interval_den.iter().any(|&d| d == 0) {
        return Err(err("Downbeat spans do not cover every beat interval"));
    }
    let mut times = Vec::with_capacity(interval_den.len() * SUBBEAT_DIVISION + 1);
    let mut dens = Vec::with_capacity(times.capacity());
    let mut quarters = vec![0.0f64];
    let mut current = 0.0f64;
    for i in 0..beats.len() - 1 {
        let (start, end) = (beats[i].time, beats[i + 1].time);
        let den = interval_den[i];
        // np.linspace(start, end, 5)[:-1] = start + k·step
        let step = (end - start) / SUBBEAT_DIVISION as f64;
        for k in 0..SUBBEAT_DIVISION {
            times.push(k as f64 * step + start);
        }
        dens.extend(std::iter::repeat(den).take(SUBBEAT_DIVISION));
        let quarter_step = 4.0 / den as f64 / SUBBEAT_DIVISION as f64;
        for _ in 0..SUBBEAT_DIVISION {
            current += quarter_step;
            quarters.push(current);
        }
    }
    times.push(beats[beats.len() - 1].time);
    dens.push(*interval_den.last().unwrap());
    Ok((times, quarters, dens))
}

/// Номер шага сетки для момента времени (`searchsorted` по серединам шагов).
fn quantize(time: f64, boundaries: &[f64]) -> usize {
    boundaries.partition_point(|&b| b < time)
}

fn boundaries(times: &[f64]) -> Vec<f64> {
    times.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect()
}

fn fill_intervals(rows: &[Interval], times: &[f64], bounds: &[f64], default: &str) -> R<Vec<String>> {
    let n = times.len();
    let mut out = vec![default.to_string(); n];
    for (start, end, value) in rows {
        let s = quantize(*start, bounds).min(n - 1);
        let e = quantize(*end, bounds).min(n - 1);
        if e <= s {
            return Err(err(format!(
                "Interval {start:.6}-{end:.6} ({value}) is shorter than the ABC subbeat grid"
            )));
        }
        for v in &mut out[s..e] {
            *v = value.clone();
        }
    }
    if n > 1 {
        out[n - 1] = out[n - 2].clone();
    }
    Ok(out)
}

fn notes_to_arr(notes: &[MidiNote], times: &[f64], bounds: &[f64], voice: &str) -> R<Vec<i32>> {
    let n = times.len();
    let mut out = vec![0i32; n];
    let mut sorted = notes.to_vec();
    sorted.sort_by(|a, b| {
        a.start.total_cmp(&b.start).then(a.end.total_cmp(&b.end)).then(a.pitch.cmp(&b.pitch))
    });
    for note in &sorted {
        let s = quantize(note.start, bounds).min(n - 1);
        let e = quantize(note.end, bounds).min(n - 1);
        if e <= s {
            return Err(err(format!(
                "{voice}: MIDI note pitch={} at {:.6}-{:.6} cannot be represented on the decoded subbeat grid",
                note.pitch, note.start, note.end
            )));
        }
        if out[s..e].iter().any(|&v| v != 0) {
            return Err(err(format!("{voice}: overlapping quantized melody notes at subbeats {s}:{e}")));
        }
        let sustain = note.pitch as i32 * 2 + 2;
        for v in &mut out[s..e] {
            *v = sustain;
        }
        out[s] = sustain + 1;
    }
    Ok(out)
}

/// Партитура из долей (`[время, номер доли, числитель, знаменатель]`), нот
/// голосов и интервалов. `melody_only` — аккорды не читаются вовсе.
pub fn build_score(
    voices: [&[MidiNote]; 2],
    beat_rows: &[(f64, i64, i64, i64)],
    chords: &[Interval],
    keys: &[Interval],
    structures: &[Interval],
    melody_only: bool,
) -> R<Score> {
    let beats = parse_beats(beat_rows)?;
    let mut key_rows = parse_intervals(keys, "key")?;
    for row in &mut key_rows {
        row.2 = key_symbol_to_abc(&row.2)?;
    }
    if key_rows.is_empty() {
        return Err(err("keys: at least one key interval is required"));
    }
    let structures = parse_intervals(structures, "structure")?;
    let chords = if melody_only {
        Vec::new()
    } else {
        let rows = parse_intervals(chords, "chord")?;
        for row in &rows {
            chord_symbol_to_abc(&row.2)?;
        }
        rows
    };
    let (measures, diagnostics) = infer_measures(&beats)?;
    let (times, quarters, dens) = build_grid(&beats, &measures)?;
    let bounds = boundaries(&times);
    let voice_arrs = [
        notes_to_arr(voices[0], &times, &bounds, VOICE_IDS[0])?,
        notes_to_arr(voices[1], &times, &bounds, VOICE_IDS[1])?,
    ];
    let key_default = key_rows[0].2.clone();
    let key_arr: Vec<String> = fill_intervals(&key_rows, &times, &bounds, &key_default)?
        .into_iter()
        .map(|k| k.chars().take(16).collect())
        .collect();
    let chord_arr = if melody_only {
        vec!["N".to_string(); times.len()]
    } else {
        fill_intervals(&chords, &times, &bounds, "N")?
            .into_iter()
            .map(|c| c.chars().take(64).collect())
            .collect()
    };
    let structure_events = structures
        .iter()
        .map(|(start, _, label)| (quantize(*start, &bounds).min(times.len() - 1), label.clone()))
        .collect();
    Ok(Score {
        beats,
        measures,
        subbeat_times: times,
        subbeat_quarters: quarters,
        subbeat_denominators: dens,
        key_arr,
        chord_arr,
        structure_events,
        voice_arrs,
        diagnostics,
    })
}

// ── Сериализация ──────────────────────────────────────────────────────────

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

pub fn abc_unit_denominator(score: &Score) -> R<i64> {
    let mut l = 1i64;
    for m in &score.measures {
        for d in [m.denominator, m.abc_denominator()] {
            let v = d * SUBBEAT_DIVISION as i64;
            l = l / gcd(l, v) * v;
        }
    }
    if l > 1024 {
        return Err(err(format!("Required ABC unit length 1/{l} is unreasonably small")));
    }
    Ok(l)
}

fn measure_actual_units(m: &Measure, unit: i64) -> i64 {
    m.numerator * unit / m.denominator
}

fn measure_abc_units(m: &Measure, unit: i64) -> i64 {
    m.abc_numerator() * unit / m.abc_denominator()
}

fn measure_padding_units(m: &Measure, unit: i64) -> i64 {
    measure_abc_units(m, unit) - measure_actual_units(m, unit)
}

fn duration_units(score: &Score, start_t: usize, end_t: usize, unit: i64) -> R<i64> {
    let mut units = 0;
    for &den in &score.subbeat_denominators[start_t..end_t] {
        let divisor = den * SUBBEAT_DIVISION as i64;
        if unit % divisor != 0 {
            return Err(err(format!("ABC L:1/{unit} cannot express a 1/{divisor} subbeat exactly")));
        }
        units += unit / divisor;
    }
    Ok(units)
}

pub fn estimate_tempo(score: &Score) -> R<f64> {
    let seconds = score.subbeat_times[score.subbeat_times.len() - 1] - score.subbeat_times[0];
    let quarters = score.subbeat_quarters[score.subbeat_quarters.len() - 1] - score.subbeat_quarters[0];
    if seconds <= 0.0 || quarters <= 0.0 {
        return Err(err("Cannot estimate tempo from a zero-duration score"));
    }
    Ok(quarters / seconds * 60.0)
}

fn continues_pitch(value: i32, next: i32) -> bool {
    if value <= 0 {
        return false;
    }
    let pitch = value.div_euclid(2) - 1;
    next == pitch * 2 + 2
}

fn same_note_segment(value: i32, next: i32) -> bool {
    if value == 0 {
        return next == 0;
    }
    let pitch = value.div_euclid(2) - 1;
    next == pitch * 2 + 2
}

const SUPPORTED_DURATIONS: [i64; 11] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48];

fn split_duration_units(duration: i64) -> R<Vec<i64>> {
    if duration <= 0 {
        return Err(err(format!("Cannot serialize non-positive duration {duration}")));
    }
    let mut out = Vec::new();
    let mut remaining = duration;
    while remaining > 0 {
        if SUPPORTED_DURATIONS.contains(&remaining) {
            out.push(remaining);
            break;
        }
        let Some(chunk) = SUPPORTED_DURATIONS.iter().copied().filter(|&v| v < remaining).max() else {
            return Err(err(format!("Duration {duration} cannot be split into representable ABC values")));
        };
        out.push(chunk);
        remaining -= chunk;
    }
    Ok(out)
}

fn render_duration_tokens(prefix: &str, note: &str, duration: i64, tie_out: bool) -> R<Vec<String>> {
    let chunks = split_duration_units(duration)?;
    let n = chunks.len();
    Ok(chunks
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| {
            let continues = note != "z" && (i + 1 < n || tie_out);
            let mut s = String::new();
            if i == 0 {
                s.push_str(prefix);
            }
            s.push_str(note);
            if chunk != 1 {
                s.push_str(&chunk.to_string());
            }
            if continues {
                s.push('-');
            }
            s
        })
        .collect())
}

fn render_voice_measure(score: &Score, voice: usize, m: &Measure, unit: i64) -> R<String> {
    let arr = &score.voice_arrs[voice];
    let show_chords = voice == 0;
    let mut measure_acc: HashMap<usize, i64> = HashMap::new();
    let mut current_key = score.key_arr[m.start_t()].clone();
    let mut key_acc = get_key_accidentals(&current_key)?;
    let mut parts: Vec<String> = Vec::new();
    let padding = measure_padding_units(m, unit);
    if padding < 0 {
        return Err(err(format!("Measure {}: notated meter is shorter than its decoded span", m.index)));
    }
    let mut leading = if m.pad_before { padding } else { 0 };
    let mut trailing = if m.pad_before { 0 } else { padding };
    let (start_t, end_t) = (m.start_t(), m.end_t());
    let mut t = start_t;
    while t < end_t {
        let mut next_t = end_t;
        if let Some(p) = (t + 1..end_t).find(|&p| !same_note_segment(arr[t], arr[p])) {
            next_t = next_t.min(p);
        }
        if let Some(p) = (t + 1..end_t).find(|&p| score.key_arr[p] != score.key_arr[p - 1]) {
            next_t = next_t.min(p);
        }
        if show_chords {
            if let Some(p) = (t + 1..end_t).find(|&p| score.chord_arr[p] != score.chord_arr[p - 1]) {
                next_t = next_t.min(p);
            }
        }
        let mut prefix = String::new();
        let key = &score.key_arr[t];
        if t > start_t && *key != current_key {
            current_key = key.clone();
            key_acc = get_key_accidentals(&current_key)?;
            measure_acc.clear();
            prefix.push_str(&format!("[K:{current_key}]"));
        }
        if show_chords && (t == start_t || score.chord_arr[t] != score.chord_arr[t - 1]) {
            if let Some(text) = chord_symbol_to_abc(&score.chord_arr[t])? {
                prefix.push_str(&format!("\"{text}\""));
            }
        }
        let value = arr[t];
        let note_text = if value == 0 {
            "z".to_string()
        } else {
            note_to_abc((value.div_euclid(2) - 1) as i64, &key_acc, &mut measure_acc)?
        };
        let mut duration = duration_units(score, t, next_t, unit)?;
        if t == start_t && leading != 0 {
            if value == 0 && prefix.is_empty() {
                duration += leading;
            } else {
                parts.extend(render_duration_tokens("", "z", leading, false)?);
            }
            leading = 0;
        }
        if value == 0 && next_t == end_t && trailing != 0 {
            duration += trailing;
            trailing = 0;
        }
        if duration <= 0 {
            return Err(err(format!("Non-positive ABC duration at subbeats {t}:{next_t}")));
        }
        let tie_out = value > 0 && next_t < arr.len() && continues_pitch(value, arr[next_t]);
        parts.extend(render_duration_tokens(&prefix, &note_text, duration, tie_out)?);
        t = next_t;
    }
    if leading != 0 {
        return Err(err(format!("Measure {}: leading rest padding was not serialized", m.index)));
    }
    if trailing != 0 {
        parts.extend(render_duration_tokens("", "z", trailing, false)?);
    }
    Ok(parts.concat())
}

/// Элемент такта по регулярке релиза:
/// `"…"` | `[K:…]` | `[_=^]*[A-Ga-gz][,']*` + `\d*` + `-?`.
#[derive(Debug, Clone, PartialEq)]
enum Element {
    Quoted(String),
    Key(String),
    Note { note: String, duration: String, tie: bool },
}

/// Разбор такта в элементы; `Err(позиция)` — там, где регулярка бы не совпала
/// (пропуск символа в `finditer`).
fn scan_elements(s: &str) -> Vec<(usize, usize, Element)> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        // "(?P<quoted>[^"]*)"
        if b[i] == b'"' {
            if let Some(close) = s[i + 1..].find('"') {
                let end = i + 1 + close + 1;
                out.push((i, end, Element::Quoted(s[i + 1..i + 1 + close].to_string())));
                i = end;
                continue;
            }
        }
        // \[K:(?P<key>[^\]]+)\]
        if s[i..].starts_with("[K:") {
            let body_start = i + 3;
            let body_len = s[body_start..].find(']').unwrap_or(0);
            if body_len > 0 {
                let end = body_start + body_len + 1;
                out.push((i, end, Element::Key(s[body_start..body_start + body_len].to_string())));
                i = end;
                continue;
            }
        }
        // (?P<note>[_=^]*[A-Ga-gz][,']*)(?P<duration>\d*)(?P<tie>-?)
        let mut j = i;
        while j < b.len() && matches!(b[j], b'_' | b'=' | b'^') {
            j += 1;
        }
        if j < b.len() && (matches!(b[j], b'A'..=b'G' | b'a'..=b'g') || b[j] == b'z') {
            j += 1;
            while j < b.len() && matches!(b[j], b',' | b'\'') {
                j += 1;
            }
            let note = s[i..j].to_string();
            let d0 = j;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            let duration = s[d0..j].to_string();
            let tie = j < b.len() && b[j] == b'-';
            if tie {
                j += 1;
            }
            out.push((i, j, Element::Note { note, duration, tie }));
            i = j;
            continue;
        }
        i += 1;
    }
    out
}

fn is_compressible_full_rest(rendered: &str) -> bool {
    let mut cursor = 0usize;
    let mut saw_note = false;
    for (start, end, el) in scan_elements(rendered) {
        if start != cursor {
            return false;
        }
        cursor = end;
        match el {
            Element::Quoted(_) | Element::Key(_) => return false,
            Element::Note { note, tie, .. } => {
                saw_note = true;
                if note != "z" || tie {
                    return false;
                }
            }
        }
    }
    saw_note && cursor == rendered.len()
}

fn render_voice_group(score: &Score, voice: usize, measures: &[Measure], unit: i64) -> R<String> {
    let rendered: Vec<String> =
        measures.iter().map(|m| render_voice_measure(score, voice, m, unit)).collect::<R<_>>()?;
    let mut out = String::new();
    let mut i = 0usize;
    while i < rendered.len() {
        if !is_compressible_full_rest(&rendered[i]) {
            out.push_str(&rendered[i]);
            out.push('|');
            i += 1;
            continue;
        }
        let mut end = i + 1;
        while end < rendered.len() && is_compressible_full_rest(&rendered[end]) {
            end += 1;
        }
        let count = end - i;
        out.push('Z');
        if count > 1 {
            out.push_str(&count.to_string());
        }
        out.push('|');
        i = end;
    }
    Ok(out)
}

#[derive(Debug, Clone)]
struct MeasureGroup {
    measures: Vec<Measure>,
    structure_labels: Vec<String>,
    meter_changed: bool,
    key_changed: bool,
}

fn sanitize_label(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn measure_groups(score: &Score) -> Vec<MeasureGroup> {
    let first = &score.measures[0];
    let mut active_meter = (first.abc_numerator(), first.abc_denominator());
    let mut active_key = score.key_arr[first.start_t()].clone();
    let mut active_structure = String::new();
    let mut groups: Vec<MeasureGroup> = Vec::new();
    for m in &score.measures {
        let meter = (m.abc_numerator(), m.abc_denominator());
        let key = score.key_arr[m.start_t()].clone();
        let meter_changed = meter != active_meter;
        let key_changed = key != active_key;
        let mut new_labels = Vec::new();
        for (t, label) in &score.structure_events {
            if !(m.start_t() <= *t && *t < m.end_t()) {
                continue;
            }
            let clean = sanitize_label(label);
            if !clean.is_empty() && clean != active_structure {
                new_labels.push(clean.clone());
                active_structure = clean;
            }
        }
        let start_group = groups.last().map(|g| g.measures.len() >= 4).unwrap_or(true)
            || meter_changed
            || key_changed
            || !new_labels.is_empty();
        if start_group {
            groups.push(MeasureGroup {
                measures: vec![*m],
                structure_labels: new_labels,
                meter_changed,
                key_changed,
            });
        } else {
            groups.last_mut().unwrap().measures.push(*m);
        }
        active_meter = meter;
        active_key = score.key_arr[m.end_t() - 1].clone();
    }
    groups
}

/// Партитура → текст ABC (с самопроверкой, как у релиза).
pub fn score_to_abc(score: &Score) -> R<String> {
    let unit = abc_unit_denominator(score)?;
    let first = &score.measures[0];
    let first_key = &score.key_arr[first.start_t()];
    let tempo = py_round(estimate_tempo(score)?) as i64;
    let mut lines: Vec<String> = vec![
        "X:1".into(),
        "T:".into(),
        format!("M:{}/{}", first.abc_numerator(), first.abc_denominator()),
        format!("L:1/{unit}"),
        format!("Q:1/4={tempo}"),
        "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"".into(),
        "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"".into(),
        format!("K:{first_key}"),
    ];
    for group in measure_groups(score) {
        lines.extend(group.structure_labels.iter().map(|l| format!("% {l}")));
        let head = &group.measures[0];
        for (voice, id) in VOICE_IDS.iter().enumerate() {
            lines.push(format!("V: {id}"));
            if group.meter_changed {
                lines.push(format!("M:{}/{}", head.abc_numerator(), head.abc_denominator()));
            }
            if group.key_changed {
                lines.push(format!("K:{}", score.key_arr[head.start_t()]));
            }
            lines.push(render_voice_group(score, voice, &group.measures, unit)?);
        }
    }
    let text = lines.join("\n") + "\n";
    validate_serialized_abc(&text, score)?;
    Ok(text)
}

// ── Самопроверка ──────────────────────────────────────────────────────────

fn parse_music_measure(line: &str, expected: i64, context: &str) -> R<(Vec<(i64, String)>, Vec<(i64, String)>)> {
    if line == "Z" {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut position = 0i64;
    let mut cursor = 0usize;
    let mut quoted = Vec::new();
    let mut keys = Vec::new();
    for (start, end, el) in scan_elements(line) {
        let gap = &line[cursor..start];
        if !gap.trim().is_empty() {
            return Err(err(format!("{context}: unsupported serialized ABC tokens {gap:?}")));
        }
        cursor = end;
        match el {
            Element::Quoted(q) => quoted.push((position, q)),
            Element::Key(k) => keys.push((position, k)),
            Element::Note { note, duration, tie } => {
                if tie && note == "z" {
                    return Err(err(format!("{context}: a rest cannot be tied")));
                }
                let d: i64 = if duration.is_empty() { 1 } else { duration.parse().unwrap_or(0) };
                if !SUPPORTED_DURATIONS.contains(&d) {
                    return Err(err(format!("{context}: duration {d} is not parser-representable")));
                }
                position += d;
            }
        }
    }
    if !line[cursor..].trim().is_empty() {
        return Err(err(format!("{context}: unsupported serialized ABC tokens {:?}", &line[cursor..])));
    }
    if position != expected {
        return Err(err(format!("{context}: duration {position} does not match meter duration {expected}")));
    }
    // (^|[\s|])-[_=^A-Ga-g]
    let bytes = line.as_bytes();
    for (i, &c) in bytes.iter().enumerate() {
        if c == b'-'
            && (i == 0 || bytes[i - 1].is_ascii_whitespace() || bytes[i - 1] == b'|')
            && bytes.get(i + 1).map(|&n| matches!(n, b'_' | b'=' | b'^' | b'A'..=b'G' | b'a'..=b'g')).unwrap_or(false)
        {
            return Err(err(format!("{context}: tie is written before its second note")));
        }
    }
    Ok((quoted, keys))
}

fn expected_measure_chords(score: &Score, m: &Measure, unit: i64) -> R<Vec<(i64, String)>> {
    let leading = if m.pad_before { measure_padding_units(m, unit) } else { 0 };
    let mut out = Vec::new();
    for t in m.start_t()..m.end_t() {
        if t != m.start_t() && score.chord_arr[t] == score.chord_arr[t - 1] {
            continue;
        }
        let position = leading + duration_units(score, m.start_t(), t, unit)?;
        if let Some(text) = chord_symbol_to_abc(&score.chord_arr[t])? {
            out.push((position, text));
        }
    }
    Ok(out)
}

fn expected_measure_keys(score: &Score, m: &Measure, unit: i64) -> R<Vec<(i64, String)>> {
    let leading = if m.pad_before { measure_padding_units(m, unit) } else { 0 };
    let mut out = Vec::new();
    for t in m.start_t() + 1..m.end_t() {
        if score.key_arr[t] != score.key_arr[t - 1] {
            out.push((leading + duration_units(score, m.start_t(), t, unit)?, score.key_arr[t].clone()));
        }
    }
    Ok(out)
}

type VoiceBlock = (usize, Vec<(String, String)>, Vec<String>);

fn parse_voice_group(lines: &[&str], mut cursor: usize, voice: &str, group: usize) -> R<VoiceBlock> {
    let expected = format!("V: {voice}");
    if lines.get(cursor).copied() != Some(expected.as_str()) {
        let observed = lines.get(cursor).copied().unwrap_or("<end>");
        return Err(err(format!("Group {group}: expected {expected}, got {observed:?}")));
    }
    cursor += 1;
    let mut fields: Vec<(String, String)> = Vec::new();
    while cursor < lines.len() && (lines[cursor].starts_with("M:") || lines[cursor].starts_with("K:")) {
        let (name, value) = lines[cursor].split_once(':').unwrap();
        if fields.iter().any(|(n, _)| n == name) {
            return Err(err(format!("Group {group} {voice}: repeated {name}: field")));
        }
        fields.push((name.to_string(), value.to_string()));
        cursor += 1;
    }
    let Some(music) = lines.get(cursor).copied() else {
        return Err(err(format!("Group {group} {voice}: missing music line")));
    };
    if ["V:", "M:", "K:", "%"].iter().any(|p| music.starts_with(p)) {
        return Err(err(format!("Group {group} {voice}: invalid music line {music:?}")));
    }
    cursor += 1;
    let split: Vec<&str> = music.split('|').collect();
    if split.last().map(|s| !s.trim().is_empty()).unwrap_or(true) {
        return Err(err(format!("Group {group} {voice}: music line must end with a barline")));
    }
    let serialized: Vec<&str> = split[..split.len() - 1].iter().map(|b| b.trim()).collect();
    if serialized.iter().any(|b| b.is_empty()) {
        return Err(err(format!("Group {group} {voice}: empty serialized measure")));
    }
    let mut bars = Vec::new();
    for bar in serialized {
        let compressed = bar.strip_prefix('Z').filter(|rest| rest.is_empty() || matches!(*rest, "1" | "2" | "3" | "4"));
        match compressed {
            None => bars.push(bar.to_string()),
            Some("1") => return Err(err(format!("Group {group} {voice}: Z1 must be written as Z"))),
            Some(rest) => {
                let count: usize = if rest.is_empty() { 1 } else { rest.parse().unwrap() };
                bars.extend(std::iter::repeat("Z".to_string()).take(count));
            }
        }
    }
    if !(1..=4).contains(&bars.len()) {
        return Err(err(format!("Group {group} {voice}: expected 1-4 semantic measures")));
    }
    Ok((cursor, fields, bars))
}

/// Инварианты, которые обязаны держаться до записи ABC (как у релиза).
pub fn validate_serialized_abc(text: &str, score: &Score) -> R<()> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.first().copied() != Some("X:1") {
        return Err(err("ABC must start with X:1"));
    }
    if lines.get(1).copied() != Some("T:") {
        return Err(err("ABC title must be fixed as empty T:"));
    }
    let header_voices: Vec<&str> = lines
        .iter()
        .filter_map(|l| {
            l.strip_prefix("V: Vocal ").map(|_| "Vocal").or_else(|| l.strip_prefix("V: Ins ").map(|_| "Ins"))
        })
        .collect();
    if header_voices != VOICE_IDS {
        return Err(err(format!("Expected fixed Vocal/Ins voice definitions, got {header_voices:?}")));
    }
    let ins_header = lines.iter().position(|l| l.starts_with("V: Ins ")).unwrap();
    let header_key_index = lines
        .iter()
        .enumerate()
        .position(|(i, l)| i > 0 && l.starts_with("K:") && ins_header < i)
        .ok_or_else(|| err("ABC header K: field is missing"))?;
    let first = &score.measures[0];
    let expected_meter = format!("M:{}/{}", first.abc_numerator(), first.abc_denominator());
    let meters: Vec<&str> = lines[..=header_key_index].iter().copied().filter(|l| l.starts_with("M:")).collect();
    if meters != [expected_meter.as_str()] {
        return Err(err(format!("ABC header meters {meters:?} != [{expected_meter:?}]")));
    }
    let expected_key = format!("K:{}", score.key_arr[first.start_t()]);
    let keys: Vec<&str> = lines[..=header_key_index].iter().copied().filter(|l| l.starts_with("K:")).collect();
    if keys != [expected_key.as_str()] {
        return Err(err(format!("ABC header keys {keys:?} != [{expected_key:?}]")));
    }
    let unit = abc_unit_denominator(score)?;
    let mut cursor = header_key_index + 1;
    for (gi, group) in measure_groups(score).iter().enumerate() {
        let mut labels = Vec::new();
        while cursor < lines.len() && lines[cursor].starts_with("% ") {
            labels.push(lines[cursor][2..].trim().to_string());
            cursor += 1;
        }
        if labels != group.structure_labels {
            return Err(err(format!("Group {gi}: structure labels {labels:?} != {:?}", group.structure_labels)));
        }
        let (c, vocal_fields, vocal_bars) = parse_voice_group(&lines, cursor, "Vocal", gi)?;
        let (c, ins_fields, ins_bars) = parse_voice_group(&lines, c, "Ins", gi)?;
        cursor = c;
        if vocal_fields != ins_fields {
            return Err(err(format!("Group {gi}: meter/key changes must be scoped to both voices")));
        }
        let head = &group.measures[0];
        let mut expected_fields: Vec<(String, String)> = Vec::new();
        if group.meter_changed {
            expected_fields.push(("M".into(), format!("{}/{}", head.abc_numerator(), head.abc_denominator())));
        }
        if group.key_changed {
            expected_fields.push(("K".into(), score.key_arr[head.start_t()].clone()));
        }
        let mut sorted_fields = vocal_fields.clone();
        sorted_fields.sort();
        let mut sorted_expected = expected_fields.clone();
        sorted_expected.sort();
        if sorted_fields != sorted_expected {
            return Err(err(format!("Group {gi}: fields {vocal_fields:?} != required changes {expected_fields:?}")));
        }
        if vocal_bars.len() != group.measures.len() || ins_bars.len() != group.measures.len() {
            return Err(err(format!("Group {gi}: both voices must contain {} measures", group.measures.len())));
        }
        for (bi, m) in group.measures.iter().enumerate() {
            let expected = m.abc_numerator() * unit / m.abc_denominator();
            for (voice, bars) in [("Vocal", &vocal_bars), ("Ins", &ins_bars)] {
                let context = format!("measure {} {voice}", m.index);
                let (quoted, key_events) = parse_music_measure(&bars[bi], expected, &context)?;
                if key_events != expected_measure_keys(score, m, unit)? {
                    return Err(err(format!("Measure {} {voice}: inline keys differ", m.index)));
                }
                if voice == "Vocal" {
                    if quoted != expected_measure_chords(score, m, unit)? {
                        return Err(err(format!("Measure {}: chord symbols differ", m.index)));
                    }
                } else if !quoted.is_empty() {
                    return Err(err(format!("Measure {}: chords must only be in Vocal", m.index)));
                }
            }
        }
    }
    if cursor != lines.len() {
        return Err(err(format!(
            "Unexpected trailing ABC body lines: {:?}",
            &lines[cursor..lines.len().min(cursor + 5)]
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chord_symbols_match_release() {
        assert_eq!(chord_symbol_to_abc("C:maj").unwrap().as_deref(), Some("C"));
        assert_eq!(chord_symbol_to_abc("A:min7").unwrap().as_deref(), Some("Am7"));
        assert_eq!(chord_symbol_to_abc("C:maj/3").unwrap().as_deref(), Some("C/E"));
        assert_eq!(chord_symbol_to_abc("D:min/b3").unwrap().as_deref(), Some("Dm/F"));
        assert_eq!(chord_symbol_to_abc("F#:7/b7").unwrap().as_deref(), Some("F#7/E"));
        assert_eq!(chord_symbol_to_abc("G:sus4(b7)").unwrap().as_deref(), Some("G7sus4"));
        assert_eq!(chord_symbol_to_abc("N").unwrap(), None);
    }

    #[test]
    fn keys_match_release() {
        assert_eq!(key_symbol_to_abc("G:minor").unwrap(), "Gm");
        assert_eq!(key_symbol_to_abc("A#:major").unwrap(), "Bb");
        assert_eq!(key_symbol_to_abc("D#:minor").unwrap(), "D#m");
        assert_eq!(key_symbol_to_abc("G#:major").unwrap(), "Ab");
    }

    #[test]
    fn notes_use_key_relative_spelling() {
        let acc = get_key_accidentals("Gm").unwrap();
        let mut bar = HashMap::new();
        assert_eq!(note_to_abc(70, &acc, &mut bar).unwrap(), "B"); // B♭ по ключу
        assert_eq!(note_to_abc(71, &acc, &mut bar).unwrap(), "=B");
        assert_eq!(note_to_abc(59, &acc, &mut bar).unwrap(), "B,"); // бекар действует на букву
        assert_eq!(note_to_abc(74, &acc, &mut bar).unwrap(), "d");
        assert_eq!(note_to_abc(86, &acc, &mut bar).unwrap(), "d'");
    }

    #[test]
    fn durations_split_like_release() {
        assert_eq!(split_duration_units(10).unwrap(), vec![8, 2]);
        assert_eq!(split_duration_units(64).unwrap(), vec![48, 16]);
        assert_eq!(render_duration_tokens("\"C\"", "E", 10, false).unwrap(), vec!["\"C\"E8-", "E2"]);
    }
}
