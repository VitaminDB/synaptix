//! Родная партитура YuE2: ограниченный двухголосный диалект ABC (`Vocal`, `Ins`).
//!
//! Порт официальных инструментов YuE2 (`skills/yue2-music/instrumental/scripts`:
//! `abc_tools.py`, `compile_score.py`, `instrumentalize.py`) один к одному:
//! разбор диалекта в точные события (время — дроби четвертей), сборка
//! партитуры из событий, перенос мелодии `Vocal` → `Ins` (родной путь
//! инструментала), снятие аккордов (кавер в режиме `melody`), метки разделов
//! для лирики и нормализация стиля инструментала. Это не общий парсер ABC:
//! всё, что вне диалекта, — ошибка, как и в оригинале.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

/// Голоса родной партитуры по порядку.
pub const VOICES: [&str; 2] = ["Vocal", "Ins"];
/// Допустимые длительности в единицах `L:`.
pub const DURATIONS: [i64; 11] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48];
const QUALITIES: [&str; 15] = ["", "m", "dim", "aug", "7", "maj7", "m7", "dim7", "m7b5", "sus4", "sus2", "6", "m6", "7sus4", "m(maj7)"];
/// Метки разделов (`% verse`), которые знает модель.
pub const SECTIONS: [&str; 8] = ["intro", "verse", "pre-chorus", "chorus", "bridge", "interlude", "outro", "instrumental"];
const VOCAL_DEF: &str = "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"";
const INS_DEF: &str = "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"";

/// Ошибка диалекта: неподдерживаемая запись или нарушенный инвариант.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbcError(pub String);

impl fmt::Display for AbcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AbcError {}

pub type R<T> = Result<T, AbcError>;

fn fail(cond: bool, msg: impl FnOnce() -> String) -> R<()> {
    if cond {
        Err(AbcError(msg()))
    } else {
        Ok(())
    }
}

/// Точная дробь (как `fractions.Fraction`): знаменатель > 0, несократима.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Q {
    n: i64,
    d: i64,
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn lcm(a: i64, b: i64) -> i64 {
    a / gcd(a, b) * b
}

impl Q {
    pub fn new(n: i64, d: i64) -> Q {
        assert!(d != 0, "дробь со знаменателем 0");
        let s = if d < 0 { -1 } else { 1 };
        let g = gcd(n, d).max(1);
        Q { n: s * n / g, d: s * d / g }
    }
    pub const ZERO: Q = Q { n: 0, d: 1 };
    pub fn int(n: i64) -> Q {
        Q { n, d: 1 }
    }
    pub fn numer(self) -> i64 {
        self.n
    }
    pub fn denom(self) -> i64 {
        self.d
    }
    /// Разбор `"3/2"`, `"5"`, `"-1/4"` (как `Fraction(str)` для этих форм).
    pub fn parse(s: &str) -> Option<Q> {
        match s.trim().split_once('/') {
            Some((a, b)) => {
                let (n, d): (i64, i64) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
                (d != 0).then(|| Q::new(n, d))
            }
            None => s.trim().parse().ok().map(Q::int),
        }
    }
    pub fn to_f64(self) -> f64 {
        self.n as f64 / self.d as f64
    }
}

impl fmt::Display for Q {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.d == 1 {
            write!(f, "{}", self.n)
        } else {
            write!(f, "{}/{}", self.n, self.d)
        }
    }
}

impl Ord for Q {
    fn cmp(&self, o: &Q) -> Ordering {
        ((self.n as i128) * (o.d as i128)).cmp(&((o.n as i128) * (self.d as i128)))
    }
}

impl PartialOrd for Q {
    fn partial_cmp(&self, o: &Q) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl std::ops::Add for Q {
    type Output = Q;
    fn add(self, o: Q) -> Q {
        Q::new(self.n * o.d + o.n * self.d, self.d * o.d)
    }
}

impl std::ops::Sub for Q {
    type Output = Q;
    fn sub(self, o: Q) -> Q {
        Q::new(self.n * o.d - o.n * self.d, self.d * o.d)
    }
}

impl std::ops::Mul for Q {
    type Output = Q;
    fn mul(self, o: Q) -> Q {
        Q::new(self.n * o.n, self.d * o.d)
    }
}

impl std::ops::Div for Q {
    type Output = Q;
    fn div(self, o: Q) -> Q {
        Q::new(self.n * o.d, self.d * o.n)
    }
}

/// Буква → высота ноты в до-мажоре.
fn natural(letter: char) -> i32 {
    match letter {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => unreachable!("буква ноты"),
    }
}

const LETTERS: [char; 7] = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];

/// Число знаков тональности (−7…7) или `None`.
fn key_count(key: &str) -> Option<i32> {
    const MAJ: [&str; 15] = ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"];
    const MIN: [&str; 15] = ["Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m"];
    MAJ.iter().position(|k| *k == key).or_else(|| MIN.iter().position(|k| *k == key)).map(|i| i as i32 - 7)
}

/// Имя тональности по числу знаков и ладу (как MIDI `key signature`).
pub fn key_name(sharps: i32, minor: bool) -> Option<&'static str> {
    const MAJ: [&str; 15] = ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"];
    const MIN: [&str; 15] = ["Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m"];
    let i = usize::try_from(sharps + 7).ok().filter(|i| *i < 15)?;
    Some(if minor { MIN[i] } else { MAJ[i] })
}

/// Знаки тональности по буквам (+1 диез, −1 бемоль).
pub fn key_accidentals(key: &str) -> R<BTreeMap<char, i32>> {
    let count = key_count(key).ok_or_else(|| AbcError(format!("Unsupported key {key:?}; use a standard major or minor K: field")))?;
    let mut out: BTreeMap<char, i32> = LETTERS.iter().map(|l| (*l, 0)).collect();
    let order: Vec<char> = if count > 0 { "FCGDAEB".chars().collect() } else { "BEADGCF".chars().collect() };
    for l in order.into_iter().take(count.unsigned_abs() as usize) {
        out.insert(l, if count > 0 { 1 } else { -1 });
    }
    Ok(out)
}

fn positive_int(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    (!b.is_empty() && b[0] != b'0' && b.iter().all(|c| c.is_ascii_digit())).then(|| s.parse().ok()).flatten()
}

/// Размер `n/d` (d — степень двойки ≤ 1024).
pub fn meter_value(text: &str) -> R<(i64, i64)> {
    let parsed = text.split_once('/').and_then(|(a, b)| Some((positive_int(a)?, positive_int(b)?)));
    let (n, d) = parsed.ok_or_else(|| AbcError(format!("Unsupported meter {text:?}; write an explicit fraction")))?;
    fail(d > 1024 || d & (d - 1) != 0, || format!("Unsupported meter denominator {d}"))?;
    Ok((n, d))
}

fn pitch_name_ok(s: &str) -> bool {
    let mut c = s.chars();
    matches!(c.next(), Some('A'..='G')) && matches!(c.as_str(), "" | "b" | "#" | "bb" | "##")
}

/// Символ аккорда из словаря модели (корень, качество, необязательный бас).
pub fn chord_ok(s: &str) -> bool {
    let (body, bass) = match s.split_once('/') {
        Some((a, b)) => (a, Some(b)),
        None => (s, None),
    };
    if bass.is_some_and(|b| !pitch_name_ok(b)) {
        return false;
    }
    (1..=3).any(|i| body.is_char_boundary(i) && i <= body.len() && pitch_name_ok(&body[..i]) && QUALITIES.contains(&&body[i..]))
}

/// Голос партитуры: точные события.
#[derive(Clone, Debug, Default)]
pub struct Voice {
    pub meter: (i64, i64),
    pub key: String,
    pub time: Q,
    /// (начало, MIDI-высота, длительность) в четвертях.
    pub notes: Vec<(Q, i32, Q)>,
    /// (начало, длина, размер) тактов.
    pub bars: Vec<(Q, Q, (i64, i64))>,
    pub chords: Vec<(Q, String)>,
    pub keys: Vec<(Q, String)>,
    pending: Option<(i32, i32)>,
}

impl Default for Q {
    fn default() -> Q {
        Q::ZERO
    }
}

/// Разобранная партитура.
#[derive(Clone, Debug)]
pub struct Score {
    pub text: String,
    pub unit: Q,
    pub bpm: i64,
    pub vocal: Voice,
    pub ins: Voice,
    /// Номер строки → голос (строки с музыкой).
    pub music_lines: BTreeMap<usize, &'static str>,
}

impl Score {
    pub fn voice(&self, name: &str) -> &Voice {
        if name == "Vocal" {
            &self.vocal
        } else {
            &self.ins
        }
    }

    /// Номинальная длина, секунды (по темпу).
    pub fn nominal_seconds(&self) -> f64 {
        self.vocal.time.to_f64() * 60.0 / self.bpm as f64
    }
}

/// Лексема такта (как `TOKEN` оригинала).
enum Tok<'a> {
    Chord(&'a str),
    Key(&'a str),
    Note { acc: &'a str, note: char, oct: &'a str, dur: &'a str, tie: bool },
}

/// Следующая лексема с позиции `at`: (лексема, конец) или `None`.
fn token(s: &str, at: usize) -> Option<(Tok<'_>, usize)> {
    let rest = &s[at..];
    if let Some(r) = rest.strip_prefix('"') {
        let end = r.find(['"', '\n'])?;
        if r.as_bytes()[end] != b'"' {
            return None;
        }
        return Some((Tok::Chord(&r[..end]), at + 1 + end + 1));
    }
    if let Some(r) = rest.strip_prefix("[K:") {
        let end = r.find([']', '\n'])?;
        if r.as_bytes()[end] != b']' || end == 0 {
            return None;
        }
        return Some((Tok::Key(&r[..end]), at + 3 + end + 1));
    }
    let b = rest.as_bytes();
    let mut i = 0;
    let acc_len = if rest.starts_with("^^") || rest.starts_with("__") {
        2
    } else if matches!(b.first(), Some(b'^' | b'_' | b'=')) {
        1
    } else {
        0
    };
    i += acc_len;
    let note = *b.get(i)? as char;
    if !matches!(note, 'A'..='G' | 'a'..='g' | 'z') {
        return None;
    }
    i += 1;
    let o0 = i;
    while matches!(b.get(i), Some(b',' | b'\'')) {
        i += 1;
    }
    let d0 = i;
    while b.get(i).is_some_and(|c| c.is_ascii_digit()) {
        i += 1;
    }
    let tie = b.get(i) == Some(&b'-');
    if tie {
        i += 1;
    }
    Some((Tok::Note { acc: &rest[..acc_len], note, oct: &rest[o0..d0], dur: &rest[d0..i - usize::from(tie)], tie }, at + i))
}

fn parse_bar(body: &str, voice: &mut Voice, unit: Q, ctx: &str) -> R<()> {
    let (n, d) = voice.meter;
    let length = Q::new(4 * n, d);
    let start = voice.time;
    let mut offset = Q::ZERO;
    let mut local: BTreeMap<char, i32> = BTreeMap::new();
    if body == "Z" {
        fail(voice.pending.is_some(), || format!("{ctx}: tie enters a full-measure rest"))?;
        offset = length;
    } else {
        let mut cursor = 0;
        while cursor < body.len() {
            let c = body[cursor..].chars().next().expect("символ");
            if c.is_whitespace() {
                cursor += c.len_utf8();
                continue;
            }
            let (tok, end) = token(body, cursor).ok_or_else(|| {
                AbcError(format!("{ctx}: unsupported token at {:?}", body[cursor..].chars().take(24).collect::<String>()))
            })?;
            cursor = end;
            fail(offset >= length, || format!("{ctx}: event after the measure end"))?;
            match tok {
                Tok::Chord(ch) => {
                    fail(!chord_ok(ch), || format!("{ctx}: unsupported chord {ch:?}"))?;
                    voice.chords.push((start + offset, ch.to_string()));
                }
                Tok::Key(k) => {
                    key_accidentals(k)?;
                    voice.key = k.to_string();
                    voice.keys.push((start + offset, k.to_string()));
                    local.clear();
                }
                Tok::Note { acc, note, oct, dur, tie } => {
                    let units: i64 = if dur.is_empty() { 1 } else { dur.parse().unwrap_or(-1) };
                    fail(!DURATIONS.contains(&units), || format!("{ctx}: unsupported duration {units}; split it into tied supported lengths"))?;
                    let duration = Q::int(units) * unit * Q::int(4);
                    fail(offset + duration > length, || format!("{ctx}: note/rest exceeds meter duration"))?;
                    fail(oct.contains(',') && oct.contains('\''), || format!("{ctx}: mixed octave marks"))?;
                    if note == 'z' {
                        fail(!acc.is_empty() || !oct.is_empty() || tie, || format!("{ctx}: a rest cannot have accidentals, octave marks or ties"))?;
                        fail(voice.pending.is_some(), || format!("{ctx}: tie enters a rest"))?;
                    } else {
                        let letter = note.to_ascii_uppercase();
                        let mut written = 60 + natural(letter) + if note.is_ascii_lowercase() { 12 } else { 0 };
                        written += 12 * (oct.matches('\'').count() as i32 - oct.matches(',').count() as i32);
                        let mut alteration = match local.get(&letter) {
                            Some(a) => *a,
                            None => key_accidentals(&voice.key)?[&letter],
                        };
                        if !acc.is_empty() {
                            alteration = match acc {
                                "=" => 0,
                                "_" => -1,
                                "__" => -2,
                                "^" => 1,
                                _ => 2,
                            };
                            local.insert(letter, alteration);
                        }
                        let mut pitch = written + alteration;
                        if let Some((old_pitch, old_written)) = voice.pending {
                            if acc.is_empty() && written == old_written {
                                pitch = old_pitch;
                            }
                            fail(pitch != old_pitch, || format!("{ctx}: tie changes pitch from {old_pitch} to {pitch}"))?;
                            let last = voice.notes.last_mut().expect("нота под лигой");
                            last.2 = last.2 + duration;
                        } else {
                            fail(!(0..=127).contains(&pitch), || format!("{ctx}: pitch {pitch} is outside MIDI range"))?;
                            voice.notes.push((start + offset, pitch, duration));
                        }
                        voice.pending = tie.then_some((pitch, written));
                    }
                    offset = offset + duration;
                }
            }
        }
    }
    fail(offset != length, || format!("{ctx}: duration {offset} quarter notes != meter duration {length}"))?;
    voice.bars.push((start, length, voice.meter));
    voice.time = voice.time + length;
    Ok(())
}

/// `Z`, `Z2`…`Z4` — число тактов полной паузы.
fn full_rest(bar: &str) -> Option<usize> {
    match bar {
        "Z" => Some(1),
        "Z2" => Some(2),
        "Z3" => Some(3),
        "Z4" => Some(4),
        _ => None,
    }
}

/// Разбор родной партитуры; всё вне диалекта — ошибка.
pub fn parse(text: &str) -> R<Score> {
    let lines: Vec<&str> = text.lines().collect();
    fail(lines.len() < 12, || "Incomplete native two-voice ABC".into())?;
    fail(lines[0] != "X:1" || lines[1] != "T:", || "Expected native X:1 and blank T: header".into())?;
    fail(!lines[2].starts_with("M:"), || "Missing header M:".into())?;
    let meter = meter_value(&lines[2][2..])?;
    let denominator = lines[3].strip_prefix("L:1/").and_then(positive_int).ok_or_else(|| AbcError("Expected L:1/<power of two>, usually L:1/32".into()))?;
    fail(denominator > 1024 || denominator & (denominator - 1) != 0, || "Unsupported L: denominator".into())?;
    let unit = Q::new(1, denominator);
    let bpm = lines[4].strip_prefix("Q:1/4=").and_then(positive_int).ok_or_else(|| AbcError("Expected integer quarter-note tempo Q:1/4=<BPM>".into()))?;
    fail(lines[5] != VOCAL_DEF || lines[6] != INS_DEF, || "Preserve native Vocal and Ins voice definitions".into())?;
    fail(!lines[7].starts_with("K:"), || "Missing header K:".into())?;
    let key = lines[7][2..].to_string();
    key_accidentals(&key)?;
    let fresh = || Voice { meter, key: key.clone(), keys: vec![(Q::ZERO, key.clone())], ..Voice::default() };
    let (mut vocal, mut ins) = (fresh(), fresh());
    let mut music_lines = BTreeMap::new();
    let mut cursor = 8;
    let mut group = 0;
    while cursor < lines.len() {
        while cursor < lines.len() && lines[cursor].starts_with("% ") {
            cursor += 1;
        }
        fail(cursor == lines.len(), || "Dangling section comment without music".into())?;
        group += 1;
        let mut counts = Vec::new();
        for name in VOICES {
            let ctx = format!("group {group}, {name}");
            fail(cursor >= lines.len() || lines[cursor] != format!("V: {name}"), || format!("{ctx}: expected V: {name}"))?;
            cursor += 1;
            let voice = if name == "Vocal" { &mut vocal } else { &mut ins };
            let mut fields = Vec::new();
            while cursor < lines.len() && (lines[cursor].starts_with("M:") || lines[cursor].starts_with("K:")) {
                let (field, value) = lines[cursor].split_once(':').expect("поле");
                fail(fields.contains(&field), || format!("{ctx}: duplicate {field}: field"))?;
                fields.push(field);
                if field == "M" {
                    voice.meter = meter_value(value)?;
                } else {
                    key_accidentals(value)?;
                    voice.key = value.to_string();
                    voice.keys.push((voice.time, value.to_string()));
                }
                cursor += 1;
            }
            fail(cursor >= lines.len(), || format!("{ctx}: missing music line"))?;
            let line = lines[cursor];
            fail(!line.ends_with('|'), || format!("{ctx}: music line must end with a plain barline"))?;
            music_lines.insert(cursor, name);
            cursor += 1;
            let mut bars: Vec<&str> = Vec::new();
            for bar in line[..line.len() - 1].split('|') {
                let bar = bar.trim();
                fail(bar.is_empty(), || format!("{ctx}: empty measure or unsupported double/repeat barline"))?;
                match full_rest(bar) {
                    Some(k) => bars.extend(std::iter::repeat_n("Z", k)),
                    None => bars.push(bar),
                }
            }
            fail(!(1..=4).contains(&bars.len()), || format!("{ctx}: expected 1–4 measures after expanding Z rests"))?;
            counts.push(bars.len());
            for bar in bars {
                let c = format!("{ctx}, bar {}", voice.bars.len() + 1);
                parse_bar(bar, voice, unit, &c)?;
            }
        }
        fail(counts[0] != counts[1], || format!("group {group}: voices have different measure counts"))?;
    }
    for (name, v) in [("Vocal", &vocal), ("Ins", &ins)] {
        fail(v.pending.is_some(), || format!("{name}: unresolved tie at end of score"))?;
    }
    fail(!ins.chords.is_empty(), || "Native chord symbols belong in Vocal, not Ins".into())?;
    fail(vocal.bars != ins.bars, || "Voice meter/time grids differ".into())?;
    fail(vocal.keys != ins.keys, || "Voice key-change timelines differ".into())?;
    Ok(Score { text: text.to_string(), unit, bpm, vocal, ins, music_lines })
}

/// Такт событийной записи: размер, тональность, раздел (наследуются).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EventBar {
    pub meter: Option<String>,
    pub key: Option<String>,
    pub section: Option<String>,
}

/// Событийная запись мелодии (формат `events.json` оригинала): времена — в
/// четвертях от начала пьесы, ноты — одна линия без наложений.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Events {
    pub bpm: i64,
    pub key: String,
    pub bars: Vec<EventBar>,
    /// (начало, длительность, MIDI-высота).
    pub notes: Vec<(Q, Q, i32)>,
    pub chords: Vec<(Q, String)>,
}

fn lengths(value: Q) -> R<Vec<i64>> {
    fail(value.d != 1 || value.n <= 0, || format!("Unrepresentable duration in ABC units: {value}"))?;
    let mut left = value.n;
    let mut out = Vec::new();
    for part in DURATIONS.iter().rev() {
        while left >= *part {
            out.push(*part);
            left -= part;
        }
    }
    Ok(out)
}

fn spelling(pitch: i32, key: &str, active: &mut BTreeMap<char, i32>) -> R<String> {
    let sig = key_accidentals(key)?;
    let prefer_sharp = sig.values().sum::<i32>() >= 0;
    let mut best: Option<((bool, i32, bool, char, i32, i32), char, i32, i32)> = None;
    for letter in LETTERS {
        for alteration in [-1, 0, 1] {
            let base = pitch - alteration - 60 - natural(letter);
            if base.rem_euclid(12) != 0 {
                continue;
            }
            let octave = base.div_euclid(12);
            let score = (alteration != sig[&letter], alteration.abs(), if prefer_sharp { alteration < 0 } else { alteration > 0 }, letter, alteration, octave);
            if best.as_ref().is_none_or(|b| score < b.0) {
                best = Some((score, letter, alteration, octave));
            }
        }
    }
    let (_, letter, alteration, octave) = best.expect("у любой высоты есть написание");
    let mut accidental = "";
    if alteration != *active.get(&letter).unwrap_or(&sig[&letter]) {
        accidental = match alteration {
            -1 => "_",
            0 => "=",
            _ => "^",
        };
        active.insert(letter, alteration);
    }
    let name = match octave.cmp(&0) {
        Ordering::Less => format!("{letter}{}", ",".repeat((-octave) as usize)),
        Ordering::Equal => letter.to_string(),
        Ordering::Greater => format!("{}{}", letter.to_ascii_lowercase(), "'".repeat((octave - 1) as usize)),
    };
    Ok(format!("{accidental}{name}"))
}

fn compress(bars: &[String]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < bars.len() {
        if bars[i] != "Z" {
            out.push_str(&bars[i]);
            out.push('|');
            i += 1;
            continue;
        }
        let mut end = i + 1;
        while end < bars.len() && bars[end] == "Z" {
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
    out
}

fn rests(value: Q) -> R<Vec<String>> {
    Ok(lengths(value)?.into_iter().map(|n| if n != 1 { format!("z{n}") } else { "z".into() }).collect())
}

/// Сборка родной партитуры из событий (мелодия — в `Ins`, аккорды — в
/// `Vocal`), с проверкой обратного разбора — `compile_events` оригинала.
pub fn compile_events(ev: &Events) -> R<String> {
    fail(ev.bpm <= 0, || "bpm must be a positive integer quarter-note tempo".into())?;
    let mut key = if ev.key.is_empty() { "C".to_string() } else { ev.key.clone() };
    key_accidentals(&key)?;
    fail(ev.bars.is_empty(), || "bars must be a nonempty list".into())?;
    struct Bar {
        start: Q,
        end: Q,
        meter: String,
        key: String,
        section: Option<String>,
    }
    let mut bars: Vec<Bar> = Vec::new();
    let (mut time, mut meter, mut section) = (Q::ZERO, "4/4".to_string(), None::<String>);
    let mut unit_den: i64 = 32;
    for (i, item) in ev.bars.iter().enumerate() {
        if let Some(m) = &item.meter {
            meter = m.clone();
        }
        if let Some(k) = &item.key {
            key = k.clone();
        }
        let (n, d) = meter_value(&meter)?;
        key_accidentals(&key)?;
        unit_den = lcm(unit_den, 4 * d);
        if let Some(s) = &item.section {
            fail(!SECTIONS.contains(&s.as_str()), || format!("bar {}: section must be one of {SECTIONS:?}", i + 1))?;
            section = Some(s.clone());
        }
        let end = time + Q::new(4 * n, d);
        bars.push(Bar { start: time, end, meter: meter.clone(), key: key.clone(), section: section.clone() });
        time = end;
    }
    let mut notes: Vec<(Q, i32, Q)> = Vec::new();
    for (i, &(start, duration, pitch)) in ev.notes.iter().enumerate() {
        fail(!(0..=127).contains(&pitch), || format!("note {}: pitch must be one MIDI integer; reduce polyphony first", i + 1))?;
        fail(start < Q::ZERO || duration <= Q::ZERO || start + duration > time, || format!("note {}: onset/duration is outside the score", i + 1))?;
        notes.push((start, pitch, duration));
        unit_den = lcm(lcm(unit_den, (start / Q::int(4)).d), (duration / Q::int(4)).d);
    }
    notes.sort();
    fail(notes.is_empty(), || "Instrumental score has no sounding notes".into())?;
    for w in notes.windows(2) {
        fail(w[0].0 + w[0].2 > w[1].0, || format!("Overlapping melody notes at quarter {}; choose one lead line before compilation", w[1].0))?;
    }
    let mut chords: Vec<(Q, String)> = Vec::new();
    for (i, (when, symbol)) in ev.chords.iter().enumerate() {
        fail(*when < Q::ZERO || *when >= time || !chord_ok(symbol), || format!("chord {}: unsupported symbol or onset", i + 1))?;
        chords.push((*when, symbol.clone()));
        unit_den = lcm(unit_den, (*when / Q::int(4)).d);
    }
    chords.sort();
    fail(chords.windows(2).any(|w| w[0].0 == w[1].0), || "Conflicting/duplicate chord events at the same onset; choose one harmony timeline".into())?;
    fail(unit_den > 1024 || unit_den & (unit_den - 1) != 0, || "Rhythm is outside the power-of-two training grid; explicitly quantize tuplets and record the changes".into())?;
    let unit_q = Q::new(4, unit_den);
    let (mut ins, mut vocal) = (Vec::new(), Vec::new());
    for bar in &bars {
        let (start, end) = (bar.start, bar.end);
        let events: Vec<_> = notes.iter().filter(|x| x.0 < end && x.0 + x.2 > start).collect();
        let (mut pieces, mut cursor, mut active) = (Vec::<String>::new(), start, BTreeMap::new());
        for &&(onset, pitch, duration) in &events {
            let (a, b) = (start.max(onset), end.min(onset + duration));
            if a > cursor {
                pieces.extend(rests((a - cursor) / unit_q)?);
            }
            let parts = lengths((b - a) / unit_q)?;
            for (i, count) in parts.iter().enumerate() {
                let tie = if i < parts.len() - 1 || b < onset + duration { "-" } else { "" };
                let cnt = if *count != 1 { count.to_string() } else { String::new() };
                pieces.push(format!("{}{cnt}{tie}", spelling(pitch, &bar.key, &mut active)?));
            }
            cursor = b;
        }
        if cursor < end {
            pieces.extend(rests((end - cursor) / unit_q)?);
        }
        ins.push(if events.is_empty() { "Z".to_string() } else { pieces.concat() });
        let current = chords.iter().rev().find(|(w, _)| *w <= start).map(|(_, s)| s.clone());
        let mut changes: Vec<(Q, Option<String>)> = vec![(start, current)];
        changes.extend(chords.iter().filter(|(w, _)| start < *w && *w < end).map(|(w, s)| (*w, Some(s.clone()))));
        let mut chunks: Vec<String> = Vec::new();
        for (i, (when, symbol)) in changes.iter().enumerate() {
            let stop = changes.get(i + 1).map(|c| c.0).unwrap_or(end);
            chunks.push(symbol.as_ref().map(|s| format!("\"{s}\"")).unwrap_or_default());
            chunks.extend(rests((stop - *when) / unit_q)?);
        }
        vocal.push(if changes.iter().any(|c| c.1.is_some()) { chunks.concat() } else { "Z".to_string() });
    }
    let mut lines = vec![
        "X:1".to_string(),
        "T:".to_string(),
        format!("M:{}", bars[0].meter),
        format!("L:1/{unit_den}"),
        format!("Q:1/4={}", ev.bpm),
        VOCAL_DEF.to_string(),
        INS_DEF.to_string(),
        format!("K:{}", bars[0].key),
    ];
    let mut index = 0;
    while index < bars.len() {
        let bar = &bars[index];
        let previous = if index > 0 { &bars[index - 1] } else { bar };
        let mut end = index + 1;
        while end < bars.len() && end - index < 4 && bars[end].meter == bar.meter && bars[end].key == bar.key && bars[end].section == bar.section {
            end += 1;
        }
        if let Some(s) = &bar.section {
            if index == 0 || Some(s) != previous.section.as_ref() {
                lines.push(format!("% {s}"));
            }
        }
        for (voice, rendered) in [("Vocal", &vocal), ("Ins", &ins)] {
            lines.push(format!("V: {voice}"));
            if bar.meter != previous.meter {
                lines.push(format!("M:{}", bar.meter));
            }
            if bar.key != previous.key {
                lines.push(format!("K:{}", bar.key));
            }
            lines.push(compress(&rendered[index..end]));
        }
        index = end;
    }
    let text = lines.join("\n") + "\n";
    let parsed = parse(&text)?;
    fail(parsed.ins.notes != notes || !parsed.vocal.notes.is_empty(), || "Internal note roundtrip failed".into())?;
    fail(dedup(&parsed.vocal.chords) != dedup(&chords), || "Internal harmony roundtrip failed".into())?;
    Ok(text)
}

/// Аккорды без повторов подряд (повтор действующего на начале такта — родной).
fn dedup(events: &[(Q, String)]) -> Vec<(Q, String)> {
    let mut out: Vec<(Q, String)> = Vec::new();
    for e in events {
        if out.last().is_none_or(|l| l.1 != e.1) {
            out.push(e.clone());
        }
    }
    out
}

/// Как поступать с нотами `Ins`, звучащими одновременно с `Vocal`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overlap {
    /// Ноты `Vocal` главнее: от нот `Ins` остаются только непересекающиеся куски.
    Vocal,
    /// Любое пересечение — ошибка.
    Error,
}

/// Отчёт переноса голосов (`transfer.json` оригинала, кратко).
#[derive(Clone, Debug, PartialEq)]
pub struct Transfer {
    pub vocal_notes_before: usize,
    pub original_ins_notes: usize,
    pub unaltered_ins_notes: usize,
    pub output_ins_notes: usize,
    pub affected_ins_notes: usize,
    pub chords_kept: bool,
    pub nominal_seconds: f64,
}

fn subtract_intervals(start: Q, duration: Q, occupied: &[(Q, Q)]) -> Vec<(Q, Q)> {
    let mut pieces = vec![(start, start + duration)];
    for &(left, right) in occupied {
        let mut remaining = Vec::new();
        for (a, b) in pieces {
            if right <= a || left >= b {
                remaining.push((a, b));
            } else {
                if a < left {
                    remaining.push((a, left));
                }
                if right < b {
                    remaining.push((right, b));
                }
            }
        }
        pieces = remaining;
    }
    pieces
}

/// Начала тактов с метками разделов (по строкам `% метка` перед музыкой `Vocal`).
fn section_starts(score: &Score) -> R<BTreeMap<Q, String>> {
    let (mut cursor, mut pending) = (0usize, None::<String>);
    let mut out = BTreeMap::new();
    for (index, line) in score.text.lines().enumerate() {
        if let Some(label) = line.strip_prefix("% ") {
            fail(!SECTIONS.contains(&label), || format!("Unknown section label: {label}"))?;
            pending = Some(label.to_string());
        }
        if score.music_lines.get(&index) != Some(&"Vocal") {
            continue;
        }
        if let Some(p) = pending.take() {
            out.insert(score.vocal.bars[cursor].0, p);
        }
        for bar in line[..line.len() - 1].split('|') {
            cursor += full_rest(bar.trim()).unwrap_or(1);
        }
    }
    Ok(out)
}

/// Родной перенос инструментала: все ноты `Vocal` (высота, начало, длительность)
/// уходят в `Ins`, в `Vocal` остаются паузы и (при `keep_chords`) аккорды;
/// ноты `Ins`, не задетые мелодией, сохраняются — `convert_score` оригинала.
pub fn instrumentalize(text: &str, overlap: Overlap, keep_chords: bool) -> R<(String, Transfer)> {
    fail(text.trim().is_empty(), || "Need a nonempty native score".into())?;
    let source = parse(text)?;
    let (vocal, ins) = (&source.vocal, &source.ins);
    fail(vocal.notes.is_empty() && ins.notes.is_empty(), || "The score contains no sounding notes".into())?;
    let occupied: Vec<(Q, Q)> = vocal.notes.iter().map(|&(t, _, d)| (t, t + d)).collect();
    let mut merged: Vec<(Q, i32, Q)> = vocal.notes.clone();
    let (mut affected, mut unchanged) = (0usize, 0usize);
    for &(start, pitch, duration) in &ins.notes {
        let pieces = subtract_intervals(start, duration, &occupied);
        if pieces == [(start, start + duration)] {
            unchanged += 1;
        } else {
            fail(overlap == Overlap::Error, || "Vocal and Ins overlap; choose vocal priority or edit the arrangement explicitly".into())?;
            affected += 1;
        }
        merged.extend(pieces.into_iter().map(|(a, b)| (a, pitch, b - a)));
    }
    merged.sort();
    let sections = section_starts(&source)?;
    let starts: Vec<Q> = vocal.bars.iter().map(|b| b.0).collect();
    fail(vocal.keys.iter().any(|(w, _)| !starts.contains(w)), || "An inline key change occurs inside a bar; normalize its spelling without changing pitches first".into())?;
    let mut bars = Vec::new();
    for &(start, _, (n, d)) in &vocal.bars {
        let key = vocal.keys.iter().rev().find(|(t, _)| *t <= start).map(|(_, k)| k.clone());
        bars.push(EventBar { meter: Some(format!("{n}/{d}")), key, section: sections.get(&start).cloned() });
    }
    let mut chords: Vec<(Q, String)> = Vec::new();
    if keep_chords {
        for (when, symbol) in &vocal.chords {
            if let Some(last) = chords.last() {
                if last.0 == *when {
                    fail(last.1 != *symbol, || "Conflicting chord labels at the same time".into())?;
                    continue;
                }
            }
            chords.push((*when, symbol.clone()));
        }
    }
    let events = Events {
        bpm: source.bpm,
        key: vocal.keys[0].1.clone(),
        bars,
        notes: merged.iter().map(|&(t, p, d)| (t, d, p)).collect(),
        chords,
    };
    let converted = compile_events(&events)?;
    let target = parse(&converted)?;
    fail(target.ins.notes != merged || !target.vocal.notes.is_empty(), || "Voice transfer changed the intended note events".into())?;
    fail(target.ins.bars != ins.bars || target.bpm != source.bpm, || "Voice transfer changed meter, timing or tempo".into())?;
    fail(vocal.notes.iter().any(|n| !target.ins.notes.contains(n)), || "A vocal note was not preserved exactly".into())?;
    fail(keep_chords && dedup(&target.vocal.chords) != dedup(&vocal.chords), || "Voice transfer changed harmony".into())?;
    let check = Transfer {
        vocal_notes_before: vocal.notes.len(),
        original_ins_notes: ins.notes.len(),
        unaltered_ins_notes: unchanged,
        output_ins_notes: merged.len(),
        affected_ins_notes: affected,
        chords_kept: keep_chords,
        nominal_seconds: ins.time.to_f64() * 60.0 / source.bpm as f64,
    };
    Ok((converted, check))
}

/// Убрать символы аккордов (кавер с `cot = melody`); `keep_voice` — оставить
/// мелодию только этого голоса (другой — паузами). Мелодия и размеры обязаны
/// остаться прежними — `strip_chords` оригинала.
pub fn strip_chords(text: &str, keep_voice: Option<&str>) -> R<String> {
    let source = parse(text)?;
    let mut lines: Vec<String> = text.split_inclusive('\n').map(String::from).collect();
    for (&index, &name) in &source.music_lines {
        let line = &lines[index];
        let mut out = String::with_capacity(line.len());
        let mut at = 0;
        while at < line.len() {
            match token(line, at) {
                Some((Tok::Chord(_), end)) => at = end,
                Some((Tok::Note { note, dur, .. }, end)) if note != 'z' && keep_voice.is_some_and(|k| k != name) => {
                    out.push('z');
                    out.push_str(dur);
                    at = end;
                }
                Some((_, end)) => {
                    out.push_str(&line[at..end]);
                    at = end;
                }
                None => {
                    let c = line[at..].chars().next().expect("символ");
                    out.push(c);
                    at += c.len_utf8();
                }
            }
        }
        lines[index] = out;
    }
    let output = lines.concat();
    let result = parse(&output)?;
    for name in VOICES {
        if keep_voice.is_none_or(|k| k == name) {
            fail(source.voice(name).bars != result.voice(name).bars || source.voice(name).notes != result.voice(name).notes, || format!("Chord removal changed melody: {name}"))?;
        }
    }
    fail(!result.vocal.chords.is_empty(), || "Chord removal left a chord symbol".into())?;
    if let Some(k) = keep_voice {
        fail(VOICES.iter().any(|n| *n != k && !result.voice(n).notes.is_empty()), || "Unselected voice was not silenced".into())?;
    }
    Ok(output)
}

/// Лирика из меток разделов партитуры (`[Verse]\n\n[Chorus]\n`) — у
/// инструментала она пустая или ровно такая, без слов.
pub fn lyric_tags(text: &str) -> String {
    let labels: Vec<String> = text.lines().filter_map(|l| l.strip_prefix("% ")).map(title).collect();
    if labels.is_empty() {
        return String::new();
    }
    labels.iter().map(|l| format!("[{l}]")).collect::<Vec<_>>().join("\n\n") + "\n"
}

/// `str.title()` Python: заглавная — первая буква каждого буквенного отрезка.
fn title(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_alpha = false;
    for c in s.chars() {
        if c.is_alphabetic() {
            if prev_alpha {
                out.extend(c.to_lowercase());
            } else {
                out.extend(c.to_uppercase());
            }
            prev_alpha = true;
        } else {
            out.push(c);
            prev_alpha = false;
        }
    }
    out
}

/// Стиль инструментала как в родном процессе: «Instrumental, …» и явные
/// запреты голоса (no vocals, no singing, no choir, no spoken words), точка в конце.
pub fn instrumental_style(style: &str) -> String {
    let mut s = style.trim().trim_end_matches(['.', ',']).to_string();
    if s.is_empty() {
        s = "Expressive instrumental music".into();
    }
    let lower = s.to_lowercase();
    let starts = lower.strip_prefix("instrumental").is_some_and(|r| !r.chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_'));
    if !starts {
        s = format!("Instrumental, {s}");
    }
    for condition in ["no vocals", "no singing", "no choir", "no spoken words"] {
        if !s.to_lowercase().contains(condition) {
            s.push_str(", ");
            s.push_str(condition);
        }
    }
    s.push('.');
    s
}

/// Лирика по умолчанию для партитуры инструментала, когда своих меток нет.
pub const INSTRUMENTAL_PLAN_LYRICS: &str = "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n";

/// Проверка партитуры инструментала перед рендером (`validate_score` оригинала).
pub fn validate_instrumental(text: &str) -> R<Score> {
    fail(text.trim().is_empty(), || "ABC must be nonempty text".into())?;
    let score = parse(text)?;
    fail(!score.vocal.notes.is_empty(), || "Vocal contains sounding notes; move the intended lead to Ins and replace Vocal notes with rests".into())?;
    fail(score.ins.notes.is_empty(), || "Ins has no sounding notes".into())?;
    for &(_, _, (_, d)) in &score.ins.bars {
        fail(score.unit.d % (4 * d) != 0, || "L is too coarse for the four-subbeat grid; scale ALL durations with a finer L".into())?;
    }
    let mut previous: Option<&str> = None;
    for line in text.lines() {
        if let Some(label) = line.strip_prefix("% ") {
            fail(!SECTIONS.contains(&label) || previous == Some(label), || "Use native section labels, without consecutive duplicate section comments".into())?;
            previous = Some(label);
        }
    }
    let lines: Vec<&str> = text.lines().collect();
    for &index in score.music_lines.keys() {
        let line = lines[index];
        let bars: Vec<&str> = line[..line.len() - 1].split('|').collect();
        fail(bars.windows(2).any(|w| full_rest(w[0]).is_some() && full_rest(w[1]).is_some()), || "Compress adjacent full-bar rests to Z2/Z3/Z4 inside each group".into())?;
        for bar in bars {
            let only_rests = !bar.is_empty() && {
                let mut at = 0;
                let mut ok = true;
                while at < bar.len() {
                    match token(bar, at) {
                        Some((Tok::Note { note: 'z', acc: "", oct: "", tie: false, .. }, end)) => at = end,
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                }
                ok
            };
            fail(only_rests, || "Use Z for an unannotated full-bar rest".into())?;
            let mut at = 0;
            while at < bar.len() {
                match token(bar, at) {
                    Some((tok, end)) => {
                        fail(matches!(tok, Tok::Note { dur: "1", .. }), || "Omit explicit duration 1 in the native notation".into())?;
                        at = end;
                    }
                    None => at += bar[at..].chars().next().map_or(1, char::len_utf8),
                }
            }
        }
    }
    Ok(score)
}

/// Мелодия партитуры без аккордов из `Ins` в `Vocal` — чтобы её спели (песня
/// по нотам из MIDI или по своей мелодии). Строки голосов в каждой группе
/// меняются местами; размеры и тональности у голосов и так общие.
pub fn melody_to_vocal(text: &str) -> R<String> {
    let source = parse(text)?;
    fail(!source.vocal.chords.is_empty() || !source.vocal.notes.is_empty(), || "Vocal уже занят (ноты или аккорды) — переносить некуда".into())?;
    let mut lines: Vec<String> = text.lines().map(String::from).collect();
    let vocal: Vec<usize> = source.music_lines.iter().filter(|(_, v)| **v == "Vocal").map(|(i, _)| *i).collect();
    let ins: Vec<usize> = source.music_lines.iter().filter(|(_, v)| **v == "Ins").map(|(i, _)| *i).collect();
    for (a, b) in vocal.into_iter().zip(ins) {
        lines.swap(a, b);
    }
    let out = lines.join("\n") + "\n";
    let result = parse(&out)?;
    fail(result.vocal.notes != source.ins.notes || !result.ins.notes.is_empty(), || "Перенос в Vocal изменил ноты".into())?;
    Ok(out)
}
