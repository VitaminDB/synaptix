//! MIDI → родная партитура YuE2.
//!
//! Оригинал YuE2 принимает ноты только своей ABC (два мелодических голоса,
//! `Vocal` и `Ins`, аккорды в `full`); MIDI и MusicXML он не читает. Здесь —
//! мост через событийную запись официального сборщика ([`crate::score::compile_events`]):
//! из стандартного MIDI-файла берётся одна мелодическая линия, квантуется на
//! сетку 1/32 и собирается в родную партитуру. Размеры, темп и тональность —
//! из метасобытий файла.
//!
//! Многоголосие сводится к верхнему голосу («небосвод»): звучащая нота
//! обрывается на начале более высокой, более низкая во время звучащей
//! отбрасывается. Ударные (10-й канал) не берутся. Сводка изменений — в
//! [`MidiReport`].

use crate::score::{compile_events, key_name, AbcError, EventBar, Events, Q, R};

/// Какую линию брать из MIDI.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MidiPick {
    /// Дорожка с наибольшим числом нот (кроме ударных).
    #[default]
    Auto,
    /// Дорожка по номеру (с 0, как в файле).
    Track(usize),
    /// Канал (0–15) по всем дорожкам.
    Channel(u8),
}

/// Что сделано с нотами при переводе.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MidiReport {
    pub track: Option<usize>,
    pub notes_in: usize,
    pub notes_out: usize,
    /// Отброшены как нижние голоса аккордов и наложений.
    pub dropped_polyphony: usize,
    /// Укорочены, чтобы не налезать на следующую ноту.
    pub truncated: usize,
    /// Сдвинуты квантованием на сетку 1/32.
    pub quantized: usize,
    pub bpm: i64,
    pub key: String,
    pub bars: usize,
}

#[derive(Clone, Copy, Debug)]
struct RawNote {
    start: u64,
    end: u64,
    pitch: u8,
    track: usize,
    channel: u8,
}

struct Smf {
    division: u64,
    notes: Vec<RawNote>,
    /// (тик, мкс на четверть).
    tempos: Vec<(u64, u32)>,
    /// (тик, числитель, знаменатель).
    meters: Vec<(u64, i64, i64)>,
    /// (тик, знаков, минор).
    keys: Vec<(u64, i32, bool)>,
}

fn err(s: impl Into<String>) -> AbcError {
    AbcError(format!("MIDI: {}", s.into()))
}

fn be16(b: &[u8]) -> u16 {
    u16::from_be_bytes([b[0], b[1]])
}

fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

fn vlq(b: &[u8], at: &mut usize) -> R<u64> {
    let mut v: u64 = 0;
    for _ in 0..4 {
        let c = *b.get(*at).ok_or_else(|| err("обрыв в числе переменной длины"))?;
        *at += 1;
        v = (v << 7) | u64::from(c & 0x7f);
        if c & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err(err("слишком длинное число переменной длины"))
}

fn read_smf(b: &[u8]) -> R<Smf> {
    if b.len() < 14 || &b[..4] != b"MThd" {
        return Err(err("не стандартный MIDI-файл (нет MThd)"));
    }
    let hlen = be32(&b[4..8]) as usize;
    let ntrks = be16(&b[10..12]) as usize;
    let division = be16(&b[12..14]);
    if division & 0x8000 != 0 {
        return Err(err("время в SMPTE-кадрах не поддерживается — нужен файл с тиками на четверть"));
    }
    let mut smf = Smf { division: u64::from(division.max(1)), notes: Vec::new(), tempos: Vec::new(), meters: Vec::new(), keys: Vec::new() };
    let mut at = 8 + hlen;
    for track in 0..ntrks {
        if at + 8 > b.len() {
            break;
        }
        let len = be32(&b[at + 4..at + 8]) as usize;
        let body = b.get(at + 8..at + 8 + len).ok_or_else(|| err("дорожка короче заявленной"))?;
        let is_track = &b[at..at + 4] == b"MTrk";
        at += 8 + len;
        if !is_track {
            continue;
        }
        let (mut i, mut tick, mut status) = (0usize, 0u64, 0u8);
        // Звучащие ноты: (канал, высота) → начало.
        let mut on: std::collections::BTreeMap<(u8, u8), Vec<u64>> = Default::default();
        while i < body.len() {
            tick += vlq(body, &mut i)?;
            let mut s = *body.get(i).ok_or_else(|| err("обрыв события"))?;
            if s & 0x80 != 0 {
                i += 1;
                if s < 0xf0 {
                    status = s;
                }
            } else {
                // Бегущий статус.
                s = status;
                if s == 0 {
                    return Err(err("данные без статуса"));
                }
            }
            match s {
                0xff => {
                    let kind = *body.get(i).ok_or_else(|| err("обрыв метасобытия"))?;
                    i += 1;
                    let n = vlq(body, &mut i)? as usize;
                    let data = body.get(i..i + n).ok_or_else(|| err("обрыв метасобытия"))?;
                    i += n;
                    match kind {
                        0x51 if n == 3 => smf.tempos.push((tick, (u32::from(data[0]) << 16) | (u32::from(data[1]) << 8) | u32::from(data[2]))),
                        0x58 if n >= 2 => smf.meters.push((tick, i64::from(data[0]), 1i64 << data[1].min(10))),
                        0x59 if n == 2 => smf.keys.push((tick, i32::from(data[0] as i8), data[1] == 1)),
                        0x2f => break,
                        _ => {}
                    }
                }
                0xf0 | 0xf7 => {
                    let n = vlq(body, &mut i)? as usize;
                    i += n;
                }
                _ => {
                    let kind = s & 0xf0;
                    let channel = s & 0x0f;
                    let need = if kind == 0xc0 || kind == 0xd0 { 1 } else { 2 };
                    let d = body.get(i..i + need).ok_or_else(|| err("обрыв события канала"))?;
                    i += need;
                    match kind {
                        0x90 if d[1] > 0 => on.entry((channel, d[0])).or_default().push(tick),
                        0x80 | 0x90 => {
                            if let Some(start) = on.get_mut(&(channel, d[0])).and_then(|v| (!v.is_empty()).then(|| v.remove(0))) {
                                if tick > start {
                                    smf.notes.push(RawNote { start, end: tick, pitch: d[0], track, channel });
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    smf.notes.sort_by_key(|n| (n.start, std::cmp::Reverse(n.pitch)));
    Ok(smf)
}

/// Мелодия из MIDI-файла → родная партитура YuE2 (мелодия в `Ins`, как у
/// официального сборщика; для песни её переносит в `Vocal`
/// [`crate::score::melody_to_vocal`]).
pub fn midi_to_abc(bytes: &[u8], pick: MidiPick) -> R<(String, MidiReport)> {
    let smf = read_smf(bytes)?;
    let mut rep = MidiReport::default();
    let picked: Vec<RawNote> = match pick {
        MidiPick::Channel(c) => smf.notes.iter().copied().filter(|n| n.channel == c).collect(),
        MidiPick::Track(t) => {
            rep.track = Some(t);
            smf.notes.iter().copied().filter(|n| n.track == t && n.channel != 9).collect()
        }
        MidiPick::Auto => {
            let mut count: std::collections::BTreeMap<usize, usize> = Default::default();
            for n in smf.notes.iter().filter(|n| n.channel != 9) {
                *count.entry(n.track).or_default() += 1;
            }
            let best = count.iter().max_by_key(|(t, c)| (**c, std::cmp::Reverse(**t))).map(|(t, _)| *t);
            rep.track = best;
            smf.notes.iter().copied().filter(|n| Some(n.track) == best && n.channel != 9).collect()
        }
    };
    if picked.is_empty() {
        return Err(err("в выбранной дорожке нет нот"));
    }
    rep.notes_in = picked.len();
    // Верхний голос.
    let mut line: Vec<RawNote> = Vec::new();
    for n in picked {
        match line.last_mut() {
            Some(cur) if n.start < cur.end => {
                if n.pitch > cur.pitch {
                    cur.end = n.start;
                    if cur.end <= cur.start {
                        line.pop();
                    } else {
                        rep.truncated += 1;
                    }
                    line.push(n);
                } else {
                    rep.dropped_polyphony += 1;
                }
            }
            _ => line.push(n),
        }
    }
    // Сетка 1/32 = 1/8 четверти.
    let ppq = smf.division as i64;
    let grid = |t: u64| -> Q { Q::new(((t as i64) * 8 + ppq / 2).div_euclid(ppq), 8) };
    let mut notes: Vec<(Q, Q, i32)> = Vec::new();
    for n in line {
        let (a, mut b) = (grid(n.start), grid(n.end));
        if Q::new(n.start as i64, ppq) != a || Q::new(n.end as i64, ppq) != b {
            rep.quantized += 1;
        }
        if b <= a {
            b = a + Q::new(1, 8);
        }
        if let Some(last) = notes.last_mut() {
            if a < last.0 + last.1 {
                last.1 = a - last.0;
                rep.truncated += 1;
                if last.1 <= Q::ZERO {
                    notes.pop();
                }
            }
        }
        notes.push((a, b - a, i32::from(n.pitch)));
    }
    let end = notes.iter().map(|n| n.0 + n.1).max().unwrap_or(Q::ZERO);
    // Такты по размерам файла; смена посреди такта — с ближайшей черты.
    let tick_q = |t: u64| Q::new(t as i64, ppq);
    let mut bars = Vec::new();
    let (mut time, mut meter) = (Q::ZERO, (4i64, 4i64));
    let mut key = smf.keys.first().filter(|k| k.0 == 0).and_then(|k| key_name(k.1, k.2)).unwrap_or("C").to_string();
    let mut last_key = key.clone();
    let first_key = key.clone();
    let mut last_meter = None;
    while time < end {
        if let Some(m) = smf.meters.iter().rev().find(|m| tick_q(m.0) <= time) {
            meter = (m.1.max(1), m.2);
        }
        if let Some(k) = smf.keys.iter().rev().find(|k| tick_q(k.0) <= time).and_then(|k| key_name(k.1, k.2)) {
            key = k.to_string();
        }
        let meter_s = format!("{}/{}", meter.0, meter.1);
        bars.push(EventBar {
            meter: (last_meter.as_ref() != Some(&meter_s)).then(|| meter_s.clone()),
            key: (key != last_key).then(|| key.clone()),
            section: None,
        });
        last_meter = Some(meter_s);
        last_key = key.clone();
        time = time + Q::new(4 * meter.0, meter.1);
    }
    let bpm = smf.tempos.first().map(|t| (60_000_000.0 / f64::from(t.1.max(1))).round() as i64).unwrap_or(120).max(1);
    rep.notes_out = notes.len();
    rep.bpm = bpm;
    rep.key = first_key.clone();
    rep.bars = bars.len();
    let text = compile_events(&Events { bpm, key: first_key, bars, notes, chords: Vec::new() })?;
    Ok((text, rep))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::parse;

    /// Маленький SMF формата 1: дорожка темпа/размера и мелодическая с
    /// аккордом в начале (нижние ноты уходят) и наложением.
    fn smf() -> Vec<u8> {
        let mut t0 = Vec::new();
        t0.extend([0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20]); // 120 BPM
        t0.extend([0x00, 0xff, 0x58, 0x04, 0x03, 0x02, 0x18, 0x08]); // 3/4
        t0.extend([0x00, 0xff, 0x59, 0x02, 0x01, 0x00]); // G
        t0.extend([0x00, 0xff, 0x2f, 0x00]);
        let mut t1 = Vec::new();
        // Аккорд G4-B4-D5 на 0, четверть (480): остаётся D5.
        t1.extend([0x00, 0x90, 67, 90, 0x00, 69 + 2, 90, 0x00, 74, 90]);
        t1.extend([0x83, 0x60, 0x80, 67, 0, 0x00, 71, 0, 0x00, 74, 0]);
        // E5 восьмая (240), потом F#5 половинная, начавшаяся на 60 тиков раньше конца E5.
        t1.extend([0x00, 0x90, 76, 90, 0x81, 0x34, 0x90, 78, 90, 0x3c, 0x80, 76, 0]);
        t1.extend([0x87, 0x40, 0x80, 78, 0]);
        // Ударные на 10-м канале не берутся.
        t1.extend([0x00, 0x99, 36, 100, 0x10, 0x89, 36, 0]);
        t1.extend([0x00, 0xff, 0x2f, 0x00]);
        let mut out = Vec::new();
        out.extend(b"MThd");
        out.extend([0, 0, 0, 6, 0, 1, 0, 2, 0x01, 0xe0]); // формат 1, 2 дорожки, 480 тиков
        for t in [t0, t1] {
            out.extend(b"MTrk");
            out.extend((t.len() as u32).to_be_bytes());
            out.extend(t);
        }
        out
    }

    #[test]
    fn melody_from_midi() {
        let (abc, rep) = midi_to_abc(&smf(), MidiPick::Auto).expect("перевод");
        let s = parse(&abc).expect("родная партитура");
        assert_eq!((s.bpm, rep.key.as_str(), rep.track), (120, "G", Some(1)));
        assert_eq!(s.ins.bars[0].2, (3, 4));
        let q = |a: i64, b: i64| Q::new(a, b);
        // D5 четверть, E5 укорочена до начала F#5 (3/8), F#5 = 60+960 тиков.
        assert_eq!(s.ins.notes, vec![(q(0, 1), 74, q(1, 1)), (q(1, 1), 76, q(3, 8)), (q(11, 8), 78, q(17, 8))]);
        assert_eq!(rep.dropped_polyphony, 2);
        assert!(s.vocal.notes.is_empty());
        assert!(midi_to_abc(b"not midi", MidiPick::Auto).is_err());
        assert!(midi_to_abc(&smf(), MidiPick::Track(0)).is_err(), "в дорожке темпа нот нет");
    }
}
