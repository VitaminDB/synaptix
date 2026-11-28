//! Родная партитура YuE2 против официальных инструментов (`skills/yue2-music/
//! instrumental/scripts`): эталоны в `tests/data` сняты их скриптами
//! (`instrumentalize.convert_score`, `abc_tools.strip_chords`,
//! `compile_score.compile_events`, `instrumental.lyric_tags`) на партитуре
//! `native.abc` — случайные знаки, лига через такт, наложение `Ins` на `Vocal`,
//! смена размера и тональности, аккорд с басом. Порт обязан совпасть байт в байт.

use synaptix_music_yue2::score::{
    compile_events, instrumental_style, instrumentalize, lyric_tags, parse, strip_chords, validate_instrumental, EventBar, Events, Overlap, Q,
};

fn data(name: &str) -> String {
    std::fs::read_to_string(format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"))).expect(name)
}

fn q(s: &str) -> Q {
    Q::parse(s).expect(s)
}

#[test]
fn parse_matches_reference_events() {
    let s = parse(&data("native.abc")).expect("разбор");
    let want_vocal = [
        ("8", 71, "1"), ("9", 73, "1"), ("10", 74, "1"), ("11", 74, "3/2"), ("25/2", 78, "1/2"), ("13", 79, "1"), ("16", 69, "1"), ("17", 70, "1"),
        ("18", 71, "2"), ("20", 70, "1"), ("21", 72, "1"), ("22", 74, "1"), ("23", 72, "3/2"), ("49/2", 70, "1/2"), ("25", 69, "1"),
    ];
    let want_ins = [("0", 74, "1"), ("1", 78, "1"), ("2", 81, "1"), ("3", 78, "1"), ("4", 76, "3"), ("7", 73, "1"), ("10", 55, "2"), ("12", 57, "4"), ("16", 62, "4")];
    let conv = |v: &[(&str, i32, &str)]| v.iter().map(|(a, p, d)| (q(a), *p, q(d))).collect::<Vec<_>>();
    assert_eq!(s.vocal.notes, conv(&want_vocal));
    assert_eq!(s.ins.notes, conv(&want_ins));
    assert_eq!(s.bpm, 120);
    assert_eq!(s.nominal_seconds(), 13.0);
}

#[test]
fn instrumentalize_matches_reference() {
    let src = data("native.abc");
    let (out, check) = instrumentalize(&src, Overlap::Vocal, true).expect("перенос");
    assert_eq!(out, data("native.ins.abc"));
    assert_eq!((check.vocal_notes_before, check.original_ins_notes, check.unaltered_ins_notes, check.output_ins_notes, check.affected_ins_notes), (15, 9, 6, 22, 3));
    assert_eq!(check.nominal_seconds, 13.0);
    let (out, _) = instrumentalize(&src, Overlap::Vocal, false).expect("без аккордов");
    assert_eq!(out, data("native.ins-nochords.abc"));
    assert!(instrumentalize(&src, Overlap::Error, true).is_err(), "наложение при Overlap::Error — ошибка");
    // Результат — годная партитура инструментала: в Vocal только паузы и аккорды.
    validate_instrumental(&data("native.ins.abc")).expect("проверка инструментала");
    assert!(validate_instrumental(&src).is_err(), "у исходника в Vocal есть ноты");
}

#[test]
fn strip_chords_and_tags_match_reference() {
    let src = data("native.abc");
    assert_eq!(strip_chords(&src, None).expect("без аккордов"), data("native.strip.abc"));
    assert_eq!(strip_chords(&src, Some("Vocal")).expect("только Vocal"), data("native.strip-vocal.abc"));
    assert_eq!(lyric_tags(&src), data("native.tags.txt"));
}

#[test]
fn compile_events_matches_reference() {
    // Пример из `references/event-schema.md` оригинала.
    let bar = |meter: Option<&str>, section: Option<&str>| EventBar { meter: meter.map(String::from), key: None, section: section.map(String::from) };
    let ev = Events {
        bpm: 81,
        key: "Bm".into(),
        bars: vec![bar(Some("4/4"), Some("verse")), bar(None, None), bar(Some("5/8"), Some("outro"))],
        notes: [("0", "1", 71), ("1", "1/2", 74), ("3/2", "1/2", 78), ("2", "3", 74), ("5", "1", 73), ("6", "2", 71), ("8", "5/2", 71)]
            .iter()
            .map(|(a, d, p)| (q(a), q(d), *p))
            .collect(),
        chords: vec![(q("0"), "Bm".into()), (q("4"), "F#".into()), (q("8"), "Bm".into())],
    };
    assert_eq!(compile_events(&ev).expect("сборка"), data("events.abc"));
    let mut bad = ev.clone();
    bad.notes.push((q("1/4"), q("1"), 60));
    assert!(compile_events(&bad).is_err(), "наложение нот — ошибка");
    let mut bad = ev;
    bad.notes[0].1 = q("1/3");
    assert!(compile_events(&bad).is_err(), "триоль без явного квантования — ошибка");
}

#[test]
fn style_normalized_like_reference() {
    // Правило `prepare` оригинала.
    assert_eq!(instrumental_style("warm solo piano."), "Instrumental, warm solo piano, no vocals, no singing, no choir, no spoken words.");
    assert_eq!(
        instrumental_style("Instrumental, cinematic strings, no vocals,"),
        "Instrumental, cinematic strings, no vocals, no singing, no choir, no spoken words."
    );
    assert_eq!(instrumental_style("instrumentalist jazz"), "Instrumental, instrumentalist jazz, no vocals, no singing, no choir, no spoken words.");
    assert_eq!(instrumental_style(""), "Instrumental, Expressive instrumental music, no vocals, no singing, no choir, no spoken words.");
}

#[test]
fn melody_moves_to_vocal_for_singing() {
    use synaptix_music_yue2::score::melody_to_vocal;
    let ins = data("native.ins-nochords.abc");
    let sung = melody_to_vocal(&ins).expect("перенос");
    let (a, b) = (parse(&ins).unwrap(), parse(&sung).unwrap());
    assert_eq!(b.vocal.notes, a.ins.notes);
    assert!(b.ins.notes.is_empty());
    assert!(melody_to_vocal(&data("native.ins.abc")).is_err(), "аккорды в Vocal — переносить некуда");
}
