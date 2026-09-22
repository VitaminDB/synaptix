//! Склейка окон и сборка ABC против питоновского релиза.
//!
//! Фикстуры — токены, которые выдал релиз (CPU, FP32) на песнях, сгенерированных
//! YuE2: 60, 164 и 209 с (одно окно) и склейка двух песен на 419 с (три окна с
//! префиксами перекрытия). В каждой — ожидаемые префиксы, число и времена
//! событий и итоговый ABC релиза в обоих режимах (с аккордами и без).
//! Нейросеть здесь не участвует: проверяется вся детерминированная часть
//! конвейера от токенов до текста партитуры.

use serde_json::Value;
use synaptix_music_sheetsage2::export::{export_abc, VoiceSelect};
use synaptix_music_sheetsage2::stitch::{Stitcher, Window};
use synaptix_music_sheetsage2::vocab::{Vocab, FULL_TASK_PROMPTS};

fn load(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_slice(&std::fs::read(&path).expect("фикстура")).expect("json")
}

/// Числа в фикстурах — строками `repr`: парсер serde_json по умолчанию
/// округляет дроби с ошибкой до ULP, а сверка здесь побитовая.
fn f(v: &Value) -> f64 {
    v.as_str().expect("число строкой").parse().unwrap()
}

fn opt_f(v: &Value) -> Option<f64> {
    v.as_str().map(|s| s.parse().unwrap())
}

fn check(name: &str) {
    let fx = load(name);
    let vocab = Vocab::new(300.0, 100);
    let prompts = Vocab::normalize_prompts(&FULL_TASK_PROMPTS).unwrap();
    let duration = f(&fx["duration"]);
    let mut stitcher = Stitcher::new(&vocab, &prompts, duration, 300.0);
    for (index, w) in fx["windows"].as_array().unwrap().iter().enumerate() {
        let window = Window {
            start: f(&w["start"]),
            end: f(&w["end"]),
            accept_start: f(&w["accept_start"]),
            accept_end: f(&w["accept_end"]),
            prefix_end: f(&w["prefix_end"]),
            generation_stop: opt_f(&w["generation_stop"]),
        };
        let tokens: Vec<u32> = w["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
        let expected_prefix = w["prefix_tokens"].as_u64().unwrap() as usize;
        let base = match stitcher.prefix_for(index, &window).unwrap() {
            Some((prefix, base)) => {
                assert_eq!(prefix.len(), expected_prefix, "{name}: длина префикса окна {index}");
                assert_eq!(prefix[..], tokens[..expected_prefix], "{name}: префикс окна {index}");
                base
            }
            None => {
                assert_eq!(expected_prefix, 0, "{name}: у окна {index} ожидался префикс");
                0
            }
        };
        stitcher.accept(index, &window, &tokens, base).unwrap();
    }
    let (events, warnings) = stitcher.finish();
    assert!(warnings.is_empty(), "{name}: {warnings:?}");
    assert_eq!(events.len() as u64, fx["events"].as_u64().unwrap(), "{name}: число событий");
    for (i, (e, t)) in events.iter().zip(fx["event_times"].as_array().unwrap()).enumerate() {
        assert_eq!(e.time.unwrap(), f(t), "{name}: время события {i}");
    }
    for (melody_only, key) in [(true, "melody"), (false, "full")] {
        let out = export_abc(&events, duration, melody_only, VoiceSelect::Both);
        assert_eq!(out.abc_error.as_deref(), fx[format!("abc_error_{key}")].as_str(), "{name}/{key}: ошибка ABC");
        let expected = fx[format!("abc_{key}")].as_str().unwrap_or("");
        let got = out.abc.as_deref().unwrap_or("");
        if got != expected {
            let diff = got
                .lines()
                .zip(expected.lines())
                .enumerate()
                .find(|(_, (a, b))| a != b)
                .map(|(i, (a, b))| format!("строка {i}:\n  у нас: {a}\n  релиз: {b}"))
                .unwrap_or_else(|| "разная длина".into());
            panic!("{name}/{key}: ABC отличается от релиза, {diff}");
        }
        let diagnostics: Vec<String> = fx[format!("diagnostics_{key}")]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_str().unwrap().to_string())
            .collect();
        assert_eq!(out.diagnostics, diagnostics, "{name}/{key}: диагностика");
        assert_eq!(out.measures as u64, fx[format!("measures_{key}")].as_u64().unwrap());
        let notes: Vec<u64> = fx["notes"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap()).collect();
        assert_eq!(vec![out.melody_notes as u64, out.vocal_notes as u64, out.instrumental_notes as u64], notes);
    }
}

#[test]
fn one_window_60s() {
    check("rap1");
}

#[test]
fn one_window_164s() {
    check("rap330");
}

#[test]
fn one_window_209s() {
    check("magistral");
}

#[test]
fn three_windows_with_overlap_prefixes() {
    check("concat");
}

/// Выбор голоса: второй голос становится паузами на той же сетке, и партитура
/// проходит ту же самопроверку.
#[test]
fn voice_selection_keeps_grid() {
    let fx = load("rap1");
    let vocab = Vocab::new(300.0, 100);
    let prompts = Vocab::normalize_prompts(&FULL_TASK_PROMPTS).unwrap();
    let duration = f(&fx["duration"]);
    let w = &fx["windows"][0];
    let window = Window {
        start: 0.0,
        end: f(&w["end"]),
        accept_start: 0.0,
        accept_end: f(&w["accept_end"]),
        prefix_end: 0.0,
        generation_stop: None,
    };
    let tokens: Vec<u32> = w["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
    let mut stitcher = Stitcher::new(&vocab, &prompts, duration, 300.0);
    stitcher.accept(0, &window, &tokens, 0).unwrap();
    let (events, _) = stitcher.finish();
    let both = export_abc(&events, duration, true, VoiceSelect::Both).abc.unwrap();
    let vocal = export_abc(&events, duration, true, VoiceSelect::Vocal).abc.unwrap();
    let ins = export_abc(&events, duration, true, VoiceSelect::Instrumental).abc.unwrap();
    let voice_lines = |abc: &str, voice: &str| -> Vec<String> {
        let lines: Vec<&str> = abc.lines().collect();
        lines
            .iter()
            .enumerate()
            .filter(|(_, l)| **l == format!("V: {voice}"))
            .map(|(i, _)| lines[i + 1].to_string())
            .collect()
    };
    assert_eq!(voice_lines(&vocal, "Vocal"), voice_lines(&both, "Vocal"));
    assert_eq!(voice_lines(&ins, "Ins"), voice_lines(&both, "Ins"));
    assert!(voice_lines(&vocal, "Ins").iter().all(|l| l.chars().all(|c| "Z0123456789|".contains(c))));
    assert!(voice_lines(&ins, "Vocal").iter().all(|l| l.chars().all(|c| "Z0123456789|".contains(c))));
}
