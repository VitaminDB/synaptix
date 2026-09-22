//! Транскрипция целиком на GPU (`#[ignore]`: нужен бандл с компонентом
//! `sheetsage2`).
//!
//! Запись длиннее окна (330 с синтетики) — два окна, второе продолжает с
//! префикса перекрытия; партитура собирается или возвращается внятная ошибка.
//! Заодно — отмена: кооперативный флаг обрывает прогон до конца первого окна.
//!
//! ```sh
//! cargo test --release -p synaptix-music-sheetsage2 --test transcribe_e2e -- --ignored --nocapture
//! ```

use synaptix_core::{device::Device, dtype::DType};
use synaptix_music_sheetsage2::pipeline::Progress;
use synaptix_music_sheetsage2::{SheetError, SheetSage2, TranscribeOptions};

const SR: usize = 24000;

/// Мелодия по гамме (нота на долю при 120 уд/мин), бас и щелчки на долях.
fn synth(seconds: f64) -> Vec<f32> {
    let n = (seconds * SR as f64) as usize;
    let scale = [0i64, 2, 4, 5, 7, 9, 11, 12];
    (0..n)
        .map(|i| {
            let t = i as f64 / SR as f64;
            let idx = (t / 0.5) as i64;
            let midi = 60 + scale[((idx * 3) % 8) as usize];
            let f = 440.0 * 2f64.powf((midi - 69) as f64 / 12.0);
            let phase_in = t - idx as f64 * 0.5;
            let env = (-3.0 * phase_in).exp();
            let x = 0.3 * env * (2.0 * std::f64::consts::PI * f * t).sin()
                + 0.1 * (2.0 * std::f64::consts::PI * 110.0 * t).sin();
            let click = if phase_in < 0.005 { 0.5 * (2.0 * std::f64::consts::PI * 2000.0 * t).sin() } else { 0.0 };
            (x + click) as f32
        })
        .collect()
}

fn bundle() -> std::path::PathBuf {
    std::env::var("SHEETSAGE2_BUNDLE").map(Into::into).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        std::path::PathBuf::from(home).join("Storage/syn_models/yue2-3b.syn")
    })
}

#[test]
#[ignore]
fn long_recording_two_windows() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let model = SheetSage2::open(bundle(), Device::Cuda(0), DType::BF16).expect("SheetSage2 из бандла");
    eprintln!("загрузка {:.2} с", model.load_seconds);
    let audio = synth(330.0);
    let mut stages = Vec::new();
    let result = model
        .transcribe(&audio, &TranscribeOptions::default(), &mut |p| stages.push(p), &|| false)
        .expect("транскрипция");
    assert_eq!(result.windows.len(), 2, "330 с — два окна");
    assert_eq!(result.windows[0].prefix_tokens, 0);
    assert!(result.windows[1].prefix_tokens > 8, "второе окно продолжает с префикса перекрытия");
    for (i, w) in result.windows.iter().enumerate() {
        eprintln!(
            "окно {}: {:.0}–{:.0} с, префикс {}, токенов {}, событий {} (принято {}), энкодер {:.2} с, декодер {:.2} с",
            i + 1,
            w.window.start,
            w.window.end,
            w.prefix_tokens,
            w.tokens.len(),
            w.events,
            w.accepted_events,
            w.encode_seconds,
            w.decode_seconds
        );
        assert_eq!(w.tokens.last(), Some(&synaptix_music_sheetsage2::vocab::EOS));
    }
    // События склеены по времени и не выходят за запись.
    let times: Vec<f64> = result.events.iter().map(|e| e.time.unwrap()).collect();
    assert!(times.windows(2).all(|w| w[0] <= w[1]), "события не по порядку");
    assert!(times.last().copied().unwrap_or(0.0) < 330.0);
    assert!(matches!(stages.last(), Some(Progress::Notation)));
    match (&result.abc, &result.abc_error) {
        (Some(abc), _) => {
            eprintln!("ABC: {} тактов, {} символов, всего {:.1} с", result.export.measures, abc.len(), result.seconds);
            assert!(abc.starts_with("X:1\nT:\n"));
            // Режим кавера: аккордов в партитуре нет.
            assert!(abc.lines().filter(|l| !l.starts_with("V:")).all(|l| !l.contains('"')));
        }
        (None, Some(err)) => eprintln!("ABC не собран (синтетика): {err}"),
        (None, None) => panic!("ни партитуры, ни ошибки"),
    }
}

#[test]
#[ignore]
fn cancel_stops_before_first_window_ends() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let model = SheetSage2::open(bundle(), Device::Cuda(0), DType::BF16).expect("SheetSage2 из бандла");
    let audio = synth(30.0);
    let calls = std::cell::Cell::new(0usize);
    let cancel = || {
        calls.set(calls.get() + 1);
        calls.get() > 40
    };
    let err = model
        .transcribe(&audio, &TranscribeOptions::default(), &mut |_| {}, &cancel)
        .err()
        .expect("прогон должен оборваться");
    assert!(matches!(err, SheetError::Cancelled(_)), "{err}");
}
