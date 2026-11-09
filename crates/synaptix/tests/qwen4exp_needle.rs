//! «Иголка в стоге сена» для Qwen4Exp: в длинный текст (по умолчанию 200k
//! токенов) на заданную глубину вставляется уникальный факт, модель просят его
//! назвать. Проверяется извлечение на длинном контексте целиком (QSA,
//! индексатор, KV, кэш экспертов).
//!
//!   SYN_QWEN4EXP_BUNDLE=…syn [SYN_NEEDLE_CTX=200000] [SYN_NEEDLE_DEPTHS=10,50,90] \
//!   cargo test -p synaptix --profile fast-release --test qwen4exp_needle -- --nocapture

use std::path::Path;
use std::time::Instant;

use synaptix::facade::llm::{load_llm_with_policy, optimal_profile, Device, GenerationOptions, LlmGeneration};

fn filler_text(tok: &synaptix::facade::llm::LlmTokenizer, want: usize) -> Vec<u32> {
    let paras = [
        "Отчёт склада: поступило 214 коробок, отгружено 187, остаток сверен с учётной системой, расхождений нет.",
        "Журнал сборки: cargo build --release, 214 крейтов, предупреждений 37, ошибок 0, время 2 мин 41 с.",
        "Заметка: в понедельник обновить сертификаты на шлюзе, проверить резервные копии и журналы доступа.",
        "Протокол встречи: обсудили перенос релиза, решили провести дополнительное нагрузочное тестирование.",
        "Ошибка: connection refused (os error 111) при обращении к 127.0.0.1:8080, служба перезапущена.",
        "Сводка погоды: облачно, ветер северо-западный 4 м/с, давление 752 мм рт. ст., влажность 68 %.",
        "def parse(line):\n    key, _, value = line.partition('=')\n    return key.strip(), value.strip()",
        "Таблица задач: миграция базы — в работе, обновление UI — готово, тесты API — в очереди.",
    ];
    let mut text = String::new();
    let mut i = 0usize;
    loop {
        text.push_str(&format!("[{i}] {}\n", paras[i % paras.len()]));
        i += 1;
        if i % 64 == 0 && tok.encode(&text).expect("encode").len() >= want {
            break;
        }
    }
    let mut ids = tok.encode(&text).expect("encode");
    ids.truncate(want);
    ids
}

#[test]
fn qwen4exp_needle_in_haystack() {
    let Ok(path) = std::env::var("SYN_QWEN4EXP_BUNDLE") else {
        eprintln!("SYN_QWEN4EXP_BUNDLE не задан — пропускаем");
        return;
    };
    let ctx: usize = std::env::var("SYN_NEEDLE_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000);
    let depths: Vec<usize> = std::env::var("SYN_NEEDLE_DEPTHS")
        .unwrap_or_else(|_| "10,50,90".into())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    let policy = optimal_profile(Path::new(&path)).policy;
    let (model, tok) = load_llm_with_policy(Path::new(&path), policy, &Device::Cuda(0)).expect("load");

    let head = tok
        .encode("<|im_start|>system\nТы внимательный ассистент. Отвечай кратко.<|im_end|>\n<|im_start|>user\nНиже длинный журнал. Прочитай его.\n")
        .expect("encode");
    let question = tok
        .encode("\nКакой пароль от сейфа в архиве назван в журнале? Ответь только паролем.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        .expect("encode");
    let body = filler_text(&tok, ctx.saturating_sub(head.len() + question.len() + 64));
    let mut passed = 0usize;
    for (n, depth) in depths.iter().enumerate() {
        // Своя иголка на каждую глубину — ответ не угадать по прошлому ходу.
        let code = format!("{}-{}", ["изумрудный-кит", "медный-филин", "северный-маяк"][n % 3], 4817 + 1111 * n);
        let needle = tok.encode(&format!("\nВажно: пароль от сейфа в архиве — «{code}». Запомни его.\n")).expect("encode");
        let at = body.len() * depth / 100;
        let mut prompt = head.clone();
        prompt.extend_from_slice(&body[..at]);
        prompt.extend_from_slice(&needle);
        prompt.extend_from_slice(&body[at..]);
        prompt.extend_from_slice(&question);
        let opts = GenerationOptions {
            max_new_tokens: 32,
            max_seq_len: prompt.len() + 64,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 7,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        };
        let mut session = model.new_kv_session(prompt.len() + 64, 32).expect("session").expect("префикс-KV");
        let mut r = LlmGeneration::new(&model, opts);
        let t = Instant::now();
        let mut text = String::new();
        r.generate_streaming_cached(&mut session, &prompt, &tok, |_, s| {
            text.push_str(s);
            true
        })
        .expect("generate");
        let ok = text.contains(&code);
        passed += ok as usize;
        eprintln!(
            "[иголка] глубина {depth} %, промпт {} ток, {:.0} с: «{}» → {}",
            prompt.len(),
            t.elapsed().as_secs_f64(),
            text.trim().chars().take(80).collect::<String>(),
            if ok { format!("НАЙДЕНО ({code})") } else { format!("НЕ найдено (ждали {code})") }
        );
    }
    eprintln!("[иголка] итог: {passed} из {}", depths.len());
    assert_eq!(passed, depths.len(), "модель не нашла иголку");
}
