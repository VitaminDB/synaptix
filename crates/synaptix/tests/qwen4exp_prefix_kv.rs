//! Префикс-KV Qwen4Exp: ход, продолживший с сохранённого кэша, обязан выдать
//! ровно те же токены, что ход с полным префиллом.
//!
//! Гоняется на бандле: `SYN_QWEN4EXP_BUNDLE=… cargo test -p synaptix --release
//! --test qwen4exp_prefix_kv -- --nocapture --test-threads=1`. Без переменной
//! тест проходит как no-op (в CI без GPU и весов).
//!
//! Здесь ломается всё, если точка возврата снята не там: у этой архитектуры в
//! кэше живут четыре разных состояния сразу — KV и ключи индексатора QSA,
//! рекуррентное состояние GDN и свёртка PLE.

use std::path::Path;

use synaptix::facade::llm::{
    load_llm_with_policy, Device, GenerationOptions, LlmGeneration, QuantPolicy,
};

fn opts(max_seq_len: usize, max_new: usize) -> GenerationOptions {
    GenerationOptions {
        max_new_tokens: max_new,
        max_seq_len,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: 7,
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
    }
}

fn reclaim_vram() {
    synaptix::facade::llm::cuda_release_kernel_caches();
    synaptix::facade::llm::cuda_trim_pool(0);
}

#[test]
fn qwen4exp_prefix_kv_matches_full_prefill() {
    let Ok(path) = std::env::var("SYN_QWEN4EXP_BUNDLE") else {
        eprintln!("SYN_QWEN4EXP_BUNDLE не задан — пропускаем");
        return;
    };
    reclaim_vram();
    let (model, tok) =
        load_llm_with_policy(Path::new(&path), QuantPolicy::balance(), &Device::Cuda(0))
            .expect("load");

    // Ход 1 — «история», ход 2 — она же плюс дописанный хвост: ровно то, что
    // делает чат, добавляя ответ модели и результат инструмента.
    let head = "Сформулируй кратко: чем отличается TCP от UDP? Отвечай по-русски. ".repeat(4);
    let tail = "Дополнение: расскажи ещё про QUIC и его отличие от TCP, тоже кратко. ".repeat(4);
    let ids1 = tok.encode(&head).expect("encode 1");
    // Хвост дописываем в токенах, а не пере-кодированием склейки: пробел на
    // стыке слился бы с первым словом, и префикс перестал бы быть префиксом.
    let mut ids2 = ids1.clone();
    ids2.extend_from_slice(&tok.encode(&tail).expect("encode 2"));
    assert_eq!(&ids2[..ids1.len()], &ids1[..]);
    let max_new = 16usize;
    let ctx = 4096usize;

    // A. С нуля: ход 1, затем ход 2 — обычным путём.
    let mut fresh2 = Vec::new();
    {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.generate_streaming(&ids1, &tok, |_, _| true).expect("fresh turn 1");
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.generate_streaming(&ids2, &tok, |id, _| {
            fresh2.push(id);
            true
        })
        .expect("fresh turn 2");
    }

    // B. С префикс-KV: ход 1 наполняет кэш, ход 2 продолжает с него.
    let mut session = model
        .new_kv_session(ctx, max_new)
        .expect("session")
        .expect("Qwen4Exp умеет префикс-KV");
    let mut cached2 = Vec::new();
    let t_fresh;
    let t_cached;
    {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        let t = std::time::Instant::now();
        let reused = r
            .generate_streaming_cached(&mut session, &ids1, &tok, |_, _| true)
            .expect("cached turn 1");
        t_fresh = t.elapsed().as_millis();
        assert_eq!(reused, 0, "первый ход переиспользовать нечего");

        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        let t = std::time::Instant::now();
        let reused = r
            .generate_streaming_cached(&mut session, &ids2, &tok, |id, _| {
                cached2.push(id);
                true
            })
            .expect("cached turn 2");
        t_cached = t.elapsed().as_millis();
        assert_eq!(
            reused,
            ids1.len(),
            "должен переиспользоваться весь промпт прошлого хода"
        );
    }
    println!(
        "qwen4exp: история {} ток, новый промпт {} ток; ход {t_fresh} мс без кэша, \
         {t_cached} мс с префикс-KV",
        ids1.len(),
        ids2.len()
    );
    assert_eq!(
        tok.decode(&cached2).unwrap_or_default(),
        tok.decode(&fresh2).unwrap_or_default(),
        "продолжение с префикс-KV расходится с полным префиллом"
    );
    assert_eq!(cached2, fresh2, "токены расходятся");

    // C. Расхождение промпта: кэш обязан честно сброситься.
    let other = tok
        .encode("Совсем другой промпт про кофе. Отвечай по-русски.")
        .expect("encode other");
    let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
    let reused = r
        .generate_streaming_cached(&mut session, &other, &tok, |_, _| true)
        .expect("cached turn 3");
    assert_eq!(reused, 0, "при расхождении префикса переиспользовать нельзя");
}

/// Кэш Qwen4Exp переживает переезд в host-RAM: ход после `park`/`unpark`
/// совпадает с ходом без переезда.
#[test]
fn qwen4exp_prefix_kv_survives_host_park() {
    let Ok(path) = std::env::var("SYN_QWEN4EXP_BUNDLE") else {
        eprintln!("SYN_QWEN4EXP_BUNDLE не задан — пропускаем");
        return;
    };
    reclaim_vram();
    let (model, tok) =
        load_llm_with_policy(Path::new(&path), QuantPolicy::balance(), &Device::Cuda(0))
            .expect("load");

    let head = "Сформулируй кратко: чем отличается TCP от UDP? Отвечай по-русски. ".repeat(4);
    let tail = "Дополнение: расскажи ещё про QUIC и его отличие от TCP, тоже кратко. ".repeat(4);
    let ids1 = tok.encode(&head).expect("encode 1");
    let mut ids2 = ids1.clone();
    ids2.extend_from_slice(&tok.encode(&tail).expect("encode 2"));
    let max_new = 16usize;
    let ctx = 4096usize;

    // A. Кэш без переезда — эталон.
    let mut resident2 = Vec::new();
    {
        let mut session = model.new_kv_session(ctx, max_new).expect("session").expect("есть");
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.generate_streaming_cached(&mut session, &ids1, &tok, |_, _| true)
            .expect("resident turn 1");
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.generate_streaming_cached(&mut session, &ids2, &tok, |id, _| {
            resident2.push(id);
            true
        })
        .expect("resident turn 2");
    }
    reclaim_vram();

    // B. Тот же кэш, но между ходами он съездил в RAM и обратно.
    let mut session = model.new_kv_session(ctx, max_new).expect("session").expect("есть");
    let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
    r.generate_streaming_cached(&mut session, &ids1, &tok, |_, _| true)
        .expect("parked turn 1");

    let held = session.device_bytes();
    assert!(held > 0, "после хода кэш обязан держать VRAM");
    let t = std::time::Instant::now();
    let moved = session.park_to_host().expect("park");
    let park_ms = t.elapsed().as_millis();
    assert!(session.is_parked());
    assert_eq!(session.device_bytes(), 0, "припаркованный кэш не держит VRAM");
    reclaim_vram();

    let t = std::time::Instant::now();
    let back = session.unpark_to(Device::Cuda(0)).expect("unpark");
    let unpark_ms = t.elapsed().as_millis();
    assert!(!session.is_parked());
    println!(
        "qwen4exp: освобождено {} MB за {park_ms} мс, вернулось {} MB за {unpark_ms} мс \
         (держал {} MB)",
        moved / (1024 * 1024),
        back / (1024 * 1024),
        held / (1024 * 1024)
    );

    let mut parked2 = Vec::new();
    let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
    let reused = r
        .generate_streaming_cached(&mut session, &ids2, &tok, |id, _| {
            parked2.push(id);
            true
        })
        .expect("parked turn 2");
    assert_eq!(reused, ids1.len(), "переезд не должен ломать префикс");
    assert_eq!(
        tok.decode(&parked2).unwrap_or_default(),
        tok.decode(&resident2).unwrap_or_default(),
        "ход после переезда кэша разошёлся с ходом без переезда"
    );
    assert_eq!(parked2, resident2, "токены расходятся");
}

/// Сколько префикс-KV экономит на диалоге чатового объёма: ход, у которого
/// история уже посчитана, префиллит только хвост.
///
/// У этой модели цена префилла особая: эксперты MoE стримятся с хоста, и
/// каждый повторный проход по истории тащит через шину десятки гигабайт.
#[test]
fn qwen4exp_prefix_kv_saves_prefill_time() {
    let Ok(path) = std::env::var("SYN_QWEN4EXP_BUNDLE") else {
        eprintln!("SYN_QWEN4EXP_BUNDLE не задан — пропускаем");
        return;
    };
    reclaim_vram();
    let (model, tok) =
        load_llm_with_policy(Path::new(&path), QuantPolicy::balance(), &Device::Cuda(0))
            .expect("load");

    let head = "Разбери подробно устройство HTTP: методы, заголовки, коды ответов, \
                keep-alive, чанковую передачу, кэширование и согласование содержимого. "
        .repeat(80);
    let ids1 = tok.encode(&head).expect("encode 1");
    let mut ids2 = ids1.clone();
    ids2.extend_from_slice(&tok.encode("И ещё про HTTP/3 отдельно. ").expect("encode 2"));
    let ctx = 16384usize;
    let max_new = 8usize;
    println!("история {} ток, новый промпт {} ток", ids1.len(), ids2.len());

    // Прогрев: первый ход поднимает экспертов в кэш, иначе замер меряет
    // подкачку весов, а не префилл.
    {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.generate_streaming(&ids2, &tok, |_, _| true).expect("warmup");
    }
    let t = std::time::Instant::now();
    {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.generate_streaming(&ids2, &tok, |_, _| true).expect("fresh");
    }
    let fresh_ms = t.elapsed().as_millis();

    let mut session = model
        .new_kv_session(ctx, max_new)
        .expect("session")
        .expect("Qwen4Exp умеет префикс-KV");
    {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.generate_streaming_cached(&mut session, &ids1, &tok, |_, _| true)
            .expect("warm turn");
    }
    let t = std::time::Instant::now();
    let reused = {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.generate_streaming_cached(&mut session, &ids2, &tok, |_, _| true)
            .expect("cached turn")
    };
    let cached_ms = t.elapsed().as_millis();
    println!(
        "qwen4exp: ход целиком — без кэша {fresh_ms} мс, с префикс-KV {cached_ms} мс \
         (переиспользовано {reused} из {} ток)",
        ids2.len()
    );
    assert!(reused > 0, "префикс обязан переиспользоваться");
}
