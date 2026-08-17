//! Префикс-KV даёт ровно то же, что полный префилл.
//!
//! Гоняется на бандле гибрида: `SYN_QWEN38_BUNDLE=… cargo test -p synaptix
//! --release --test prefix_kv_equivalence -- --nocapture`. Без переменной тест
//! проходит как no-op (в CI без GPU и весов).
//!
//! Проверяем главное свойство: ход, продолживший с сохранённого кэша, выдаёт
//! те же токены, что ход с нуля. Именно здесь ломается всё, если linear-состояние
//! GDN восстановлено не на ту границу или кэш MTP-головы уехал.

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

fn collect(n: usize) -> (Vec<u32>, impl FnMut(u32, &str) -> bool) {
    let out = Vec::with_capacity(n);
    (out, |_id: u32, _d: &str| true)
}

#[test]
fn prefix_kv_matches_full_prefill() {
    let Ok(path) = std::env::var("SYN_QWEN38_BUNDLE") else {
        eprintln!("SYN_QWEN38_BUNDLE не задан — пропускаем");
        return;
    };
    let (model, tok) = load_llm_with_policy(
        Path::new(&path),
        QuantPolicy::balance(),
        &Device::Cuda(0),
    )
    .expect("load");

    // Ход 1 — «история», ход 2 — она же плюс дописанный хвост: ровно то, что
    // делает чат, когда добавляет ответ модели и результат инструмента.
    let head = "Сформулируй кратко: чем отличается TCP от UDP? Отвечай по-русски. "
        .repeat(8);
    let tail = "Дополнение: ещё расскажи про QUIC и его отличие от TCP, тоже кратко. "
        .repeat(8);
    let ids1 = tok.encode(&head).expect("encode 1");
    // Хвост дописываем в токенах, а не пере-кодированием склейки: в реальном
    // чате граница хода — спецтокен (`<|im_start|>`), а тут пробел на стыке
    // слился бы с первым словом и префикс перестал бы быть префиксом. Проверяем
    // механику кэша, а не устойчивость BPE к склейке.
    let mut ids2 = ids1.clone();
    ids2.extend_from_slice(&tok.encode(&tail).expect("encode 2"));
    assert_eq!(&ids2[..ids1.len()], &ids1[..]);
    let max_new = 24usize;
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
        .expect("гибрид с MTP умеет префикс-KV");
    let mut cached2 = Vec::new();
    {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        let reused = r
            .generate_streaming_cached(&mut session, &ids1, &tok, |_, _| true)
            .expect("cached turn 1");
        assert_eq!(reused, 0, "первый ход переиспользовать нечего");
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        let reused = r
            .generate_streaming_cached(&mut session, &ids2, &tok, |id, _| {
                cached2.push(id);
                true
            })
            .expect("cached turn 2");
        println!(
            "переиспользовано {reused} из {} токенов промпта (история {} ток)",
            ids2.len(),
            ids1.len()
        );
        // Точка возврата стоит на границе, кратной чанку GDN-скана (64), —
        // хвост прошлого промпта после неё считается заново.
        let expect = (ids1.len() / 64) * 64;
        assert_eq!(reused, expect, "должен переиспользоваться префикс до границы 64");
    }

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

/// Сколько экономит префикс-KV на реальном по объёму диалоге: ход, у которого
/// история уже посчитана, префиллит только хвост.
#[test]
fn prefix_kv_saves_prefill_time() {
    let Ok(path) = std::env::var("SYN_QWEN38_BUNDLE") else {
        eprintln!("SYN_QWEN38_BUNDLE не задан — пропускаем");
        return;
    };
    let (model, tok) = load_llm_with_policy(
        Path::new(&path),
        QuantPolicy::balance(),
        &Device::Cuda(0),
    )
    .expect("load");

    let head = "Разбери подробно устройство HTTP: методы, заголовки, коды ответов, \
                keep-alive, чанковую передачу, кэширование и согласование содержимого. "
        .repeat(120);
    let ids1 = tok.encode(&head).expect("encode 1");
    let mut ids2 = ids1.clone();
    ids2.extend_from_slice(&tok.encode("И ещё про HTTP/3 отдельно. ").expect("encode 2"));
    let ctx = 16384usize;
    let max_new = 8usize;
    println!("история {} ток, новый промпт {} ток", ids1.len(), ids2.len());

    let t = std::time::Instant::now();
    {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.generate_streaming(&ids2, &tok, |_, _| true).expect("fresh");
    }
    let fresh_ms = t.elapsed().as_millis();

    let mut session = model
        .new_kv_session(ctx, max_new)
        .expect("session")
        .expect("гибрид с MTP умеет префикс-KV");
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
        "ход целиком: без кэша {fresh_ms} мс, с префикс-KV {cached_ms} мс \
         (переиспользовано {reused} ток)"
    );
    assert!(
        cached_ms * 2 < fresh_ms,
        "префикс-KV должен экономить кратно: {cached_ms} мс против {fresh_ms} мс"
    );
}
