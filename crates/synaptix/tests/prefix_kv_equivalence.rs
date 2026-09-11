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

/// Вернуть драйверу VRAM модели предыдущего теста: пулы держат её с
/// `RELEASE_THRESHOLD=MAX`, и без трима следующая модель в этом же процессе не
/// влезает (в приложении это делает выгрузка модели).
fn reclaim_vram() {
    synaptix::facade::llm::cuda_release_kernel_caches();
    synaptix::facade::llm::cuda_trim_pool(0);
}

#[test]
fn prefix_kv_matches_full_prefill() {
    let Ok(path) = std::env::var("SYN_QWEN38_BUNDLE") else {
        eprintln!("SYN_QWEN38_BUNDLE не задан — пропускаем");
        return;
    };
    reclaim_vram();
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
    reclaim_vram();
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

/// Muse-Glimmer БЕЗ спекуляции (temperature>0 → graph/eager-декод): здесь путь
/// численно один и тот же, поэтому продолжение с кэша обязано совпасть с полным
/// префиллом токен в токен. Это и есть проверка самой логики префикс-KV.
#[test]
fn prefix_kv_muse_matches_full_prefill() {
    let Ok(path) = std::env::var("SYN_MUSE_BUNDLE") else {
        eprintln!("SYN_MUSE_BUNDLE не задан — пропускаем");
        return;
    };
    reclaim_vram();
    let (model, tok) = load_llm_with_policy(
        Path::new(&path),
        QuantPolicy::balance(),
        &Device::Cuda(0),
    )
    .expect("load");

    // temperature>0 уводит с DFlash/lookup-путей на обычный декод.
    let sampled = |ctx: usize, max_new: usize| GenerationOptions {
        temperature: 0.7,
        top_k: 0,
        top_p: 1.0,
        seed: 12345,
        ..opts(ctx, max_new)
    };
    let head = "Расскажи, чем отличается TCP от UDP. Отвечай по-русски. ".repeat(10);
    let ids1 = tok.encode(&head).expect("encode 1");
    let mut ids2 = ids1.clone();
    ids2.extend_from_slice(&tok.encode("Добавь абзац про QUIC. ").expect("encode 2"));
    let ctx = 8192usize;
    let max_new = 24usize;

    let mut fresh2 = Vec::new();
    {
        let mut r = LlmGeneration::new(&model, sampled(ctx, max_new));
        r.generate_streaming(&ids1, &tok, |_, _| true).expect("fresh turn 1");
        let mut r = LlmGeneration::new(&model, sampled(ctx, max_new));
        r.generate_streaming(&ids2, &tok, |id, _| {
            fresh2.push(id);
            true
        })
        .expect("fresh turn 2");
    }

    let mut session = model
        .new_kv_session(ctx, max_new)
        .expect("session")
        .expect("Muse-Glimmer умеет префикс-KV");
    let mut cached2 = Vec::new();
    {
        let mut r = LlmGeneration::new(&model, sampled(ctx, max_new));
        r.generate_streaming_cached(&mut session, &ids1, &tok, |_, _| true)
            .expect("cached turn 1");
        let mut r = LlmGeneration::new(&model, sampled(ctx, max_new));
        let reused = r
            .generate_streaming_cached(&mut session, &ids2, &tok, |id, _| {
                cached2.push(id);
                true
            })
            .expect("cached turn 2");
        println!(
            "muse (без спекуляции): переиспользовано {reused} из {} ток",
            ids2.len()
        );
        assert_eq!(reused, ids1.len(), "у Muse точка возврата — весь промпт");
    }
    assert_eq!(
        tok.decode(&cached2).unwrap_or_default(),
        tok.decode(&fresh2).unwrap_or_default(),
        "продолжение с префикс-KV расходится с полным префиллом"
    );
    assert_eq!(cached2, fresh2, "токены расходятся");
}

/// Muse-Glimmer с DFlash: контекст драфтера при возобновлении набирается только
/// по хвосту промпта (роллинг-окно кэша драфтера откатывается на границу), и
/// продолжение обязано совпасть с полным префиллом токен в токен.
///
/// Важно, что оба хода идут ОДНИМ путём декода: спекуляция «lossless» лишь в
/// точной арифметике — длина verify-чанка меняет план GEMM, а с ним последние
/// биты логитов, так что сравнение спекулятивного пути с обычным на ничьих
/// расходится (ловилось именно так, когда DFlash не загрузился и кэш-ход уходил
/// на graph-декод).
#[test]
fn prefix_kv_muse_dflash_matches_full_prefill() {
    let Ok(path) = std::env::var("SYN_MUSE_BUNDLE") else {
        eprintln!("SYN_MUSE_BUNDLE не задан — пропускаем");
        return;
    };
    reclaim_vram();
    let (model, tok) = load_llm_with_policy(
        Path::new(&path),
        QuantPolicy::balance(),
        &Device::Cuda(0),
    )
    .expect("load");

    let head = "Опиши кратко разницу между TCP и UDP. Отвечай по-русски. ".repeat(60);
    let tail = "Добавь абзац про QUIC. ".repeat(6);
    let ids1 = tok.encode(&head).expect("encode 1");
    let mut ids2 = ids1.clone();
    ids2.extend_from_slice(&tok.encode(&tail).expect("encode 2"));
    let ctx = 8192usize;
    let max_new = 24usize;

    let mut fresh2 = Vec::new();
    let fresh_ms;
    {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.generate_streaming(&ids1, &tok, |_, _| true).expect("fresh turn 1");
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        let t = std::time::Instant::now();
        r.generate_streaming(&ids2, &tok, |id, _| {
            fresh2.push(id);
            true
        })
        .expect("fresh turn 2");
        fresh_ms = t.elapsed().as_millis();
    }

    let mut session = model
        .new_kv_session(ctx, max_new)
        .expect("session")
        .expect("Muse-Glimmer умеет префикс-KV");
    let mut cached2 = Vec::new();
    {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        let reused = r
            .generate_streaming_cached(&mut session, &ids1, &tok, |_, _| true)
            .expect("cached turn 1");
        assert_eq!(reused, 0, "первый ход переиспользовать нечего");
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        let t = std::time::Instant::now();
        let reused = r
            .generate_streaming_cached(&mut session, &ids2, &tok, |id, _| {
                cached2.push(id);
                true
            })
            .expect("cached turn 2");
        let cached_ms = t.elapsed().as_millis();
        println!(
            "muse: переиспользовано {reused} из {} токенов промпта (история {} ток); \
             ход: без кэша {fresh_ms} мс, с префикс-KV {cached_ms} мс",
            ids2.len(),
            ids1.len()
        );
        // У Muse точка возврата — весь промпт прошлого хода (linear-слоёв нет,
        // выравнивать не нужно).
        assert_eq!(reused, ids1.len());
    }

    println!(
        "muse+dflash: с кэшем {:?}\n              без кэша {:?}",
        tok.decode(&cached2).unwrap_or_default(),
        tok.decode(&fresh2).unwrap_or_default()
    );
    assert_eq!(cached2, fresh2, "продолжение с префикс-KV расходится с полным префиллом");

    let other = tok
        .encode("Совсем другой промпт про кофе. Отвечай по-русски.")
        .expect("encode other");
    let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
    let reused = r
        .generate_streaming_cached(&mut session, &other, &tok, |_, _| true)
        .expect("cached turn 3");
    assert_eq!(reused, 0, "при расхождении префикса переиспользовать нельзя");
}

/// Кэш переживает переезд в host-RAM: `park_to_host` + `unpark_to` дают ровно
/// тот же ход, что и без переезда.
///
/// Зачем такой переезд: пока идёт вложенная генерация (субагент чата), кэш
/// диалога держит гигабайты VRAM, а вложенному прогону их не хватает.
/// Альтернатива — выбросить кэш и префиллить историю заново, поэтому цена
/// ошибки здесь — не «медленно», а «тихо другой ответ».
#[test]
fn prefix_kv_survives_host_park() {
    let Ok(path) = std::env::var("SYN_QWEN38_BUNDLE") else {
        eprintln!("SYN_QWEN38_BUNDLE не задан — пропускаем");
        return;
    };
    reclaim_vram();
    let (model, tok) = load_llm_with_policy(
        Path::new(&path),
        QuantPolicy::balance(),
        &Device::Cuda(0),
    )
    .expect("load");

    let head = "Сформулируй кратко: чем отличается TCP от UDP? Отвечай по-русски. ".repeat(8);
    let tail = "Дополнение: ещё расскажи про QUIC и его отличие от TCP, тоже кратко. ".repeat(8);
    let ids1 = tok.encode(&head).expect("encode 1");
    let mut ids2 = ids1.clone();
    ids2.extend_from_slice(&tok.encode(&tail).expect("encode 2"));
    let max_new = 24usize;
    let ctx = 4096usize;

    // A. Кэш без переезда — эталон.
    let mut resident2 = Vec::new();
    let resident_reused;
    {
        let mut session = model
            .new_kv_session(ctx, max_new)
            .expect("session")
            .expect("гибрид с MTP умеет префикс-KV");
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.generate_streaming_cached(&mut session, &ids1, &tok, |_, _| true)
            .expect("resident turn 1");
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        resident_reused = r
            .generate_streaming_cached(&mut session, &ids2, &tok, |id, _| {
                resident2.push(id);
                true
            })
            .expect("resident turn 2");
    }
    reclaim_vram();

    // B. Тот же кэш, но между ходами он съездил в RAM и обратно.
    let mut session = model
        .new_kv_session(ctx, max_new)
        .expect("session")
        .expect("гибрид с MTP умеет префикс-KV");
    let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
    r.generate_streaming_cached(&mut session, &ids1, &tok, |_, _| true)
        .expect("parked turn 1");

    let held = session.device_bytes();
    assert!(held > 0, "после хода кэш обязан держать VRAM");
    let t = std::time::Instant::now();
    let moved = session.park_to_host().expect("park");
    let park_ms = t.elapsed().as_millis();
    assert!(session.is_parked(), "сессия должна считаться припаркованной");
    assert_eq!(session.device_bytes(), 0, "припаркованный кэш не держит VRAM");
    reclaim_vram();

    let t = std::time::Instant::now();
    let back = session.unpark_to(Device::Cuda(0)).expect("unpark");
    let unpark_ms = t.elapsed().as_millis();
    assert!(!session.is_parked());
    // Возвращается меньше, чем освободилось: device-зеркала GDN-состояния не
    // возят через шину, их пересеет из host-векторов первый же forward
    // (`sync_to_device`). Всё остальное — K/V слоёв — обязано вернуться.
    assert!(back > 0, "K/V слоёв не вернулись");
    assert_eq!(session.device_bytes(), back, "на устройстве не то, что привезли");
    assert!(back <= held);
    println!(
        "префикс-KV: освобождено {} MB за {park_ms} мс, вернулось {} MB за {unpark_ms} мс \
         (держал {} MB)",
        moved / (1024 * 1024),
        back / (1024 * 1024),
        held / (1024 * 1024)
    );

    let mut parked2 = Vec::new();
    let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
    let parked_reused = r
        .generate_streaming_cached(&mut session, &ids2, &tok, |id, _| {
            parked2.push(id);
            true
        })
        .expect("parked turn 2");

    assert_eq!(
        parked_reused, resident_reused,
        "переезд не должен менять размер переиспользуемого префикса"
    );
    assert_eq!(
        tok.decode(&parked2).unwrap_or_default(),
        tok.decode(&resident2).unwrap_or_default(),
        "ход после переезда кэша разошёлся с ходом без переезда"
    );
    assert_eq!(parked2, resident2, "токены расходятся");
}

/// Любой бандл с сессией (плотный Qwen3, Llama, Gemma-3, Gemma-4, гибрид,
/// Muse, Qwen4Exp): два хода чата по шаблону модели, второй обязан
/// продолжить с кэша и выдать то же начало ответа, что ход с нуля.
///
/// `SYN_PREFIX_KV_BUNDLE=… cargo test -p synaptix --release --test
/// prefix_kv_equivalence prefix_kv_bundle -- --nocapture --test-threads=1`.
#[test]
fn prefix_kv_bundle_matches_full_prefill() {
    use synaptix::facade::llm::{optimal_profile, Message};
    let Ok(path) = std::env::var("SYN_PREFIX_KV_BUNDLE") else {
        eprintln!("SYN_PREFIX_KV_BUNDLE не задан — пропускаем");
        return;
    };
    reclaim_vram();
    let mut policy = optimal_profile(Path::new(&path)).policy;
    // A/B: `SYN_PREFIX_KV_KV_DTYPE=bf16|f16` — плотный KV вместо профильного.
    match std::env::var("SYN_PREFIX_KV_KV_DTYPE").ok().as_deref() {
        Some("bf16") => policy.kv_dtype = synaptix::facade::llm::KvDtypePolicy::BF16,
        Some("f16") => policy.kv_dtype = synaptix::facade::llm::KvDtypePolicy::F16,
        _ => {}
    }
    let (model, tok) =
        load_llm_with_policy(Path::new(&path), policy, &Device::Cuda(0)).expect("load");

    let msgs1 = [
        Message::system("Отвечай кратко, одним предложением."),
        Message::user("Чем TCP отличается от UDP?"),
    ];
    let prompt1 = tok
        .apply_chat_template_ex_tools(&msgs1, true, false, None)
        .expect("chat template 1");
    let ids1 = tok.encode(&prompt1).expect("encode 1");
    let ctx = 4096usize;
    let max_new = 32usize;
    let eos = tok.eos_ids().to_vec();

    let mut session = model
        .new_kv_session(ctx, max_new)
        .expect("session")
        .expect("архитектура умеет префикс-KV");
    let mut a1 = Vec::new();
    let mut t1 = String::new();
    {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.set_stop_tokens(eos.clone());
        r.generate_streaming_cached(&mut session, &ids1, &tok, |id, s| {
            a1.push(id);
            t1.push_str(s);
            true
        })
        .expect("cached turn 1");
    }
    eprintln!(
        "ход 1: промпт {} ток → {t1:?}; id {:?}; eos {:?}; промпт {:?}",
        ids1.len(),
        a1,
        eos,
        prompt1
    );
    while a1.last().is_some_and(|t| eos.contains(t)) {
        a1.pop();
    }
    // Хвост промпта хода 2 — по шаблону, за маркером ответа; сам промпт
    // собирается из токенов хода 1 (шаблон может отрисовать прошлый ход иначе,
    // чем выглядел его generation prompt).
    const MARK: &str = "QQXZ";
    let msgs2 = [
        msgs1[0].clone(),
        msgs1[1].clone(),
        Message::assistant(MARK),
        Message::user("А что такое QUIC?"),
    ];
    let prompt2 = tok
        .apply_chat_template_ex_tools(&msgs2, true, false, None)
        .expect("chat template 2");
    let at = prompt2.rfind(MARK).expect("маркер ответа в промпте 2");
    let mut ids2 = ids1.clone();
    ids2.extend_from_slice(&a1);
    ids2.extend(tok.encode(&prompt2[at + MARK.len()..]).expect("encode tail"));

    let mut cached2 = Vec::new();
    let mut t2 = String::new();
    let reused = {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.set_stop_tokens(eos.clone());
        r.generate_streaming_cached(&mut session, &ids2, &tok, |id, s| {
            cached2.push(id);
            t2.push_str(s);
            true
        })
        .expect("cached turn 2")
    };
    eprintln!("ход 2: промпт {} ток, из кэша {reused} → {t2:?}", ids2.len());
    assert!(
        reused > 0 && reused <= ids1.len() && ids1.len() - reused < 64,
        "ход 2 переиспользовал {reused} из {} ток промпта 1",
        ids1.len()
    );

    let mut fresh2 = Vec::new();
    let mut t2_ref = String::new();
    {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.set_stop_tokens(eos);
        r.generate_streaming(&ids2, &tok, |id, s| {
            fresh2.push(id);
            t2_ref.push_str(s);
            true
        })
        .expect("fresh turn 2");
    }
    let diff = cached2.iter().zip(&fresh2).position(|(x, y)| x != y);
    eprintln!("эталон без сессии → {t2_ref:?}; первое расхождение: {diff:?}");
    // Детерминизм самого стека: второй ход с нуля обязан повторить первый.
    // `SYN_PREFIX_KV_CHUNK=<n>` режет его префилл чанками по n — если с
    // чанком, равным длине префикса, он повторяет кэшированный ход, то
    // расхождение выше — только от формы GEMM, а не от самого кэша.
    if let Some(n) = std::env::var("SYN_PREFIX_KV_CHUNK").ok().and_then(|v| v.parse().ok()) {
        synaptix::facade::llm::set_prefill_chunk_size(n);
    }
    let mut fresh2b = Vec::new();
    {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.set_stop_tokens(tok.eos_ids().to_vec());
        r.generate_streaming(&ids2, &tok, |id, _| {
            fresh2b.push(id);
            true
        })
        .expect("fresh turn 2b");
    }
    let diff_ff = fresh2b.iter().zip(&fresh2).position(|(x, y)| x != y);
    let diff_cb = fresh2b.iter().zip(&cached2).position(|(x, y)| x != y);
    eprintln!(
        "два хода с нуля: первое расхождение {diff_ff:?}; второй против кэшированного: {diff_cb:?}"
    );
    assert!(
        diff.map_or(true, |d| d >= 4),
        "кэшированный ход разошёлся с полным префиллом с токена {diff:?}"
    );
}
