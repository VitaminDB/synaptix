//! Префикс-KV с картинкой в истории — для любой архитектуры с сессией и
//! башней зрения (гибрид Qwen3.6/3.8, Muse-Glimmer, Gemma-4, Qwen4Exp):
//! ход, продолжающий диалог с вложением, обязан переиспользовать посчитанный
//! префикс, а подмена картинки при тех же токенах — сбросить его.
//!
//! До 11.09.2026 медиа-промпт выключал сессию целиком; у Qwen4Exp это
//! поправлено первым (`qwen4exp_prefix_kv_media`), остальные архитектуры
//! до этого теста префиллили всю историю с картинкой на каждом ходу.
//!
//! Запуск (бандл любой из четырёх архитектур):
//! `SYN_PREFIX_KV_MEDIA_BUNDLE=… SYN_PREFIX_KV_MEDIA_IMAGE=/path/to.png
//! cargo test -p synaptix --release --test prefix_kv_media -- --nocapture
//! --test-threads=1` (для flash-next добавить `SYN_QWEN4EXP_EXPERT_CACHE_GB=6`).

use std::path::Path;

use synaptix::facade::llm::{
    load_llm_with_policy, optimal_profile, Device, GenerationOptions, LlmGeneration,
    MediaEmbedding, Message,
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

/// Хвост промпта следующего хода после текста ответа модели: рендерим
/// шаблон с ответом-маркером и берём то, что идёт за маркером. Так тест не
/// знает разметку хода конкретной архитектуры (ChatML у Qwen, `<|turn>` у
/// Gemma-4, каналы у Muse). Сам промпт хода 2 собирается из ТОКЕНОВ хода 1
/// плюс ответ плюс этот хвост — как в synthos: шаблон может отрисовать
/// прошлый ход иначе, чем выглядел его generation prompt (Gemma-4 опускает
/// пустой канал размышлений), и сравнивать надо не тексты, а кэш.
fn turn_tail(tok: &synaptix::facade::llm::LlmTokenizer, msgs: &[Message], question: &str) -> String {
    const MARK: &str = "QQXZ";
    let mut full = msgs.to_vec();
    full.push(Message::assistant(MARK));
    full.push(Message::user(question));
    let rendered = tok
        .apply_chat_template_ex_tools(&full, true, false, None)
        .expect("chat template (следующий ход)");
    let at = rendered
        .rfind(MARK)
        .unwrap_or_else(|| panic!("маркер ответа не найден в промпте:\n{rendered}"));
    rendered[at + MARK.len()..].to_string()
}

#[test]
fn prefix_kv_survives_media_in_history() {
    let (Ok(path), Ok(image)) = (
        std::env::var("SYN_PREFIX_KV_MEDIA_BUNDLE"),
        std::env::var("SYN_PREFIX_KV_MEDIA_IMAGE"),
    ) else {
        eprintln!("SYN_PREFIX_KV_MEDIA_BUNDLE / SYN_PREFIX_KV_MEDIA_IMAGE не заданы — пропускаем");
        return;
    };
    synaptix::facade::llm::cuda_release_kernel_caches();
    synaptix::facade::llm::cuda_trim_pool(0);
    let policy = optimal_profile(Path::new(&path)).policy;
    let (model, tok) =
        load_llm_with_policy(Path::new(&path), policy, &Device::Cuda(0)).expect("load");
    assert!(model.supports_media(), "бандл без башни зрения");
    assert!(model.kv_session_media_ok(), "архитектура без сессии с медиа");
    assert!(model.ensure_media_tower().expect("башня"), "башня не загрузилась");
    let media = model.encode_image(Path::new(&image), Some(512)).expect("encode_image");
    model.release_media_tower();
    eprintln!("картинка: {} vision-токенов", media.tokens());

    let msgs = [
        Message::system("Отвечай кратко, одним предложением."),
        Message::user(format!("{}Что изображено на картинке?", media.prompt_block)),
    ];
    let prompt1 = tok
        .apply_chat_template_ex_tools(&msgs, true, false, None)
        .expect("chat template");
    let ids1 = tok.encode(&prompt1).expect("encode");
    let tail2 = turn_tail(&tok, &msgs, "Назови главный цвет на ней одним словом.");

    let ctx = 8192usize;
    let max_new = 48usize;
    let mut session = model
        .new_kv_session(ctx, max_new)
        .expect("session")
        .expect("архитектура умеет префикс-KV");

    let eos = tok.eos_ids().to_vec();
    let run = |session: &mut _, ids: &[u32], media: &MediaEmbedding| -> (Vec<u32>, String, usize) {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.set_stop_tokens(eos.clone());
        let mut got = Vec::new();
        let mut text = String::new();
        let reused = r
            .generate_streaming_cached_media(session, ids, &tok, &[media], |id, s| {
                got.push(id);
                text.push_str(s);
                true
            })
            .expect("turn");
        (got, text, reused)
    };

    // Ход 1: полный префилл, точка возврата на конце промпта.
    let (mut a1, t1, reused1) = run(&mut session, &ids1, &media);
    eprintln!("ход 1: промпт {} ток, из кэша {reused1} → {t1:?}", ids1.len());
    assert_eq!(reused1, 0);
    assert!(!t1.trim().is_empty(), "модель ничего не сказала");
    // Стоп-токен хода уже стоит в хвосте шаблона.
    while a1.last().is_some_and(|t| eos.contains(t)) {
        a1.pop();
    }

    // Ход 2: та же история + ответ + новый вопрос — префикс обязан
    // переиспользоваться. У гибрида точка возврата стоит на границе,
    // кратной чанку GDN (64), — остаток хвоста промпта 1 считается заново.
    let mut ids2 = ids1.clone();
    ids2.extend_from_slice(&a1);
    ids2.extend(tok.encode(&tail2).expect("encode tail"));
    let (a2, t2, reused2) = run(&mut session, &ids2, &media);
    eprintln!("ход 2: промпт {} ток, из кэша {reused2} → {t2:?}", ids2.len());
    assert!(
        reused2 > 0 && reused2 <= ids1.len() && ids1.len() - reused2 < 64,
        "ход с картинкой в истории переиспользовал {reused2} из {} ток промпта 1",
        ids1.len()
    );
    assert!(!t2.trim().is_empty());

    // Эталон хода 2 без сессии: полный префилл по тем же эмбеддингам.
    let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
    r.set_stop_tokens(eos.clone());
    let mut a2_ref = Vec::new();
    let mut t2_ref = String::new();
    r.generate_streaming_media(&ids2, &tok, &[&media], |id, s| {
        a2_ref.push(id);
        t2_ref.push_str(s);
        true
    })
    .expect("fresh turn");
    let diff = a2.iter().zip(&a2_ref).position(|(x, y)| x != y);
    eprintln!("эталон без сессии → {t2_ref:?}; первое расхождение: {diff:?}");
    // Стек не бит-детерминирован, но начало ответа обязано совпасть.
    assert!(
        diff.map_or(true, |d| d >= 4),
        "кэшированный ход разошёлся с полным префиллом с токена {diff:?}"
    );

    // Ход 3: те же токены, но другая картинка на месте заполнителей —
    // префикс недействителен, кэш обязан это заметить.
    let other = MediaEmbedding {
        kind: media.kind,
        embeds: media.embeds.mul_scalar(0.5).expect("scale"),
        tokens: media.tokens,
        prompt_block: media.prompt_block.clone(),
        grid_hw: media.grid_hw,
        blocks: media.blocks,
    };
    let (_, t3, reused3) = run(&mut session, &ids2, &other);
    eprintln!("ход 3 (другая картинка): из кэша {reused3} → {t3:?}");
    assert_eq!(reused3, 0, "подменённая картинка не сбросила префикс");

    // Ход 4: снова исходная картинка и тот же промпт — после сброса точка
    // возврата стоит на конце промпта хода 3 с чужими строками, так что и
    // здесь полный префилл; а вот следующий за ним ход 5 обязан продолжить.
    let (mut a4, _, reused4) = run(&mut session, &ids2, &media);
    eprintln!("ход 4 (исходная картинка): из кэша {reused4}");
    assert_eq!(reused4, 0);
    while a4.last().is_some_and(|t| eos.contains(t)) {
        a4.pop();
    }
    let mut msgs5 = msgs.to_vec();
    msgs5.push(Message::assistant(&t1));
    msgs5.push(Message::user("Назови главный цвет на ней одним словом."));
    let tail5 = turn_tail(&tok, &msgs5, "А сколько на ней объектов?");
    let mut ids5 = ids2.clone();
    ids5.extend_from_slice(&a4);
    ids5.extend(tok.encode(&tail5).expect("encode tail 5"));
    let (_, t5, reused5) = run(&mut session, &ids5, &media);
    eprintln!("ход 5: промпт {} ток, из кэша {reused5} → {t5:?}", ids5.len());
    assert!(
        reused5 > 0 && reused5 <= ids2.len() && ids2.len() - reused5 < 64,
        "ход 5 переиспользовал {reused5} из {} ток промпта 4",
        ids2.len()
    );
}
