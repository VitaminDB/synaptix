//! Префикс-KV Inkling-2 для ходов с картинкой: второй ход (уточнение про ту
//! же картинку) берёт из кэша весь первый — промпт со строками картинки и
//! ответ; другая картинка при той же разметке кэш не берёт.
//!
//! `SYN_INKLING2_DIR=<бандл или HF-каталог> SYN_INKLING2_IMAGE=<картинка>
//! [SYN_INKLING2_IMAGE2=<другая>] cargo test -p synaptix --profile
//! fast-release --test inkling2_media_kv -- --nocapture`. Без переменных —
//! no-op.

use std::path::Path;

use synaptix::facade::llm::{load_llm_with_policy, Device, GenerationOptions, LlmGeneration, Message, QuantPolicy};

fn opts(ctx: usize, max_new: usize) -> GenerationOptions {
    GenerationOptions {
        max_new_tokens: max_new,
        max_seq_len: ctx,
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

#[test]
fn inkling2_media_turn_reuses_previous_turn() {
    let (Ok(path), Ok(image)) = (std::env::var("SYN_INKLING2_DIR"), std::env::var("SYN_INKLING2_IMAGE")) else {
        eprintln!("SYN_INKLING2_DIR / SYN_INKLING2_IMAGE не заданы — пропускаем");
        return;
    };
    let (model, tok) = load_llm_with_policy(Path::new(&path), QuantPolicy::balance(), &Device::Cuda(0)).expect("load");
    assert!(model.ensure_media_tower().expect("башня"), "в модели нет башни зрения");
    assert!(model.kv_session_media_ok());
    let img = model.encode_image(Path::new(&image), None).expect("картинка");
    // Предел ответа — во весь контекст, как у `synaptix chat`: ход дорастит
    // сессию (копия KV с картинкой в больший кэш).
    let (ctx, max_new) = (4096usize, 4096usize);
    let mut session = model.new_kv_session(ctx, max_new).expect("session").expect("префикс-KV");

    let mut msgs = vec![Message::user(format!("{}Что на картинке? Ответь одним предложением.", img.prompt_block))];
    let run = |session: &mut synaptix::facade::llm::LlmKvSession, msgs: &[Message], media: &[&synaptix::facade::llm::MediaEmbedding]| {
        let prompt = tok.apply_chat_template_reasoning(msgs, true, true, None, None).expect("template");
        let ids = tok.encode(&prompt).expect("encode");
        let mut out = Vec::new();
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.set_stop_tokens(tok.eos_ids().to_vec());
        let t = std::time::Instant::now();
        let reused = r
            .generate_streaming_cached_media(session, &ids, &tok, media, |id, _| {
                out.push(id);
                true
            })
            .expect("ход");
        (ids, out, reused, t.elapsed().as_millis())
    };
    let split = |out: &[u32]| {
        let mut p = synaptix::facade::chat::StreamParser::for_model(&tok, true, "");
        let (mut text, mut think) = (String::new(), String::new());
        let mut acc = Vec::new();
        let mut prev = String::new();
        for &id in out {
            acc.push(id);
            let s = tok.decode(&acc).unwrap_or_default();
            let delta = s.strip_prefix(prev.as_str()).unwrap_or("").to_string();
            prev = s;
            let (b, t, _) = p.feed(id, &delta);
            text += &b;
            think += &t;
        }
        (think, text)
    };

    let (ids1, out1, reused1, ms1) = run(&mut session, &msgs, &[&img]);
    let (think1, answer1) = split(&out1);
    println!("ход 1: промпт {} ток, из кэша {reused1}, {ms1} мс; ответ {answer1:?}", ids1.len());
    assert_eq!(reused1, 0);

    msgs.push(Message::assistant_turn(answer1, (!think1.is_empty()).then_some(think1), Vec::new()));
    msgs.push(Message::user("Какого цвета на ней неоновые вывески? Коротко."));
    let (ids2, out2, reused2, ms2) = run(&mut session, &msgs, &[&img]);
    let (_, answer2) = split(&out2);
    println!("ход 2: промпт {} ток, из кэша {reused2}, {ms2} мс; ответ {answer2:?}", ids2.len());
    assert_eq!(reused2, ids1.len() + out1.len() - 1, "весь прошлый ход должен браться из кэша");

    // Та же разметка, другая картинка — кэш не годится.
    if let Ok(other) = std::env::var("SYN_INKLING2_IMAGE2") {
        let img2 = model.encode_image(Path::new(&other), None).expect("картинка 2");
        let mut r = LlmGeneration::new(&model, opts(ctx, 4));
        let prompt = tok.apply_chat_template_reasoning(&msgs, true, true, None, None).unwrap();
        let ids = tok.encode(&prompt).unwrap();
        let reused = r.generate_streaming_cached_media(&mut session, &ids, &tok, &[&img2], |_, _| true).expect("ход 3");
        println!("ход 3 (другая картинка): из кэша {reused}");
        assert_eq!(reused, 0, "другая картинка при той же разметке не должна браться из кэша");
    }
}
