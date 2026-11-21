//! Префикс-KV Inkling-2 в чате: второй ход берёт из кэша весь первый (промпт
//! и ответ) и выдаёт те же токены, что ход с полным префиллом.
//!
//! Гоняется на HF-каталоге: `SYN_INKLING2_DIR=…/Inkling-2-Mini cargo test -p
//! synaptix --profile fast-release --test inkling2_prefix_kv -- --nocapture`.
//! Без переменной тест проходит как no-op (в CI без GPU и весов).
//!
//! Короткие свёртки назад не откатываются, поэтому кэш годен, только если
//! новый промпт продолжает посчитанное токен в токен. Шаблон Inkling-2
//! рендерит реплику ассистента с размышлениями ровно так, как модель её
//! сгенерировала, — это и проверяется: история из разобранного ответа
//! (размышления + текст) обязана дать те же токены.

use std::path::Path;

use synaptix::facade::llm::{load_llm_with_policy, Device, GenerationOptions, LlmGeneration, Message, QuantPolicy};

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

#[test]
fn inkling2_chat_turn_reuses_previous_turn() {
    let Ok(path) = std::env::var("SYN_INKLING2_DIR") else {
        eprintln!("SYN_INKLING2_DIR не задан — пропускаем");
        return;
    };
    let (model, tok) = load_llm_with_policy(Path::new(&path), QuantPolicy::balance(), &Device::Cuda(0)).expect("load");
    let single = |s: &str| {
        let ids = tok.encode(s).expect("encode");
        assert_eq!(ids.len(), 1, "{s} — один токен");
        ids[0]
    };
    let (message, thinking, text, end) =
        (single("<|message_model|>"), single("<|content_thinking|>"), single("<|content_text|>"), single("<|end_message|>"));
    let (ctx, max_new) = (4096usize, 400usize);

    let mut msgs = vec![Message::user("Сколько будет 2+2? Ответь одним числом.")];
    let prompt1 = tok.apply_chat_template_reasoning(&msgs, true, true, None, None).expect("template 1");
    let ids1 = tok.encode(&prompt1).expect("encode 1");

    let mut session = model.new_kv_session(ctx, max_new).expect("session").expect("Inkling-2 умеет префикс-KV");
    let mut out1 = Vec::new();
    let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
    r.set_stop_tokens(tok.eos_ids().to_vec());
    let reused = r
        .generate_streaming_cached(&mut session, &ids1, &tok, |id, _| {
            out1.push(id);
            true
        })
        .expect("turn 1");
    assert_eq!(reused, 0);
    assert!(tok.eos_ids().contains(out1.last().unwrap()), "ход 1 должен кончиться EOS");

    // Разбор ответа по спецтокенам — как `Splitter` в CLI.
    let (mut reasoning, mut answer, mut cur) = (Vec::new(), Vec::new(), None::<bool>);
    for &id in &out1 {
        if id == thinking {
            cur = Some(true);
        } else if id == text {
            cur = Some(false);
        } else if id == end || id == message {
            cur = None;
        } else if let Some(t) = cur {
            if t { reasoning.push(id) } else { answer.push(id) }
        }
    }
    let reasoning = tok.decode(&reasoning).unwrap();
    let answer = tok.decode(&answer).unwrap();
    println!("ход 1: размышления {reasoning:?}, ответ {answer:?}");
    assert!(answer.contains('4'), "{answer}");

    msgs.push(Message::assistant_turn(answer, (!reasoning.is_empty()).then_some(reasoning), Vec::new()));
    msgs.push(Message::user("А если умножить результат на 10? Ответь одним числом."));
    let prompt2 = tok.apply_chat_template_reasoning(&msgs, true, true, None, None).expect("template 2");
    let ids2 = tok.encode(&prompt2).expect("encode 2");
    let head: Vec<u32> = ids1.iter().chain(&out1).copied().collect();
    assert_eq!(&ids2[..head.len()], &head[..], "история не продолжает сгенерированное токен в токен");

    let mut cached2 = Vec::new();
    let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
    r.set_stop_tokens(tok.eos_ids().to_vec());
    let t = std::time::Instant::now();
    let reused = r
        .generate_streaming_cached(&mut session, &ids2, &tok, |id, _| {
            cached2.push(id);
            true
        })
        .expect("turn 2");
    let t_cached = t.elapsed().as_millis();
    // В кэше всё, кроме EOS прошлого хода: последний сэмплированный токен
    // через модель не прогонялся.
    assert_eq!(reused, head.len() - 1, "должен переиспользоваться весь прошлый ход");

    let mut fresh2 = Vec::new();
    let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
    r.set_stop_tokens(tok.eos_ids().to_vec());
    let t = std::time::Instant::now();
    r.generate_streaming(&ids2, &tok, |id, _| {
        fresh2.push(id);
        true
    })
    .expect("fresh turn 2");
    let t_fresh = t.elapsed().as_millis();
    println!(
        "ход 2: промпт {} ток, из кэша {reused}; {t_cached} мс с кэшем, {t_fresh} мс с нуля; ответ {:?}",
        ids2.len(),
        tok.decode(&cached2).unwrap_or_default()
    );
    // Нарезка префилла разная (103 + 19 против 122 одним куском), поэтому
    // логиты отличаются в младших разрядах и greedy-ничья через десятки
    // токенов может уйти иначе (замер: расхождение на ~60-м токене). Порча
    // кэша дала бы расхождение с первого токена.
    let same = cached2.iter().zip(&fresh2).take_while(|(a, b)| a == b).count();
    assert!(same >= 32.min(fresh2.len()), "с кэшем и с нуля расходятся уже на {same}-м токене");
    assert!(tok.decode(&cached2).unwrap_or_default().contains("40"));

    // Расхождение промпта: кэш честно сбрасывается.
    let other = tok
        .encode(&tok.apply_chat_template_reasoning(&[Message::user("Привет!")], true, true, None, None).unwrap())
        .unwrap();
    let mut r = LlmGeneration::new(&model, opts(ctx, 8));
    let reused = r.generate_streaming_cached(&mut session, &other, &tok, |_, _| true).expect("turn 3");
    assert_eq!(reused, 0, "при расхождении префикса переиспользовать нельзя");
}
